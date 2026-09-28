"""
Кандидатогенерация: инвертированный BM25 + dense (e5), склейка RRF.

rank_bm25.BM25Okapi на ~189k объявлений раздувает RAM до 10+ ГБ — не используем.
Эмбеддинги корпуса — float16 memmap (~145 МБ); dense — matmul на GPU при наличии CUDA,
иначе чанками на CPU (без FAISS).
"""

from __future__ import annotations

import gc
import math
import os
import time
from array import array
from collections import Counter, defaultdict
from typing import Any, Literal

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd
from tqdm import tqdm

from .config import (
    BM25_WEIGHT,
    CACHE_DIR,
    CANDIDATE_N,
    CLICK_PER_TOKEN,
    DENSE_WEIGHT,
    EMBED_BATCH,
    EMBED_DESC_CHARS,
    EMBED_MAX_SEQ,
    EMBEDDINGS_PATH,
    HISTORY_TOP_N,
    ITEM_IDS_CACHE,
    ITEMS_PATH,
    MAX_DESC_CHARS,
    MAX_ITEM_TOKENS,
    MODEL_DIR,
    MODEL_NAME,
    QUERY_BATCH,
    RRF_K,
    TOP_K,
)
from .metrics import dedupe_preserve
from .text_utils import (
    e5_passage_text,
    e5_query_text,
    item_search_text,
    min_rating_from_params,
    tokenize,
)

BM25_K1 = 1.5
BM25_B = 0.75
LOC_BOOST = 0.8

Mode = Literal["hybrid", "bm25", "dense"]


def model_weights_ready() -> bool:
    return (MODEL_DIR / "model.safetensors").exists() or (MODEL_DIR / "pytorch_model.bin").exists()


def _rss_mb() -> float:
    """Грубый пик RSS процесса (Windows: WorkingSet)."""
    try:
        import psutil

        return psutil.Process().memory_info().rss / (1024 * 1024)
    except Exception:
        try:
            import ctypes

            class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ("cb", ctypes.c_ulong),
                    ("PageFaultCount", ctypes.c_ulong),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            pmc = PROCESS_MEMORY_COUNTERS()
            pmc.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
            ctypes.windll.psapi.GetProcessMemoryInfo(
                ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb
            )
            return pmc.WorkingSetSize / (1024 * 1024)
        except Exception:
            return -1.0


class HybridRetriever:
    """BM25 по категории + dense e5 (CPU/GPU), склейка RRF до 50 id."""

    def __init__(self, items: pd.DataFrame):
        items = items.reset_index(drop=True)
        self.n_items = len(items)
        self.item_ids = items["item_id"].astype(str).str.lower().tolist()
        self._item_id_set = set(self.item_ids)

        self.locations = (
            pd.to_numeric(items["item_location_id"], errors="coerce").fillna(-1).astype(np.int32).to_numpy()
        )
        self.ratings = (
            pd.to_numeric(items["item_rating"], errors="coerce").fillna(0).astype(np.float32).to_numpy()
        )
        self.categories = (
            pd.to_numeric(items["item_category_id"], errors="coerce").fillna(-1).astype(np.int32).to_numpy()
        )

        cat_buf: dict[int, array] = defaultdict(lambda: array("I"))
        for i, cat in enumerate(self.categories):
            cat_buf[int(cat)].append(i)
        self._cat_to_idx = {c: np.array(buf, dtype=np.int32) for c, buf in cat_buf.items()}
        del cat_buf

        self._build_inverted_index(items)
        del items
        gc.collect()

        self._emb: np.ndarray | None = None
        self._emb_gpu = None  # torch.Tensor на GPU, нормализованный fp16
        self._encoder = None
        self._device = "cpu"
        self._history: dict[str, list[int]] = {}
        self._click_items: dict[str, np.ndarray] = {}
        self._click_counts: dict[str, np.ndarray] = {}
        self.mode: Mode = "hybrid"
        self.bm25_weight = BM25_WEIGHT
        self.dense_weight = DENSE_WEIGHT
        self.query_with_params = True

    def _build_inverted_index(self, items: pd.DataFrame) -> None:
        n = self.n_items
        post_ids: dict[str, array] = defaultdict(lambda: array("I"))
        post_tf: dict[str, array] = defaultdict(lambda: array("H"))
        title_ids: dict[str, array] = defaultdict(lambda: array("I"))
        doc_len = np.zeros(n, dtype=np.int32)
        chunk = 15000

        title_stem = [""] * n
        titles = items["item_title_raw"]
        descs = items["item_description_raw"]
        params = items["item_infm_params_text"]

        for start in tqdm(range(0, n, chunk), desc="индекс BM25"):
            end = min(start + chunk, n)
            t_chunk = titles.iloc[start:end].fillna("").astype(str).tolist()
            d_chunk = descs.iloc[start:end].fillna("").astype(str).tolist()
            p_chunk = params.iloc[start:end].fillna("").astype(str).tolist()
            for offset, (title, desc, par) in enumerate(zip(t_chunk, d_chunk, p_chunk)):
                gi = start + offset
                title_toks = tokenize(title)
                title_stem[gi] = " ".join(title_toks)
                for term in set(title_toks):
                    title_ids[term].append(gi)
                text = item_search_text(title, desc, par, MAX_DESC_CHARS)
                toks = tokenize(text, max_tokens=MAX_ITEM_TOKENS)
                doc_len[gi] = max(len(toks), 1)
                for term, tf in Counter(toks).items():
                    post_ids[term].append(gi)
                    post_tf[term].append(min(int(tf), 65535))
            del t_chunk, d_chunk, p_chunk

        self._doc_len = doc_len
        self._avgdl = float(doc_len.mean()) if n else 1.0
        self._post_ids = {t: np.asarray(buf, dtype=np.int32) for t, buf in post_ids.items()}
        self._post_tf = {t: np.asarray(buf, dtype=np.uint16) for t, buf in post_tf.items()}
        self._title_ids = {t: np.asarray(buf, dtype=np.int32) for t, buf in title_ids.items()}
        self._title_stem = title_stem
        del post_ids, post_tf, title_ids
        gc.collect()

    def fit_query_history(self, train: pd.DataFrame, top_n: int = HISTORY_TOP_N) -> None:
        """
        Клики из train, у которых item есть в корпусе.
        Точный текст запроса и «это слово уже кликали». query_id бенчмарка не зашивается.
        """
        t = train.copy()
        t["item_id"] = t["item_id"].astype(str).str.lower()
        t = t[t["item_id"].isin(self._item_id_set)]
        if t.empty:
            self._history = {}
            self._click_items = {}
            self._click_counts = {}
            return

        id_to_idx = {iid: i for i, iid in enumerate(self.item_ids)}
        t["idx"] = t["item_id"].map(id_to_idx).astype(np.int32)
        t["qkey"] = t["search_query"].map(lambda s: " ".join(tokenize(str(s))))

        vc = t.groupby(["qkey", "idx"], sort=False).size().rename("c").reset_index()
        vc = vc[vc["qkey"] != ""]
        vc = vc.sort_values(["qkey", "c"], ascending=[True, False])
        vc = vc.groupby("qkey", sort=False).head(top_n)
        self._history = vc.groupby("qkey")["idx"].apply(lambda s: s.astype(int).tolist()).to_dict()

        buckets: dict[str, Counter] = defaultdict(Counter)
        for qkey, idx in zip(t["qkey"].tolist(), t["idx"].tolist()):
            if not qkey:
                continue
            for tok in set(qkey.split()):
                buckets[tok][int(idx)] += 1

        click_items: dict[str, np.ndarray] = {}
        click_counts: dict[str, np.ndarray] = {}
        for tok, counter in buckets.items():
            top = counter.most_common(CLICK_PER_TOKEN)
            click_items[tok] = np.asarray([i for i, _ in top], dtype=np.int32)
            click_counts[tok] = np.asarray([c for _, c in top], dtype=np.float32)
        self._click_items = click_items
        self._click_counts = click_counts
        del t, vc, buckets
        gc.collect()

    def fit_bm25(self) -> None:
        return

    def _candidate_indices(self, query_row: dict[str, Any]) -> np.ndarray:
        try:
            cat = int(query_row.get("search_category", -1))
        except (TypeError, ValueError):
            cat = -1
        # в бенчмарке search_category=0 — нет фильтра, такой категории в корпусе нет
        if cat <= 0:
            idxs = np.arange(self.n_items, dtype=np.int32)
        else:
            idxs = self._cat_to_idx.get(cat)
            if idxs is None or len(idxs) == 0:
                idxs = np.arange(self.n_items, dtype=np.int32)

        min_rating = min_rating_from_params(query_row.get("search_infm_params_text"))
        if min_rating is None:
            return idxs
        filtered = idxs[self.ratings[idxs] >= min_rating]
        return filtered if len(filtered) else idxs

    def _query_terms(self, query_row: dict[str, Any]) -> tuple[list[str], str]:
        main = tokenize(str(query_row.get("search_query") or ""))
        extra = [
            t
            for t in tokenize(str(query_row.get("search_infm_params_text") or ""))
            if t not in main
        ]
        terms = main + extra
        uniq: list[str] = []
        seen: set[str] = set()
        for t in terms:
            if t not in seen:
                uniq.append(t)
                seen.add(t)
        return uniq, " ".join(main)

    def _bm25_rank(self, query_row: dict[str, Any], n: int) -> list[int]:
        idxs = self._candidate_indices(query_row)
        q_tokens, q_phrase = self._query_terms(query_row)
        if len(idxs) == 0:
            return []
        if not q_tokens:
            return idxs[:n].tolist()

        allowed = np.zeros(self.n_items, dtype=bool)
        allowed[idxs] = True
        scores = np.zeros(self.n_items, dtype=np.float32)
        doc_cov = np.zeros(self.n_items, dtype=np.int8)
        title_cov = np.zeros(self.n_items, dtype=np.int8)
        avgdl = self._avgdl
        k1, b = BM25_K1, BM25_B

        for term in q_tokens:
            post = self._post_ids.get(term)
            if post is None:
                continue
            n_t = len(post)
            idf = math.log((self.n_items - n_t + 0.5) / (n_t + 0.5) + 1.0)
            tfs = self._post_tf[term]
            mask = allowed[post]
            if np.any(mask):
                docs = post[mask]
                tf = tfs[mask].astype(np.float32)
                dl = self._doc_len[docs].astype(np.float32)
                denom = tf + k1 * (1.0 - b + b * dl / avgdl)
                scores[docs] += np.float32(idf) * (tf * (k1 + 1.0) / denom)
                doc_cov[docs] += 1

            tpost = self._title_ids.get(term)
            if tpost is not None:
                tmask = allowed[tpost]
                if np.any(tmask):
                    title_cov[tpost[tmask]] += 1

            c_items = self._click_items.get(term)
            if c_items is not None:
                cmask = allowed[c_items]
                if np.any(cmask):
                    bump = np.float32(idf) * np.log1p(self._click_counts[term][cmask]) * np.float32(1.1)
                    scores[c_items[cmask]] += np.minimum(bump, np.float32(3.5))

        nq = len(q_tokens)
        scores += title_cov.astype(np.float32) * np.float32(2.2)
        scores += doc_cov.astype(np.float32) * np.float32(0.6)
        full_title = np.flatnonzero(title_cov == nq)
        if len(full_title):
            scores[full_title] += np.float32(6.0)
            if nq >= 2 and q_phrase:
                for gi in full_title.tolist():
                    if q_phrase in self._title_stem[gi]:
                        scores[gi] += 4.0
        full_doc = np.flatnonzero((doc_cov == nq) & (title_cov < nq))
        if len(full_doc):
            scores[full_doc] += np.float32(2.0)

        loc = query_row.get("search_location_id")
        if loc is not None and not (isinstance(loc, float) and np.isnan(loc)):
            try:
                loc_i = int(loc)
            except (TypeError, ValueError):
                loc_i = None
            if loc_i is not None:
                hit = idxs[(self.locations[idxs] == loc_i) & (scores[idxs] > 0)]
                if len(hit):
                    scores[hit] += np.float32(LOC_BOOST)

        for gi in self._history.get(q_phrase, ()):
            if allowed[gi]:
                scores[gi] += np.float32(14.0)

        sub = scores[idxs]
        take = min(n, len(idxs))
        if take <= 0:
            return []
        if take == len(idxs):
            order = np.argsort(-sub, kind="mergesort")
        else:
            part = np.argpartition(sub, -take)[-take:]
            order = part[np.argsort(-sub[part], kind="mergesort")]
        return idxs[order].tolist()

    @classmethod
    def _torch_device(cls) -> str:
        """CUDA только если доступна; иначе CPU (офлайн-инференс без GPU)."""
        force_cpu = os.environ.get("AVITO_FORCE_CPU", "").strip().lower() in ("1", "true", "yes")
        if force_cpu:
            return "cpu"
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
        except Exception:
            pass
        return "cpu"

    def _resolve_device(self) -> str:
        return self._torch_device()

    def _load_encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer

            self._device = self._resolve_device()
            if MODEL_DIR.exists() and model_weights_ready():
                # локальный кэш — без сети
                os.environ.setdefault("HF_HUB_OFFLINE", "1")
                self._encoder = SentenceTransformer(str(MODEL_DIR), device=self._device)
            else:
                os.environ.pop("HF_HUB_OFFLINE", None)
                print(f"Качаю модель {MODEL_NAME}...")
                self._encoder = SentenceTransformer(MODEL_NAME, device=self._device)
                CACHE_DIR.mkdir(parents=True, exist_ok=True)
                MODEL_DIR.mkdir(parents=True, exist_ok=True)
                self._encoder.save(str(MODEL_DIR))
                print(f"Модель сохранена в {MODEL_DIR}")
            self._encoder.max_seq_length = EMBED_MAX_SEQ
            try:
                import torch

                torch.set_num_threads(2)
            except Exception:
                pass
            print(f"Энкодер на {self._device}")
        return self._encoder

    def unload_encoder(self) -> None:
        if self._encoder is not None:
            try:
                import torch

                del self._encoder
                self._encoder = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                self._encoder = None
        gc.collect()

    def _cache_ok(self) -> bool:
        if not (EMBEDDINGS_PATH.exists() and ITEM_IDS_CACHE.exists()):
            return False
        cached_ids = ITEM_IDS_CACHE.read_text(encoding="utf-8").splitlines()
        return cached_ids == self.item_ids

    def _load_emb_cache(self) -> bool:
        if not self._cache_ok():
            return False
        self._emb = np.load(EMBEDDINGS_PATH, mmap_mode="r")
        print(f"Кэш эмбеддингов: {self._emb.shape}, dtype={self._emb.dtype}")
        return True

    def _upload_emb_gpu(self) -> None:
        """Векторы на GPU один раз — дальше matmul без FAISS (~145 МБ fp16)."""
        if self._emb is None:
            return
        try:
            import torch

            if not torch.cuda.is_available():
                self._emb_gpu = None
                return
            # читаем memmap кусками, чтобы не держать float32-копию всего корпуса в RAM
            n, dim = self._emb.shape
            buf = np.empty((n, dim), dtype=np.float16)
            chunk = 20000
            for start in range(0, n, chunk):
                end = min(start + chunk, n)
                buf[start:end] = np.asarray(self._emb[start:end], dtype=np.float16)
            self._emb_gpu = torch.from_numpy(buf).to("cuda", non_blocking=True)
            del buf
            gc.collect()
            print(f"Эмбеддинги на GPU: {tuple(self._emb_gpu.shape)}")
        except Exception as e:
            print(f"GPU upload не удался ({e}), dense на CPU")
            self._emb_gpu = None

    def fit_embeddings(self, use_cache: bool = True, items_path=None) -> bool:
        """
        Подхватывает fp16-кэш или считает эмбеддинги потоково из parquet.
        Кодировщик после прохода выгружается.
        """
        if use_cache and self._load_emb_cache():
            self._upload_emb_gpu()
            return True

        path = items_path or ITEMS_PATH
        print(f"Считаю эмбеддинги корпуса ({MODEL_NAME}), fp16 -> {EMBEDDINGS_PATH}")
        t0 = time.time()
        peak = _rss_mb()

        encoder = self._load_encoder()
        dim = int(encoder.get_sentence_embedding_dimension())
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

        # пишем сразу float16 — меньше диск и RAM при mmap
        out = np.lib.format.open_memmap(
            EMBEDDINGS_PATH, mode="w+", dtype=np.float16, shape=(self.n_items, dim)
        )

        cols = ["item_id", "item_title_raw", "item_description_raw", "item_infm_params_text"]
        chunk = 4096
        written = 0
        import pyarrow.parquet as pq

        pf = pq.ParquetFile(path)
        # порядок строк parquet должен совпасть с load_items(); сверяем item_id на лету
        for batch in tqdm(
            pf.iter_batches(batch_size=chunk, columns=cols),
            total=(self.n_items + chunk - 1) // chunk,
            desc="эмбеддинги",
        ):
            sl = batch.to_pandas()
            ids = sl["item_id"].astype(str).str.lower().tolist()
            expect = self.item_ids[written : written + len(ids)]
            if ids != expect:
                raise RuntimeError(
                    f"порядок item_id в parquet разъехался на offset={written}"
                )
            texts = [
                e5_passage_text(t, d, p, EMBED_DESC_CHARS)
                for t, d, p in zip(
                    sl["item_title_raw"].fillna("").tolist(),
                    sl["item_description_raw"].fillna("").tolist(),
                    sl["item_infm_params_text"].fillna("").tolist(),
                )
            ]
            emb = encoder.encode(
                texts,
                batch_size=EMBED_BATCH,
                show_progress_bar=False,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
            end = written + len(ids)
            out[written:end] = np.asarray(emb, dtype=np.float16)
            written = end
            cur = _rss_mb()
            if cur > peak:
                peak = cur
            del texts, emb, sl, batch
            gc.collect()

        if written != self.n_items:
            raise RuntimeError(f"записано {written} эмбеддингов, ожидали {self.n_items}")

        out.flush()
        del out
        ITEM_IDS_CACHE.write_text("\n".join(self.item_ids), encoding="utf-8")
        self.unload_encoder()
        self._emb = np.load(EMBEDDINGS_PATH, mmap_mode="r")
        self._upload_emb_gpu()
        elapsed = time.time() - t0
        print(
            f"Эмбеддинги готовы: {written} x {dim} fp16, "
            f"{elapsed:.1f} с, пик RSS ~{peak:.0f} МБ"
        )
        gc.collect()
        return True

    def encode_from_texts(self, texts: list[str], batch_size: int = EMBED_BATCH) -> None:
        """Совместимость: тексты уже с префиксом passage:."""
        encoder = self._load_encoder()
        dim = int(encoder.get_sentence_embedding_dimension())
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        out = np.lib.format.open_memmap(
            EMBEDDINGS_PATH, mode="w+", dtype=np.float16, shape=(len(texts), dim)
        )
        for start in tqdm(range(0, len(texts), batch_size), desc="эмбеддинги"):
            batch = texts[start : start + batch_size]
            emb = encoder.encode(
                batch,
                batch_size=len(batch),
                show_progress_bar=False,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
            out[start : start + len(batch)] = np.asarray(emb, dtype=np.float16)
        out.flush()
        ITEM_IDS_CACHE.write_text("\n".join(self.item_ids), encoding="utf-8")
        del out
        self.unload_encoder()
        self._emb = np.load(EMBEDDINGS_PATH, mmap_mode="r")
        self._upload_emb_gpu()
        gc.collect()

    def _encode_queries(self, rows: list[dict[str, Any]]) -> np.ndarray:
        encoder = self._load_encoder()
        if self.query_with_params:
            texts = [e5_query_text(r) for r in rows]
        else:
            texts = [
                e5_query_text(
                    {"search_query": r.get("search_query"), "search_infm_params_text": ""}
                )
                for r in rows
            ]
        vecs = encoder.encode(
            texts,
            batch_size=min(QUERY_BATCH, max(1, len(texts))),
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return np.asarray(vecs, dtype=np.float32)

    def _semantic_from_vecs(
        self, q_vecs: np.ndarray, rows: list[dict[str, Any]], n: int
    ) -> list[list[int]]:
        """Косинус = dot product (векторы уже L2-нормированы). Без FAISS — матрица на GPU."""
        assert self._emb is not None
        import torch

        results: list[list[int]] = [[] for _ in rows]
        take = n

        if self._emb_gpu is not None:
            emb = self._emb_gpu  # [N, D] fp16
            q_t = torch.from_numpy(q_vecs).to(emb.device, dtype=emb.dtype)
            # [B, N]
            sims = q_t @ emb.T
            # категория/рейтинг: маскируем недопустимых кандидатов очень низким скором
            for j, row in enumerate(rows):
                idxs = self._candidate_indices(row)
                if len(idxs) == 0:
                    results[j] = []
                    continue
                if len(idxs) < self.n_items:
                    mask = torch.full((self.n_items,), -1e4, device=emb.device, dtype=sims.dtype)
                    mask[torch.as_tensor(idxs, device=emb.device, dtype=torch.long)] = 0
                    scores = sims[j] + mask
                else:
                    scores = sims[j]
                k = min(take, scores.numel())
                topv, topi = torch.topk(scores, k=k)
                results[j] = [int(i) for i in topi.detach().cpu().tolist()]
            del sims, q_t
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return results

        # CPU fallback: чанками
        for j, row in enumerate(rows):
            idxs = self._candidate_indices(row)
            if len(idxs) == 0:
                results[j] = []
                continue
            q = q_vecs[j].astype(np.float32, copy=False)
            take_pool = min(take, len(idxs))
            best_s = np.full(take_pool, -1e9, dtype=np.float32)
            best_i = np.full(take_pool, -1, dtype=np.int32)
            chunk = 8192
            for start in range(0, len(idxs), chunk):
                sl = idxs[start : start + chunk]
                mat = np.asarray(self._emb[sl], dtype=np.float32)
                sims = mat @ q
                k = min(take_pool, len(sims))
                part = np.argpartition(sims, -k)[-k:]
                cand_s = np.concatenate([best_s, sims[part]])
                cand_i = np.concatenate([best_i, sl[part]])
                keep = np.argpartition(cand_s, -take_pool)[-take_pool:]
                best_s = cand_s[keep]
                best_i = cand_i[keep]
            order = np.argsort(-best_s)
            results[j] = [int(i) for i in best_i[order] if i >= 0][:take]
        return results

    def _semantic_rank(self, query_row: dict[str, Any], n: int) -> list[int]:
        if self._emb is None:
            return []
        q_vecs = self._encode_queries([query_row])
        return self._semantic_from_vecs(q_vecs, [query_row], n)[0]

    @staticmethod
    def _merge_rrf(
        bm25_ids: list[str],
        sem_ids: list[str],
        k: int,
        rrf_k: int = RRF_K,
        bm25_w: float = 1.0,
        dense_w: float = 1.0,
    ) -> list[str]:
        """
        Reciprocal Rank Fusion: score = w/(rrf_k+rank).
        Ранги независимы, веса позволяют усилить densе без переобучения скоров.
        """
        scores: dict[str, float] = {}
        for rank, iid in enumerate(bm25_ids):
            scores[iid] = scores.get(iid, 0.0) + bm25_w / (rrf_k + rank + 1)
        for rank, iid in enumerate(sem_ids):
            scores[iid] = scores.get(iid, 0.0) + dense_w / (rrf_k + rank + 1)
        ordered = sorted(scores.keys(), key=lambda x: (-scores[x], x))
        return dedupe_preserve(ordered)[:k]

    # старое имя на случай импортов из ноутбука
    _merge_hybrid = _merge_rrf

    def _apply_history(self, query_row: dict[str, Any], ids: list[str], k: int) -> list[str]:
        return ids[:k]

    def retrieve_one(self, query_row: dict[str, Any], k: int = TOP_K) -> list[str]:
        mode = self.mode
        need_bm25 = mode in ("hybrid", "bm25")
        need_dense = mode in ("hybrid", "dense") and self._emb is not None

        bm25_ids: list[str] = []
        sem_ids: list[str] = []
        if need_bm25:
            bm25_g = self._bm25_rank(query_row, n=CANDIDATE_N)
            bm25_ids = [self.item_ids[i] for i in bm25_g]
        if need_dense:
            sem_g = self._semantic_rank(query_row, n=CANDIDATE_N)
            sem_ids = [self.item_ids[i] for i in sem_g]

        if mode == "bm25" or not sem_ids:
            merged = dedupe_preserve(bm25_ids)[:k]
        elif mode == "dense" or not bm25_ids:
            merged = dedupe_preserve(sem_ids)[:k]
        else:
            merged = self._merge_rrf(
                bm25_ids, sem_ids, k, RRF_K, self.bm25_weight, self.dense_weight
            )
        return self._apply_history(query_row, merged, k)

    def retrieve_batch(
        self, queries: pd.DataFrame, id_col: str = "query_id", query_batch: int = QUERY_BATCH
    ) -> dict[str, list[str]]:
        rows = [row.to_dict() for _, row in queries.iterrows()]
        qids = [str(row[id_col]) for row in rows]
        mode = self.mode
        need_bm25 = mode in ("hybrid", "bm25")
        need_dense = mode in ("hybrid", "dense") and self._emb is not None

        bm25_lists: list[list[str]] = [[] for _ in rows]
        if need_bm25:
            for i, row in enumerate(tqdm(rows, desc="bm25")):
                g_idx = self._bm25_rank(row, n=CANDIDATE_N)
                bm25_lists[i] = [self.item_ids[j] for j in g_idx]

        sem_lists: list[list[str]] = [[] for _ in rows]
        if need_dense:
            for start in tqdm(range(0, len(rows), query_batch), desc="dense"):
                batch_rows = rows[start : start + query_batch]
                q_vecs = self._encode_queries(batch_rows)
                ranked = self._semantic_from_vecs(q_vecs, batch_rows, CANDIDATE_N)
                for j, g_idx in enumerate(ranked):
                    sem_lists[start + j] = [self.item_ids[i] for i in g_idx]
            self.unload_encoder()

        out: dict[str, list[str]] = {}
        for row, qid, bm25_ids, sem_ids in zip(rows, qids, bm25_lists, sem_lists):
            if mode == "bm25" or not sem_ids:
                merged = dedupe_preserve(bm25_ids)[:TOP_K]
            elif mode == "dense" or not bm25_ids:
                merged = dedupe_preserve(sem_ids)[:TOP_K]
            else:
                merged = self._merge_rrf(
                    bm25_ids, sem_ids, TOP_K, RRF_K, self.bm25_weight, self.dense_weight
                )
            out[qid] = self._apply_history(row, merged, TOP_K)
        return out


def make_query_key(row) -> str:
    if isinstance(row, pd.Series):
        getter = row.get
    else:
        getter = row.get
    parts = [
        str(getter("search_query", "") or ""),
        str(getter("search_location_id", "") or ""),
        str(getter("search_category", "") or ""),
        str(getter("search_infm_params_text", "") or ""),
        str(getter("search_is_delivery_search", "") or ""),
    ]
    return "|".join(parts)


def make_query_key_frame(df: pd.DataFrame) -> pd.Series:
    return (
        df["search_query"].fillna("").astype(str)
        + "|"
        + df["search_location_id"].fillna("").astype(str)
        + "|"
        + df["search_category"].fillna("").astype(str)
        + "|"
        + df["search_infm_params_text"].fillna("").astype(str)
        + "|"
        + df["search_is_delivery_search"].fillna("").astype(str)
    )


def build_validation_split(
    train: pd.DataFrame,
    seed: int,
    max_val_queries: int,
    corpus: set[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, set[str]]]:
    t = train.copy()
    t["item_id"] = t["item_id"].astype(str).str.lower()
    if corpus is not None:
        t = t[t["item_id"].isin(corpus)]
    t["query_key"] = make_query_key_frame(t)

    keys = t["query_key"].drop_duplicates().sample(frac=1.0, random_state=seed)
    n_val = min(max_val_queries, max(1, int(len(keys) * 0.2)))
    val_keys = set(keys.iloc[:n_val].tolist())

    val_rows = t[t["query_key"].isin(val_keys)]
    relevant: dict[str, set[str]] = (
        val_rows.groupby("query_key")["item_id"].apply(lambda s: set(s.tolist())).to_dict()
    )
    relevant = {k: v for k, v in relevant.items() if v}
    val_queries = (
        val_rows[val_rows["query_key"].isin(relevant)]
        .drop_duplicates("query_key")
        .rename(columns={"query_key": "query_id"})
    )
    del t
    gc.collect()
    return val_queries, relevant

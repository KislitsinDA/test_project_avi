"""
Пайплайн: локальный Recall@50 и answer.csv.

Запуск из папки avito:
    python run_solution.py --validate
    python run_solution.py --validate --semantic
    python run_solution.py --submit --semantic
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.config import (  # noqa: E402
    ANSWER_PATH,
    BM25_WEIGHT,
    DENSE_WEIGHT,
    ITEM_COLS,
    ITEMS_PATH,
    MAX_VAL_QUERIES,
    QUERIES_PATH,
    RANDOM_SEED,
    TRAIN_COLS,
    TRAIN_PATH,
)
from src.metrics import recall_at_k  # noqa: E402
from src.retriever import HybridRetriever, build_validation_split  # noqa: E402
from src.text_utils import norm_text  # noqa: E402
from src.validate_submission import validate_answer_csv  # noqa: E402


def load_items() -> pd.DataFrame:
    items = pd.read_parquet(ITEMS_PATH, columns=ITEM_COLS)
    items["item_id"] = items["item_id"].astype(str).str.lower()
    return items


def load_queries() -> pd.DataFrame:
    q = pd.read_parquet(QUERIES_PATH)
    q["query_id"] = q["query_id"].astype(str)
    return q


def load_train() -> pd.DataFrame:
    train = pd.read_parquet(TRAIN_PATH, columns=TRAIN_COLS)
    train["item_id"] = train["item_id"].astype(str).str.lower()
    return train


def build_retriever(
    items: pd.DataFrame,
    use_semantic: bool = False,
    mode: str = "hybrid",
    bm25_weight: float | None = None,
    dense_weight: float | None = None,
    query_with_params: bool = True,
) -> HybridRetriever:
    print("Строю BM25-индекс по обрезанным текстам...")
    retriever = HybridRetriever(items)
    retriever.mode = mode  # type: ignore[assignment]
    if bm25_weight is not None:
        retriever.bm25_weight = bm25_weight
    if dense_weight is not None:
        retriever.dense_weight = dense_weight
    retriever.query_with_params = query_with_params
    if use_semantic or mode == "dense":
        # кэш или пересчёт — энкодер выгружается внутри
        ok = retriever.fit_embeddings(use_cache=True, items_path=ITEMS_PATH)
        if not ok:
            raise RuntimeError("не удалось подготовить эмбеддинги")
    else:
        print("Семантика выключена. Включить: --semantic")
    return retriever


def run_validation(retriever: HybridRetriever) -> float:
    print(f"Hold-out Recall@50, до {MAX_VAL_QUERIES} запросов (тексты запроса вырезаны из train)...")
    t0 = time.time()
    train = load_train()
    val_queries, relevant = build_validation_split(
        train,
        seed=RANDOM_SEED,
        max_val_queries=MAX_VAL_QUERIES,
        corpus=retriever._item_id_set,
    )
    val_texts = set(val_queries["search_query"].fillna("").map(norm_text))
    keep = ~train["search_query"].fillna("").map(norm_text).isin(val_texts)
    retriever.fit_query_history(train.loc[keep])
    del train
    gc.collect()
    preds = retriever.retrieve_batch(val_queries, id_col="query_id")
    score = recall_at_k(preds, relevant, k=50)
    print(
        f"Recall@50 на hold-out ({len(relevant)} запросов, mode={retriever.mode}, "
        f"bm25_w={retriever.bm25_weight}, dense_w={retriever.dense_weight}): "
        f"{score:.4f}  ({time.time() - t0:.1f} с)"
    )
    return score


def run_compare(retriever: HybridRetriever) -> dict[str, float]:
    """Один hold-out: bm25, dense, hybrid (нужны эмбеддинги)."""
    print(f"Сравнение режимов, до {MAX_VAL_QUERIES} запросов hold-out...")
    t0 = time.time()
    train = load_train()
    val_queries, relevant = build_validation_split(
        train,
        seed=RANDOM_SEED,
        max_val_queries=MAX_VAL_QUERIES,
        corpus=retriever._item_id_set,
    )
    val_texts = set(val_queries["search_query"].fillna("").map(norm_text))
    keep = ~train["search_query"].fillna("").map(norm_text).isin(val_texts)
    retriever.fit_query_history(train.loc[keep])
    del train
    gc.collect()

    scores: dict[str, float] = {}
    for mode in ("bm25", "dense", "hybrid"):
        retriever.mode = mode  # type: ignore[assignment]
        tm = time.time()
        preds = retriever.retrieve_batch(val_queries, id_col="query_id")
        scores[mode] = recall_at_k(preds, relevant, k=50)
        print(
            f"  {mode:6s} Recall@50 = {scores[mode]:.4f}  "
            f"(bm25_w={retriever.bm25_weight}, dense_w={retriever.dense_weight}, "
            f"{time.time() - tm:.1f} с)"
        )
    print(f"Сравнение заняло {time.time() - t0:.1f} с")
    return scores


def run_submit(retriever: HybridRetriever, items: pd.DataFrame) -> None:
    queries = load_queries()
    preds = retriever.retrieve_batch(queries, id_col="query_id")

    rows = []
    for qid in queries["query_id"].astype(str):
        top = preds.get(qid, [])
        rows.append({"query_id": qid, "answer": " ".join(top)})

    answer = pd.DataFrame(rows)
    answer.to_csv(ANSWER_PATH, index=False, encoding="utf-8")
    print(f"Сохранено: {ANSWER_PATH} ({len(answer)} строк)")

    errs = validate_answer_csv(str(ANSWER_PATH), queries, items)
    if errs:
        print("Проверка формата: есть проблемы:")
        for e in errs[:20]:
            print(" -", e)
        if len(errs) > 20:
            print(f" ... ещё {len(errs) - 20}")
    else:
        print("Проверка формата: OK")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Recall@50 на hold-out для bm25, dense и hybrid (нужен --semantic)",
    )
    parser.add_argument(
        "--semantic",
        action="store_true",
        help="dense e5 + RRF (кэш эмбеддингов или пересчёт)",
    )
    parser.add_argument(
        "--mode",
        choices=["hybrid", "bm25", "dense"],
        default=None,
        help="hybrid=RRF, bm25, dense; по умолчанию hybrid при --semantic иначе bm25",
    )
    parser.add_argument("--bm25-weight", type=float, default=None)
    parser.add_argument("--dense-weight", type=float, default=None)
    parser.add_argument(
        "--no-query-params",
        action="store_true",
        help="в e5-запрос не класть search_infm_params_text",
    )
    parser.add_argument(
        "--bm25-only",
        action="store_true",
        help="устаревший флаг, BM25 и так режим без --semantic",
    )
    args = parser.parse_args()

    if not args.validate and not args.submit and not args.compare:
        args.submit = True

    if args.compare and not args.semantic:
        args.semantic = True

    if args.mode is None:
        mode = "hybrid" if args.semantic else "bm25"
    else:
        mode = args.mode
    # эмбеддинги нужны для dense и hybrid
    use_semantic = mode in ("dense", "hybrid")

    items = load_items()
    retriever = build_retriever(
        items,
        use_semantic=use_semantic,
        mode=mode,
        bm25_weight=args.bm25_weight if args.bm25_weight is not None else BM25_WEIGHT,
        dense_weight=args.dense_weight if args.dense_weight is not None else DENSE_WEIGHT,
        query_with_params=not args.no_query_params,
    )
    items_ids = items[["item_id"]].copy()
    del items
    gc.collect()

    if args.validate:
        run_validation(retriever)

    if args.compare:
        run_compare(retriever)

    if args.submit:
        print("История кликов из train...")
        train = load_train()
        retriever.fit_query_history(train)
        del train
        gc.collect()
        run_submit(retriever, items_ids)


if __name__ == "__main__":
    main()

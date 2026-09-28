"""Проверка answer.csv перед отправкой на платформу."""

from __future__ import annotations

import re

import pandas as pd

ITEM_ID_RE = re.compile(r"^[0-9a-f]{16}$")


def validate_answer_csv(
    answer_path: str,
    queries: pd.DataFrame,
    items: pd.DataFrame,
    k: int = 50,
) -> list[str]:
    """
    Возвращает список ошибок; пустой список — всё ок.
    """
    errors: list[str] = []
    ans = pd.read_csv(answer_path, dtype=str)
    expected_qids = set(queries["query_id"].astype(str))
    corpus = set(items["item_id"].astype(str).str.lower())

    if list(ans.columns) != ["query_id", "answer"]:
        errors.append(f"ожидались колонки query_id, answer; получено {list(ans.columns)}")

    if len(ans) != len(expected_qids):
        errors.append(f"строк {len(ans)}, ожидалось {len(expected_qids)}")

    got_qids = ans["query_id"].astype(str)
    if got_qids.duplicated().any():
        errors.append("есть дубликаты query_id")

    missing = expected_qids - set(got_qids)
    extra = set(got_qids) - expected_qids
    if missing:
        errors.append(f"нет query_id: {len(missing)} шт.")
    if extra:
        errors.append(f"лишние query_id: {len(extra)} шт.")

    for i, row in ans.iterrows():
        qid = str(row["query_id"])
        raw = row.get("answer", "")
        if pd.isna(raw) or raw == "":
            parts: list[str] = []
        else:
            parts = str(raw).split()
        if len(parts) > k:
            errors.append(f"query_id {qid}: больше {k} item_id ({len(parts)})")
        if len(parts) != len(set(parts)):
            errors.append(f"query_id {qid}: дубликаты item_id в answer")
        for iid in parts:
            if not ITEM_ID_RE.match(iid):
                errors.append(f"query_id {qid}: неверный формат item_id {iid!r}")
            elif iid not in corpus:
                errors.append(f"query_id {qid}: item_id {iid} нет в корпусе")

    return errors

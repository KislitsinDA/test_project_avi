from __future__ import annotations

from typing import Iterable


def recall_at_k(
    predictions: dict[str, list[str]],
    relevant: dict[str, set[str]],
    k: int = 50,
) -> float:
    """
    Recall@k по ТЗ: для каждого запроса |pred[:k] ∩ rel| / |rel|, затем среднее.
    """
    scores: list[float] = []
    for qid, rel_set in relevant.items():
        if not rel_set:
            continue
        pred = predictions.get(qid, [])[:k]
        hit = len(set(pred) & rel_set)
        scores.append(hit / len(rel_set))
    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def dedupe_preserve(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item_id in items:
        if item_id in seen:
            continue
        seen.add(item_id)
        out.append(item_id)
    return out

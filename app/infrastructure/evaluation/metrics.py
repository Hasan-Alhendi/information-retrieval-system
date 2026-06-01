"""IR evaluation metrics."""

import math


def precision_at_k(retrieved: list[str], relevant: set[str], k: int = 10) -> float:
    """Compute Precision@k."""
    if k <= 0:
        return 0.0
    top_k = retrieved[:k]
    if not top_k:
        return 0.0
    return sum(1 for doc_id in top_k if doc_id in relevant) / len(top_k)


def recall_at_k(retrieved: list[str], relevant: set[str], k: int | None = None) -> float:
    """Compute recall at k or over the full retrieved list."""
    if not relevant:
        return 0.0
    candidates = retrieved if k is None else retrieved[:k]
    return sum(1 for doc_id in candidates if doc_id in relevant) / len(relevant)


def average_precision(retrieved: list[str], relevant: set[str]) -> float:
    """Compute average precision for one query."""
    if not relevant:
        return 0.0

    hits = 0
    precision_sum = 0.0
    for index, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            hits += 1
            precision_sum += hits / index

    return precision_sum / len(relevant)


def dcg_at_k(retrieved: list[str], relevance_scores: dict[str, float], k: int = 10) -> float:
    """Compute Discounted Cumulative Gain at k."""
    score = 0.0
    for index, doc_id in enumerate(retrieved[:k], start=1):
        relevance = relevance_scores.get(doc_id, 0.0)
        if relevance > 0:
            score += relevance / math.log2(index + 1)
    return score


def ndcg_at_k(retrieved: list[str], relevance_scores: dict[str, float], k: int = 10) -> float:
    """Compute normalized DCG at k."""
    ideal_docs = sorted(relevance_scores, key=relevance_scores.get, reverse=True)
    ideal_dcg = dcg_at_k(ideal_docs, relevance_scores, k=k)
    if ideal_dcg == 0:
        return 0.0
    return dcg_at_k(retrieved, relevance_scores, k=k) / ideal_dcg

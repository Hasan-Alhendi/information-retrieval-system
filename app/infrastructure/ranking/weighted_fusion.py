"""Weighted score fusion."""


def weighted_sum(score_groups: list[dict[str, float]], weights: list[float]) -> dict[str, float]:
    """Combine multiple score dictionaries using weighted sum."""
    if len(score_groups) != len(weights):
        raise ValueError("score_groups and weights must have the same length")

    fused: dict[str, float] = {}
    for scores, weight in zip(score_groups, weights, strict=True):
        for doc_id, score in scores.items():
            fused[doc_id] = fused.get(doc_id, 0.0) + weight * score
    return fused

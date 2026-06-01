"""Score normalization helpers."""


def min_max_normalize(scores: dict[str, float]) -> dict[str, float]:
    """Scale scores to a common zero to one range."""
    if not scores:
        return {}

    lowest = min(scores.values())
    highest = max(scores.values())
    if highest == lowest:
        return {key: 1.0 for key in scores}

    scale = highest - lowest
    normalized: dict[str, float] = {}
    for key, value in scores.items():
        normalized[key] = (value - lowest) / scale
    return normalized

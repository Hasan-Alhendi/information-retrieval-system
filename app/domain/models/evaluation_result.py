"""Evaluation result domain model."""

from dataclasses import dataclass


@dataclass(frozen=True)
class EvaluationResult:
    """Represents aggregate retrieval evaluation metrics."""

    dataset_name: str
    model_name: str
    map_score: float
    recall: float
    precision_at_10: float
    ndcg: float
    evaluated_queries: int

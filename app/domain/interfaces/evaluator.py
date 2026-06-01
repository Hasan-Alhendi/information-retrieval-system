"""Evaluator interface."""

from typing import Protocol

from app.domain.models.evaluation_result import EvaluationResult


class Evaluator(Protocol):
    """Contract implemented by retrieval evaluators."""

    def evaluate(self, dataset_name: str, model_name: str) -> EvaluationResult:
        """Evaluate a retrieval model on a dataset."""
        ...

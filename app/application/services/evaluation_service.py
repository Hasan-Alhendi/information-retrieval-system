"""Application evaluation service."""

from app.domain.interfaces.evaluator import Evaluator
from app.domain.models.evaluation_result import EvaluationResult


class EvaluationService:
    """Coordinates retrieval evaluation."""

    def __init__(self, evaluator: Evaluator) -> None:
        self._evaluator = evaluator

    def evaluate(self, dataset_name: str, model_name: str) -> EvaluationResult:
        """Evaluate a model on a dataset."""
        return self._evaluator.evaluate(dataset_name=dataset_name, model_name=model_name)

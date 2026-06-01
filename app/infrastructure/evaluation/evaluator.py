"""Retrieval evaluator skeleton."""

from app.domain.models.evaluation_result import EvaluationResult


class RetrievalEvaluator:
    """Evaluates retrieval models using benchmark qrels."""

    def evaluate(self, dataset_name: str, model_name: str) -> EvaluationResult:
        """Evaluate a retrieval model.

        The full implementation will be migrated from the previous project after retrieval
        components are ready.
        """
        return EvaluationResult(
            dataset_name=dataset_name,
            model_name=model_name,
            map_score=0.0,
            recall=0.0,
            precision_at_10=0.0,
            ndcg=0.0,
            evaluated_queries=0,
        )

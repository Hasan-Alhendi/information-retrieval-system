"""Evaluation API routes."""

from fastapi import APIRouter, HTTPException, Query

from app.infrastructure.evaluation.evaluator import RetrievalEvaluator

router = APIRouter(prefix="/evaluation", tags=["evaluation"])


@router.get("/{dataset_name}/{model_name}")
def evaluate_model(
    dataset_name: str,
    model_name: str,
    max_docs: int | None = Query(default=None, ge=1),
    max_queries: int | None = Query(default=25, ge=1),
    top_k: int = Query(default=10, ge=1, le=100),
    bm25_k1: float = Query(default=1.5, gt=0),
    bm25_b: float = Query(default=0.75, ge=0, le=1),
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> dict[str, object]:
    """Evaluate a model on a dataset."""
    evaluator = RetrievalEvaluator(
        max_docs=max_docs,
        max_queries=max_queries,
        top_k=top_k,
        bm25_k1=bm25_k1,
        bm25_b=bm25_b,
        embedding_model=embedding_model,
    )
    try:
        result = evaluator.evaluate(dataset_name=dataset_name, model_name=model_name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "dataset_name": result.dataset_name,
        "model_name": result.model_name,
        "metrics": {
            "MAP": result.map_score,
            "Recall": result.recall,
            "Precision@10": result.precision_at_10,
            "nDCG": result.ndcg,
        },
        "evaluated_queries": result.evaluated_queries,
    }

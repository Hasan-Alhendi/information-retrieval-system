"""Evaluation API routes."""

from fastapi import APIRouter, HTTPException, Query

from app.infrastructure.evaluation.evaluator_v2 import RetrievalEvaluatorV2

router = APIRouter(prefix="/evaluation", tags=["evaluation"])


@router.get("/{dataset_name}/{model_name}")
def evaluate_model(
    dataset_name: str,
    model_name: str,
    max_docs: int | None = Query(
        default=None,
        ge=1,
        description="Leave null to evaluate finalized full-corpus indexes.",
    ),
    max_queries: int | None = Query(default=25, ge=1),
    top_k: int = Query(default=10, ge=1, le=100),
    bm25_k1: float = Query(default=1.5, gt=0),
    bm25_b: float = Query(default=0.75, ge=0, le=1),
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> dict[str, object]:
    """Evaluate any corrected retrieval model using benchmark qrels."""
    evaluator = RetrievalEvaluatorV2(
        max_docs=max_docs,
        max_queries=max_queries,
        top_k=top_k,
        bm25_k1=bm25_k1,
        bm25_b=bm25_b,
        embedding_model=embedding_model,
    )
    try:
        result = evaluator.evaluate(dataset_name=dataset_name, model_name=model_name)
    except (RuntimeError, ValueError, OSError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    return {
        "dataset_name": result.dataset_name,
        "model_name": result.model_name,
        "index_scope": "full" if max_docs is None else "development",
        "cutoff": top_k,
        "metrics": {
            f"MAP@{top_k}": result.map_score,
            f"Recall@{top_k}": result.recall,
            "Precision@10": result.precision_at_10,
            f"nDCG@{top_k}": result.ndcg,
        },
        "evaluated_queries": result.evaluated_queries,
    }

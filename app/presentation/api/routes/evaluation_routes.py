"""Evaluation API routes."""

from fastapi import APIRouter

router = APIRouter(prefix="/evaluation", tags=["evaluation"])


@router.get("/{dataset_name}/{model_name}")
def evaluate_model(dataset_name: str, model_name: str) -> dict[str, object]:
    """Evaluate a model on a dataset."""
    return {
        "dataset_name": dataset_name,
        "model_name": model_name,
        "metrics": {},
    }

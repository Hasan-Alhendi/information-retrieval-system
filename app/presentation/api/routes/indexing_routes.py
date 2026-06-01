"""Indexing API routes."""

from fastapi import APIRouter

router = APIRouter(prefix="/indexes", tags=["indexing"])


@router.post("/{dataset_name}/{model_name}")
def build_index(dataset_name: str, model_name: str) -> dict[str, str]:
    """Build an index for a dataset and model."""
    return {
        "dataset_name": dataset_name,
        "model_name": model_name,
        "status": "scheduled",
    }

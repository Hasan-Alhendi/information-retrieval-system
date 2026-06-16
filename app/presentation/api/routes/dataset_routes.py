"""Dataset API routes."""

from fastapi import APIRouter

from app.infrastructure.datasets.dataset_registry import SUPPORTED_DATASETS

router = APIRouter(prefix="/datasets", tags=["datasets"])


@router.get("")
def list_datasets() -> dict[str, object]:
    """List official datasets and dataset-specific processing profiles."""
    return {
        "datasets": [
            {
                "name": config.name,
                "display_name": config.display_name,
                "source": config.source,
                "external_id": config.external_id,
                "task_type": config.task_type,
                "processing_profile": config.processing_profile,
            }
            for config in SUPPORTED_DATASETS.values()
        ]
    }

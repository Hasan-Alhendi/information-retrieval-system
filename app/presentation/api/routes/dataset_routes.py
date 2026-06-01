"""Dataset API routes."""

from fastapi import APIRouter

from app.infrastructure.datasets.dataset_registry import SUPPORTED_DATASETS

router = APIRouter(prefix="/datasets", tags=["datasets"])


@router.get("")
def list_datasets() -> dict[str, object]:
    """List supported datasets."""
    return {
        "datasets": [
            {
                "name": config.name,
                "display_name": config.display_name,
                "source": config.source,
                "document_limit": config.document_limit,
            }
            for config in SUPPORTED_DATASETS.values()
        ]
    }

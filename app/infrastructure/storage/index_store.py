"""Index storage path helpers."""

from pathlib import Path

from app.config import INDEXES_DIR


def get_index_dir(dataset_name: str, model_name: str) -> Path:
    """Return the storage directory for a model index."""
    return INDEXES_DIR / dataset_name / model_name

"""BEIR dataset loading infrastructure.

The full implementation will be migrated from the previous project in the dataset phase.
"""

from app.domain.models.document import Document
from app.infrastructure.datasets.dataset_registry import get_dataset_config


class BeirDatasetLoader:
    """Loads BEIR-compatible datasets."""

    def load_documents(self, dataset_name: str) -> list[Document]:
        """Load documents for a dataset.

        This placeholder validates the dataset name and returns an empty list until the
        full dataset loading implementation is migrated.
        """
        get_dataset_config(dataset_name)
        return []

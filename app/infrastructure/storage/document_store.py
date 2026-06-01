"""Document storage skeleton."""

from app.domain.models.document import Document


class DocumentStore:
    """Stores and retrieves documents for datasets."""

    def load_documents(self, dataset_name: str) -> list[Document]:
        """Load documents for a dataset."""
        _ = dataset_name
        return []

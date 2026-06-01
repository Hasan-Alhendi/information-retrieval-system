"""Application retrieval service."""

from app.domain.interfaces.retriever import Retriever
from app.domain.models.search_result import SearchResult


class RetrievalService:
    """Coordinates retrieval operations."""

    def __init__(self, retrievers: dict[str, Retriever]) -> None:
        self._retrievers = retrievers

    def search(
        self,
        model_name: str,
        query: str,
        dataset_name: str,
        top_k: int = 10,
    ) -> list[SearchResult]:
        """Search using a registered retrieval model."""
        if model_name not in self._retrievers:
            raise ValueError(f"Unsupported retrieval model: {model_name}")
        return self._retrievers[model_name].search(query, dataset_name, top_k)

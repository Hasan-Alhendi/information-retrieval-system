"""Retriever interface."""

from typing import Protocol

from app.domain.models.search_result import SearchResult


class Retriever(Protocol):
    """Contract implemented by all retrieval models."""

    def search(self, query: str, dataset_name: str, top_k: int = 10) -> list[SearchResult]:
        """Return ranked search results for a query."""
        ...

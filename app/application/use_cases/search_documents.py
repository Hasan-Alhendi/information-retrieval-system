"""Search documents use case."""

from app.domain.interfaces.retriever import Retriever
from app.domain.models.search_result import SearchResult


class SearchDocumentsUseCase:
    """Use case for searching documents with a selected retriever."""

    def __init__(self, retriever: Retriever) -> None:
        self._retriever = retriever

    def execute(self, query: str, dataset_name: str, top_k: int = 10) -> list[SearchResult]:
        """Search documents for the provided query."""
        return self._retriever.search(query=query, dataset_name=dataset_name, top_k=top_k)

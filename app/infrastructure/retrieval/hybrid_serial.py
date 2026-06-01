"""Hybrid serial retriever skeleton."""

from app.domain.models.search_result import SearchResult


class HybridSerialRetriever:
    """Serial hybrid retrieval: candidate retrieval followed by reranking."""

    def search(self, query: str, dataset_name: str, top_k: int = 10) -> list[SearchResult]:
        """Search using a serial hybrid strategy."""
        _ = query
        _ = dataset_name
        _ = top_k
        return []

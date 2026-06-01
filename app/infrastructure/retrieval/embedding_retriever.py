"""Embedding-based retriever implementation skeleton."""

from app.domain.models.search_result import SearchResult


class EmbeddingRetriever:
    """Semantic retriever based on dense document and query embeddings."""

    def search(self, query: str, dataset_name: str, top_k: int = 10) -> list[SearchResult]:
        """Search documents using dense embeddings."""
        _ = query
        _ = dataset_name
        _ = top_k
        return []

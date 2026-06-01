"""BM25 retriever implementation skeleton."""

from app.config import DEFAULT_BM25_B, DEFAULT_BM25_K1
from app.domain.models.search_result import SearchResult


class BM25Retriever:
    """BM25 probabilistic retriever with tunable k1 and b parameters."""

    def __init__(self, k1: float = DEFAULT_BM25_K1, b: float = DEFAULT_BM25_B) -> None:
        self.k1 = k1
        self.b = b

    def search(self, query: str, dataset_name: str, top_k: int = 10) -> list[SearchResult]:
        """Search documents using BM25."""
        _ = query
        _ = dataset_name
        _ = top_k
        return []

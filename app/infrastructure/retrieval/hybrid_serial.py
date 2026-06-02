"""Hybrid serial retriever implementation."""

from app.domain.models.search_result import SearchResult
from app.infrastructure.retrieval.bm25_retriever import BM25Retriever
from app.infrastructure.retrieval.embedding_retriever import EmbeddingRetriever


class HybridSerialRetriever:
    """Serial hybrid retrieval: BM25 candidate retrieval followed by embedding reranking."""

    model_name = "hybrid_serial"

    def __init__(
        self,
        bm25_retriever: BM25Retriever | None = None,
        embedding_retriever: EmbeddingRetriever | None = None,
        max_docs: int | None = None,
        candidate_k: int = 100,
        bm25_weight: float = 0.3,
        embedding_weight: float = 0.7,
    ) -> None:
        self._bm25 = bm25_retriever or BM25Retriever(max_docs=max_docs)
        self._embedding = embedding_retriever or EmbeddingRetriever(max_docs=max_docs)
        self._max_docs = max_docs
        self.candidate_k = candidate_k
        self.bm25_weight = bm25_weight
        self.embedding_weight = embedding_weight

    def build(self, dataset_name: str, force: bool = False, max_docs: int | None = None) -> None:
        """Build indexes required by the serial hybrid retriever."""
        self._bm25.build(dataset_name=dataset_name, force=force, max_docs=max_docs)
        self._embedding.build(dataset_name=dataset_name, force=force, max_docs=max_docs)

    def search(self, query: str, dataset_name: str, top_k: int = 10) -> list[SearchResult]:
        """Search using serial hybrid retrieval.

        BM25 first selects candidate documents. Embedding retrieval then provides semantic
        scores, and candidates are reranked using a weighted combination.
        """
        candidate_count = max(top_k, self.candidate_k)
        bm25_candidates = self._bm25.search(query, dataset_name, top_k=candidate_count)
        embedding_results = self._embedding.search(query, dataset_name, top_k=candidate_count)

        embedding_scores = {result.doc_id: result.score for result in embedding_results}
        bm25_scores = _normalize({result.doc_id: result.score for result in bm25_candidates})
        normalized_embedding_scores = _normalize(embedding_scores)

        reranked: list[SearchResult] = []
        for result in bm25_candidates:
            combined_score = (
                self.bm25_weight * bm25_scores.get(result.doc_id, 0.0)
                + self.embedding_weight * normalized_embedding_scores.get(result.doc_id, 0.0)
            )
            reranked.append(
                SearchResult(
                    doc_id=result.doc_id,
                    score=combined_score,
                    rank=result.rank,
                    text=result.text,
                    title=result.title,
                    metadata={
                        "model": self.model_name,
                        "bm25_score": result.score,
                        "embedding_score": embedding_scores.get(result.doc_id, 0.0),
                    },
                )
            )

        reranked.sort(key=lambda item: item.score, reverse=True)
        return [
            SearchResult(
                doc_id=result.doc_id,
                score=result.score,
                rank=rank,
                text=result.text,
                title=result.title,
                metadata=result.metadata,
            )
            for rank, result in enumerate(reranked[:top_k], start=1)
        ]


def _normalize(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    lowest = min(scores.values())
    highest = max(scores.values())
    if highest == lowest:
        return {key: 1.0 for key in scores}
    scale = highest - lowest
    return {key: (value - lowest) / scale for key, value in scores.items()}

"""Hybrid parallel retriever implementation."""

from app.domain.models.search_result import SearchResult
from app.infrastructure.retrieval.bm25_retriever import BM25Retriever
from app.infrastructure.retrieval.embedding_retriever import EmbeddingRetriever
from app.infrastructure.retrieval.tfidf_retriever import TFIDFRetriever


class HybridParallelRetriever:
    """Parallel hybrid retrieval with result fusion."""

    model_name = "hybrid_parallel"

    def __init__(
        self,
        tfidf_retriever: TFIDFRetriever | None = None,
        bm25_retriever: BM25Retriever | None = None,
        embedding_retriever: EmbeddingRetriever | None = None,
        max_docs: int | None = None,
        fusion_k: int = 60,
    ) -> None:
        self._tfidf = tfidf_retriever or TFIDFRetriever(max_docs=max_docs)
        self._bm25 = bm25_retriever or BM25Retriever(max_docs=max_docs)
        self._embedding = embedding_retriever or EmbeddingRetriever(max_docs=max_docs)
        self._max_docs = max_docs
        self.fusion_k = fusion_k

    def build(self, dataset_name: str, force: bool = False, max_docs: int | None = None) -> None:
        """Build all indexes required by the parallel hybrid retriever."""
        self._tfidf.build(dataset_name=dataset_name, force=force, max_docs=max_docs)
        self._bm25.build(dataset_name=dataset_name, force=force, max_docs=max_docs)
        self._embedding.build(dataset_name=dataset_name, force=force, max_docs=max_docs)

    def search(self, query: str, dataset_name: str, top_k: int = 10) -> list[SearchResult]:
        """Search using parallel hybrid retrieval and reciprocal rank fusion."""
        candidate_k = max(top_k * 3, 30)
        result_groups = [
            self._tfidf.search(query, dataset_name, top_k=candidate_k),
            self._bm25.search(query, dataset_name, top_k=candidate_k),
            self._embedding.search(query, dataset_name, top_k=candidate_k),
        ]

        fused_scores: dict[str, float] = {}
        result_lookup: dict[str, SearchResult] = {}
        source_models: dict[str, list[str]] = {}

        for results in result_groups:
            for rank, result in enumerate(results, start=1):
                fused_scores[result.doc_id] = fused_scores.get(result.doc_id, 0.0) + 1.0 / (
                    self.fusion_k + rank
                )
                result_lookup.setdefault(result.doc_id, result)
                source_models.setdefault(result.doc_id, []).append(result.metadata.get("model", "unknown"))

        ranked_doc_ids = sorted(fused_scores, key=fused_scores.get, reverse=True)[:top_k]
        return [
            SearchResult(
                doc_id=doc_id,
                score=fused_scores[doc_id],
                rank=rank,
                text=result_lookup[doc_id].text,
                title=result_lookup[doc_id].title,
                metadata={
                    "model": self.model_name,
                    "fusion": "reciprocal_rank_fusion",
                    "source_models": source_models.get(doc_id, []),
                },
            )
            for rank, doc_id in enumerate(ranked_doc_ids, start=1)
        ]

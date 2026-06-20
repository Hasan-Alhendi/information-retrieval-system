"""Correct serial hybrid retrieval implementation."""

from __future__ import annotations

import time

from app.domain.models.search_result import SearchResult
from app.infrastructure.retrieval.bm25_retriever import BM25Retriever
from app.infrastructure.retrieval.dense_candidate_scorer import DenseCandidateScorer
from app.infrastructure.retrieval.embedding_retriever import EmbeddingRetriever


class HybridSerialRetriever:
    """Retrieve BM25 candidates, then rerank those same candidates semantically."""

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
        self._scorer = DenseCandidateScorer(self._embedding.embedding_model_name)
        self._max_docs = max_docs
        self.candidate_k = candidate_k
        self.bm25_weight = bm25_weight
        self.embedding_weight = embedding_weight

    def build(
        self,
        dataset_name: str,
        force: bool = False,
        max_docs: int | None = None,
    ) -> None:
        """Build the lexical and dense indexes required by serial hybrid retrieval."""
        self._bm25.build(dataset_name, force=force, max_docs=max_docs)
        self._embedding.build(dataset_name, force=force, max_docs=max_docs)

    def prepare(self, dataset_name: str) -> None:
        """Warm the underlying indexes and dense candidate scorer."""
        self._bm25.ensure_ready(dataset_name)
        self._embedding.ensure_ready(dataset_name)
        if self._max_docs is None:
            self._scorer.prepare(dataset_name)

    def search(
        self,
        query: str,
        dataset_name: str,
        top_k: int = 10,
    ) -> list[SearchResult]:
        """Retrieve with BM25, score the candidates semantically, and rerank them."""
        started = time.perf_counter()
        candidate_count = max(top_k, self.candidate_k)
        candidates = self._bm25.search(
            query,
            dataset_name,
            top_k=candidate_count,
        )
        if not candidates:
            return []

        if self._max_docs is None:
            dense_scores = self._scorer.score(
                query,
                dataset_name,
                [item.doc_id for item in candidates],
            )
        else:
            dense_scores = {
                item.doc_id: item.score
                for item in self._embedding.search(
                    query,
                    dataset_name,
                    top_k=candidate_count,
                )
            }

        lexical_scores = _normalize({item.doc_id: item.score for item in candidates})
        semantic_scores = _normalize(dense_scores)

        reranked: list[tuple[float, SearchResult, float]] = []
        for item in candidates:
            dense_score = dense_scores.get(item.doc_id, 0.0)
            combined_score = (
                self.bm25_weight * lexical_scores.get(item.doc_id, 0.0)
                + self.embedding_weight * semantic_scores.get(item.doc_id, 0.0)
            )
            reranked.append((combined_score, item, dense_score))

        reranked.sort(key=lambda row: row[0], reverse=True)
        latency_ms = round((time.perf_counter() - started) * 1000, 3)

        return [
            SearchResult(
                doc_id=item.doc_id,
                score=combined_score,
                rank=rank,
                text=item.text,
                title=item.title,
                metadata={
                    **item.metadata,
                    "model": self.model_name,
                    "pipeline": "bm25_then_dense_reranking",
                    "candidate_k": candidate_count,
                    "bm25_score": item.score,
                    "embedding_score": dense_score,
                    "bm25_weight": self.bm25_weight,
                    "embedding_weight": self.embedding_weight,
                    "query_time_ms": latency_ms,
                },
            )
            for rank, (combined_score, item, dense_score) in enumerate(
                reranked[:top_k],
                start=1,
            )
        ]


def _normalize(scores: dict[str, float]) -> dict[str, float]:
    """Min-max normalize a score mapping to the range [0, 1]."""
    if not scores:
        return {}
    lowest = min(scores.values())
    highest = max(scores.values())
    if lowest == highest:
        return {key: 1.0 for key in scores}
    scale = highest - lowest
    return {key: (value - lowest) / scale for key, value in scores.items()}

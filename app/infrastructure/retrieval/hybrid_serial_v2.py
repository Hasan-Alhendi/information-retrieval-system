"""Correct serial hybrid retrieval."""

import time

from app.domain.models.search_result import SearchResult
from app.infrastructure.retrieval.bm25_retriever import BM25Retriever
from app.infrastructure.retrieval.dense_candidate_scorer import DenseCandidateScorer
from app.infrastructure.retrieval.embedding_retriever import EmbeddingRetriever


class HybridSerialRetrieverV2:
    """Retrieve BM25 candidates and rerank those same candidates semantically."""

    model_name = "hybrid_serial"

    def __init__(
        self,
        bm25_retriever=None,
        embedding_retriever=None,
        max_docs=None,
        candidate_k=100,
        bm25_weight=0.3,
        embedding_weight=0.7,
    ):
        self.bm25 = bm25_retriever or BM25Retriever(max_docs=max_docs)
        self.embedding = embedding_retriever or EmbeddingRetriever(max_docs=max_docs)
        self.scorer = DenseCandidateScorer(self.embedding.embedding_model_name)
        self.max_docs = max_docs
        self.candidate_k = candidate_k
        self.bm25_weight = bm25_weight
        self.embedding_weight = embedding_weight

    def build(self, dataset_name, force=False, max_docs=None):
        self.bm25.build(dataset_name, force=force, max_docs=max_docs)
        self.embedding.build(dataset_name, force=force, max_docs=max_docs)

    def prepare(self, dataset_name):
        self.bm25.ensure_ready(dataset_name)
        self.embedding.ensure_ready(dataset_name)
        if self.max_docs is None:
            self.scorer.prepare(dataset_name)

    def search(self, query, dataset_name, top_k=10):
        started = time.perf_counter()
        candidates = self.bm25.search(
            query,
            dataset_name,
            top_k=max(top_k, self.candidate_k),
        )
        if not candidates:
            return []

        if self.max_docs is None:
            dense = self.scorer.score(
                query,
                dataset_name,
                [item.doc_id for item in candidates],
            )
        else:
            dense = {
                item.doc_id: item.score
                for item in self.embedding.search(
                    query,
                    dataset_name,
                    top_k=max(top_k, self.candidate_k),
                )
            }

        lexical = _normalize({item.doc_id: item.score for item in candidates})
        semantic = _normalize(dense)
        ranked = []
        for item in candidates:
            score = (
                self.bm25_weight * lexical.get(item.doc_id, 0.0)
                + self.embedding_weight * semantic.get(item.doc_id, 0.0)
            )
            ranked.append((score, item, dense.get(item.doc_id, 0.0)))
        ranked.sort(key=lambda row: row[0], reverse=True)

        latency = round((time.perf_counter() - started) * 1000, 3)
        return [
            SearchResult(
                doc_id=item.doc_id,
                score=score,
                rank=rank,
                text=item.text,
                title=item.title,
                metadata={
                    **item.metadata,
                    "model": self.model_name,
                    "pipeline": "bm25_then_dense_reranking",
                    "candidate_k": max(top_k, self.candidate_k),
                    "bm25_score": item.score,
                    "embedding_score": dense_score,
                    "bm25_weight": self.bm25_weight,
                    "embedding_weight": self.embedding_weight,
                    "query_time_ms": latency,
                },
            )
            for rank, (score, item, dense_score) in enumerate(ranked[:top_k], 1)
        ]


def _normalize(scores):
    if not scores:
        return {}
    low, high = min(scores.values()), max(scores.values())
    if low == high:
        return {key: 1.0 for key in scores}
    return {key: (value - low) / (high - low) for key, value in scores.items()}

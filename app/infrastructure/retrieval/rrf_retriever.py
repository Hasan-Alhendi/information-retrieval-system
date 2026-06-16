"""Reciprocal-rank fusion retriever."""

import time

from app.domain.models.search_result import SearchResult
from app.infrastructure.retrieval.bm25_retriever import BM25Retriever
from app.infrastructure.retrieval.embedding_retriever import EmbeddingRetriever
from app.infrastructure.retrieval.tfidf_retriever import TFIDFRetriever


class ReciprocalRankFusionRetriever:
    """Fuse TF-IDF, BM25, and embedding rankings with RRF."""

    model_name = "hybrid_parallel"

    def __init__(
        self,
        tfidf_retriever=None,
        bm25_retriever=None,
        embedding_retriever=None,
        max_docs=None,
        fusion_k=60,
    ):
        self.tfidf = tfidf_retriever or TFIDFRetriever(max_docs=max_docs)
        self.bm25 = bm25_retriever or BM25Retriever(max_docs=max_docs)
        self.embedding = embedding_retriever or EmbeddingRetriever(max_docs=max_docs)
        self.max_docs = max_docs
        self.fusion_k = fusion_k

    def build(self, dataset_name, force=False, max_docs=None):
        self.tfidf.build(dataset_name, force=force, max_docs=max_docs)
        self.bm25.build(dataset_name, force=force, max_docs=max_docs)
        self.embedding.build(dataset_name, force=force, max_docs=max_docs)

    def prepare(self, dataset_name):
        self.tfidf.ensure_ready(dataset_name)
        self.bm25.ensure_ready(dataset_name)
        self.embedding.ensure_ready(dataset_name)
        self.embedding.search("initialization", dataset_name, top_k=1)

    def search(self, query, dataset_name, top_k=10):
        started = time.perf_counter()
        candidate_k = max(top_k * 3, 30)
        groups = [
            self.tfidf.search(query, dataset_name, top_k=candidate_k),
            self.bm25.search(query, dataset_name, top_k=candidate_k),
            self.embedding.search(query, dataset_name, top_k=candidate_k),
        ]

        fused, lookup, sources = {}, {}, {}
        for results in groups:
            for rank, item in enumerate(results, 1):
                fused[item.doc_id] = fused.get(item.doc_id, 0.0) + 1.0 / (
                    self.fusion_k + rank
                )
                lookup.setdefault(item.doc_id, item)
                sources.setdefault(item.doc_id, []).append(
                    item.metadata.get("model", "unknown")
                )

        ranked_ids = sorted(fused, key=fused.get, reverse=True)[:top_k]
        latency = round((time.perf_counter() - started) * 1000, 3)
        return [
            SearchResult(
                doc_id=doc_id,
                score=fused[doc_id],
                rank=rank,
                text=lookup[doc_id].text,
                title=lookup[doc_id].title,
                metadata={
                    **lookup[doc_id].metadata,
                    "model": self.model_name,
                    "fusion": "reciprocal_rank_fusion",
                    "fusion_k": self.fusion_k,
                    "candidate_k": candidate_k,
                    "source_models": sources.get(doc_id, []),
                    "query_time_ms": latency,
                },
            )
            for rank, doc_id in enumerate(ranked_ids, 1)
        ]

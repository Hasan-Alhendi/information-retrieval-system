"""Cluster-aware semantic reranking for before/after retrieval comparison."""

from __future__ import annotations

import time

import numpy as np

from app.domain.models.search_result import SearchResult
from app.infrastructure.clustering.cluster_store import (
    ClusterArtifactStore,
    ClusterArtifacts,
)
from app.infrastructure.clustering.clusterer import DocumentClusterer
from app.infrastructure.retrieval.embedding_retriever import EmbeddingRetriever


class ClusterAwareRetriever:
    """Rerank embedding candidates using the query-to-cluster similarity."""

    model_name = "embedding_clustered"

    def __init__(
        self,
        *,
        embedding_retriever: EmbeddingRetriever | None = None,
        max_docs: int | None,
        number_of_clusters: int = 5,
        cluster_weight: float = 0.2,
        candidate_k: int = 100,
    ) -> None:
        if max_docs is None:
            raise ValueError(
                "Embedding + Clustering is currently a development-subset experiment. "
                "Choose a positive max_docs value."
            )
        if not 0.0 <= cluster_weight <= 1.0:
            raise ValueError("cluster_weight must be between 0 and 1.")
        if number_of_clusters < 2:
            raise ValueError("number_of_clusters must be at least 2.")
        if candidate_k < 1:
            raise ValueError("candidate_k must be positive.")

        self._embedding = embedding_retriever or EmbeddingRetriever(max_docs=max_docs)
        self._max_docs = max_docs
        self.number_of_clusters = number_of_clusters
        self.cluster_weight = cluster_weight
        self.candidate_k = candidate_k
        self._artifacts: dict[str, ClusterArtifacts] = {}

    def build(
        self,
        dataset_name: str,
        force: bool = False,
        max_docs: int | None = None,
    ) -> None:
        """Build the embedding index and clustering artifacts for the same subset."""
        active_max_docs = max_docs if max_docs is not None else self._max_docs
        if active_max_docs != self._max_docs:
            raise ValueError(
                "ClusterAwareRetriever must use the same max_docs value used at construction."
            )

        self._embedding.build(
            dataset_name=dataset_name,
            force=force,
            max_docs=active_max_docs,
        )
        store = self._store(dataset_name)
        if force or not store.exists():
            DocumentClusterer(
                embedding_model_name=self._embedding.embedding_model_name,
            ).cluster(
                dataset_name=dataset_name,
                number_of_clusters=self.number_of_clusters,
                max_docs=self._max_docs,
                sample_size=1,
                persist_artifacts=True,
            )
        self._artifacts.pop(dataset_name, None)

    def prepare(self, dataset_name: str) -> None:
        """Ensure the embedding index and clustering artifacts are ready."""
        self._embedding.ensure_ready(dataset_name)
        store = self._store(dataset_name)
        if not store.exists():
            self.build(dataset_name)
        self._artifacts[dataset_name] = store.load()

    def search(
        self,
        query: str,
        dataset_name: str,
        top_k: int = 10,
    ) -> list[SearchResult]:
        """Retrieve semantic candidates and rerank them with cluster proximity."""
        if top_k <= 0:
            return []

        started = time.perf_counter()
        self.prepare(dataset_name)
        candidate_count = max(top_k, self.candidate_k)
        candidates = self._embedding.search(
            query=query,
            dataset_name=dataset_name,
            top_k=candidate_count,
        )
        if not candidates:
            return []

        artifacts = self._artifacts[dataset_name]
        query_vector = self._embedding.encode_query(query)
        centroid_scores = _centroid_scores(query_vector, artifacts.centroids)
        base_scores = _min_max_normalize(
            {candidate.doc_id: candidate.score for candidate in candidates}
        )

        reranked: list[tuple[float, SearchResult, int | None, float]] = []
        base_weight = 1.0 - self.cluster_weight
        for candidate in candidates:
            cluster_id = artifacts.doc_to_cluster.get(candidate.doc_id)
            cluster_score = (
                centroid_scores[cluster_id]
                if cluster_id is not None and 0 <= cluster_id < len(centroid_scores)
                else 0.0
            )
            final_score = (
                base_weight * base_scores.get(candidate.doc_id, 0.0)
                + self.cluster_weight * cluster_score
            )
            reranked.append((final_score, candidate, cluster_id, cluster_score))

        reranked.sort(key=lambda row: row[0], reverse=True)
        latency_ms = round((time.perf_counter() - started) * 1000.0, 3)

        return [
            SearchResult(
                doc_id=candidate.doc_id,
                score=final_score,
                rank=rank,
                text=candidate.text,
                title=candidate.title,
                metadata={
                    **candidate.metadata,
                    "model": self.model_name,
                    "pipeline": "embedding_then_cluster_aware_reranking",
                    "base_embedding_score": candidate.score,
                    "normalized_base_score": base_scores.get(candidate.doc_id, 0.0),
                    "cluster_id": cluster_id,
                    "cluster_score": cluster_score,
                    "cluster_weight": self.cluster_weight,
                    "number_of_clusters": self.number_of_clusters,
                    "candidate_k": candidate_count,
                    "query_time_ms": latency_ms,
                },
            )
            for rank, (final_score, candidate, cluster_id, cluster_score) in enumerate(
                reranked[:top_k],
                start=1,
            )
        ]

    def _store(self, dataset_name: str) -> ClusterArtifactStore:
        return ClusterArtifactStore(
            dataset_name=dataset_name,
            embedding_model_name=self._embedding.embedding_model_name,
            max_docs=self._max_docs,
            number_of_clusters=self.number_of_clusters,
        )


def _centroid_scores(query_vector: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Convert cosine similarities from [-1, 1] to stable scores in [0, 1]."""
    query = np.asarray(query_vector, dtype="float32")
    matrix = np.asarray(centroids, dtype="float32")
    if query.ndim != 1 or matrix.ndim != 2:
        raise ValueError("query_vector must be 1D and centroids must be 2D.")
    similarities = matrix @ query
    return np.clip((similarities + 1.0) / 2.0, 0.0, 1.0)


def _min_max_normalize(scores: dict[str, float]) -> dict[str, float]:
    """Normalize candidate scores to [0, 1] before mixing them with cluster scores."""
    if not scores:
        return {}
    lowest = min(scores.values())
    highest = max(scores.values())
    if lowest == highest:
        return {doc_id: 1.0 for doc_id in scores}
    scale = highest - lowest
    return {
        doc_id: (score - lowest) / scale
        for doc_id, score in scores.items()
    }

"""Tests for cluster-aware embedding reranking."""

from __future__ import annotations

import numpy as np

from app.domain.models.search_result import SearchResult
from app.infrastructure.clustering import cluster_store
from app.infrastructure.clustering.cluster_store import ClusterArtifactStore
from app.infrastructure.retrieval.cluster_aware_retriever import (
    ClusterAwareRetriever,
    _centroid_scores,
)


class FakeEmbeddingRetriever:
    embedding_model_name = "fake/model"

    def build(self, dataset_name: str, force: bool = False, max_docs: int | None = None):
        del dataset_name, force, max_docs

    def ensure_ready(self, dataset_name: str) -> None:
        del dataset_name

    def encode_query(self, query: str) -> np.ndarray:
        del query
        return np.asarray([1.0, 0.0], dtype="float32")

    def search(self, query: str, dataset_name: str, top_k: int = 10):
        del query, dataset_name
        rows = [
            SearchResult(doc_id="far", score=0.90, rank=1, text="far"),
            SearchResult(doc_id="near", score=0.80, rank=2, text="near"),
        ]
        return rows[:top_k]


def test_centroid_scores_are_bounded() -> None:
    query = np.asarray([1.0, 0.0], dtype="float32")
    centroids = np.asarray([[1.0, 0.0], [-1.0, 0.0]], dtype="float32")

    scores = _centroid_scores(query, centroids)

    assert scores.tolist() == [1.0, 0.0]


def test_cluster_signal_can_rerank_embedding_candidates(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cluster_store, "CLUSTERS_DIR", tmp_path)
    ClusterArtifactStore(
        dataset_name="quora",
        embedding_model_name="fake/model",
        max_docs=2,
        number_of_clusters=2,
    ).save(
        doc_ids=["far", "near"],
        labels=np.asarray([0, 1], dtype="int32"),
        centroids=np.asarray([[-1.0, 0.0], [1.0, 0.0]], dtype="float32"),
    )

    retriever = ClusterAwareRetriever(
        embedding_retriever=FakeEmbeddingRetriever(),
        max_docs=2,
        number_of_clusters=2,
        cluster_weight=0.8,
        candidate_k=2,
    )
    results = retriever.search("query", "quora", top_k=2)

    assert [result.doc_id for result in results] == ["near", "far"]
    assert results[0].metadata["cluster_score"] == 1.0
    assert results[1].metadata["cluster_score"] == 0.0

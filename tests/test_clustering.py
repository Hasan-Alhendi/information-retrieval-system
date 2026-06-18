"""Tests for document clustering evaluation outputs."""

from __future__ import annotations

import numpy as np

from app.infrastructure.clustering.clusterer import DocumentClusterer


class FakeDatasetLoader:
    """Return a small deterministic development subset."""

    def prepare_dataset(self, dataset_name: str, max_docs: int | None = None):
        del dataset_name, max_docs
        doc_ids = [f"d{index}" for index in range(6)]
        documents = [
            "python programming code",
            "learn software programming",
            "coding with python",
            "healthy food nutrition",
            "nutrition and healthy diet",
            "food diet health",
        ]
        return doc_ids, documents, {}, {}


EMBEDDINGS = np.asarray(
    [
        [1.00, 0.00, 0.02],
        [0.98, 0.03, 0.00],
        [0.97, 0.01, 0.04],
        [0.00, 1.00, 0.02],
        [0.03, 0.98, 0.00],
        [0.01, 0.97, 0.04],
    ],
    dtype="float32",
)


def test_cluster_returns_metrics_projection_and_sizes() -> None:
    clusterer = DocumentClusterer(dataset_loader=FakeDatasetLoader())
    clusterer._encode = lambda documents: EMBEDDINGS  # type: ignore[method-assign]

    result = clusterer.cluster(
        dataset_name="quora",
        number_of_clusters=2,
        max_docs=6,
        sample_size=2,
    )

    evaluation = result["evaluation"]
    assert evaluation["silhouette_score"] is not None
    assert evaluation["silhouette_score"] > 0.8
    assert evaluation["davies_bouldin_index"] is not None
    assert evaluation["davies_bouldin_index"] >= 0
    assert evaluation["inertia"] >= 0
    assert evaluation["silhouette_sample_size"] == 6

    assert len(result["projection"]) == 6
    assert {point["cluster"] for point in result["projection"]} == {"0", "1"}
    assert sum(cluster["size"] for cluster in result["clusters"]) == 6


def test_single_cluster_marks_separation_metrics_unavailable() -> None:
    clusterer = DocumentClusterer(dataset_loader=FakeDatasetLoader())
    clusterer._encode = lambda documents: EMBEDDINGS  # type: ignore[method-assign]

    result = clusterer.cluster(
        dataset_name="quora",
        number_of_clusters=1,
        max_docs=6,
    )

    evaluation = result["evaluation"]
    assert evaluation["silhouette_score"] is None
    assert evaluation["davies_bouldin_index"] is None
    assert evaluation["silhouette_sample_size"] == 0
    assert evaluation["inertia"] >= 0


def test_empty_dataset_returns_stable_evaluation_schema() -> None:
    class EmptyLoader:
        def prepare_dataset(self, dataset_name: str, max_docs: int | None = None):
            del dataset_name, max_docs
            return [], [], {}, {}

    result = DocumentClusterer(dataset_loader=EmptyLoader()).cluster("quora")

    assert result["documents_count"] == 0
    assert result["clusters"] == []
    assert result["projection"] == []
    assert result["evaluation"]["silhouette_score"] is None

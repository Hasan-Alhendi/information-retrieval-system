"""Persistence for cluster-aware retrieval artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np

from app.config import CLUSTERS_DIR


@dataclass(frozen=True)
class ClusterArtifacts:
    """Persisted document-to-cluster assignments and normalized centroids."""

    dataset_name: str
    embedding_model_name: str
    max_docs: int
    number_of_clusters: int
    doc_to_cluster: dict[str, int]
    centroids: np.ndarray


class ClusterArtifactStore:
    """Read and write clustering artifacts used by cluster-aware retrieval."""

    def __init__(
        self,
        *,
        dataset_name: str,
        embedding_model_name: str,
        max_docs: int | None,
        number_of_clusters: int,
    ) -> None:
        if max_docs is None:
            raise ValueError(
                "Cluster-aware retrieval currently requires a development subset. "
                "Set max_docs to a positive integer."
            )
        if max_docs <= 0:
            raise ValueError("max_docs must be positive.")
        if number_of_clusters < 2:
            raise ValueError("number_of_clusters must be at least 2.")

        self.dataset_name = dataset_name
        self.embedding_model_name = embedding_model_name
        self.max_docs = max_docs
        self.number_of_clusters = number_of_clusters

    @property
    def path(self) -> Path:
        """Return the deterministic artifact path for this configuration."""
        safe_model_name = self.embedding_model_name.replace("/", "__")
        return (
            CLUSTERS_DIR
            / self.dataset_name
            / safe_model_name
            / f"dev_{self.max_docs}_k{self.number_of_clusters}.joblib"
        )

    def exists(self) -> bool:
        """Return whether the configured artifact file exists."""
        return self.path.exists()

    def save(
        self,
        *,
        doc_ids: list[str],
        labels: np.ndarray,
        centroids: np.ndarray,
    ) -> Path:
        """Persist assignments and L2-normalized cluster centroids."""
        label_array = np.asarray(labels, dtype="int32")
        centroid_matrix = np.asarray(centroids, dtype="float32")
        if len(doc_ids) != len(label_array):
            raise ValueError("doc_ids and labels must have the same length.")
        if centroid_matrix.ndim != 2:
            raise ValueError("centroids must be a two-dimensional matrix.")
        if centroid_matrix.shape[0] != self.number_of_clusters:
            raise ValueError(
                "centroid count does not match the configured number_of_clusters."
            )

        norms = np.linalg.norm(centroid_matrix, axis=1, keepdims=True)
        normalized_centroids = centroid_matrix / np.maximum(norms, 1e-12)
        payload = {
            "dataset_name": self.dataset_name,
            "embedding_model_name": self.embedding_model_name,
            "max_docs": self.max_docs,
            "number_of_clusters": self.number_of_clusters,
            "doc_to_cluster": {
                str(doc_id): int(label)
                for doc_id, label in zip(doc_ids, label_array, strict=True)
            },
            "centroids": normalized_centroids.astype("float32"),
        }

        self.path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(payload, self.path)
        return self.path

    def load(self) -> ClusterArtifacts:
        """Load and validate the configured artifact file."""
        if not self.exists():
            raise RuntimeError(
                "Clustering artifacts are missing. Run the before/after comparison "
                "or build the cluster-aware retriever on a development subset first. "
                f"Expected: {self.path}"
            )

        payload = joblib.load(self.path)
        required = {
            "dataset_name",
            "embedding_model_name",
            "max_docs",
            "number_of_clusters",
            "doc_to_cluster",
            "centroids",
        }
        missing = required.difference(payload)
        if missing:
            raise RuntimeError(
                f"Invalid clustering artifact; missing keys: {', '.join(sorted(missing))}"
            )

        return ClusterArtifacts(
            dataset_name=str(payload["dataset_name"]),
            embedding_model_name=str(payload["embedding_model_name"]),
            max_docs=int(payload["max_docs"]),
            number_of_clusters=int(payload["number_of_clusters"]),
            doc_to_cluster={
                str(doc_id): int(cluster_id)
                for doc_id, cluster_id in dict(payload["doc_to_cluster"]).items()
            },
            centroids=np.asarray(payload["centroids"], dtype="float32"),
        )

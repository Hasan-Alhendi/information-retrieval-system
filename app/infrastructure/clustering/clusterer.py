"""Document clustering implementation with quantitative evaluation."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.metrics import davies_bouldin_score, silhouette_score

from app.infrastructure.clustering.cluster_store import ClusterArtifactStore
from app.infrastructure.datasets.dataset_loader import DatasetLoader

DEFAULT_CLUSTERING_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MAX_SILHOUETTE_SAMPLES = 2000
MAX_PROJECTION_POINTS = 2000


class DocumentClusterer:
    """Cluster documents using dense vectors and MiniBatchKMeans."""

    def __init__(
        self,
        dataset_loader: DatasetLoader | None = None,
        embedding_model_name: str = DEFAULT_CLUSTERING_EMBEDDING_MODEL,
        batch_size: int = 64,
        clustering_batch_size: int = 1024,
    ) -> None:
        self._dataset_loader = dataset_loader or DatasetLoader()
        self.embedding_model_name = embedding_model_name
        self.batch_size = batch_size
        self.clustering_batch_size = clustering_batch_size
        self._model: SentenceTransformer | None = None

    def cluster(
        self,
        dataset_name: str,
        number_of_clusters: int = 10,
        max_docs: int | None = 1000,
        sample_size: int = 5,
        persist_artifacts: bool = False,
    ) -> dict[str, Any]:
        """Cluster documents and return summaries, metrics, and PCA points."""
        doc_ids, documents, _, _ = self._dataset_loader.prepare_dataset(
            dataset_name,
            max_docs=max_docs,
        )
        if not documents:
            return {
                "dataset_name": dataset_name,
                "number_of_clusters": 0,
                "documents_count": 0,
                "embedding_model": self.embedding_model_name,
                "clustering_algorithm": "MiniBatchKMeans",
                "evaluation": {
                    "silhouette_score": None,
                    "davies_bouldin_index": None,
                    "inertia": None,
                    "silhouette_sample_size": 0,
                },
                "projection": [],
                "clusters": [],
                "artifacts_path": None,
            }

        cluster_count = max(1, min(number_of_clusters, len(documents)))
        if persist_artifacts and cluster_count != number_of_clusters:
            raise ValueError(
                "The requested number of clusters must not exceed the available documents."
            )

        embeddings = self._encode(documents)
        kmeans = MiniBatchKMeans(
            n_clusters=cluster_count,
            random_state=42,
            n_init="auto",
            batch_size=min(self.clustering_batch_size, len(documents)),
        )
        labels = kmeans.fit_predict(embeddings)

        artifact_path = None
        if persist_artifacts:
            artifact_path = ClusterArtifactStore(
                dataset_name=dataset_name,
                embedding_model_name=self.embedding_model_name,
                max_docs=max_docs,
                number_of_clusters=cluster_count,
            ).save(
                doc_ids=doc_ids,
                labels=labels,
                centroids=kmeans.cluster_centers_,
            )

        grouped_indexes: dict[int, list[int]] = defaultdict(list)
        for index, label in enumerate(labels):
            grouped_indexes[int(label)].append(index)

        clusters = []
        for cluster_id, indexes in sorted(grouped_indexes.items()):
            samples = []
            for index in indexes[:sample_size]:
                text = documents[index]
                samples.append(
                    {
                        "doc_id": doc_ids[index],
                        "preview": text[:500] + ("..." if len(text) > 500 else ""),
                    }
                )

            clusters.append(
                {
                    "cluster_id": cluster_id,
                    "size": len(indexes),
                    "top_terms": _top_terms([documents[index] for index in indexes]),
                    "samples": samples,
                }
            )

        return {
            "dataset_name": dataset_name,
            "number_of_clusters": cluster_count,
            "documents_count": len(documents),
            "embedding_model": self.embedding_model_name,
            "clustering_algorithm": "MiniBatchKMeans",
            "evaluation": _evaluate_clusters(
                embeddings=embeddings,
                labels=labels,
                inertia=float(kmeans.inertia_),
            ),
            "projection": _pca_projection(
                embeddings=embeddings,
                labels=labels,
                doc_ids=doc_ids,
            ),
            "clusters": clusters,
            "artifacts_path": None if artifact_path is None else str(artifact_path),
        }

    def _encode(self, texts: list[str]) -> np.ndarray:
        model = self._get_model()
        embeddings = model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return np.asarray(embeddings, dtype="float32")

    def _get_model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.embedding_model_name)
        return self._model


def _evaluate_clusters(
    *,
    embeddings: np.ndarray,
    labels: np.ndarray,
    inertia: float,
) -> dict[str, float | int | None]:
    """Compute internal clustering metrics with bounded Silhouette cost."""
    document_count = int(len(embeddings))
    unique_clusters = np.unique(labels)
    if len(unique_clusters) < 2 or len(unique_clusters) >= document_count:
        return {
            "silhouette_score": None,
            "davies_bouldin_index": None,
            "inertia": round(float(inertia), 6),
            "silhouette_sample_size": 0,
        }

    silhouette_sample_size = min(document_count, MAX_SILHOUETTE_SAMPLES)
    silhouette = silhouette_score(
        embeddings,
        labels,
        metric="euclidean",
        sample_size=(
            silhouette_sample_size
            if silhouette_sample_size < document_count
            else None
        ),
        random_state=42,
    )
    davies_bouldin = davies_bouldin_score(embeddings, labels)
    return {
        "silhouette_score": round(float(silhouette), 6),
        "davies_bouldin_index": round(float(davies_bouldin), 6),
        "inertia": round(float(inertia), 6),
        "silhouette_sample_size": silhouette_sample_size,
    }


def _pca_projection(
    *,
    embeddings: np.ndarray,
    labels: np.ndarray,
    doc_ids: list[str],
) -> list[dict[str, Any]]:
    """Project a bounded deterministic sample to two dimensions for display."""
    document_count = len(embeddings)
    if document_count < 2:
        return []

    point_count = min(document_count, MAX_PROJECTION_POINTS)
    if point_count == document_count:
        selected_indexes = np.arange(document_count)
    else:
        random_generator = np.random.default_rng(42)
        selected_indexes = np.sort(
            random_generator.choice(
                document_count,
                size=point_count,
                replace=False,
            )
        )

    selected_embeddings = embeddings[selected_indexes]
    coordinates = PCA(n_components=2, random_state=42).fit_transform(
        selected_embeddings
    )
    return [
        {
            "doc_id": str(doc_ids[int(index)]),
            "cluster": str(int(labels[int(index)])),
            "pc1": float(coordinates[position, 0]),
            "pc2": float(coordinates[position, 1]),
        }
        for position, index in enumerate(selected_indexes)
    ]


def _top_terms(documents: list[str], limit: int = 8) -> list[str]:
    counter: Counter[str] = Counter()
    for document in documents:
        for raw_token in document.lower().split():
            token = "".join(character for character in raw_token if character.isalpha())
            if len(token) < 4 or token in ENGLISH_STOP_WORDS:
                continue
            counter[token] += 1
    return [term for term, _ in counter.most_common(limit)]

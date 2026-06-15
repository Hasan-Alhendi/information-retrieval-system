"""Document clustering implementation."""

from collections import Counter, defaultdict
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from app.infrastructure.datasets.dataset_loader import DatasetLoader

DEFAULT_CLUSTERING_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class DocumentClusterer:
    """Clusters documents using dense vectors and MiniBatchKMeans."""

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
    ) -> dict[str, Any]:
        """Cluster documents for a dataset and return cluster summaries."""
        doc_ids, documents, _, _ = self._dataset_loader.prepare_dataset(
            dataset_name,
            max_docs=max_docs,
        )
        if not documents:
            return {
                "dataset_name": dataset_name,
                "number_of_clusters": 0,
                "documents_count": 0,
                "clusters": [],
            }

        cluster_count = max(1, min(number_of_clusters, len(documents)))
        embeddings = self._encode(documents)
        kmeans = MiniBatchKMeans(
            n_clusters=cluster_count,
            random_state=42,
            n_init="auto",
            batch_size=min(self.clustering_batch_size, len(documents)),
        )
        labels = kmeans.fit_predict(embeddings)

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
            "clusters": clusters,
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


def _top_terms(documents: list[str], limit: int = 8) -> list[str]:
    counter: Counter[str] = Counter()
    for document in documents:
        for raw_token in document.lower().split():
            token = "".join(character for character in raw_token if character.isalpha())
            if len(token) < 4 or token in ENGLISH_STOP_WORDS:
                continue
            counter[token] += 1
    return [term for term, _ in counter.most_common(limit)]

"""Dense scoring for a selected set of document identifiers."""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from app.infrastructure.vector_store.candidate_faiss_store import CandidateFaissVectorStore

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class DenseCandidateScorer:
    """Encode one query and score only lexical candidates from the full FAISS store."""

    def __init__(
        self,
        embedding_model_name: str = DEFAULT_MODEL,
        batch_size: int = 16,
    ) -> None:
        self.embedding_model_name = embedding_model_name
        self.batch_size = max(1, batch_size)
        self._model: SentenceTransformer | None = None
        self._stores: dict[str, CandidateFaissVectorStore] = {}

    def score(
        self,
        query: str,
        dataset_name: str,
        doc_ids: list[str],
    ) -> dict[str, float]:
        """Return cosine similarity for the requested document identifiers."""
        vector = self._encode(query)
        return self._store(dataset_name).score_doc_ids(vector, doc_ids)

    def prepare(self, dataset_name: str) -> None:
        """Load model and index once before interactive use or benchmarking."""
        self._get_model()
        self._store(dataset_name).load_index()
        self._encode("initialization")

    def _encode(self, query: str) -> np.ndarray:
        vectors = self._get_model().encode(
            [query],
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return np.asarray(vectors[0], dtype="float32")

    def _get_model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.embedding_model_name, device="cpu")
        return self._model

    def _store(self, dataset_name: str) -> CandidateFaissVectorStore:
        if dataset_name not in self._stores:
            safe_model_name = self.embedding_model_name.replace("/", "__")
            self._stores[dataset_name] = CandidateFaissVectorStore(
                dataset_name=dataset_name,
                model_name=f"embedding_{safe_model_name}",
            )
        return self._stores[dataset_name]

"""Embedding-based retriever implementation."""

import numpy as np
from sentence_transformers import SentenceTransformer

from app.domain.models.search_result import SearchResult
from app.infrastructure.datasets.beir_loader import BeirDatasetLoader
from app.infrastructure.vector_store.faiss_store import FaissVectorStore

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class EmbeddingRetriever:
    """Semantic retriever based on dense document and query embeddings."""

    model_name = "embedding"

    def __init__(
        self,
        dataset_loader: BeirDatasetLoader | None = None,
        embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
        max_docs: int | None = None,
        batch_size: int = 64,
    ) -> None:
        self._dataset_loader = dataset_loader or BeirDatasetLoader()
        self.embedding_model_name = embedding_model_name
        self._max_docs = max_docs
        self._batch_size = batch_size
        self._model: SentenceTransformer | None = None

    def build(self, dataset_name: str, force: bool = False, max_docs: int | None = None) -> None:
        """Build and persist the embedding vector store for a dataset."""
        active_max_docs = max_docs if max_docs is not None else self._max_docs
        vector_store = self._vector_store(dataset_name, active_max_docs)
        if not force and vector_store.exists():
            return

        doc_ids, documents, _, _ = self._dataset_loader.prepare_dataset(
            dataset_name,
            max_docs=active_max_docs,
        )
        embeddings = self._encode(documents)
        vector_store.build(embeddings=embeddings, doc_ids=doc_ids, documents=documents)

    def search(self, query: str, dataset_name: str, top_k: int = 10) -> list[SearchResult]:
        """Search documents using dense embeddings and FAISS."""
        self.ensure_ready(dataset_name)
        vector_store = self._vector_store(dataset_name, self._max_docs)
        doc_ids, documents = vector_store.load_metadata()

        query_vector = self._encode([query])[0]
        matches = vector_store.search(query_vector=query_vector, top_k=top_k)

        results: list[SearchResult] = []
        for rank, (row_index, score) in enumerate(matches, start=1):
            if score <= 0:
                continue
            results.append(
                SearchResult(
                    doc_id=doc_ids[row_index],
                    score=score,
                    rank=rank,
                    text=documents[row_index],
                    metadata={
                        "model": self.model_name,
                        "embedding_model": self.embedding_model_name,
                    },
                )
            )
        return results

    def ensure_ready(self, dataset_name: str) -> None:
        """Build the embedding vector store if it does not exist."""
        vector_store = self._vector_store(dataset_name, self._max_docs)
        if not vector_store.exists():
            self.build(dataset_name)

    def _encode(self, texts: list[str]) -> np.ndarray:
        model = self._get_model()
        embeddings = model.encode(
            texts,
            batch_size=self._batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return np.asarray(embeddings, dtype="float32")

    def _get_model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.embedding_model_name)
        return self._model

    def _vector_store(self, dataset_name: str, max_docs: int | None = None) -> FaissVectorStore:
        safe_model_name = self.embedding_model_name.replace("/", "__")
        store_name = f"{self.model_name}_{safe_model_name}"
        return FaissVectorStore(dataset_name=dataset_name, model_name=store_name, max_docs=max_docs)

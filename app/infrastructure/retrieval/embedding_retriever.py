"""Embedding-based retriever implementation."""

from __future__ import annotations

import time

import numpy as np
from sentence_transformers import SentenceTransformer

from app.domain.models.search_result import SearchResult
from app.infrastructure.datasets.dataset_loader import DatasetLoader
from app.infrastructure.datasets.dataset_registry import get_dataset_config
from app.infrastructure.retrieval.full_embedding_index import FullEmbeddingIndexBuilder
from app.infrastructure.vector_store.faiss_store import FaissVectorStore
from app.infrastructure.vector_store.incremental_faiss_store import (
    IncrementalFaissVectorStore,
)

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class EmbeddingRetriever:
    """Semantic retriever based on dense document and query embeddings."""

    model_name = "embedding"

    def __init__(
        self,
        dataset_loader: DatasetLoader | None = None,
        embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
        max_docs: int | None = None,
        batch_size: int = 64,
        full_checkpoint_size: int = 5000,
    ) -> None:
        self._dataset_loader = dataset_loader or DatasetLoader()
        self.embedding_model_name = embedding_model_name
        self._max_docs = max_docs
        self._batch_size = max(1, batch_size)
        self._full_checkpoint_size = max(1, full_checkpoint_size)
        self._model: SentenceTransformer | None = None
        self._full_stores: dict[str, IncrementalFaissVectorStore] = {}
        self._development_stores: dict[tuple[str, int | None], FaissVectorStore] = {}

    def build(self, dataset_name: str, force: bool = False, max_docs: int | None = None) -> None:
        """Build and persist the embedding vector store for a dataset."""
        active_max_docs = max_docs if max_docs is not None else self._max_docs
        if self._uses_full_store(dataset_name, active_max_docs):
            vector_store = self._full_vector_store(dataset_name)
            builder = FullEmbeddingIndexBuilder(
                dataset_name,
                encoder=self._encode_documents,
                vector_store=vector_store,
                dataset_loader=self._dataset_loader,
                checkpoint_size=self._full_checkpoint_size,
            )
            builder.build(force=force)
            return

        vector_store = self._development_vector_store(dataset_name, active_max_docs)
        if not force and vector_store.exists():
            return

        doc_ids, documents, _, _ = self._dataset_loader.prepare_dataset(
            dataset_name,
            max_docs=active_max_docs,
        )
        embeddings = self._encode(documents, show_progress=True)
        vector_store.build(embeddings=embeddings, doc_ids=doc_ids, documents=documents)

    def search(self, query: str, dataset_name: str, top_k: int = 10) -> list[SearchResult]:
        """Search documents using dense embeddings and FAISS."""
        if self._uses_full_store(dataset_name, self._max_docs):
            return self._search_full(query, dataset_name, top_k)
        return self._search_development(query, dataset_name, top_k)

    def ensure_ready(self, dataset_name: str) -> None:
        """Ensure the selected vector-store mode is ready."""
        if self._uses_full_store(dataset_name, self._max_docs):
            vector_store = self._full_vector_store(dataset_name)
            if not vector_store.exists():
                status = vector_store.status()
                raise RuntimeError(
                    "The full embedding index is not finalized. Run "
                    "scripts/build_indexes.py with --model embedding and no "
                    f"--max-docs. Current progress: {status['processed_documents']} documents."
                )
            return

        vector_store = self._development_vector_store(dataset_name, self._max_docs)
        if not vector_store.exists():
            self.build(dataset_name)

    def full_status(self, dataset_name: str) -> dict[str, object]:
        """Return full dense-index progress for CLI diagnostics."""
        return self._full_vector_store(dataset_name).status()

    def _search_full(
        self,
        query: str,
        dataset_name: str,
        top_k: int,
    ) -> list[SearchResult]:
        self.ensure_ready(dataset_name)
        vector_store = self._full_vector_store(dataset_name)
        start = time.perf_counter()
        query_vector = self._encode([query], show_progress=False)[0]
        matches = vector_store.search(query_vector=query_vector, top_k=top_k)
        records = vector_store.load_records([row_index for row_index, _ in matches])
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        results: list[SearchResult] = []
        for rank, (row_index, score) in enumerate(matches, start=1):
            if score <= 0 or row_index not in records:
                continue
            record = records[row_index]
            title = record["title"]
            text = record["text"]
            display_text = f"{title}\n{text}".strip() if title else text
            results.append(
                SearchResult(
                    doc_id=record["doc_id"],
                    score=score,
                    rank=rank,
                    title=title,
                    text=display_text,
                    metadata={
                        **record["metadata"],
                        "model": self.model_name,
                        "embedding_model": self.embedding_model_name,
                        "storage": "incremental_faiss_full",
                        "query_time_ms": round(elapsed_ms, 3),
                    },
                )
            )
        return results

    def _search_development(
        self,
        query: str,
        dataset_name: str,
        top_k: int,
    ) -> list[SearchResult]:
        self.ensure_ready(dataset_name)
        vector_store = self._development_vector_store(dataset_name, self._max_docs)
        doc_ids, documents = vector_store.load_metadata()

        query_vector = self._encode([query], show_progress=False)[0]
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
                        "storage": "faiss_development",
                    },
                )
            )
        return results

    def _encode_documents(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts, show_progress=True)

    def _encode(self, texts: list[str], *, show_progress: bool) -> np.ndarray:
        model = self._get_model()
        embeddings = model.encode(
            texts,
            batch_size=self._batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return np.asarray(embeddings, dtype="float32")

    def _get_model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.embedding_model_name, device="cpu")
        return self._model

    def _full_vector_store(self, dataset_name: str) -> IncrementalFaissVectorStore:
        if dataset_name not in self._full_stores:
            safe_model_name = self.embedding_model_name.replace("/", "__")
            store_name = f"{self.model_name}_{safe_model_name}"
            self._full_stores[dataset_name] = IncrementalFaissVectorStore(
                dataset_name=dataset_name,
                model_name=store_name,
            )
        return self._full_stores[dataset_name]

    def _development_vector_store(
        self,
        dataset_name: str,
        max_docs: int | None,
    ) -> FaissVectorStore:
        cache_key = (dataset_name, max_docs)
        if cache_key not in self._development_stores:
            safe_model_name = self.embedding_model_name.replace("/", "__")
            store_name = f"{self.model_name}_{safe_model_name}"
            self._development_stores[cache_key] = FaissVectorStore(
                dataset_name=dataset_name,
                model_name=store_name,
                max_docs=max_docs,
            )
        return self._development_stores[cache_key]

    @staticmethod
    def _uses_full_store(dataset_name: str, max_docs: int | None) -> bool:
        config = get_dataset_config(dataset_name, include_experimental=True)
        return max_docs is None and config.external_id is not None

"""Batch builder for full-corpus dense embedding indexes."""

from __future__ import annotations

from collections.abc import Callable
from itertools import islice

import numpy as np

from app.domain.models.document import Document
from app.infrastructure.datasets.dataset_loader import DatasetLoader
from app.infrastructure.datasets.dataset_registry import get_dataset_config
from app.infrastructure.vector_store.incremental_faiss_store import (
    IncrementalFaissVectorStore,
)

Encoder = Callable[[list[str]], np.ndarray]


class FullEmbeddingIndexBuilder:
    """Stream documents, encode batches, and checkpoint them into FAISS."""

    def __init__(
        self,
        dataset_name: str,
        *,
        encoder: Encoder,
        vector_store: IncrementalFaissVectorStore,
        dataset_loader: DatasetLoader | None = None,
        checkpoint_size: int = 5000,
        document_limit: int | None = None,
    ) -> None:
        self.dataset_name = dataset_name
        self.encoder = encoder
        self.vector_store = vector_store
        self.dataset_loader = dataset_loader or DatasetLoader()
        self.checkpoint_size = max(1, checkpoint_size)
        self.document_limit = document_limit
        self.config = get_dataset_config(dataset_name, include_experimental=True)

    def build(self, *, force: bool = False) -> int:
        """Build or resume the dense index and return the final document count."""
        if force:
            self.vector_store.reset()
        if self.vector_store.exists():
            return self.vector_store.processed_documents()

        processed = self.vector_store.prepare_resume()
        source = self.dataset_loader.iter_documents(
            self.dataset_name,
            max_docs=self.document_limit,
        )
        if processed:
            source = islice(source, processed, None)

        while True:
            documents = list(islice(source, self.checkpoint_size))
            if not documents:
                break

            texts = [
                dense_document_text(document, self.config.processing_profile)
                for document in documents
            ]
            embeddings = self.encoder(texts)
            processed = self.vector_store.add_checkpoint(embeddings, documents)
            print(
                f"Encoded and checkpointed {processed:,} documents "
                f"for {self.dataset_name}"
            )

        self.vector_store.finalize(expected_count=processed)
        return processed


def dense_document_text(document: Document, processing_profile: str) -> str:
    """Create natural-language input for a dense encoder.

    Argument documents repeat the title once to preserve its topical signal while
    keeping the original body untouched for semantic encoding.
    """
    title = (document.title or "").strip()
    body = document.text.strip()
    if title and processing_profile == "argument":
        return f"{title}. {title}. {body}".strip()
    if title:
        return f"{title}. {body}".strip()
    return body

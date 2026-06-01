"""FAISS vector store implementation."""

from pathlib import Path

import faiss
import numpy as np

from app.config import VECTOR_STORES_DIR
from app.infrastructure.storage.joblib_store import load_object, save_object


class FaissVectorStore:
    """Vector store abstraction backed by FAISS inner-product search."""

    def __init__(self, dataset_name: str, model_name: str, max_docs: int | None = None) -> None:
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.max_docs = max_docs

    def build(self, embeddings: np.ndarray, doc_ids: list[str], documents: list[str]) -> None:
        """Build and persist a FAISS index.

        Embeddings are normalized so inner product becomes cosine similarity.
        """
        matrix = np.asarray(embeddings, dtype="float32")
        faiss.normalize_L2(matrix)

        index = faiss.IndexFlatIP(matrix.shape[1])
        index.add(matrix)

        paths = self._paths()
        paths["base_dir"].mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(paths["index"]))
        save_object(doc_ids, paths["doc_ids"])
        save_object(documents, paths["documents"])

    def search(self, query_vector: np.ndarray, top_k: int = 10) -> list[tuple[int, float]]:
        """Search the FAISS index and return row indexes with scores."""
        index = self.load_index()
        query = np.asarray(query_vector, dtype="float32")
        if query.ndim == 1:
            query = query.reshape(1, -1)
        faiss.normalize_L2(query)

        scores, indexes = index.search(query, top_k)
        return [
            (int(row_index), float(score))
            for row_index, score in zip(indexes[0], scores[0], strict=True)
            if row_index >= 0
        ]

    def load_index(self):
        """Load the persisted FAISS index."""
        return faiss.read_index(str(self._paths()["index"]))

    def load_metadata(self) -> tuple[list[str], list[str]]:
        """Load document IDs and original document texts."""
        paths = self._paths()
        return load_object(paths["doc_ids"]), load_object(paths["documents"])

    def exists(self) -> bool:
        """Return true if the vector store artifacts exist."""
        paths = self._paths()
        return paths["index"].exists() and paths["doc_ids"].exists() and paths["documents"].exists()

    def _paths(self) -> dict[str, Path]:
        base_dir = VECTOR_STORES_DIR / self.dataset_name / self.model_name
        if self.max_docs is not None:
            base_dir = base_dir / f"dev_{self.max_docs}"
        return {
            "base_dir": base_dir,
            "index": base_dir / "faiss.index",
            "doc_ids": base_dir / "doc_ids.joblib",
            "documents": base_dir / "documents.joblib",
        }

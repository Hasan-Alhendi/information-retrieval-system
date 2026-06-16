"""Candidate-level scoring on top of the full FAISS vector store."""

from __future__ import annotations

import faiss
import numpy as np

from app.infrastructure.vector_store.incremental_faiss_store import (
    IncrementalFaissVectorStore,
)


class CandidateFaissVectorStore(IncrementalFaissVectorStore):
    """Score selected document IDs without running a full-index nearest-neighbor search."""

    def score_doc_ids(
        self,
        query_vector: np.ndarray,
        doc_ids: list[str],
    ) -> dict[str, float]:
        """Return cosine scores for the requested documents in input order."""
        if not doc_ids:
            return {}
        if not self.exists():
            raise RuntimeError("The full FAISS vector store is not finalized.")

        query = np.asarray(query_vector, dtype="float32")
        if query.ndim == 1:
            query = query.reshape(1, -1)
        faiss.normalize_L2(query)

        row_indexes = self._row_indexes_for_doc_ids(doc_ids)
        index = self.load_index()
        scores: dict[str, float] = {}
        for doc_id in doc_ids:
            row_index = row_indexes.get(doc_id)
            if row_index is None:
                continue
            vector = np.asarray(index.reconstruct(row_index), dtype="float32")
            scores[doc_id] = float(np.dot(query[0], vector))
        return scores

    def _row_indexes_for_doc_ids(self, doc_ids: list[str]) -> dict[str, int]:
        mapping: dict[str, int] = {}
        for start in range(0, len(doc_ids), 900):
            chunk = doc_ids[start : start + 900]
            placeholders = ",".join("?" for _ in chunk)
            with self._connect(read_only=True) as connection:
                rows = connection.execute(
                    f"SELECT doc_id, row_index FROM documents WHERE doc_id IN ({placeholders})",
                    chunk,
                )
                mapping.update({str(doc_id): int(row_index) for doc_id, row_index in rows})
        return mapping

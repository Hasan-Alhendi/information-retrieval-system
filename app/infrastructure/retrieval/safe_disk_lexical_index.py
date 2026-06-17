"""Safer finalization for large SQLite lexical indexes."""

from __future__ import annotations

import math
import sqlite3

from app.infrastructure.retrieval.disk_lexical_index import (
    DiskLexicalIndex,
    _tfidf_idf,
)


class SafeDiskLexicalIndex(DiskLexicalIndex):
    """Finalize TF-IDF norms in bounded chunks using one SQLite connection.

    The original implementation kept a very large read cursor open on one
    connection while committing norm updates through a second connection. On
    long Windows builds this can stress WAL/temp-file handling. This version
    fetches bounded document ranges, closes each read cursor before writing,
    and checkpoints progress after every chunk.
    """

    norm_chunk_documents = 5000

    def _finalize_tfidf_norms(self) -> None:
        with self._connect() as connection:
            quick_check = connection.execute("PRAGMA quick_check").fetchone()
            if not quick_check or str(quick_check[0]).lower() != "ok":
                raise sqlite3.DatabaseError(
                    "SQLite quick_check failed before TF-IDF finalization: "
                    f"{quick_check[0] if quick_check else 'unknown error'}"
                )

            total_documents = self._metadata_int(connection, "total_documents")
            last_completed = self._metadata_int(
                connection,
                "tfidf_norms_processed_document",
            )
            idf_by_term = {
                int(term_id): _tfidf_idf(total_documents, int(document_frequency))
                for term_id, document_frequency in connection.execute(
                    "SELECT term_id, df FROM terms"
                ).fetchall()
            }

            start_document = last_completed + 1
            while start_document <= total_documents:
                end_document = min(
                    total_documents,
                    start_document + self.norm_chunk_documents - 1,
                )
                rows = connection.execute(
                    """
                    SELECT document_id, term_id, tf
                    FROM postings
                    WHERE document_id BETWEEN ? AND ?
                    ORDER BY document_id, term_id
                    """,
                    (start_document, end_document),
                ).fetchall()

                norm_squared: dict[int, float] = {}
                for document_id, term_id, term_frequency in rows:
                    document_id = int(document_id)
                    inverse_document_frequency = idf_by_term[int(term_id)]
                    weight = int(term_frequency) * inverse_document_frequency
                    norm_squared[document_id] = (
                        norm_squared.get(document_id, 0.0) + weight * weight
                    )

                updates = [
                    (math.sqrt(norm_squared.get(document_id, 0.0)), document_id)
                    for document_id in range(start_document, end_document + 1)
                ]
                self._write_norm_updates(connection, updates)
                self._set_metadata(
                    connection,
                    "tfidf_norms_processed_document",
                    str(end_document),
                )
                connection.commit()
                print(
                    f"Finalized TF-IDF norms for documents "
                    f"{start_document:,}-{end_document:,}"
                )
                start_document = end_document + 1

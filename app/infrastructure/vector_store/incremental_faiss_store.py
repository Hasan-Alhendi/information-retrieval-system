"""Checkpointed FAISS vector store for full-corpus dense retrieval."""

from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from app.config import VECTOR_STORES_DIR
from app.domain.models.document import Document


class IncrementalFaissVectorStore:
    """Persist an exact cosine-similarity FAISS index incrementally.

    Document metadata is stored in a small companion SQLite database. Each
    checkpoint commits metadata first and then atomically replaces the FAISS
    index file. If a process stops between those steps, the next run rolls the
    metadata back to the last durable FAISS row count and safely resumes.
    """

    def __init__(
        self,
        dataset_name: str,
        model_name: str,
        *,
        base_dir: Path | None = None,
    ) -> None:
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.base_dir = base_dir or (
            VECTOR_STORES_DIR / dataset_name / model_name / "full"
        )
        self.index_path = self.base_dir / "faiss.index"
        self.metadata_path = self.base_dir / "metadata.sqlite3"
        self._index: Any | None = None

    def prepare_resume(self) -> int:
        """Create storage, repair an interrupted checkpoint, and return progress."""
        self.base_dir.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            self._create_schema(connection)
            connection.commit()
        return self._repair_state()

    def reset(self) -> None:
        """Delete all full-vector artifacts and cached state."""
        self._index = None
        for path in (
            self.index_path,
            self.metadata_path,
            Path(f"{self.metadata_path}-wal"),
            Path(f"{self.metadata_path}-shm"),
            self.index_path.with_suffix(".index.tmp"),
        ):
            if path.exists():
                path.unlink()

    def add_checkpoint(
        self,
        embeddings: np.ndarray,
        documents: list[Document],
    ) -> int:
        """Append one durable checkpoint and return the total row count."""
        if len(documents) == 0:
            return self.processed_documents()

        matrix = np.asarray(embeddings, dtype="float32")
        if matrix.ndim != 2:
            raise ValueError("Embeddings must be a two-dimensional matrix.")
        if matrix.shape[0] != len(documents):
            raise ValueError("Embedding and document counts must match.")
        faiss.normalize_L2(matrix)

        processed = self.prepare_resume()
        index = self._load_or_create_index(matrix.shape[1])
        if int(index.ntotal) != processed:
            raise RuntimeError(
                "FAISS row count and metadata progress disagree after repair."
            )
        if int(index.d) != int(matrix.shape[1]):
            raise ValueError(
                f"Embedding dimension changed from {index.d} to {matrix.shape[1]}."
            )

        target_count = processed + len(documents)
        with self._connect() as connection:
            rows = [
                (
                    processed + offset,
                    document.doc_id,
                    document.title,
                    document.text,
                    json.dumps(document.metadata, ensure_ascii=False),
                )
                for offset, document in enumerate(documents)
            ]
            connection.executemany(
                """
                INSERT INTO documents(row_index, doc_id, title, text, metadata_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                rows,
            )
            self._set_metadata(connection, "processed_documents", str(target_count))
            self._set_metadata(connection, "dimension", str(matrix.shape[1]))
            self._set_metadata(connection, "finalized", "0")
            connection.commit()

        index.add(matrix)
        self._write_index_atomically(index)
        self._index = index

        with self._connect() as connection:
            self._set_metadata(connection, "last_checkpoint_unix", str(time.time()))
            connection.commit()
        return target_count

    def finalize(self, expected_count: int | None = None) -> None:
        """Mark the store ready after validating index and metadata counts."""
        processed = self.prepare_resume()
        index = self.load_index()
        if int(index.ntotal) != processed:
            raise RuntimeError("Cannot finalize inconsistent FAISS metadata.")
        if expected_count is not None and processed != expected_count:
            raise RuntimeError(
                f"Expected {expected_count} vectors but found {processed}."
            )

        with self._connect() as connection:
            self._set_metadata(connection, "processed_documents", str(processed))
            self._set_metadata(connection, "finalized", "1")
            self._set_metadata(connection, "completed_at_unix", str(time.time()))
            connection.commit()

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
    ) -> list[tuple[int, float]]:
        """Search the finalized index and return row indexes with scores."""
        if not self.exists():
            raise RuntimeError("The full FAISS vector store is not finalized.")
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

    def load_records(self, row_indexes: list[int]) -> dict[int, dict[str, Any]]:
        """Load only metadata for the requested FAISS rows."""
        if not row_indexes:
            return {}
        placeholders = ",".join("?" for _ in row_indexes)
        with self._connect(read_only=True) as connection:
            rows = connection.execute(
                f"""
                SELECT row_index, doc_id, title, text, metadata_json
                FROM documents
                WHERE row_index IN ({placeholders})
                """,
                row_indexes,
            )
            return {
                int(row_index): {
                    "doc_id": str(doc_id),
                    "title": title,
                    "text": str(text),
                    "metadata": json.loads(metadata_json or "{}"),
                }
                for row_index, doc_id, title, text, metadata_json in rows
            }

    def load_index(self):
        """Load and cache the FAISS index once per Python process."""
        if self._index is None:
            if not self.index_path.exists():
                raise RuntimeError("FAISS index file does not exist.")
            self._index = faiss.read_index(str(self.index_path))
        return self._index

    def processed_documents(self) -> int:
        """Return the last durable document count."""
        if not self.metadata_path.exists():
            return 0
        with self._connect(read_only=True) as connection:
            return self._metadata_int(connection, "processed_documents")

    def exists(self) -> bool:
        """Return whether the vector store is finalized and searchable."""
        if not self.index_path.exists() or not self.metadata_path.exists():
            return False
        with self._connect(read_only=True) as connection:
            finalized = self._metadata_value(connection, "finalized") == "1"
            processed = self._metadata_int(connection, "processed_documents")
        if not finalized:
            return False
        try:
            return int(self.load_index().ntotal) == processed
        except RuntimeError:
            return False

    def status(self) -> dict[str, Any]:
        """Return progress and artifact sizes for diagnostics."""
        processed = self.processed_documents()
        index_rows = 0
        dimension = 0
        if self.index_path.exists():
            index = self.load_index()
            index_rows = int(index.ntotal)
            dimension = int(index.d)
        finalized = False
        if self.metadata_path.exists():
            with self._connect(read_only=True) as connection:
                finalized = self._metadata_value(connection, "finalized") == "1"
        return {
            "dataset_name": self.dataset_name,
            "model_name": self.model_name,
            "base_dir": str(self.base_dir),
            "processed_documents": processed,
            "index_rows": index_rows,
            "dimension": dimension,
            "finalized": finalized and processed == index_rows,
            "index_size_bytes": self.index_path.stat().st_size
            if self.index_path.exists()
            else 0,
            "metadata_size_bytes": self.metadata_path.stat().st_size
            if self.metadata_path.exists()
            else 0,
        }

    def _repair_state(self) -> int:
        index_count = 0
        if self.index_path.exists():
            self._index = faiss.read_index(str(self.index_path))
            index_count = int(self._index.ntotal)

        with self._connect() as connection:
            self._create_schema(connection)
            row = connection.execute("SELECT COUNT(*) FROM documents").fetchone()
            metadata_count = int(row[0]) if row else 0

            if metadata_count > index_count:
                connection.execute(
                    "DELETE FROM documents WHERE row_index >= ?",
                    (index_count,),
                )
                metadata_count = index_count
                self._set_metadata(connection, "finalized", "0")
            elif index_count > metadata_count:
                raise RuntimeError(
                    "FAISS contains rows that have no durable metadata. Rebuild with --force."
                )

            self._set_metadata(
                connection,
                "processed_documents",
                str(metadata_count),
            )
            connection.commit()
            return metadata_count

    def _load_or_create_index(self, dimension: int):
        if self.index_path.exists():
            return self.load_index()
        self._index = faiss.IndexFlatIP(dimension)
        return self._index

    def _write_index_atomically(self, index: Any) -> None:
        temporary_path = self.index_path.with_suffix(".index.tmp")
        faiss.write_index(index, str(temporary_path))
        os.replace(temporary_path, self.index_path)

    def _connect(self, *, read_only: bool = False) -> sqlite3.Connection:
        self.base_dir.mkdir(parents=True, exist_ok=True)
        if read_only:
            uri = f"file:{self.metadata_path.resolve().as_posix()}?mode=ro"
            connection = sqlite3.connect(uri, uri=True, timeout=60.0)
        else:
            connection = sqlite3.connect(self.metadata_path, timeout=60.0)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("PRAGMA temp_store=MEMORY")
        return connection

    @staticmethod
    def _create_schema(connection: sqlite3.Connection) -> None:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS documents (
                row_index INTEGER PRIMARY KEY,
                doc_id TEXT NOT NULL UNIQUE,
                title TEXT,
                text TEXT NOT NULL,
                metadata_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            """
        )

    @staticmethod
    def _metadata_value(connection: sqlite3.Connection, key: str) -> str | None:
        row = connection.execute(
            "SELECT value FROM metadata WHERE key = ?",
            (key,),
        ).fetchone()
        return None if row is None else str(row[0])

    def _metadata_int(self, connection: sqlite3.Connection, key: str) -> int:
        value = self._metadata_value(connection, key)
        return int(value) if value is not None else 0

    @staticmethod
    def _set_metadata(connection: sqlite3.Connection, key: str, value: str) -> None:
        connection.execute(
            """
            INSERT INTO metadata(key, value) VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (key, value),
        )

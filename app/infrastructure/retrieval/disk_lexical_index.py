"""Disk-backed lexical index for full-corpus BM25 and TF-IDF retrieval."""

from __future__ import annotations

import heapq
import json
import math
import sqlite3
import time
from collections import Counter, defaultdict
from itertools import islice
from pathlib import Path
from typing import Any, Iterable

from app.domain.models.document import Document
from app.domain.models.search_result import SearchResult
from app.infrastructure.datasets.dataset_loader import DatasetLoader
from app.infrastructure.datasets.dataset_registry import get_dataset_config
from app.infrastructure.preprocessing.spacy_preprocessor import (
    PREPROCESSING_BACKEND_TAG,
    SpacyPreprocessor,
)
from app.infrastructure.storage.index_store import get_index_dir


class DiskLexicalIndex:
    """Build and search a compact SQLite inverted index in batches."""

    def __init__(
        self,
        dataset_name: str,
        *,
        dataset_loader: DatasetLoader | None = None,
        preprocessor: SpacyPreprocessor | None = None,
        batch_size: int = 128,
        database_path: Path | None = None,
    ) -> None:
        self.dataset_name = dataset_name
        self.config = get_dataset_config(dataset_name, include_experimental=True)
        if not self.config.external_id:
            raise ValueError(
                f"Dataset '{dataset_name}' does not expose a streaming external_id."
            )
        self.dataset_loader = dataset_loader or DatasetLoader()
        self.preprocessor = preprocessor or SpacyPreprocessor()
        self.batch_size = max(1, batch_size)
        self.base_dir = get_index_dir(dataset_name, "disk_lexical") / "full"
        self.database_path = database_path or (self.base_dir / "lexical.sqlite3")

    def exists(self) -> bool:
        """Return whether the full index is finalized and ready for search."""
        if not self.database_path.exists():
            return False
        with self._connect(read_only=True) as connection:
            return self._metadata_value(connection, "finalized") == "1"

    def build(self, *, force: bool = False) -> None:
        """Build the full inverted index with checkpointed batch commits."""
        if force:
            self._remove_database_files()

        self.base_dir.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            self._create_schema(connection)
            if self._metadata_value(connection, "finalized") == "1":
                return

            processed_documents = self._metadata_int(connection, "processed_documents")
            term_cache = {
                str(term): int(term_id)
                for term_id, term in connection.execute("SELECT term_id, term FROM terms")
            }
            next_term_id = max(term_cache.values(), default=0) + 1
            next_document_id = processed_documents + 1

            source = self.dataset_loader.iter_documents(self.dataset_name)
            if processed_documents:
                source = islice(source, processed_documents, None)

            while True:
                documents = list(islice(source, self.batch_size))
                if not documents:
                    break

                next_term_id, next_document_id = self._index_batch(
                    connection=connection,
                    documents=documents,
                    term_cache=term_cache,
                    next_term_id=next_term_id,
                    next_document_id=next_document_id,
                )
                processed_documents += len(documents)
                self._set_metadata(connection, "processed_documents", str(processed_documents))
                self._set_metadata(connection, "next_term_id", str(next_term_id))
                self._set_metadata(connection, "preprocessing", PREPROCESSING_BACKEND_TAG)
                self._set_metadata(connection, "processing_profile", self.config.processing_profile)
                connection.commit()
                print(f"Indexed {processed_documents:,} documents for {self.dataset_name}")

            total_documents = self._document_count(connection)
            average_document_length = self._average_document_length(connection)
            self._set_metadata(connection, "total_documents", str(total_documents))
            self._set_metadata(connection, "average_document_length", str(average_document_length))
            connection.commit()

        self._finalize_tfidf_norms()

        with self._connect() as connection:
            self._set_metadata(connection, "finalized", "1")
            self._set_metadata(connection, "completed_at_unix", str(time.time()))
            connection.commit()
            connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")

    def search_bm25(
        self,
        query: str,
        *,
        top_k: int = 10,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> list[SearchResult]:
        """Search the full corpus using BM25 over the disk-backed postings."""
        self._require_ready()
        start = time.perf_counter()
        processed_query = self.preprocessor.preprocess(
            query,
            profile=self.config.processing_profile,
        )
        terms = list(dict.fromkeys(processed_query.split()))
        if not terms:
            return []

        with self._connect(read_only=True) as connection:
            total_documents = self._metadata_int(connection, "total_documents")
            average_document_length = self._metadata_float(
                connection,
                "average_document_length",
            )
            rows = self._matching_postings(connection, terms)
            scores: dict[int, float] = defaultdict(float)

            for term, document_id, term_frequency, document_length, _, document_frequency in rows:
                del term
                if document_frequency <= 0 or total_documents <= 0:
                    continue
                inverse_document_frequency = math.log(
                    1.0
                    + (
                        total_documents
                        - document_frequency
                        + 0.5
                    )
                    / (document_frequency + 0.5)
                )
                length_ratio = (
                    document_length / average_document_length
                    if average_document_length > 0
                    else 1.0
                )
                denominator = term_frequency + k1 * (1.0 - b + b * length_ratio)
                if denominator <= 0:
                    continue
                scores[document_id] += inverse_document_frequency * (
                    term_frequency * (k1 + 1.0)
                ) / denominator

            ranked = heapq.nlargest(top_k, scores.items(), key=lambda item: item[1])
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            return self._load_results(
                connection,
                ranked,
                model_name="bm25",
                elapsed_ms=elapsed_ms,
                extra_metadata={"k1": k1, "b": b},
            )

    def search_tfidf(self, query: str, *, top_k: int = 10) -> list[SearchResult]:
        """Search the full corpus using cosine-normalized TF-IDF."""
        self._require_ready()
        start = time.perf_counter()
        processed_query = self.preprocessor.preprocess(
            query,
            profile=self.config.processing_profile,
        )
        query_counts = Counter(processed_query.split())
        if not query_counts:
            return []

        with self._connect(read_only=True) as connection:
            total_documents = self._metadata_int(connection, "total_documents")
            rows = self._matching_postings(connection, list(query_counts))
            dot_products: dict[int, float] = defaultdict(float)
            query_weights: dict[str, float] = {}

            for term, document_id, term_frequency, _, document_norm, document_frequency in rows:
                if term not in query_weights:
                    inverse_document_frequency = _tfidf_idf(
                        total_documents,
                        document_frequency,
                    )
                    query_weights[term] = query_counts[term] * inverse_document_frequency
                inverse_document_frequency = _tfidf_idf(
                    total_documents,
                    document_frequency,
                )
                document_weight = term_frequency * inverse_document_frequency
                dot_products[document_id] += document_weight * query_weights[term]

            query_norm = math.sqrt(sum(weight * weight for weight in query_weights.values()))
            if query_norm <= 0:
                return []

            document_norms = self._document_norms(connection, dot_products)
            scores: dict[int, float] = {}
            for document_id, dot_product in dot_products.items():
                document_norm = document_norms.get(document_id, 0.0)
                if document_norm <= 0:
                    continue
                scores[document_id] = dot_product / (document_norm * query_norm)

            ranked = heapq.nlargest(top_k, scores.items(), key=lambda item: item[1])
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            return self._load_results(
                connection,
                ranked,
                model_name="tfidf",
                elapsed_ms=elapsed_ms,
            )

    def status(self) -> dict[str, Any]:
        """Return build status for CLI and UI diagnostics."""
        if not self.database_path.exists():
            return {
                "dataset_name": self.dataset_name,
                "database_path": str(self.database_path),
                "exists": False,
                "finalized": False,
                "processed_documents": 0,
                "size_bytes": 0,
            }
        with self._connect(read_only=True) as connection:
            return {
                "dataset_name": self.dataset_name,
                "database_path": str(self.database_path),
                "exists": True,
                "finalized": self._metadata_value(connection, "finalized") == "1",
                "processed_documents": self._metadata_int(
                    connection,
                    "processed_documents",
                ),
                "total_documents": self._metadata_int(connection, "total_documents"),
                "processing_profile": self._metadata_value(
                    connection,
                    "processing_profile",
                ),
                "size_bytes": self.database_path.stat().st_size,
            }

    def _index_batch(
        self,
        *,
        connection: sqlite3.Connection,
        documents: list[Document],
        term_cache: dict[str, int],
        next_term_id: int,
        next_document_id: int,
    ) -> tuple[int, int]:
        index_texts = [_index_text(document, self.config.processing_profile) for document in documents]
        normalized_documents = self.preprocessor.preprocess_many(
            index_texts,
            batch_size=self.batch_size,
            profile=self.config.processing_profile,
        )

        document_rows: list[tuple[Any, ...]] = []
        posting_rows: list[tuple[int, int, int]] = []
        term_df_updates: Counter[int] = Counter()
        new_term_rows: list[tuple[int, str, int]] = []

        for document, normalized_text in zip(documents, normalized_documents, strict=True):
            document_id = next_document_id
            next_document_id += 1
            token_counts = Counter(normalized_text.split())
            document_rows.append(
                (
                    document_id,
                    document.doc_id,
                    document.title,
                    document.text,
                    json.dumps(document.metadata, ensure_ascii=False),
                    sum(token_counts.values()),
                )
            )

            for term, term_frequency in token_counts.items():
                term_id = term_cache.get(term)
                if term_id is None:
                    term_id = next_term_id
                    next_term_id += 1
                    term_cache[term] = term_id
                    new_term_rows.append((term_id, term, 0))
                posting_rows.append((term_id, document_id, int(term_frequency)))
                term_df_updates[term_id] += 1

        connection.executemany(
            """
            INSERT INTO documents(id, doc_id, title, text, metadata_json, length)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            document_rows,
        )
        if new_term_rows:
            connection.executemany(
                "INSERT INTO terms(term_id, term, df) VALUES (?, ?, ?)",
                new_term_rows,
            )
        if posting_rows:
            connection.executemany(
                "INSERT INTO postings(term_id, document_id, tf) VALUES (?, ?, ?)",
                posting_rows,
            )
        connection.executemany(
            "UPDATE terms SET df = df + ? WHERE term_id = ?",
            [(increment, term_id) for term_id, increment in term_df_updates.items()],
        )
        return next_term_id, next_document_id

    def _finalize_tfidf_norms(self) -> None:
        with self._connect(read_only=True) as read_connection:
            total_documents = self._metadata_int(read_connection, "total_documents")
            last_completed_document = self._metadata_int(
                read_connection,
                "tfidf_norms_processed_document",
            )
            idf_by_term = {
                int(term_id): _tfidf_idf(total_documents, int(document_frequency))
                for term_id, document_frequency in read_connection.execute(
                    "SELECT term_id, df FROM terms"
                )
            }
            cursor = read_connection.execute(
                """
                SELECT document_id, term_id, tf
                FROM postings
                WHERE document_id > ?
                ORDER BY document_id, term_id
                """,
                (last_completed_document,),
            )

            with self._connect() as write_connection:
                current_document_id: int | None = None
                norm_squared = 0.0
                updates: list[tuple[float, int]] = []

                for document_id, term_id, term_frequency in cursor:
                    document_id = int(document_id)
                    if current_document_id is None:
                        current_document_id = document_id
                    elif document_id != current_document_id:
                        updates.append((math.sqrt(norm_squared), current_document_id))
                        if len(updates) >= 5000:
                            self._write_norm_updates(write_connection, updates)
                            self._set_metadata(
                                write_connection,
                                "tfidf_norms_processed_document",
                                str(updates[-1][1]),
                            )
                            write_connection.commit()
                            updates.clear()
                        current_document_id = document_id
                        norm_squared = 0.0

                    inverse_document_frequency = idf_by_term[int(term_id)]
                    weight = int(term_frequency) * inverse_document_frequency
                    norm_squared += weight * weight

                if current_document_id is not None:
                    updates.append((math.sqrt(norm_squared), current_document_id))
                if updates:
                    self._write_norm_updates(write_connection, updates)
                    self._set_metadata(
                        write_connection,
                        "tfidf_norms_processed_document",
                        str(updates[-1][1]),
                    )
                    write_connection.commit()

    @staticmethod
    def _write_norm_updates(
        connection: sqlite3.Connection,
        updates: list[tuple[float, int]],
    ) -> None:
        connection.executemany(
            "UPDATE documents SET tfidf_norm = ? WHERE id = ?",
            updates,
        )

    def _matching_postings(
        self,
        connection: sqlite3.Connection,
        terms: list[str],
    ) -> Iterable[tuple[str, int, int, int, float, int]]:
        placeholders = ",".join("?" for _ in terms)
        return connection.execute(
            f"""
            SELECT
                t.term,
                p.document_id,
                p.tf,
                d.length,
                COALESCE(d.tfidf_norm, 0.0),
                t.df
            FROM terms AS t
            JOIN postings AS p ON p.term_id = t.term_id
            JOIN documents AS d ON d.id = p.document_id
            WHERE t.term IN ({placeholders})
            """,
            terms,
        )

    @staticmethod
    def _document_norms(
        connection: sqlite3.Connection,
        document_scores: dict[int, float],
    ) -> dict[int, float]:
        if not document_scores:
            return {}
        document_ids = list(document_scores)
        norms: dict[int, float] = {}
        for start in range(0, len(document_ids), 900):
            chunk = document_ids[start : start + 900]
            placeholders = ",".join("?" for _ in chunk)
            rows = connection.execute(
                f"SELECT id, COALESCE(tfidf_norm, 0.0) FROM documents WHERE id IN ({placeholders})",
                chunk,
            )
            norms.update({int(document_id): float(norm) for document_id, norm in rows})
        return norms

    def _load_results(
        self,
        connection: sqlite3.Connection,
        ranked: list[tuple[int, float]],
        *,
        model_name: str,
        elapsed_ms: float,
        extra_metadata: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        if not ranked:
            return []
        document_ids = [document_id for document_id, _ in ranked]
        placeholders = ",".join("?" for _ in document_ids)
        records = {
            int(document_id): {
                "doc_id": str(doc_id),
                "title": title,
                "text": str(text),
                "metadata": json.loads(metadata_json or "{}"),
            }
            for document_id, doc_id, title, text, metadata_json in connection.execute(
                f"""
                SELECT id, doc_id, title, text, metadata_json
                FROM documents
                WHERE id IN ({placeholders})
                """,
                document_ids,
            )
        }

        results: list[SearchResult] = []
        for rank, (document_id, score) in enumerate(ranked, start=1):
            record = records[document_id]
            text = record["text"]
            if record["title"]:
                text = f"{record['title']}\n{text}".strip()
            metadata = {
                **record["metadata"],
                "model": model_name,
                "storage": "sqlite_disk",
                "processing_profile": self.config.processing_profile,
                "query_time_ms": round(elapsed_ms, 3),
            }
            if extra_metadata:
                metadata.update(extra_metadata)
            results.append(
                SearchResult(
                    doc_id=record["doc_id"],
                    score=float(score),
                    rank=rank,
                    text=text,
                    metadata=metadata,
                )
            )
        return results

    def _require_ready(self) -> None:
        if not self.exists():
            raise RuntimeError(
                f"Full disk index for '{self.dataset_name}' is not ready. "
                "Run scripts/build_indexes.py without --max-docs first."
            )

    def _create_schema(self, connection: sqlite3.Connection) -> None:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY,
                doc_id TEXT NOT NULL UNIQUE,
                title TEXT,
                text TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                length INTEGER NOT NULL,
                tfidf_norm REAL
            );

            CREATE TABLE IF NOT EXISTS terms (
                term_id INTEGER PRIMARY KEY,
                term TEXT NOT NULL UNIQUE,
                df INTEGER NOT NULL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS postings (
                term_id INTEGER NOT NULL,
                document_id INTEGER NOT NULL,
                tf INTEGER NOT NULL,
                PRIMARY KEY(term_id, document_id)
            ) WITHOUT ROWID;

            CREATE INDEX IF NOT EXISTS idx_postings_document
            ON postings(document_id);

            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            """
        )
        connection.commit()

    def _connect(self, *, read_only: bool = False) -> sqlite3.Connection:
        if read_only:
            uri = f"file:{self.database_path.as_posix()}?mode=ro"
            connection = sqlite3.connect(uri, uri=True, timeout=60.0)
        else:
            connection = sqlite3.connect(self.database_path, timeout=60.0)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        connection.execute("PRAGMA temp_store=FILE")
        connection.execute("PRAGMA cache_size=-65536")
        return connection

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

    def _metadata_float(self, connection: sqlite3.Connection, key: str) -> float:
        value = self._metadata_value(connection, key)
        return float(value) if value is not None else 0.0

    @staticmethod
    def _set_metadata(connection: sqlite3.Connection, key: str, value: str) -> None:
        connection.execute(
            """
            INSERT INTO metadata(key, value) VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (key, value),
        )

    @staticmethod
    def _document_count(connection: sqlite3.Connection) -> int:
        row = connection.execute("SELECT COUNT(*) FROM documents").fetchone()
        return int(row[0]) if row else 0

    @staticmethod
    def _average_document_length(connection: sqlite3.Connection) -> float:
        row = connection.execute("SELECT AVG(length) FROM documents").fetchone()
        return float(row[0] or 0.0) if row else 0.0

    def _remove_database_files(self) -> None:
        for path in (
            self.database_path,
            Path(f"{self.database_path}-wal"),
            Path(f"{self.database_path}-shm"),
        ):
            if path.exists():
                path.unlink()


def _index_text(document: Document, profile: str) -> str:
    """Create the searchable text, boosting argument titles by repetition."""
    if document.title and profile == "argument":
        return f"{document.title} {document.title} {document.text}".strip()
    if document.title:
        return f"{document.title} {document.text}".strip()
    return document.text.strip()


def _tfidf_idf(total_documents: int, document_frequency: int) -> float:
    return math.log((1.0 + total_documents) / (1.0 + document_frequency)) + 1.0

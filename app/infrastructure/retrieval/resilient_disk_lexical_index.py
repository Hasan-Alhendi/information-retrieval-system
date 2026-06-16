"""Recovery-aware full-corpus lexical index finalization."""

from __future__ import annotations

import math
import shutil
import sqlite3
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from app.infrastructure.retrieval.optimized_disk_lexical_index import (
    OptimizedDiskLexicalIndex,
)


class ResilientOptimizedDiskLexicalIndex(OptimizedDiskLexicalIndex):
    """Add safer SQLite connections, repair, and chunked TF-IDF finalization.

    The original finalizer kept one long read cursor open while a second
    connection committed TF-IDF norm updates. On a large corpus this can keep the
    WAL alive for a long time. This implementation reads and writes one bounded
    document range at a time using a single connection.
    """

    finalization_chunk_size = 5000

    def _connect(self, *, read_only: bool = False) -> sqlite3.Connection:
        self.base_dir.mkdir(parents=True, exist_ok=True)
        if read_only:
            uri = f"file:{self.database_path.resolve().as_posix()}?mode=ro"
            connection = sqlite3.connect(uri, uri=True, timeout=120.0)
            connection.execute("PRAGMA query_only=ON")
            connection.execute("PRAGMA temp_store=FILE")
            connection.execute("PRAGMA cache_size=-131072")
            return connection

        connection = sqlite3.connect(self.database_path, timeout=120.0)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("PRAGMA temp_store=FILE")
        connection.execute("PRAGMA cache_size=-131072")
        return connection

    def _finalize_tfidf_norms(self) -> None:
        self.ensure_healthy_for_finalization()

        with self._connect() as connection:
            total_documents = self._metadata_int(connection, "total_documents")
            last_completed_document = self._metadata_int(
                connection,
                "tfidf_norms_processed_document",
            )

            while last_completed_document < total_documents:
                end_document = min(
                    total_documents,
                    last_completed_document + self.finalization_chunk_size,
                )
                rows = list(
                    connection.execute(
                        """
                        SELECT p.document_id, p.tf, t.df
                        FROM postings AS p
                        JOIN terms AS t ON t.term_id = p.term_id
                        WHERE p.document_id > ? AND p.document_id <= ?
                        ORDER BY p.document_id, p.term_id
                        """,
                        (last_completed_document, end_document),
                    )
                )

                norm_squares: dict[int, float] = defaultdict(float)
                for document_id, term_frequency, document_frequency in rows:
                    inverse_document_frequency = _tfidf_idf(
                        total_documents,
                        int(document_frequency),
                    )
                    weight = int(term_frequency) * inverse_document_frequency
                    norm_squares[int(document_id)] += weight * weight

                updates = [
                    (math.sqrt(norm_squared), document_id)
                    for document_id, norm_squared in norm_squares.items()
                ]
                if updates:
                    connection.executemany(
                        "UPDATE documents SET tfidf_norm = ? WHERE id = ?",
                        updates,
                    )

                self._set_metadata(
                    connection,
                    "tfidf_norms_processed_document",
                    str(end_document),
                )
                connection.commit()
                last_completed_document = end_document
                print(
                    f"Finalized TF-IDF norms for {last_completed_document:,}/"
                    f"{total_documents:,} documents"
                )

            self._set_metadata(connection, "tfidf_norms_complete", "1")
            connection.commit()
            connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")

    def ensure_healthy_for_finalization(self) -> dict[str, Any]:
        """Check the database and rebuild the derived postings index if needed."""
        check = self.quick_check()
        if check["ok"]:
            return check

        repair = self.repair_secondary_index(create_backup=True)
        if not repair["ok_after_repair"]:
            details = "; ".join(repair["check_after_repair"])
            raise RuntimeError(
                "The lexical SQLite database is still damaged after rebuilding "
                f"the derived postings index: {details}. Keep the recovery backup "
                "and do not use --force until the database has been inspected."
            )
        return repair

    def quick_check(self) -> dict[str, Any]:
        """Run SQLite quick_check without modifying the database."""
        if not self.database_path.exists():
            return {"ok": False, "messages": ["database file does not exist"]}
        try:
            with self._connect(read_only=True) as connection:
                messages = [
                    str(row[0])
                    for row in connection.execute("PRAGMA quick_check")
                ]
        except sqlite3.DatabaseError as exc:
            return {"ok": False, "messages": [str(exc)]}
        return {
            "ok": messages == ["ok"],
            "messages": messages,
        }

    def repair_secondary_index(self, *, create_backup: bool = True) -> dict[str, Any]:
        """Rebuild the document-order postings index from the postings table."""
        backup_dir = self._backup_database_files() if create_backup else None
        check_before = self.quick_check()

        try:
            with self._connect() as connection:
                try:
                    connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                except sqlite3.DatabaseError:
                    pass
                connection.execute("DROP INDEX IF EXISTS idx_postings_document")
                connection.commit()
                connection.execute(
                    "CREATE INDEX idx_postings_document ON postings(document_id)"
                )
                connection.commit()
                connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        except sqlite3.DatabaseError as exc:
            return {
                "backup_dir": str(backup_dir) if backup_dir else None,
                "check_before_repair": check_before["messages"],
                "repair_error": str(exc),
                "check_after_repair": [str(exc)],
                "ok_after_repair": False,
            }

        check_after = self.quick_check()
        return {
            "backup_dir": str(backup_dir) if backup_dir else None,
            "check_before_repair": check_before["messages"],
            "check_after_repair": check_after["messages"],
            "ok_after_repair": check_after["ok"],
        }

    def _backup_database_files(self) -> Path:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        backup_dir = self.base_dir / "recovery" / timestamp
        backup_dir.mkdir(parents=True, exist_ok=False)
        for path in (
            self.database_path,
            Path(f"{self.database_path}-wal"),
            Path(f"{self.database_path}-shm"),
        ):
            if path.exists():
                shutil.copy2(path, backup_dir / path.name)
        return backup_dir


def _tfidf_idf(total_documents: int, document_frequency: int) -> float:
    return math.log((1.0 + total_documents) / (1.0 + document_frequency)) + 1.0

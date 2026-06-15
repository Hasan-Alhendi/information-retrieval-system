"""Optimized search over an existing full-corpus SQLite lexical index."""

from __future__ import annotations

import heapq
import math
import sqlite3
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

from app.domain.models.search_result import SearchResult
from app.infrastructure.retrieval.disk_lexical_index import DiskLexicalIndex


@dataclass(frozen=True)
class QueryTermStat:
    """Term statistics needed by optimized lexical scoring."""

    term_id: int
    term: str
    document_frequency: int


class OptimizedDiskLexicalIndex(DiskLexicalIndex):
    """Reuse the existing SQLite index with lower-latency query execution.

    The optimization does not rebuild or modify the index. It removes very common
    query terms when more selective terms are available, reads postings directly by
    ``term_id``, avoids a second TF-IDF norm lookup, and enables a larger SQLite
    read cache and memory mapping.
    """

    def __init__(
        self,
        *args: Any,
        max_df_ratio: float = 0.30,
        minimum_terms: int = 1,
        sqlite_cache_mib: int = 128,
        mmap_mib: int = 256,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if not 0 < max_df_ratio <= 1:
            raise ValueError("max_df_ratio must be in the range (0, 1].")
        self.max_df_ratio = max_df_ratio
        self.minimum_terms = max(1, minimum_terms)
        self.sqlite_cache_mib = max(16, sqlite_cache_mib)
        self.mmap_mib = max(0, mmap_mib)

    def search_bm25(
        self,
        query: str,
        *,
        top_k: int = 10,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> list[SearchResult]:
        """Search BM25 while pruning only extremely common query terms."""
        start = time.perf_counter()
        processed_query = self.preprocessor.preprocess(
            query,
            profile=self.config.processing_profile,
        )
        original_terms = list(dict.fromkeys(processed_query.split()))
        if not original_terms:
            return []

        connection = self._connect(read_only=True)
        try:
            self._ensure_finalized(connection)
            total_documents = self._metadata_int(connection, "total_documents")
            average_document_length = self._metadata_float(
                connection,
                "average_document_length",
            )
            selected_terms, pruned_terms = self._select_query_terms(
                connection,
                original_terms,
                total_documents,
            )

            scores: dict[int, float] = defaultdict(float)
            scoring_start = time.perf_counter()
            for stat in selected_terms:
                inverse_document_frequency = math.log(
                    1.0
                    + (
                        total_documents
                        - stat.document_frequency
                        + 0.5
                    )
                    / (stat.document_frequency + 0.5)
                )
                rows = connection.execute(
                    """
                    SELECT p.document_id, p.tf, d.length
                    FROM postings AS p
                    JOIN documents AS d ON d.id = p.document_id
                    WHERE p.term_id = ?
                    """,
                    (stat.term_id,),
                )
                for document_id, term_frequency, document_length in rows:
                    length_ratio = (
                        float(document_length) / average_document_length
                        if average_document_length > 0
                        else 1.0
                    )
                    denominator = float(term_frequency) + k1 * (
                        1.0 - b + b * length_ratio
                    )
                    if denominator <= 0:
                        continue
                    scores[int(document_id)] += inverse_document_frequency * (
                        float(term_frequency) * (k1 + 1.0)
                    ) / denominator

            scoring_ms = (time.perf_counter() - scoring_start) * 1000.0
            ranked = heapq.nlargest(top_k, scores.items(), key=lambda item: item[1])
            results = self._load_results(
                connection,
                ranked,
                model_name="bm25",
                elapsed_ms=0.0,
                extra_metadata={
                    "k1": k1,
                    "b": b,
                    "query_terms_original": original_terms,
                    "query_terms_used": [stat.term for stat in selected_terms],
                    "pruned_terms": pruned_terms,
                    "max_df_ratio": self.max_df_ratio,
                    "scoring_time_ms": round(scoring_ms, 3),
                    "search_optimization": "direct_postings_common_term_pruning",
                },
            )
            total_ms = (time.perf_counter() - start) * 1000.0
            self._set_result_latency(results, total_ms)
            return results
        finally:
            connection.close()

    def search_tfidf(self, query: str, *, top_k: int = 10) -> list[SearchResult]:
        """Search cosine TF-IDF without the previous second norm lookup."""
        start = time.perf_counter()
        processed_query = self.preprocessor.preprocess(
            query,
            profile=self.config.processing_profile,
        )
        query_counts = Counter(processed_query.split())
        original_terms = list(query_counts)
        if not original_terms:
            return []

        connection = self._connect(read_only=True)
        try:
            self._ensure_finalized(connection)
            total_documents = self._metadata_int(connection, "total_documents")
            selected_terms, pruned_terms = self._select_query_terms(
                connection,
                original_terms,
                total_documents,
            )

            dot_products: dict[int, float] = defaultdict(float)
            document_norms: dict[int, float] = {}
            query_weights: dict[str, float] = {}
            scoring_start = time.perf_counter()

            for stat in selected_terms:
                inverse_document_frequency = _tfidf_idf(
                    total_documents,
                    stat.document_frequency,
                )
                query_weight = query_counts[stat.term] * inverse_document_frequency
                query_weights[stat.term] = query_weight
                rows = connection.execute(
                    """
                    SELECT p.document_id, p.tf, COALESCE(d.tfidf_norm, 0.0)
                    FROM postings AS p
                    JOIN documents AS d ON d.id = p.document_id
                    WHERE p.term_id = ?
                    """,
                    (stat.term_id,),
                )
                for document_id, term_frequency, document_norm in rows:
                    document_id = int(document_id)
                    document_weight = float(term_frequency) * inverse_document_frequency
                    dot_products[document_id] += document_weight * query_weight
                    document_norms[document_id] = float(document_norm)

            query_norm = math.sqrt(
                sum(weight * weight for weight in query_weights.values())
            )
            if query_norm <= 0:
                return []

            scores = {
                document_id: dot_product / (document_norms[document_id] * query_norm)
                for document_id, dot_product in dot_products.items()
                if document_norms.get(document_id, 0.0) > 0
            }
            scoring_ms = (time.perf_counter() - scoring_start) * 1000.0
            ranked = heapq.nlargest(top_k, scores.items(), key=lambda item: item[1])
            results = self._load_results(
                connection,
                ranked,
                model_name="tfidf",
                elapsed_ms=0.0,
                extra_metadata={
                    "query_terms_original": original_terms,
                    "query_terms_used": [stat.term for stat in selected_terms],
                    "pruned_terms": pruned_terms,
                    "max_df_ratio": self.max_df_ratio,
                    "scoring_time_ms": round(scoring_ms, 3),
                    "search_optimization": "direct_postings_inline_norms_common_term_pruning",
                },
            )
            total_ms = (time.perf_counter() - start) * 1000.0
            self._set_result_latency(results, total_ms)
            return results
        finally:
            connection.close()

    def _select_query_terms(
        self,
        connection: sqlite3.Connection,
        terms: list[str],
        total_documents: int,
    ) -> tuple[list[QueryTermStat], list[str]]:
        placeholders = ",".join("?" for _ in terms)
        rows = connection.execute(
            f"SELECT term_id, term, df FROM terms WHERE term IN ({placeholders})",
            terms,
        )
        available = [
            QueryTermStat(
                term_id=int(term_id),
                term=str(term),
                document_frequency=int(document_frequency),
            )
            for term_id, term, document_frequency in rows
        ]
        if not available:
            return [], terms

        selected = [
            stat
            for stat in available
            if total_documents <= 0
            or stat.document_frequency / total_documents <= self.max_df_ratio
        ]
        if len(selected) < self.minimum_terms:
            selected = sorted(
                available,
                key=lambda stat: stat.document_frequency,
            )[: self.minimum_terms]

        selected_names = {stat.term for stat in selected}
        selected.sort(key=lambda stat: terms.index(stat.term))
        pruned = [term for term in terms if term not in selected_names]
        return selected, pruned

    def _connect(self, *, read_only: bool = False) -> sqlite3.Connection:
        if not read_only:
            return super()._connect(read_only=False)

        resolved_path = self.database_path.resolve().as_posix()
        uri = f"file:{resolved_path}?mode=ro&immutable=1"
        connection = sqlite3.connect(uri, uri=True, timeout=60.0)
        connection.execute("PRAGMA query_only=ON")
        connection.execute("PRAGMA temp_store=MEMORY")
        connection.execute(f"PRAGMA cache_size=-{self.sqlite_cache_mib * 1024}")
        if self.mmap_mib:
            connection.execute(f"PRAGMA mmap_size={self.mmap_mib * 1024 * 1024}")
        return connection

    def _ensure_finalized(self, connection: sqlite3.Connection) -> None:
        if self._metadata_value(connection, "finalized") != "1":
            raise RuntimeError(
                f"Full disk index for '{self.dataset_name}' is not finalized."
            )

    @staticmethod
    def _set_result_latency(results: list[SearchResult], total_ms: float) -> None:
        rounded = round(total_ms, 3)
        for result in results:
            result.metadata["query_time_ms"] = rounded


def _tfidf_idf(total_documents: int, document_frequency: int) -> float:
    return math.log((1.0 + total_documents) / (1.0 + document_frequency)) + 1.0

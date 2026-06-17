"""Optimized lexical search with safe chunked TF-IDF finalization."""

from app.infrastructure.retrieval.optimized_disk_lexical_index import (
    OptimizedDiskLexicalIndex,
)
from app.infrastructure.retrieval.safe_disk_lexical_index import SafeDiskLexicalIndex


class SafeOptimizedDiskLexicalIndex(
    SafeDiskLexicalIndex,
    OptimizedDiskLexicalIndex,
):
    """Combine optimized search with bounded single-connection finalization."""

    pass

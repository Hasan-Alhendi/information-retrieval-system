"""Tests for optimized disk-backed lexical search."""

from pathlib import Path

import spacy

from app.domain.models.document import Document
from app.infrastructure.preprocessing.spacy_preprocessor import SpacyPreprocessor
from app.infrastructure.retrieval.disk_lexical_index import DiskLexicalIndex
from app.infrastructure.retrieval.optimized_disk_lexical_index import (
    OptimizedDiskLexicalIndex,
)


class SmallLoader:
    def iter_documents(self, dataset_name: str, max_docs: int | None = None):
        del dataset_name
        documents = [
            Document(
                doc_id="d1",
                title="Teacher tenure",
                text="Teachers should not receive permanent tenure without review.",
            ),
            Document(
                doc_id="d2",
                title="Teacher tenure",
                text="Tenure can protect teachers and academic independence.",
            ),
            Document(
                doc_id="d3",
                title="Public transport",
                text="Rail networks can reduce traffic congestion.",
            ),
        ]
        yield from documents if max_docs is None else documents[:max_docs]


def test_optimized_search_reuses_existing_index(tmp_path: Path) -> None:
    database_path = tmp_path / "lexical.sqlite3"
    preprocessor = SpacyPreprocessor(nlp=spacy.blank("en"))
    base_index = DiskLexicalIndex(
        "touche2020-v2",
        dataset_loader=SmallLoader(),
        preprocessor=preprocessor,
        database_path=database_path,
        batch_size=2,
    )
    base_index.build(force=True)

    optimized = OptimizedDiskLexicalIndex(
        "touche2020-v2",
        dataset_loader=SmallLoader(),
        preprocessor=preprocessor,
        database_path=database_path,
        max_df_ratio=0.50,
    )
    bm25_results = optimized.search_bm25("Should teachers get tenure?", top_k=2)
    tfidf_results = optimized.search_tfidf("Should teachers get tenure?", top_k=2)

    assert bm25_results
    assert tfidf_results
    assert bm25_results[0].metadata["search_optimization"]
    assert tfidf_results[0].metadata["query_terms_used"]
    assert bm25_results[0].metadata["query_time_ms"] >= 0

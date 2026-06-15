"""Tests for the checkpointed disk-backed lexical index."""

from pathlib import Path

import spacy

from app.domain.models.document import Document
from app.infrastructure.preprocessing.spacy_preprocessor import SpacyPreprocessor
from app.infrastructure.retrieval.disk_lexical_index import DiskLexicalIndex


class FakeDatasetLoader:
    """Small deterministic stream used instead of downloading a dataset."""

    def iter_documents(self, dataset_name: str, max_docs: int | None = None):
        del dataset_name
        documents = [
            Document(
                doc_id="d1",
                title="Teacher tenure",
                text="Teachers should not receive permanent tenure without review.",
                metadata={"stance": "CON"},
            ),
            Document(
                doc_id="d2",
                title="Teacher tenure",
                text="Tenure can protect teachers and support academic independence.",
                metadata={"stance": "PRO"},
            ),
            Document(
                doc_id="d3",
                title="Electronic cigarettes",
                text="Vaping may create health risks for young adults.",
                metadata={"stance": "CON"},
            ),
        ]
        if max_docs is not None:
            documents = documents[:max_docs]
        yield from documents


def _build_index(database_path: Path) -> DiskLexicalIndex:
    index = DiskLexicalIndex(
        "touche2020-v2",
        dataset_loader=FakeDatasetLoader(),
        preprocessor=SpacyPreprocessor(nlp=spacy.blank("en")),
        batch_size=2,
        database_path=database_path,
    )
    index.build(force=True)
    return index


def test_disk_index_builds_and_resumes(tmp_path: Path) -> None:
    index = _build_index(tmp_path / "lexical.sqlite3")

    assert index.exists()
    assert index.status()["total_documents"] == 3

    index.build(force=False)
    assert index.status()["total_documents"] == 3


def test_disk_bm25_and_tfidf_search(tmp_path: Path) -> None:
    index = _build_index(tmp_path / "lexical.sqlite3")

    bm25_results = index.search_bm25("Should teachers not get tenure?", top_k=2)
    tfidf_results = index.search_tfidf("Should teachers not get tenure?", top_k=2)

    assert bm25_results
    assert tfidf_results
    assert bm25_results[0].doc_id == "d1"
    assert tfidf_results[0].doc_id == "d1"
    assert bm25_results[0].metadata["storage"] == "sqlite_disk"
    assert "query_time_ms" in tfidf_results[0].metadata

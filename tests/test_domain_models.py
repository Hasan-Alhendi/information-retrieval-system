"""Tests for domain models."""

from app.domain.models.document import Document
from app.domain.models.evaluation_result import EvaluationResult
from app.domain.models.search_result import SearchResult


def test_document_model() -> None:
    document = Document(doc_id="doc-1", title="Title", text="Document text")

    assert document.doc_id == "doc-1"
    assert document.title == "Title"
    assert document.text == "Document text"


def test_search_result_model() -> None:
    result = SearchResult(doc_id="doc-1", score=0.9, rank=1, text="Document text")

    assert result.rank == 1
    assert result.score == 0.9


def test_evaluation_result_model() -> None:
    result = EvaluationResult(
        dataset_name="msmarco",
        model_name="bm25",
        map_score=0.1,
        recall=0.2,
        precision_at_10=0.3,
        ndcg=0.4,
        evaluated_queries=5,
    )

    assert result.dataset_name == "msmarco"
    assert result.model_name == "bm25"
    assert result.evaluated_queries == 5

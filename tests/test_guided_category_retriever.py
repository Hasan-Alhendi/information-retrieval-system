"""Tests for guided semantic category reranking."""

from __future__ import annotations

import numpy as np

from app.domain.models.search_result import SearchResult
from app.infrastructure.clustering.category_profiles import CategoryProfile
from app.infrastructure.retrieval import guided_category_retriever as guided_module
from app.infrastructure.retrieval.guided_category_retriever import (
    GuidedCategoryRetriever,
    _select_categories,
)


class FakeEmbeddingRetriever:
    embedding_model_name = "fake/model"

    def build(self, dataset_name: str, force: bool = False, max_docs: int | None = None):
        del dataset_name, force, max_docs

    def prepare(self, dataset_name: str) -> None:
        del dataset_name

    def encode_query(self, query: str) -> np.ndarray:
        del query
        return np.asarray([1.0, 0.0], dtype="float32")

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        assert len(texts) == 2
        return np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype="float32")

    def search(self, query: str, dataset_name: str, top_k: int = 10):
        del query, dataset_name
        rows = [
            SearchResult(
                doc_id="base_first",
                score=0.90,
                rank=1,
                text="second category document",
                metadata={"row_index": 0},
            ),
            SearchResult(
                doc_id="category_match",
                score=0.80,
                rank=2,
                text="first category document",
                metadata={"row_index": 1},
            ),
        ]
        return rows[:top_k]

    def load_candidate_vectors(
        self,
        dataset_name: str,
        row_indexes: list[int],
    ) -> np.ndarray:
        del dataset_name
        vectors = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype="float32")
        return vectors[row_indexes]


def test_select_categories_returns_soft_weights() -> None:
    indexes, weights, scores = _select_categories(
        query_vector=np.asarray([1.0, 0.0], dtype="float32"),
        category_vectors=np.asarray(
            [[1.0, 0.0], [0.0, 1.0]],
            dtype="float32",
        ),
        top_categories=2,
    )

    assert indexes.tolist() == [0, 1]
    assert np.isclose(float(np.sum(weights)), 1.0)
    assert scores[0] > scores[1]


def test_guided_category_signal_can_change_candidate_order(monkeypatch) -> None:
    profiles = (
        CategoryProfile("first", "First", "First semantic category."),
        CategoryProfile("second", "Second", "Second semantic category."),
    )
    monkeypatch.setattr(
        guided_module,
        "get_category_profiles",
        lambda dataset_name: profiles,
    )

    retriever = GuidedCategoryRetriever(
        embedding_retriever=FakeEmbeddingRetriever(),
        category_weight=0.8,
        candidate_k=2,
        top_categories=1,
    )
    results = retriever.search("query", "quora", top_k=2)

    assert [result.doc_id for result in results] == [
        "category_match",
        "base_first",
    ]
    assert results[0].metadata["best_category"] == "first"
    assert results[0].metadata["category_alignment_score"] > results[1].metadata[
        "category_alignment_score"
    ]

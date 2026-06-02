"""Tests for IR evaluation metrics."""

from app.infrastructure.evaluation.metrics import (
    average_precision,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


def test_precision_at_k() -> None:
    retrieved = ["d1", "d2", "d3"]
    relevant = {"d1", "d3"}

    assert precision_at_k(retrieved, relevant, k=2) == 0.5


def test_recall_at_k() -> None:
    retrieved = ["d1", "d2", "d3"]
    relevant = {"d1", "d3", "d4"}

    assert recall_at_k(retrieved, relevant, k=3) == 2 / 3


def test_average_precision() -> None:
    retrieved = ["d1", "d2", "d3"]
    relevant = {"d1", "d3"}

    assert average_precision(retrieved, relevant) == (1 / 1 + 2 / 3) / 2


def test_ndcg_at_k() -> None:
    retrieved = ["d1", "d2"]
    relevance_scores = {"d1": 2.0, "d2": 1.0}

    assert ndcg_at_k(retrieved, relevance_scores, k=2) == 1.0

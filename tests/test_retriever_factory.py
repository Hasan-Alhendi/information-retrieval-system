"""Tests for the shared retriever factory."""

import pytest

from app.application.services.retriever_factory import (
    SUPPORTED_MODELS,
    create_retriever,
)
from app.infrastructure.retrieval.bm25_retriever import BM25Retriever
from app.infrastructure.retrieval.guided_category_retriever import GuidedCategoryRetriever
from app.infrastructure.retrieval.hybrid_serial import HybridSerialRetriever
from app.infrastructure.retrieval.rrf_retriever import ReciprocalRankFusionRetriever


def test_factory_exposes_baselines_and_guided_category_model() -> None:
    assert SUPPORTED_MODELS == (
        "tfidf",
        "bm25",
        "embedding",
        "hybrid_serial",
        "hybrid_parallel",
        "embedding_guided_categories",
    )


def test_factory_uses_corrected_hybrid_implementations() -> None:
    serial = create_retriever("hybrid_serial", max_docs=None)
    parallel = create_retriever("hybrid_parallel", max_docs=None)

    assert isinstance(serial, HybridSerialRetriever)
    assert isinstance(parallel, ReciprocalRankFusionRetriever)


def test_factory_creates_guided_categories_for_full_or_development_index() -> None:
    retriever = create_retriever(
        "embedding_guided_categories",
        max_docs=None,
        category_weight=0.35,
        category_candidate_k=200,
        top_categories=2,
    )

    assert isinstance(retriever, GuidedCategoryRetriever)
    assert retriever.category_weight == 0.35
    assert retriever.candidate_k == 200
    assert retriever.top_categories == 2


def test_factory_applies_bm25_parameters() -> None:
    retriever = create_retriever(
        "bm25",
        max_docs=1000,
        bm25_k1=1.2,
        bm25_b=0.6,
    )

    assert isinstance(retriever, BM25Retriever)
    assert retriever.k1 == 1.2
    assert retriever.b == 0.6


def test_factory_rejects_unknown_model() -> None:
    with pytest.raises(ValueError):
        create_retriever("unknown", max_docs=1000)

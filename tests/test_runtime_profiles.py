"""Tests for official dataset runtime defaults."""

import pytest

from app.infrastructure.datasets.runtime_profiles import get_runtime_profile


def test_quora_runtime_profile_uses_question_query_and_safe_batches() -> None:
    profile = get_runtime_profile("quora")

    assert "programming" in profile.sample_query.lower()
    assert profile.smoke_documents == 5000
    assert profile.lexical_batch_size == 256
    assert profile.embedding_batch_size == 32
    assert profile.embedding_checkpoint_size == 5000


def test_touche_runtime_profile_uses_argument_query() -> None:
    profile = get_runtime_profile("touche2020-v2")

    assert "tenure" in profile.sample_query.lower()
    assert profile.embedding_batch_size == 16


def test_unknown_runtime_profile_is_rejected() -> None:
    with pytest.raises(ValueError):
        get_runtime_profile("unknown")

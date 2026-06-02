"""Tests for preprocessing utilities."""

from app.infrastructure.preprocessing.text_normalizer import normalize_text, normalize_whitespace


def test_normalize_whitespace() -> None:
    assert normalize_whitespace("hello    world\nagain") == "hello world again"


def test_normalize_text_handles_empty_input() -> None:
    assert normalize_text("") == ""

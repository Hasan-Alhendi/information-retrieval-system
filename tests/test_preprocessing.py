"""Tests for preprocessing utilities."""

import spacy

from app.infrastructure.preprocessing.spacy_preprocessor import SpacyPreprocessor
from app.infrastructure.preprocessing.text_normalizer import normalize_text, normalize_whitespace


def test_normalize_whitespace() -> None:
    assert normalize_whitespace("hello    world\nagain") == "hello world again"


def test_normalize_text_handles_empty_input() -> None:
    assert normalize_text("") == ""


def test_question_profile_preserves_question_words_and_negation() -> None:
    preprocessor = SpacyPreprocessor(nlp=spacy.blank("en"))

    result = preprocessor.preprocess(
        "How can I not learn programming?",
        profile="question",
    ).split()

    assert "how" in result
    assert "not" in result
    assert "learn" in result
    assert "programming" in result


def test_argument_profile_preserves_argumentative_stopwords() -> None:
    preprocessor = SpacyPreprocessor(nlp=spacy.blank("en"))

    result = preprocessor.preprocess(
        "Teachers should not get tenure, but universities can support it.",
        profile="argument",
    ).split()

    assert "should" in result
    assert "not" in result
    assert "but" in result
    assert "can" in result
    assert "support" in result

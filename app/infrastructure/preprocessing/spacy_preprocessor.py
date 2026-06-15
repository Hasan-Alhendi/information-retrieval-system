"""spaCy-based preprocessing implementation."""

from __future__ import annotations

from functools import lru_cache
from typing import Iterable

import spacy
from spacy.language import Language

from app.infrastructure.preprocessing.text_normalizer import normalize_text

SPACY_MODEL_NAME = "en_core_web_sm"
PREPROCESSING_BACKEND_TAG = "spacy_v2_profiles"

QUESTION_PRESERVED_TERMS = {
    "how",
    "why",
    "what",
    "when",
    "where",
    "who",
    "which",
    "not",
    "no",
    "nor",
}

ARGUMENT_PRESERVED_TERMS = {
    "not",
    "no",
    "nor",
    "but",
    "however",
    "although",
    "against",
    "support",
    "oppose",
    "should",
    "must",
    "can",
    "cannot",
}


@lru_cache(maxsize=1)
def get_nlp() -> Language:
    """Load and cache the spaCy pipeline.

    If the small English model is not installed, a blank English pipeline is used as a
    safe fallback. The fallback still provides tokenization but not full lemmatization.
    """
    try:
        return spacy.load(SPACY_MODEL_NAME, disable=["parser", "ner", "textcat"])
    except OSError:
        return spacy.blank("en")


class SpacyPreprocessor:
    """Preprocess text using spaCy tokenization and dataset-aware rules."""

    def __init__(self, nlp: Language | None = None) -> None:
        self._nlp = nlp or get_nlp()

    def preprocess(self, text: str, profile: str = "default") -> str:
        """Normalize and preprocess a single text."""
        return self._normalize_doc(self._nlp(normalize_text(text)), profile=profile)

    def preprocess_many(
        self,
        texts: Iterable[str],
        batch_size: int = 64,
        profile: str = "default",
    ) -> list[str]:
        """Normalize and preprocess multiple texts efficiently."""
        normalized_texts = (normalize_text(text) for text in texts)
        return [
            self._normalize_doc(doc, profile=profile)
            for doc in self._nlp.pipe(normalized_texts, batch_size=batch_size)
        ]

    @staticmethod
    def _normalize_doc(doc, profile: str = "default") -> str:
        protected_terms = _protected_terms(profile)
        tokens: list[str] = []

        for token in doc:
            if token.is_space or token.is_punct:
                continue

            normalized = token.lemma_.strip().lower()
            if not normalized or normalized == "-pron-":
                normalized = token.lower_

            if token.is_stop and normalized not in protected_terms:
                continue

            if normalized.isalpha():
                tokens.append(normalized)

        return " ".join(tokens)


def _protected_terms(profile: str) -> set[str]:
    if profile == "question":
        return QUESTION_PRESERVED_TERMS
    if profile == "argument":
        return ARGUMENT_PRESERVED_TERMS
    return set()


_default_preprocessor = SpacyPreprocessor()


def preprocess(text: str) -> str:
    """Backward-compatible helper for preprocessing a single text."""
    return _default_preprocessor.preprocess(text)


def preprocess_texts(texts: Iterable[str]) -> list[str]:
    """Backward-compatible helper for preprocessing multiple texts."""
    return _default_preprocessor.preprocess_many(texts)

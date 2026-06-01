"""Lightweight text normalization utilities."""

import re
import unicodedata

_WHITESPACE_RE = re.compile(r"\s+")


def normalize_whitespace(text: str) -> str:
    """Collapse repeated whitespace into a single space."""
    return _WHITESPACE_RE.sub(" ", text).strip()


def normalize_unicode(text: str) -> str:
    """Normalize Unicode text using NFKC normalization."""
    return unicodedata.normalize("NFKC", text)


def normalize_text(text: str) -> str:
    """Apply common lightweight normalization steps."""
    return normalize_whitespace(normalize_unicode(text or ""))

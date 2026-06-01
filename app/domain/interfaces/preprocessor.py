"""Preprocessor interface."""

from typing import Protocol


class Preprocessor(Protocol):
    """Contract implemented by text preprocessing pipelines."""

    def preprocess(self, text: str) -> str:
        """Normalize and preprocess raw text."""
        ...

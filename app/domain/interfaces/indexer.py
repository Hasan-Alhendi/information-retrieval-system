"""Indexer interface."""

from typing import Protocol


class Indexer(Protocol):
    """Contract implemented by index builders."""

    def build(self, dataset_name: str, force: bool = False) -> None:
        """Build and persist an index for a dataset."""
        ...

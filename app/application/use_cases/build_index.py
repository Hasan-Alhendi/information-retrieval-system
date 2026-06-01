"""Build index use case."""

from app.domain.interfaces.indexer import Indexer


class BuildIndexUseCase:
    """Use case for building a dataset index."""

    def __init__(self, indexer: Indexer) -> None:
        self._indexer = indexer

    def execute(self, dataset_name: str, force: bool = False) -> None:
        """Build an index for a dataset."""
        self._indexer.build(dataset_name=dataset_name, force=force)

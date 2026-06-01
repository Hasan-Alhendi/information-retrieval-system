"""TF-IDF retriever implementation skeleton."""

from app.domain.models.search_result import SearchResult
from app.infrastructure.preprocessing.spacy_preprocessor import SpacyPreprocessor


class TFIDFRetriever:
    """TF-IDF / Vector Space Model retriever.

    The indexing and loading logic will be migrated from the previous project in the next step.
    """

    def __init__(self, preprocessor: SpacyPreprocessor | None = None) -> None:
        self._preprocessor = preprocessor or SpacyPreprocessor()

    def search(self, query: str, dataset_name: str, top_k: int = 10) -> list[SearchResult]:
        """Search documents using TF-IDF cosine similarity."""
        _ = self._preprocessor.preprocess(query)
        _ = dataset_name
        _ = top_k
        return []

"""TF-IDF retriever implementation."""

from pathlib import Path

from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.domain.models.search_result import SearchResult
from app.infrastructure.datasets.dataset_loader import DatasetLoader
from app.infrastructure.datasets.dataset_registry import get_dataset_config
from app.infrastructure.preprocessing.spacy_preprocessor import (
    PREPROCESSING_BACKEND_TAG,
    SpacyPreprocessor,
)
from app.infrastructure.storage.index_store import get_index_dir
from app.infrastructure.storage.joblib_store import load_object, save_object


class TFIDFRetriever:
    """TF-IDF / Vector Space Model retriever."""

    model_name = "tfidf"

    def __init__(
        self,
        dataset_loader: DatasetLoader | None = None,
        preprocessor: SpacyPreprocessor | None = None,
        max_docs: int | None = None,
    ) -> None:
        self._dataset_loader = dataset_loader or DatasetLoader()
        self._preprocessor = preprocessor or SpacyPreprocessor()
        self._max_docs = max_docs

    def build(self, dataset_name: str, force: bool = False, max_docs: int | None = None) -> None:
        """Build and persist the TF-IDF index for a dataset."""
        active_max_docs = max_docs if max_docs is not None else self._max_docs
        paths = self._paths(dataset_name, active_max_docs)
        if not force and self._is_ready(paths):
            return

        doc_ids, documents, _, _ = self._dataset_loader.prepare_dataset(
            dataset_name,
            max_docs=active_max_docs,
        )
        profile = self._processing_profile(dataset_name)
        normalized_documents = self._preprocessor.preprocess_many(
            documents,
            profile=profile,
        )

        vectorizer = TfidfVectorizer(lowercase=False)
        matrix = vectorizer.fit_transform(normalized_documents)

        paths["base_dir"].mkdir(parents=True, exist_ok=True)
        save_object(vectorizer, paths["vectorizer"])
        sparse.save_npz(paths["matrix"], matrix)
        save_object(doc_ids, paths["doc_ids"])
        save_object(documents, paths["documents"])

    def search(self, query: str, dataset_name: str, top_k: int = 10) -> list[SearchResult]:
        """Search documents using TF-IDF cosine similarity."""
        self.ensure_ready(dataset_name)
        vectorizer, matrix, doc_ids, documents = self.load(dataset_name)

        profile = self._processing_profile(dataset_name)
        processed_query = self._preprocessor.preprocess(query, profile=profile)
        query_vector = vectorizer.transform([processed_query])
        similarities = cosine_similarity(query_vector, matrix)[0]

        top_indexes = similarities.argsort()[::-1][:top_k]
        results: list[SearchResult] = []
        for rank, index in enumerate(top_indexes, start=1):
            score = float(similarities[index])
            if score <= 0:
                continue
            results.append(
                SearchResult(
                    doc_id=doc_ids[index],
                    score=score,
                    rank=rank,
                    text=documents[index],
                    metadata={
                        "model": self.model_name,
                        "processing_profile": profile,
                    },
                )
            )
        return results

    def ensure_ready(self, dataset_name: str) -> None:
        """Build the TF-IDF index if it does not exist."""
        paths = self._paths(dataset_name, self._max_docs)
        if not self._is_ready(paths):
            self.build(dataset_name)

    def load(self, dataset_name: str):
        """Load persisted TF-IDF artifacts."""
        paths = self._paths(dataset_name, self._max_docs)
        vectorizer = load_object(paths["vectorizer"])
        matrix = sparse.load_npz(paths["matrix"])
        doc_ids = load_object(paths["doc_ids"])
        documents = load_object(paths["documents"])
        return vectorizer, matrix, doc_ids, documents

    def _paths(self, dataset_name: str, max_docs: int | None = None) -> dict[str, Path]:
        base_dir = get_index_dir(dataset_name, self.model_name)
        if max_docs is not None:
            base_dir = base_dir / f"dev_{max_docs}"
        prefix = PREPROCESSING_BACKEND_TAG
        return {
            "base_dir": base_dir,
            "vectorizer": base_dir / f"tfidf_vectorizer_{prefix}.joblib",
            "matrix": base_dir / f"tfidf_matrix_{prefix}.npz",
            "doc_ids": base_dir / f"doc_ids_{prefix}.joblib",
            "documents": base_dir / f"documents_{prefix}.joblib",
        }

    @staticmethod
    def _processing_profile(dataset_name: str) -> str:
        return get_dataset_config(dataset_name, include_experimental=True).processing_profile

    @staticmethod
    def _is_ready(paths: dict[str, Path]) -> bool:
        required = ("vectorizer", "matrix", "doc_ids", "documents")
        return all(paths[key].exists() for key in required)

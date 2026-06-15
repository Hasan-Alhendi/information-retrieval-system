"""BM25 retriever implementation."""

from pathlib import Path

from rank_bm25 import BM25Okapi

from app.config import DEFAULT_BM25_B, DEFAULT_BM25_K1
from app.domain.models.search_result import SearchResult
from app.infrastructure.datasets.dataset_loader import DatasetLoader
from app.infrastructure.datasets.dataset_registry import get_dataset_config
from app.infrastructure.preprocessing.spacy_preprocessor import (
    PREPROCESSING_BACKEND_TAG,
    SpacyPreprocessor,
)
from app.infrastructure.retrieval.disk_lexical_index import DiskLexicalIndex
from app.infrastructure.storage.index_store import get_index_dir
from app.infrastructure.storage.joblib_store import load_object, save_object


class BM25Retriever:
    """BM25 retriever with development and full-corpus storage modes."""

    model_name = "bm25"

    def __init__(
        self,
        dataset_loader: DatasetLoader | None = None,
        preprocessor: SpacyPreprocessor | None = None,
        k1: float = DEFAULT_BM25_K1,
        b: float = DEFAULT_BM25_B,
        max_docs: int | None = None,
        full_batch_size: int = 128,
    ) -> None:
        self._dataset_loader = dataset_loader or DatasetLoader()
        self._preprocessor = preprocessor or SpacyPreprocessor()
        self.k1 = k1
        self.b = b
        self._max_docs = max_docs
        self._full_batch_size = full_batch_size

    def build(self, dataset_name: str, force: bool = False, max_docs: int | None = None) -> None:
        """Build and persist the BM25 index for a dataset."""
        active_max_docs = max_docs if max_docs is not None else self._max_docs
        if self._uses_full_disk_index(dataset_name, active_max_docs):
            self._disk_index(dataset_name).build(force=force)
            return

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
        tokenized_documents = [document.split() for document in normalized_documents]
        bm25 = BM25Okapi(tokenized_documents, k1=self.k1, b=self.b)

        paths["base_dir"].mkdir(parents=True, exist_ok=True)
        save_object(bm25, paths["bm25"])
        save_object(doc_ids, paths["doc_ids"])
        save_object(documents, paths["documents"])
        save_object(
            {
                "k1": self.k1,
                "b": self.b,
                "preprocessing": PREPROCESSING_BACKEND_TAG,
                "processing_profile": profile,
            },
            paths["metadata"],
        )

    def search(self, query: str, dataset_name: str, top_k: int = 10) -> list[SearchResult]:
        """Search documents using BM25."""
        if self._uses_full_disk_index(dataset_name, self._max_docs):
            self.ensure_ready(dataset_name)
            return self._disk_index(dataset_name).search_bm25(
                query,
                top_k=top_k,
                k1=self.k1,
                b=self.b,
            )

        self.ensure_ready(dataset_name)
        bm25, doc_ids, documents = self.load(dataset_name)

        profile = self._processing_profile(dataset_name)
        processed_query = self._preprocessor.preprocess(query, profile=profile)
        query_tokens = processed_query.split()
        scores = bm25.get_scores(query_tokens)

        top_indexes = scores.argsort()[::-1][:top_k]
        results: list[SearchResult] = []
        for rank, index in enumerate(top_indexes, start=1):
            score = float(scores[index])
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
                        "k1": self.k1,
                        "b": self.b,
                        "processing_profile": profile,
                        "storage": "joblib_memory",
                    },
                )
            )
        return results

    def ensure_ready(self, dataset_name: str) -> None:
        """Build the selected BM25 storage mode if it does not exist."""
        if self._uses_full_disk_index(dataset_name, self._max_docs):
            disk_index = self._disk_index(dataset_name)
            if not disk_index.exists():
                disk_index.build()
            return

        paths = self._paths(dataset_name, self._max_docs)
        if not self._is_ready(paths):
            self.build(dataset_name)

    def load(self, dataset_name: str):
        """Load persisted development BM25 artifacts."""
        paths = self._paths(dataset_name, self._max_docs)
        bm25 = load_object(paths["bm25"])
        doc_ids = load_object(paths["doc_ids"])
        documents = load_object(paths["documents"])
        return bm25, doc_ids, documents

    def _disk_index(self, dataset_name: str) -> DiskLexicalIndex:
        return DiskLexicalIndex(
            dataset_name,
            dataset_loader=self._dataset_loader,
            preprocessor=self._preprocessor,
            batch_size=self._full_batch_size,
        )

    @staticmethod
    def _uses_full_disk_index(dataset_name: str, max_docs: int | None) -> bool:
        config = get_dataset_config(dataset_name, include_experimental=True)
        return max_docs is None and config.external_id is not None

    def _paths(self, dataset_name: str, max_docs: int | None = None) -> dict[str, Path]:
        base_dir = get_index_dir(dataset_name, self.model_name)
        if max_docs is not None:
            base_dir = base_dir / f"dev_{max_docs}"
        prefix = PREPROCESSING_BACKEND_TAG
        params = f"k1_{self.k1}_b_{self.b}".replace(".", "_")
        return {
            "base_dir": base_dir,
            "bm25": base_dir / f"bm25_{params}_{prefix}.joblib",
            "doc_ids": base_dir / f"doc_ids_{prefix}.joblib",
            "documents": base_dir / f"documents_{prefix}.joblib",
            "metadata": base_dir / f"metadata_{params}_{prefix}.joblib",
        }

    @staticmethod
    def _processing_profile(dataset_name: str) -> str:
        return get_dataset_config(dataset_name, include_experimental=True).processing_profile

    @staticmethod
    def _is_ready(paths: dict[str, Path]) -> bool:
        required = ("bm25", "doc_ids", "documents", "metadata")
        return all(paths[key].exists() for key in required)

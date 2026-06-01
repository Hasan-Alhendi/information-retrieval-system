"""BEIR dataset loading infrastructure."""

from functools import lru_cache
from typing import Any

from beir import util
from beir.datasets.data_loader import GenericDataLoader

from app.config import DATASETS_DIR
from app.domain.models.document import Document
from app.infrastructure.datasets.dataset_registry import get_dataset_config

Corpus = dict[str, dict[str, Any]]
Queries = dict[str, str]
Qrels = dict[str, dict[str, int]]


class BeirDatasetLoader:
    """Loads BEIR-compatible datasets and prepares them for retrieval."""

    def load_raw(self, dataset_name: str) -> tuple[Corpus, Queries, Qrels]:
        """Download if needed and load raw BEIR corpus, queries, and qrels."""
        return _load_raw_dataset(dataset_name)

    def load_documents(self, dataset_name: str, max_docs: int | None = None) -> list[Document]:
        """Load documents for a dataset."""
        doc_ids, texts, _, _ = self.prepare_dataset(dataset_name, max_docs=max_docs)
        return [Document(doc_id=doc_id, text=text) for doc_id, text in zip(doc_ids, texts, strict=True)]

    def prepare_dataset(
        self,
        dataset_name: str,
        max_docs: int | None = None,
    ) -> tuple[list[str], list[str], Queries, Qrels]:
        """Prepare documents, queries, and filtered qrels for retrieval and evaluation."""
        config = get_dataset_config(dataset_name)
        corpus, queries, qrels = self.load_raw(dataset_name)

        limit = config.document_limit if max_docs is None else max_docs
        doc_ids = list(corpus.keys())
        if limit is not None:
            doc_ids = doc_ids[:limit]

        documents: list[str] = []
        for doc_id in doc_ids:
            item = corpus[doc_id]
            title = item.get("title", "")
            text = item.get("text", "")
            documents.append(f"{title} {text}".strip())

        allowed_doc_ids = set(doc_ids)
        filtered_qrels: Qrels = {}
        for query_id, relevances in qrels.items():
            filtered = {
                doc_id: score
                for doc_id, score in relevances.items()
                if doc_id in allowed_doc_ids
            }
            if filtered:
                filtered_qrels[query_id] = filtered

        return doc_ids, documents, queries, filtered_qrels

    def summary(self, dataset_name: str) -> dict[str, int]:
        """Return dataset statistics."""
        corpus, queries, qrels = self.load_raw(dataset_name)
        return {
            "documents_count": len(corpus),
            "queries_count": len(queries),
            "qrels_count": len(qrels),
        }


@lru_cache(maxsize=8)
def _load_raw_dataset(dataset_name: str) -> tuple[Corpus, Queries, Qrels]:
    config = get_dataset_config(dataset_name)
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)

    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{config.name}.zip"
    data_path = util.download_and_unzip(url, str(DATASETS_DIR))
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")
    return corpus, queries, qrels

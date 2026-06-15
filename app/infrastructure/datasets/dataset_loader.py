"""Unified dataset loading facade for BEIR and ir_datasets sources."""

from __future__ import annotations

from collections.abc import Iterator
from functools import lru_cache
from typing import Any

from app.domain.models.document import Document
from app.infrastructure.datasets.beir_loader import BeirDatasetLoader, Queries, Qrels
from app.infrastructure.datasets.dataset_registry import get_dataset_config


class DatasetLoader:
    """Dispatch dataset loading to the configured source implementation."""

    def __init__(self) -> None:
        self._beir_loader = BeirDatasetLoader()

    def load_documents(self, dataset_name: str, max_docs: int | None = None) -> list[Document]:
        """Load a development set of documents into memory."""
        config = get_dataset_config(dataset_name, include_experimental=True)
        if config.source == "beir":
            return self._beir_loader.load_documents(dataset_name, max_docs=max_docs)
        return list(self.iter_documents(dataset_name, max_docs=max_docs))

    def iter_documents(
        self,
        dataset_name: str,
        max_docs: int | None = None,
    ) -> Iterator[Document]:
        """Stream documents from the configured dataset source.

        This iterator is the foundation for later full-corpus batch indexing. For
        ir_datasets corpora it does not materialize the full corpus in memory.
        """
        config = get_dataset_config(dataset_name, include_experimental=True)
        if config.source == "beir":
            for document in self._beir_loader.load_documents(dataset_name, max_docs=max_docs):
                yield document
            return

        dataset = _load_ir_dataset(config.external_id)
        count = 0
        for item in dataset.docs_iter():
            yield Document(
                doc_id=str(item.doc_id),
                title=_optional_text(getattr(item, "title", None)),
                text=_optional_text(getattr(item, "text", "")) or "",
                metadata={
                    "stance": _optional_text(getattr(item, "stance", None)),
                    "url": _optional_text(getattr(item, "url", None)),
                    "source": config.external_id,
                    "processing_profile": config.processing_profile,
                },
            )
            count += 1
            if max_docs is not None and count >= max_docs:
                break

    def prepare_dataset(
        self,
        dataset_name: str,
        max_docs: int | None = None,
        use_config_limit: bool = True,
    ) -> tuple[list[str], list[str], Queries, Qrels]:
        """Prepare an in-memory dataset for the existing development pipeline.

        Full-corpus ir_datasets indexing must use ``iter_documents`` and a batch
        indexer. This guard prevents accidental materialization of hundreds of
        thousands of documents on an 8 GB laptop.
        """
        config = get_dataset_config(dataset_name, include_experimental=True)
        if config.source == "beir":
            return self._beir_loader.prepare_dataset(
                dataset_name,
                max_docs=max_docs,
                use_config_limit=use_config_limit,
            )

        if max_docs is None:
            raise RuntimeError(
                "Full-corpus Touché loading is intentionally blocked in the legacy "
                "in-memory pipeline. Use a development --max-docs value until the "
                "batch indexers are enabled."
            )

        dataset = _load_ir_dataset(config.external_id)
        queries = {
            str(item.query_id): str(item.text)
            for item in dataset.queries_iter()
        }
        qrels: Qrels = {}
        for item in dataset.qrels_iter():
            query_id = str(item.query_id)
            qrels.setdefault(query_id, {})[str(item.doc_id)] = int(item.relevance)

        selected_ids = _select_qrels_aware_ids(qrels, limit=max_docs)
        selected_documents: dict[str, Document] = {}

        docs_store = dataset.docs_store()
        for doc_id in selected_ids:
            try:
                item = docs_store.get(doc_id)
            except KeyError:
                continue
            if item is None:
                continue
            selected_documents[doc_id] = _map_ir_document(item, config.external_id, config.processing_profile)

        if len(selected_documents) < max_docs:
            for item in dataset.docs_iter():
                doc_id = str(item.doc_id)
                if doc_id in selected_documents:
                    continue
                selected_documents[doc_id] = _map_ir_document(
                    item,
                    config.external_id,
                    config.processing_profile,
                )
                if len(selected_documents) >= max_docs:
                    break

        doc_ids = list(selected_documents)
        documents = [_display_text(selected_documents[doc_id]) for doc_id in doc_ids]
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

    def summary(self, dataset_name: str) -> dict[str, int | str | None]:
        """Return dataset statistics and source information."""
        config = get_dataset_config(dataset_name, include_experimental=True)
        if config.source == "beir":
            summary = self._beir_loader.summary(dataset_name)
            return {**summary, "source": config.source, "task_type": config.task_type}

        dataset = _load_ir_dataset(config.external_id)
        return {
            "documents_count": _safe_count(dataset, "docs_count"),
            "queries_count": _safe_count(dataset, "queries_count"),
            "qrels_count": _safe_count(dataset, "qrels_count"),
            "source": config.source,
            "task_type": config.task_type,
        }


def _map_ir_document(item: Any, source: str | None, processing_profile: str) -> Document:
    return Document(
        doc_id=str(item.doc_id),
        title=_optional_text(getattr(item, "title", None)),
        text=_optional_text(getattr(item, "text", "")) or "",
        metadata={
            "stance": _optional_text(getattr(item, "stance", None)),
            "url": _optional_text(getattr(item, "url", None)),
            "source": source,
            "processing_profile": processing_profile,
        },
    )


def _display_text(document: Document) -> str:
    if document.title:
        return f"{document.title}\n{document.text}".strip()
    return document.text.strip()


def _select_qrels_aware_ids(qrels: Qrels, limit: int) -> list[str]:
    selected: list[str] = []
    seen: set[str] = set()
    for relevances in qrels.values():
        for doc_id, score in relevances.items():
            if score <= 0 or doc_id in seen:
                continue
            selected.append(doc_id)
            seen.add(doc_id)
            if len(selected) >= limit:
                return selected
    return selected


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _safe_count(dataset: Any, method_name: str) -> int | None:
    method = getattr(dataset, method_name, None)
    if method is None:
        return None
    try:
        return int(method())
    except (TypeError, ValueError, RuntimeError):
        return None


@lru_cache(maxsize=4)
def _load_ir_dataset(external_id: str | None):
    if not external_id:
        raise ValueError("ir_datasets external_id is required.")
    try:
        import ir_datasets
    except ImportError as exc:
        raise RuntimeError(
            "ir-datasets is not installed. Run: pip install -r requirements.txt"
        ) from exc
    return ir_datasets.load(external_id)

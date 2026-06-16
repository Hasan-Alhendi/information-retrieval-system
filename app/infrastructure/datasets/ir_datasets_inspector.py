"""Lightweight inspection utilities for datasets exposed by ir_datasets."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import islice
from typing import Any, Iterable

from app.infrastructure.datasets.dataset_registry import DatasetConfig


@dataclass(frozen=True)
class DatasetInspection:
    """Metadata and optional samples collected without indexing the dataset."""

    name: str
    external_id: str
    display_name: str
    task_type: str | None
    processing_profile: str
    documents_count: int | None
    queries_count: int | None
    qrels_count: int | None
    query_fields: list[str]
    document_fields: list[str]
    qrel_fields: list[str]
    sample_queries: list[dict[str, Any]]
    sample_documents: list[dict[str, Any]]
    sample_qrels: list[dict[str, Any]]


class IrDatasetsInspector:
    """Inspect any registry entry that exposes an ``ir_datasets`` external ID."""

    def inspect(
        self,
        config: DatasetConfig,
        *,
        sample_queries: int = 3,
        sample_documents: int = 0,
        sample_qrels: int = 5,
        positive_qrels_only: bool = True,
    ) -> DatasetInspection:
        """Return counts, schemas, and small samples.

        A dataset may keep ``source='beir'`` for development loading while also
        exposing an ``external_id`` for streaming full-corpus access. Therefore
        inspection depends on ``external_id`` rather than the source label.
        """
        if not config.external_id:
            raise ValueError("The dataset must define an ir_datasets external_id.")

        try:
            import ir_datasets
        except ImportError as exc:
            raise RuntimeError(
                "ir-datasets is not installed. Run: pip install -r requirements.txt"
            ) from exc

        dataset = ir_datasets.load(config.external_id)
        query_items = _take(dataset.queries_iter(), sample_queries)
        qrel_items = _take_qrels(
            dataset.qrels_iter(),
            sample_qrels,
            positive_only=positive_qrels_only,
        )
        document_items = _take(dataset.docs_iter(), sample_documents) if sample_documents else []

        return DatasetInspection(
            name=config.name,
            external_id=config.external_id,
            display_name=config.display_name,
            task_type=config.task_type,
            processing_profile=config.processing_profile,
            documents_count=_safe_count(dataset, "docs_count"),
            queries_count=_safe_count(dataset, "queries_count"),
            qrels_count=_safe_count(dataset, "qrels_count"),
            query_fields=_field_names(query_items),
            document_fields=_field_names(document_items),
            qrel_fields=_field_names(qrel_items),
            sample_queries=[_to_dict(item) for item in query_items],
            sample_documents=[_to_dict(item) for item in document_items],
            sample_qrels=[_to_dict(item) for item in qrel_items],
        )


def _safe_count(dataset: Any, method_name: str) -> int | None:
    method = getattr(dataset, method_name, None)
    if method is None:
        return None
    try:
        return int(method())
    except (TypeError, ValueError, RuntimeError):
        return None


def _take(items: Iterable[Any], count: int) -> list[Any]:
    if count <= 0:
        return []
    return list(islice(items, count))


def _take_qrels(
    items: Iterable[Any],
    count: int,
    *,
    positive_only: bool,
) -> list[Any]:
    """Take qrel samples, optionally keeping only relevance scores above zero."""
    if count <= 0:
        return []
    if not positive_only:
        return _take(items, count)

    selected: list[Any] = []
    for item in items:
        relevance = getattr(item, "relevance", 0)
        try:
            is_positive = float(relevance) > 0
        except (TypeError, ValueError):
            is_positive = False
        if not is_positive:
            continue
        selected.append(item)
        if len(selected) >= count:
            break
    return selected


def _field_names(items: list[Any]) -> list[str]:
    if not items:
        return []
    item = items[0]
    if hasattr(item, "_fields"):
        return list(item._fields)
    if hasattr(item, "__dict__"):
        return list(vars(item))
    return []


def _to_dict(item: Any) -> dict[str, Any]:
    if hasattr(item, "_asdict"):
        return dict(item._asdict())
    if hasattr(item, "__dict__"):
        return dict(vars(item))
    return {"value": str(item)}

"""Tests for lightweight ir_datasets inspection."""

from collections import namedtuple
from types import SimpleNamespace

from app.infrastructure.datasets.dataset_registry import DatasetConfig
from app.infrastructure.datasets.ir_datasets_inspector import IrDatasetsInspector


Query = namedtuple("Query", ["query_id", "text"])
Doc = namedtuple("Doc", ["doc_id", "text"])
Qrel = namedtuple("Qrel", ["query_id", "doc_id", "relevance", "iteration"])


class FakeDataset:
    def queries_iter(self):
        yield Query("q1", "How can I learn programming?")

    def docs_iter(self):
        yield Doc("d1", "How do you get started learning programming?")

    def qrels_iter(self):
        yield Qrel("q1", "d1", 1, "0")

    def docs_count(self):
        return 1

    def queries_count(self):
        return 1

    def qrels_count(self):
        return 1


def test_inspector_accepts_beir_config_with_external_id(monkeypatch) -> None:
    fake_module = SimpleNamespace(load=lambda external_id: FakeDataset())
    monkeypatch.setitem(__import__("sys").modules, "ir_datasets", fake_module)
    config = DatasetConfig(
        name="quora",
        display_name="Quora",
        source="beir",
        external_id="beir/quora/test",
        task_type="duplicate-question retrieval",
        processing_profile="question",
    )

    inspection = IrDatasetsInspector().inspect(
        config,
        sample_queries=1,
        sample_documents=1,
        sample_qrels=1,
    )

    assert inspection.documents_count == 1
    assert inspection.document_fields == ["doc_id", "text"]
    assert inspection.sample_qrels[0]["relevance"] == 1

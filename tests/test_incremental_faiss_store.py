"""Tests for checkpointed full-corpus FAISS storage."""

from pathlib import Path

import numpy as np

from app.domain.models.document import Document
from app.infrastructure.vector_store.incremental_faiss_store import (
    IncrementalFaissVectorStore,
)


def _documents(start: int, count: int) -> list[Document]:
    return [
        Document(
            doc_id=f"d{index}",
            title=f"Title {index}",
            text=f"Document text {index}",
            metadata={"row": index},
        )
        for index in range(start, start + count)
    ]


def test_incremental_store_checkpoints_and_searches(tmp_path: Path) -> None:
    store = IncrementalFaissVectorStore(
        "touche2020-v2",
        "test-model",
        base_dir=tmp_path / "vectors",
    )
    first_vectors = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype="float32")
    second_vectors = np.asarray([[0.8, 0.2]], dtype="float32")

    assert store.add_checkpoint(first_vectors, _documents(0, 2)) == 2
    assert store.add_checkpoint(second_vectors, _documents(2, 1)) == 3
    store.finalize(expected_count=3)

    assert store.exists()
    assert store.status()["index_rows"] == 3
    matches = store.search(np.asarray([1.0, 0.0], dtype="float32"), top_k=2)
    assert matches[0][0] == 0

    records = store.load_records([row_index for row_index, _ in matches])
    assert records[0]["doc_id"] == "d0"
    assert records[0]["metadata"]["row"] == 0


def test_incremental_store_resumes_without_duplicate_rows(tmp_path: Path) -> None:
    base_dir = tmp_path / "vectors"
    store = IncrementalFaissVectorStore(
        "touche2020-v2",
        "test-model",
        base_dir=base_dir,
    )
    store.add_checkpoint(
        np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype="float32"),
        _documents(0, 2),
    )

    resumed = IncrementalFaissVectorStore(
        "touche2020-v2",
        "test-model",
        base_dir=base_dir,
    )
    assert resumed.prepare_resume() == 2
    resumed.add_checkpoint(
        np.asarray([[0.5, 0.5]], dtype="float32"),
        _documents(2, 1),
    )
    resumed.finalize(expected_count=3)

    assert resumed.processed_documents() == 3
    assert resumed.status()["finalized"] is True

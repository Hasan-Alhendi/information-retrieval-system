"""Tests for direct candidate scoring on a finalized FAISS store."""

from pathlib import Path

import numpy as np

from app.domain.models.document import Document
from app.infrastructure.vector_store.candidate_faiss_store import CandidateFaissVectorStore


def test_candidate_store_scores_requested_documents(tmp_path: Path) -> None:
    store = CandidateFaissVectorStore(
        "touche2020-v2",
        "test-model",
        base_dir=tmp_path / "vectors",
    )
    documents = [
        Document(doc_id="d1", text="first"),
        Document(doc_id="d2", text="second"),
        Document(doc_id="d3", text="third"),
    ]
    vectors = np.asarray(
        [[1.0, 0.0], [0.0, 1.0], [0.8, 0.2]],
        dtype="float32",
    )
    store.add_checkpoint(vectors, documents)
    store.finalize(expected_count=3)

    scores = store.score_doc_ids(
        np.asarray([1.0, 0.0], dtype="float32"),
        ["d1", "d3"],
    )

    assert set(scores) == {"d1", "d3"}
    assert scores["d1"] > scores["d3"]

"""Tests for corrected hybrid retrieval components."""

from app.domain.models.search_result import SearchResult
from app.infrastructure.retrieval.hybrid_serial_v2 import HybridSerialRetrieverV2


class FakeBM25:
    def search(self, query, dataset_name, top_k=10):
        del query, dataset_name, top_k
        return [
            SearchResult(doc_id="d1", score=10.0, rank=1, text="first"),
            SearchResult(doc_id="d2", score=8.0, rank=2, text="second"),
        ]


class FakeEmbedding:
    embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"


class FakeDenseScorer:
    def score(self, query, dataset_name, doc_ids):
        del query, dataset_name
        assert doc_ids == ["d1", "d2"]
        return {"d1": 0.2, "d2": 0.9}



def test_serial_v2_reranks_only_bm25_candidates() -> None:
    retriever = HybridSerialRetrieverV2(
        bm25_retriever=FakeBM25(),
        embedding_retriever=FakeEmbedding(),
        max_docs=None,
        bm25_weight=0.3,
        embedding_weight=0.7,
    )
    retriever.scorer = FakeDenseScorer()

    results = retriever.search("query", "touche2020-v2", top_k=2)

    assert [result.doc_id for result in results] == ["d2", "d1"]
    assert results[0].metadata["pipeline"] == "bm25_then_dense_reranking"
    assert "query_time_ms" in results[0].metadata

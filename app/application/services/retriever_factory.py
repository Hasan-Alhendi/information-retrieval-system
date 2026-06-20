"""Central retriever factory shared by API, Streamlit, and evaluation."""

from __future__ import annotations

from app.infrastructure.retrieval.bm25_retriever import BM25Retriever
from app.infrastructure.retrieval.embedding_retriever import EmbeddingRetriever
from app.infrastructure.retrieval.hybrid_serial import HybridSerialRetriever
from app.infrastructure.retrieval.rrf_retriever import ReciprocalRankFusionRetriever
from app.infrastructure.retrieval.tfidf_retriever import TFIDFRetriever


SUPPORTED_MODELS = (
    "tfidf",
    "bm25",
    "embedding",
    "hybrid_serial",
    "hybrid_parallel",
)


def create_retriever(
    model_name: str,
    *,
    max_docs: int | None,
    bm25_k1: float = 1.5,
    bm25_b: float = 0.75,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
):
    """Create the requested retriever with consistent dependencies."""
    if model_name == "tfidf":
        return TFIDFRetriever(max_docs=max_docs)
    if model_name == "bm25":
        return BM25Retriever(k1=bm25_k1, b=bm25_b, max_docs=max_docs)
    if model_name == "embedding":
        return EmbeddingRetriever(
            embedding_model_name=embedding_model,
            max_docs=max_docs,
            batch_size=16,
        )

    bm25 = BM25Retriever(k1=bm25_k1, b=bm25_b, max_docs=max_docs)
    embedding = EmbeddingRetriever(
        embedding_model_name=embedding_model,
        max_docs=max_docs,
        batch_size=16,
    )

    if model_name == "hybrid_serial":
        return HybridSerialRetriever(
            bm25_retriever=bm25,
            embedding_retriever=embedding,
            max_docs=max_docs,
        )
    if model_name == "hybrid_parallel":
        return ReciprocalRankFusionRetriever(
            tfidf_retriever=TFIDFRetriever(max_docs=max_docs),
            bm25_retriever=bm25,
            embedding_retriever=embedding,
            max_docs=max_docs,
        )
    raise ValueError(
        f"Unsupported model '{model_name}'. Supported models: {', '.join(SUPPORTED_MODELS)}"
    )

"""Streamlit search page."""

import streamlit as st

from app.infrastructure.retrieval.bm25_retriever import BM25Retriever
from app.infrastructure.retrieval.embedding_retriever import EmbeddingRetriever
from app.infrastructure.retrieval.hybrid_parallel import HybridParallelRetriever
from app.infrastructure.retrieval.hybrid_serial import HybridSerialRetriever
from app.infrastructure.retrieval.tfidf_retriever import TFIDFRetriever
from app.presentation.streamlit.components.result_card import render_result_card


def render_search_page(settings: dict[str, object]) -> None:
    """Render the search page."""
    st.header("Document Search")
    st.caption("Search the selected dataset using one retrieval model.")

    query = st.text_input("Query", placeholder="e.g. information retrieval ranking")
    search_clicked = st.button("Search", type="primary", use_container_width=True)

    if not search_clicked:
        return

    if not query.strip():
        st.warning("Please enter a query.")
        return

    with st.spinner("Searching... Missing indexes will be built automatically."):
        retriever = _create_retriever(settings)
        results = retriever.search(
            query=query,
            dataset_name=str(settings["dataset_name"]),
            top_k=int(settings["top_k"]),
        )

    st.subheader(f"Results ({len(results)})")
    if not results:
        st.info("No results found.")
        return

    for result in results:
        render_result_card(result)


def _create_retriever(settings: dict[str, object]):
    model_name = str(settings["model_name"])
    max_docs = int(settings["max_docs"])
    bm25_k1 = float(settings["bm25_k1"])
    bm25_b = float(settings["bm25_b"])
    embedding_model = str(settings["embedding_model"])

    if model_name == "tfidf":
        return TFIDFRetriever(max_docs=max_docs)
    if model_name == "bm25":
        return BM25Retriever(k1=bm25_k1, b=bm25_b, max_docs=max_docs)
    if model_name == "embedding":
        return EmbeddingRetriever(embedding_model_name=embedding_model, max_docs=max_docs)
    if model_name == "hybrid_serial":
        return HybridSerialRetriever(
            bm25_retriever=BM25Retriever(k1=bm25_k1, b=bm25_b, max_docs=max_docs),
            embedding_retriever=EmbeddingRetriever(
                embedding_model_name=embedding_model,
                max_docs=max_docs,
            ),
            max_docs=max_docs,
        )
    if model_name == "hybrid_parallel":
        return HybridParallelRetriever(
            tfidf_retriever=TFIDFRetriever(max_docs=max_docs),
            bm25_retriever=BM25Retriever(k1=bm25_k1, b=bm25_b, max_docs=max_docs),
            embedding_retriever=EmbeddingRetriever(
                embedding_model_name=embedding_model,
                max_docs=max_docs,
            ),
            max_docs=max_docs,
        )
    raise ValueError(f"Unsupported model: {model_name}")

"""Streamlit sidebar controls."""

import streamlit as st

DATASET_OPTIONS = {
    "MS MARCO": "msmarco",
    "Natural Questions": "nq",
}

MODEL_OPTIONS = {
    "TF-IDF": "tfidf",
    "BM25": "bm25",
    "Embedding": "embedding",
    "Hybrid Serial": "hybrid_serial",
    "Hybrid Parallel": "hybrid_parallel",
}


def render_sidebar() -> dict[str, object]:
    """Render sidebar controls and return selected settings."""
    st.sidebar.title("IR System")
    st.sidebar.caption("Clean Architecture Retrieval Demo")

    dataset_label = st.sidebar.selectbox("Dataset", list(DATASET_OPTIONS.keys()))
    model_label = st.sidebar.selectbox("Retrieval Model", list(MODEL_OPTIONS.keys()))

    st.sidebar.divider()
    top_k = st.sidebar.slider("Top K", min_value=1, max_value=50, value=10)
    max_docs = st.sidebar.number_input(
        "Max documents for demo",
        min_value=100,
        max_value=250000,
        value=1000,
        step=100,
        help="Use a small number for fast local demos.",
    )

    st.sidebar.divider()
    st.sidebar.subheader("Query Processing")
    use_query_refinement = st.sidebar.checkbox(
        "Enable Query Refinement",
        value=False,
        help="Apply spelling correction and domain-specific query expansion.",
    )

    st.sidebar.divider()
    st.sidebar.subheader("BM25 Parameters")
    bm25_k1 = st.sidebar.slider("k1", min_value=0.1, max_value=3.0, value=1.5, step=0.1)
    bm25_b = st.sidebar.slider("b", min_value=0.0, max_value=1.0, value=0.75, step=0.05)

    st.sidebar.divider()
    st.sidebar.subheader("Embedding Parameters")
    embedding_model = st.sidebar.text_input(
        "Embedding model",
        value="sentence-transformers/all-MiniLM-L6-v2",
    )

    return {
        "dataset_name": DATASET_OPTIONS[dataset_label],
        "model_name": MODEL_OPTIONS[model_label],
        "top_k": top_k,
        "max_docs": int(max_docs),
        "use_query_refinement": bool(use_query_refinement),
        "bm25_k1": float(bm25_k1),
        "bm25_b": float(bm25_b),
        "embedding_model": embedding_model,
    }

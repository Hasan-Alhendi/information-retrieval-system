"""Streamlit sidebar controls."""

import streamlit as st

from app.infrastructure.datasets.dataset_registry import SUPPORTED_DATASETS

DATASET_OPTIONS = {
    config.display_name: dataset_name
    for dataset_name, config in SUPPORTED_DATASETS.items()
}

MODEL_OPTIONS = {
    "BM25": "bm25",
    "TF-IDF": "tfidf",
    "Embedding": "embedding",
    "Embedding + Automatic Clustering": "embedding_clustered",
    "Embedding + Guided Categories": "embedding_guided_categories",
    "Hybrid Serial": "hybrid_serial",
    "Hybrid Parallel": "hybrid_parallel",
}


def render_sidebar() -> dict[str, object]:
    """Render sidebar controls and return selected settings."""
    st.sidebar.title("IR System")
    st.sidebar.caption("Full-corpus retrieval with Clean Architecture")

    dataset_label = st.sidebar.selectbox("Dataset", list(DATASET_OPTIONS))
    model_label = st.sidebar.selectbox("Retrieval Model", list(MODEL_OPTIONS))

    st.sidebar.divider()
    top_k = st.sidebar.slider("Top K", min_value=1, max_value=50, value=10)
    scope = st.sidebar.radio(
        "Index scope",
        ["Full dataset", "Development subset"],
        help=(
            "Full dataset uses the completed disk/FAISS indexes. Development subset "
            "is intended only for quick code tests."
        ),
    )
    max_docs: int | None = None
    if scope == "Development subset":
        max_docs = int(
            st.sidebar.number_input(
                "Development documents",
                min_value=100,
                max_value=250000,
                value=1000,
                step=100,
            )
        )
    else:
        st.sidebar.success("Full-corpus mode")

    st.sidebar.divider()
    st.sidebar.subheader("Query Processing")
    use_query_refinement = st.sidebar.checkbox(
        "Enable Query Refinement",
        value=False,
        help="Apply spelling correction and query expansion before retrieval.",
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
        "max_docs": max_docs,
        "index_scope": "full" if max_docs is None else "development",
        "use_query_refinement": bool(use_query_refinement),
        "bm25_k1": float(bm25_k1),
        "bm25_b": float(bm25_b),
        "embedding_model": embedding_model,
    }

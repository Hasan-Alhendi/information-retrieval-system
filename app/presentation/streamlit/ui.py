"""Streamlit application entry point."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from app.presentation.streamlit.components.sidebar import render_sidebar
from app.presentation.streamlit.pages.clustering_page import render_clustering_page
from app.presentation.streamlit.pages.datasets_page import render_datasets_page
from app.presentation.streamlit.pages.evaluation_page import render_evaluation_page
from app.presentation.streamlit.pages.search_page import render_search_page


def main() -> None:
    """Run the Streamlit UI."""
    st.set_page_config(
        page_title="Information Retrieval System",
        page_icon="🔎",
        layout="wide",
    )

    st.title("Information Retrieval System")
    st.caption("TF-IDF · BM25 · Embeddings · Hybrid Retrieval · Evaluation")

    settings = render_sidebar()

    page = st.tabs(["Search", "Evaluation", "Datasets", "Clustering"])
    with page[0]:
        render_search_page(settings)
    with page[1]:
        render_evaluation_page(settings)
    with page[2]:
        render_datasets_page()
    with page[3]:
        render_clustering_page()


if __name__ == "__main__":
    main()

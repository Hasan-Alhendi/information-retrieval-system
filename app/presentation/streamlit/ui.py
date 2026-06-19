"""Streamlit application entry point."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import streamlit.components.v1 as components

from app.presentation.streamlit.components.sidebar import render_sidebar
from app.presentation.streamlit.views.clustering_view import render_clustering_page
from app.presentation.streamlit.views.datasets_view import render_datasets_page
from app.presentation.streamlit.views.evaluation_view import render_evaluation_page
from app.presentation.streamlit.views.search_view import render_search_page


def _hide_streamlit_toolbar() -> None:
    """Hide Deploy and Streamlit's built-in toolbar/menu controls."""
    st.markdown(
        """
        <style>
        [data-testid="stToolbar"],
        [data-testid="stAppDeployButton"],
        .stAppDeployButton,
        #MainMenu {
            display: none !important;
            visibility: hidden !important;
        }

        [data-testid="stHeader"] {
            background: transparent !important;
        }

        footer {
            display: none !important;
            visibility: hidden !important;
        }

        @media print {
            [data-testid="stSidebar"],
            [data-testid="stToolbar"],
            [data-testid="stHeader"],
            iframe[title="streamlit_components.streamlit_components.v1.html"] {
                display: none !important;
            }

            .block-container {
                max-width: 100% !important;
                padding: 0 !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_print_button() -> None:
    """Render the only top-level utility button kept in the interface."""
    components.html(
        """
        <div style="display:flex; justify-content:flex-end; width:100%;">
            <button id="print-page" type="button" style="
                background:#ff4b4b;
                color:#ffffff;
                border:0;
                border-radius:0.5rem;
                padding:0.55rem 1.1rem;
                font-family:system-ui, sans-serif;
                font-size:0.95rem;
                font-weight:600;
                cursor:pointer;
            ">
                Print
            </button>
        </div>
        <script>
            document.getElementById("print-page").addEventListener("click", function () {
                window.parent.print();
            });
        </script>
        """,
        height=48,
        scrolling=False,
    )


def main() -> None:
    """Run the Streamlit UI."""
    st.set_page_config(
        page_title="Full-Corpus Information Retrieval System",
        page_icon="🔎",
        layout="wide",
        menu_items={
            "Get Help": None,
            "Report a bug": None,
            "About": None,
        },
    )

    _hide_streamlit_toolbar()
    _render_print_button()

    st.title("Information Retrieval System")
    st.caption(
        "Quora · Webis-Touché 2020 v2 · Full-corpus BM25, TF-IDF, Embedding, and Hybrid Retrieval"
    )

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

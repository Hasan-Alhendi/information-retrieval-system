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
    """Hide Streamlit chrome and define a system-theme-aware print layout."""
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

        @page {
            size: A4 portrait;
            margin: 12mm 11mm 14mm;
        }

        @media print {
            *,
            *::before,
            *::after {
                -webkit-print-color-adjust: exact !important;
                print-color-adjust: exact !important;
            }

            [data-testid="stSidebar"],
            [data-testid="stToolbar"],
            [data-testid="stHeader"],
            [data-testid="stStatusWidget"],
            [data-testid="stTextInput"],
            [data-testid="stNumberInput"],
            [data-testid="stSelectbox"],
            [data-testid="stMultiSelect"],
            [data-testid="stSlider"],
            [data-testid="stCheckbox"],
            [data-testid="stRadio"],
            [data-testid="stBaseButton-secondary"],
            [data-testid="stBaseButton-primary"],
            iframe[title="streamlit_components.streamlit_components.v1.html"] {
                display: none !important;
            }

            html,
            body,
            .stApp,
            [data-testid="stAppViewContainer"],
            [data-testid="stMain"] {
                width: 100% !important;
                min-width: 0 !important;
                max-width: none !important;
                height: auto !important;
                overflow: visible !important;
            }

            .block-container {
                width: 100% !important;
                max-width: 188mm !important;
                margin: 0 auto !important;
                padding: 0 !important;
            }

            h1 {
                font-size: 20pt !important;
                line-height: 1.15 !important;
                margin: 0 0 5mm !important;
            }

            h2 {
                font-size: 15pt !important;
                line-height: 1.2 !important;
                margin: 5mm 0 3mm !important;
            }

            h3 {
                font-size: 12pt !important;
                line-height: 1.25 !important;
                margin: 4mm 0 2mm !important;
            }

            p,
            li,
            label,
            [data-testid="stMarkdownContainer"] {
                font-size: 9.5pt !important;
                line-height: 1.4 !important;
            }

            [data-testid="stMetric"],
            [data-testid="stVerticalBlockBorderWrapper"],
            [data-testid="stDataFrame"],
            [data-testid="stTable"],
            details,
            figure,
            table,
            img,
            svg,
            canvas {
                break-inside: avoid !important;
                page-break-inside: avoid !important;
            }

            [data-testid="stDataFrame"],
            [data-testid="stTable"],
            table {
                width: 100% !important;
                max-width: 100% !important;
                overflow: visible !important;
                font-size: 8.5pt !important;
            }

            img,
            svg,
            canvas {
                max-width: 100% !important;
                height: auto !important;
            }

            /* Laptop/system dark mode. This class is added immediately before print. */
            html.print-system-dark,
            html.print-system-dark body,
            html.print-system-dark .stApp,
            html.print-system-dark [data-testid="stAppViewContainer"],
            html.print-system-dark [data-testid="stMain"] {
                color-scheme: dark !important;
                background: #0e1117 !important;
                background-color: #0e1117 !important;
                color: #fafafa !important;
                box-shadow: inset 0 0 0 100vmax #0e1117 !important;
            }

            html.print-system-dark h1,
            html.print-system-dark h2,
            html.print-system-dark h3,
            html.print-system-dark h4,
            html.print-system-dark h5,
            html.print-system-dark h6,
            html.print-system-dark p,
            html.print-system-dark li,
            html.print-system-dark label,
            html.print-system-dark span,
            html.print-system-dark div,
            html.print-system-dark td,
            html.print-system-dark th,
            html.print-system-dark [data-testid="stMarkdownContainer"] {
                color: #fafafa !important;
            }

            html.print-system-dark [data-testid="stMetric"],
            html.print-system-dark [data-testid="stVerticalBlockBorderWrapper"],
            html.print-system-dark details,
            html.print-system-dark [data-testid="stDataFrame"],
            html.print-system-dark [data-testid="stTable"],
            html.print-system-dark table {
                background: #262730 !important;
                background-color: #262730 !important;
                color: #fafafa !important;
                border-color: #4b5563 !important;
                box-shadow: none !important;
            }

            html.print-system-dark a {
                color: #8ec5ff !important;
                text-decoration: none !important;
            }

            html.print-system-dark hr {
                border-color: #4b5563 !important;
            }

            /* Laptop/system light mode. */
            html.print-system-light,
            html.print-system-light body,
            html.print-system-light .stApp,
            html.print-system-light [data-testid="stAppViewContainer"],
            html.print-system-light [data-testid="stMain"] {
                color-scheme: light !important;
                background: #ffffff !important;
                background-color: #ffffff !important;
                color: #111827 !important;
                box-shadow: inset 0 0 0 100vmax #ffffff !important;
            }

            html.print-system-light h1,
            html.print-system-light h2,
            html.print-system-light h3,
            html.print-system-light h4,
            html.print-system-light h5,
            html.print-system-light h6,
            html.print-system-light p,
            html.print-system-light li,
            html.print-system-light label,
            html.print-system-light span,
            html.print-system-light div,
            html.print-system-light td,
            html.print-system-light th,
            html.print-system-light [data-testid="stMarkdownContainer"] {
                color: #111827 !important;
            }

            html.print-system-light [data-testid="stMetric"],
            html.print-system-light [data-testid="stVerticalBlockBorderWrapper"],
            html.print-system-light details,
            html.print-system-light [data-testid="stDataFrame"],
            html.print-system-light [data-testid="stTable"],
            html.print-system-light table {
                background: #ffffff !important;
                background-color: #ffffff !important;
                color: #111827 !important;
                border-color: #d1d5db !important;
                box-shadow: none !important;
            }

            html.print-system-light a {
                color: #2563eb !important;
                text-decoration: none !important;
            }

            html.print-system-light hr {
                border-color: #d1d5db !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_print_button() -> None:
    """Render a compact print button that follows the laptop system theme."""
    components.html(
        """
        <style>
            html, body {
                margin: 0;
                padding: 0;
                background: transparent;
                overflow: hidden;
                font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            }

            .print-wrap {
                display: flex;
                justify-content: flex-end;
                width: 100%;
                padding: 2px 1px 4px;
                box-sizing: border-box;
            }

            .print-button {
                display: inline-flex;
                align-items: center;
                gap: 8px;
                min-height: 36px;
                padding: 7px 13px;
                border: 1px solid #cbd5e1;
                border-radius: 9px;
                background: #f8fafc;
                color: #1f2937;
                box-shadow: 0 1px 2px rgba(15, 23, 42, 0.08);
                font-size: 14px;
                font-weight: 600;
                line-height: 1;
                cursor: pointer;
                transition: background 140ms ease, border-color 140ms ease, box-shadow 140ms ease, transform 140ms ease;
            }

            @media (prefers-color-scheme: dark) {
                .print-button {
                    border-color: #4b5563;
                    background: #262730;
                    color: #fafafa;
                    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.35);
                }
            }

            .print-button:hover {
                filter: brightness(1.08);
                transform: translateY(-1px);
            }

            .print-button:active {
                transform: translateY(0);
            }

            .print-button:focus-visible {
                outline: 2px solid #64748b;
                outline-offset: 2px;
            }

            .print-icon {
                width: 16px;
                height: 16px;
                stroke: currentColor;
                stroke-width: 1.8;
                fill: none;
                stroke-linecap: round;
                stroke-linejoin: round;
            }
        </style>

        <div class="print-wrap">
            <button id="print-page" class="print-button" type="button" aria-label="Print page">
                <svg class="print-icon" viewBox="0 0 24 24" aria-hidden="true">
                    <path d="M6 9V3h12v6"></path>
                    <path d="M6 18H4a2 2 0 0 1-2-2v-5a2 2 0 0 1 2-2h16a2 2 0 0 1 2 2v5a2 2 0 0 1-2 2h-2"></path>
                    <path d="M6 14h12v7H6z"></path>
                    <path d="M18 12h.01"></path>
                </svg>
                <span>Print</span>
            </button>
        </div>

        <script>
            const printButton = document.getElementById("print-page");

            printButton.addEventListener("click", function () {
                const parentWindow = window.parent;
                const parentDocument = parentWindow.document;
                const root = parentDocument.documentElement;
                const body = parentDocument.body;
                const systemUsesDarkMode = parentWindow.matchMedia(
                    "(prefers-color-scheme: dark)"
                ).matches;

                root.classList.remove("print-system-dark", "print-system-light");
                body.classList.remove("print-system-dark", "print-system-light");

                const printThemeClass = systemUsesDarkMode
                    ? "print-system-dark"
                    : "print-system-light";

                root.classList.add(printThemeClass);
                body.classList.add(printThemeClass);

                const cleanupPrintTheme = function () {
                    root.classList.remove("print-system-dark", "print-system-light");
                    body.classList.remove("print-system-dark", "print-system-light");
                };

                parentWindow.addEventListener("afterprint", cleanupPrintTheme, { once: true });

                parentWindow.requestAnimationFrame(function () {
                    parentWindow.requestAnimationFrame(function () {
                        parentWindow.print();
                    });
                });
            });
        </script>
        """,
        height=44,
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

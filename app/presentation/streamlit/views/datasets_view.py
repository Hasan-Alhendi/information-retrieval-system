"""Streamlit datasets view."""

from __future__ import annotations

import streamlit as st

from app.infrastructure.datasets.dataset_loader import DatasetLoader
from app.infrastructure.datasets.dataset_registry import SUPPORTED_DATASETS
from app.infrastructure.retrieval.disk_lexical_index import DiskLexicalIndex
from app.infrastructure.retrieval.embedding_retriever import EmbeddingRetriever


def render_datasets_page() -> None:
    """Render official dataset metadata and local full-index status."""
    st.header("Official Datasets")
    st.caption(
        "Quora provides duplicate-question retrieval; Touché provides argument retrieval."
    )

    loader = DatasetLoader()
    for dataset_name, config in SUPPORTED_DATASETS.items():
        with st.container(border=True):
            st.subheader(config.display_name)
            st.write(
                {
                    "name": config.name,
                    "source": config.source,
                    "external_id": config.external_id,
                    "task_type": config.task_type,
                    "processing_profile": config.processing_profile,
                }
            )

            button_columns = st.columns(2)
            if button_columns[0].button(
                "Load benchmark summary",
                key=f"summary_{dataset_name}",
                use_container_width=True,
            ):
                with st.spinner("Loading dataset summary..."):
                    st.json(loader.summary(dataset_name))

            if button_columns[1].button(
                "Check full indexes",
                key=f"indexes_{dataset_name}",
                use_container_width=True,
            ):
                lexical_status = DiskLexicalIndex(dataset_name).status()
                dense_status = EmbeddingRetriever(max_docs=None).full_status(dataset_name)
                status_columns = st.columns(2)
                with status_columns[0]:
                    st.markdown("**Lexical index**")
                    st.json(lexical_status)
                with status_columns[1]:
                    st.markdown("**Embedding index**")
                    st.json(dense_status)

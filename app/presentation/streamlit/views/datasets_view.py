"""Streamlit datasets view."""

import streamlit as st

from app.infrastructure.datasets.beir_loader import BeirDatasetLoader
from app.infrastructure.datasets.dataset_registry import SUPPORTED_DATASETS


def render_datasets_page() -> None:
    """Render dataset information."""
    st.header("Datasets")
    st.caption("Supported BEIR-compatible datasets.")

    loader = BeirDatasetLoader()
    for dataset_name, config in SUPPORTED_DATASETS.items():
        with st.container(border=True):
            st.subheader(config.display_name)
            st.write(
                {
                    "name": config.name,
                    "source": config.source,
                    "configured_document_limit": config.document_limit,
                }
            )
            if st.button(f"Load summary for {dataset_name}", key=f"summary_{dataset_name}"):
                with st.spinner("Loading dataset summary..."):
                    summary = loader.summary(dataset_name)
                st.json(summary)

"""Streamlit search result card component."""

import streamlit as st

from app.domain.models.search_result import SearchResult


def render_result_card(result: SearchResult) -> None:
    """Render one search result."""
    with st.container(border=True):
        st.markdown(f"### #{result.rank} — `{result.doc_id}`")
        st.metric("Score", f"{result.score:.4f}")
        if result.title:
            st.markdown(f"**Title:** {result.title}")
        if result.text:
            preview = result.text[:800] + ("..." if len(result.text) > 800 else "")
            st.write(preview)
        if result.metadata:
            with st.expander("Metadata"):
                st.json(result.metadata)

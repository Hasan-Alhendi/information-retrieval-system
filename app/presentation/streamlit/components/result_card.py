"""Streamlit search result card component."""

import streamlit as st

from app.domain.models.search_result import SearchResult


def render_result_card(result: SearchResult) -> None:
    """Render one search result with dataset metadata."""
    with st.container(border=True):
        header_columns = st.columns([4, 1, 1])
        header_columns[0].markdown(f"### #{result.rank} — `{result.doc_id}`")
        header_columns[1].metric("Score", f"{result.score:.4f}")
        query_time = result.metadata.get("query_time_ms") if result.metadata else None
        if query_time is not None:
            header_columns[2].metric("Time", f"{float(query_time):.1f} ms")

        if result.title:
            st.markdown(f"**Title:** {result.title}")

        stance = result.metadata.get("stance") if result.metadata else None
        url = result.metadata.get("url") if result.metadata else None
        if stance:
            st.caption(f"Stance: {stance}")
        if url:
            st.link_button("Open source", str(url))

        if result.text:
            preview = result.text[:800] + ("..." if len(result.text) > 800 else "")
            st.write(preview)

        if result.metadata:
            with st.expander("Metadata"):
                st.json(result.metadata)

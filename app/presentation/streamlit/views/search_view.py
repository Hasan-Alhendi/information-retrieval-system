"""Streamlit search view."""

from __future__ import annotations

import time

import streamlit as st

from app.application.services.query_refinement_service import QueryRefinementService
from app.application.services.retriever_factory import create_retriever
from app.presentation.streamlit.components.result_card import render_result_card


@st.cache_resource(show_spinner=False)
def _cached_retriever(
    model_name: str,
    dataset_name: str,
    max_docs: int | None,
    bm25_k1: float,
    bm25_b: float,
    embedding_model: str,
):
    """Keep heavy models and indexes loaded across Streamlit reruns."""
    retriever = create_retriever(
        model_name,
        max_docs=max_docs,
        bm25_k1=bm25_k1,
        bm25_b=bm25_b,
        embedding_model=embedding_model,
    )
    if max_docs is None and hasattr(retriever, "prepare"):
        retriever.prepare(dataset_name)
    return retriever


def render_search_page(settings: dict[str, object]) -> None:
    """Render the search page."""
    st.header("Document Search")
    st.caption(
        "Search the full collection or a development subset with any retrieval model."
    )

    scope_label = (
        "Full dataset"
        if settings.get("max_docs") is None
        else f"Development subset: {settings['max_docs']:,} documents"
    )
    st.info(
        f"Dataset: **{settings['dataset_name']}** · Model: **{settings['model_name']}** "
        f"· Scope: **{scope_label}**"
    )

    query = st.text_input(
        "Query",
        placeholder="e.g. Should teachers get tenure?",
    )
    search_clicked = st.button("Search", type="primary", use_container_width=True)

    if not search_clicked:
        return
    if not query.strip():
        st.warning("Please enter a query.")
        return

    active_query = query
    if bool(settings.get("use_query_refinement", False)):
        refinement = QueryRefinementService().refine(query)
        active_query = refinement.refined_query
        with st.expander("Query Refinement", expanded=True):
            st.json(
                {
                    "original_query": refinement.original_query,
                    "refined_query": refinement.refined_query,
                    "corrections": refinement.corrections,
                    "expansions": refinement.expansions,
                }
            )

    try:
        with st.spinner("Searching the selected index..."):
            retriever = _cached_retriever(
                str(settings["model_name"]),
                str(settings["dataset_name"]),
                settings.get("max_docs"),
                float(settings["bm25_k1"]),
                float(settings["bm25_b"]),
                str(settings["embedding_model"]),
            )
            started = time.perf_counter()
            results = retriever.search(
                query=active_query,
                dataset_name=str(settings["dataset_name"]),
                top_k=int(settings["top_k"]),
            )
            wall_time_ms = (time.perf_counter() - started) * 1000.0
    except (RuntimeError, ValueError, OSError) as exc:
        st.error(str(exc))
        st.info(
            "For full mode, build the selected dataset indexes before searching. "
            "Development mode can be used for quick checks."
        )
        return

    internal_time_ms = (
        float(results[0].metadata.get("query_time_ms", wall_time_ms))
        if results
        else wall_time_ms
    )
    metric_columns = st.columns(3)
    metric_columns[0].metric("Results", len(results))
    metric_columns[1].metric("Search time", f"{internal_time_ms:.1f} ms")
    metric_columns[2].metric("Wall time", f"{wall_time_ms:.1f} ms")

    if not results:
        st.info("No results found.")
        return

    for result in results:
        render_result_card(result)

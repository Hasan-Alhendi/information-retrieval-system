"""Streamlit guided category view."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.infrastructure.clustering.category_profiles import get_category_profiles
from app.infrastructure.datasets.dataset_registry import SUPPORTED_DATASETS
from app.infrastructure.evaluation.evaluator_v2 import RetrievalEvaluatorV2


def render_clustering_page() -> None:
    """Render guided semantic categories and their retrieval comparison."""
    st.header("Guided Semantic Categories")
    st.caption(
        "Compare normal embedding search with guided category reranking. "
        "The system uses manually defined category names, but assigns queries and "
        "candidate documents automatically using embedding similarity."
    )

    dataset_name = st.selectbox(
        "Dataset",
        options=list(SUPPORTED_DATASETS),
        format_func=lambda name: SUPPORTED_DATASETS[name].display_name,
        key="guided_categories_dataset",
    )
    max_docs = st.number_input(
        "Max documents",
        min_value=100,
        max_value=10000,
        value=1000,
        step=100,
        help="Use the same development subset for the baseline and guided comparison.",
    )
    embedding_model = st.text_input(
        "Embedding model",
        value="sentence-transformers/all-MiniLM-L6-v2",
        key="guided_categories_embedding_model",
    )

    _render_guided_categories(dataset_name)
    _render_retrieval_impact_comparison(
        dataset_name=dataset_name,
        max_docs=int(max_docs),
        embedding_model=embedding_model,
    )


def _render_guided_categories(dataset_name: str) -> None:
    """Show the manually defined category names used for automatic assignment."""
    profiles = get_category_profiles(dataset_name)
    with st.expander("Guided semantic categories", expanded=True):
        st.caption(
            "Only category names and descriptions are defined manually. Queries and "
            "documents are assigned automatically by embedding similarity, and may "
            "belong softly to several categories."
        )
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Category": profile.label,
                        "Description": profile.description,
                    }
                    for profile in profiles
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )


def _render_retrieval_impact_comparison(
    *,
    dataset_name: str,
    max_docs: int,
    embedding_model: str,
) -> None:
    """Compare baseline embedding retrieval with guided category reranking."""
    st.subheader("Retrieval Quality Comparison")
    st.caption(
        "Guided Categories reuses FAISS candidate vectors. It does not rebuild the "
        "document index; only the small list of category descriptions is encoded."
    )

    with st.form("guided_category_retrieval_comparison"):
        columns = st.columns(3)
        max_queries = int(
            columns[0].number_input(
                "Evaluation queries",
                min_value=1,
                max_value=1000,
                value=100,
                step=10,
            )
        )
        candidate_k = int(
            columns[1].number_input(
                "Candidates to rerank",
                min_value=10,
                max_value=1000,
                value=100,
                step=10,
            )
        )
        top_categories = int(
            columns[2].slider(
                "Top query categories",
                min_value=1,
                max_value=5,
                value=3,
            )
        )

        category_weight = float(
            st.slider(
                "Guided category weight",
                min_value=0.0,
                max_value=1.0,
                value=0.25,
                step=0.05,
            )
        )
        submitted = st.form_submit_button(
            "Compare Embedding vs Guided Categories",
            type="primary",
            use_container_width=True,
        )

    if not submitted:
        return

    evaluator = RetrievalEvaluatorV2(
        max_docs=max_docs,
        top_k=10,
        max_queries=max_queries,
        embedding_model=embedding_model,
        category_weight=category_weight,
        category_candidate_k=candidate_k,
        top_categories=top_categories,
    )

    try:
        with st.spinner("Evaluating baseline and guided categories..."):
            baseline = evaluator.evaluate(dataset_name, "embedding")
            guided = evaluator.evaluate(
                dataset_name,
                "embedding_guided_categories",
            )
    except (RuntimeError, ValueError, OSError) as exc:
        st.error(str(exc))
        return

    frame = pd.DataFrame(
        [
            _evaluation_row("Embedding baseline", baseline),
            _evaluation_row("Guided Categories", guided),
        ]
    ).set_index("Condition")
    st.dataframe(frame, use_container_width=True)

    delta_frame = pd.DataFrame(
        [_delta_row("Guided - Baseline", guided, baseline)]
    ).set_index("Comparison")
    st.subheader("Change Relative to Embedding Baseline")
    st.dataframe(delta_frame, use_container_width=True)

    quality_columns = ["MAP@10", "Recall@10", "Precision@10", "nDCG@10"]
    st.subheader("Quality Metrics")
    st.bar_chart(frame[quality_columns])

    st.caption(
        "Positive quality deltas indicate an improvement. A positive time delta "
        "indicates additional latency."
    )
    st.download_button(
        "Download guided comparison CSV",
        data=frame.reset_index().to_csv(index=False).encode("utf-8"),
        file_name=f"{dataset_name}_guided_categories_dev_{max_docs}.csv",
        mime="text/csv",
        use_container_width=True,
    )


def _evaluation_row(condition: str, result) -> dict[str, object]:
    return {
        "Condition": condition,
        "MAP@10": result.map_score,
        "Recall@10": result.recall,
        "Precision@10": result.precision_at_10,
        "nDCG@10": result.ndcg,
        "Average time (ms)": result.average_query_time_ms,
        "Queries": result.evaluated_queries,
    }


def _delta_row(comparison: str, result, baseline) -> dict[str, object]:
    return {
        "Comparison": comparison,
        "MAP@10": result.map_score - baseline.map_score,
        "Recall@10": result.recall - baseline.recall,
        "Precision@10": result.precision_at_10 - baseline.precision_at_10,
        "nDCG@10": result.ndcg - baseline.ndcg,
        "Average time (ms)": (
            result.average_query_time_ms - baseline.average_query_time_ms
        ),
    }

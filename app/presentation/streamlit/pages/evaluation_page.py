"""Streamlit evaluation page."""

import streamlit as st

from app.infrastructure.evaluation.evaluator import RetrievalEvaluator
from app.presentation.streamlit.components.metrics_table import render_metrics_table


def render_evaluation_page(settings: dict[str, object]) -> None:
    """Render the evaluation page."""
    st.header("Retrieval Evaluation")
    st.caption("Evaluate the selected model using qrels.")

    max_queries = st.number_input(
        "Max queries for evaluation",
        min_value=1,
        max_value=1000,
        value=25,
        step=1,
    )
    evaluate_clicked = st.button("Run Evaluation", type="primary", use_container_width=True)

    if not evaluate_clicked:
        return

    with st.spinner("Evaluating... Missing indexes will be built automatically."):
        evaluator = RetrievalEvaluator(
            max_docs=int(settings["max_docs"]),
            top_k=int(settings["top_k"]),
            max_queries=int(max_queries),
            bm25_k1=float(settings["bm25_k1"]),
            bm25_b=float(settings["bm25_b"]),
            embedding_model=str(settings["embedding_model"]),
        )
        result = evaluator.evaluate(
            dataset_name=str(settings["dataset_name"]),
            model_name=str(settings["model_name"]),
        )

    st.success("Evaluation completed.")
    render_metrics_table(result)

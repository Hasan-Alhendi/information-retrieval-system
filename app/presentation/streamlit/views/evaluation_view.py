"""Streamlit evaluation view."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.application.services.retriever_factory import SUPPORTED_MODELS
from app.infrastructure.evaluation.evaluator_v2 import RetrievalEvaluatorV2
from app.presentation.streamlit.components.metrics_table import render_metrics_table

MODEL_LABELS = {
    "bm25": "BM25",
    "tfidf": "TF-IDF",
    "embedding": "Embedding",
    "hybrid_serial": "Hybrid Serial",
    "hybrid_parallel": "Hybrid Parallel",
}


def render_evaluation_page(settings: dict[str, object]) -> None:
    """Render single-model or comparative evaluation on qrels."""
    st.header("Retrieval Evaluation")
    st.caption("Evaluate models using the selected dataset queries and qrels.")

    model_names = st.multiselect(
        "Models",
        options=list(SUPPORTED_MODELS),
        default=[str(settings["model_name"])],
        format_func=lambda value: MODEL_LABELS[value],
    )
    max_queries = st.number_input(
        "Max queries for evaluation",
        min_value=1,
        max_value=10000,
        value=49 if settings["dataset_name"] == "touche2020-v2" else 100,
        step=1,
    )
    evaluate_clicked = st.button(
        "Run Evaluation",
        type="primary",
        use_container_width=True,
    )

    if not evaluate_clicked:
        return
    if not model_names:
        st.warning("Select at least one model.")
        return

    max_docs = settings.get("max_docs")
    evaluator = RetrievalEvaluatorV2(
        max_docs=max_docs,
        top_k=int(settings["top_k"]),
        max_queries=int(max_queries),
        bm25_k1=float(settings["bm25_k1"]),
        bm25_b=float(settings["bm25_b"]),
        embedding_model=str(settings["embedding_model"]),
    )

    results = []
    progress = st.progress(0, text="Starting evaluation...")
    try:
        for index, model_name in enumerate(model_names, start=1):
            progress.progress(
                (index - 1) / len(model_names),
                text=f"Evaluating {MODEL_LABELS[model_name]}...",
            )
            results.append(
                evaluator.evaluate(
                    dataset_name=str(settings["dataset_name"]),
                    model_name=model_name,
                )
            )
        progress.progress(1.0, text="Evaluation completed.")
    except (RuntimeError, ValueError, OSError) as exc:
        progress.empty()
        st.error(str(exc))
        return

    if len(results) == 1:
        st.success("Evaluation completed.")
        render_metrics_table(results[0])
        return

    frame = pd.DataFrame(
        [
            {
                "Model": MODEL_LABELS[result.model_name],
                f"MAP@{settings['top_k']}": result.map_score,
                f"Recall@{settings['top_k']}": result.recall,
                "Precision@10": result.precision_at_10,
                f"nDCG@{settings['top_k']}": result.ndcg,
                "Queries": result.evaluated_queries,
            }
            for result in results
        ]
    )
    st.success("Comparative evaluation completed.")
    st.dataframe(frame, use_container_width=True, hide_index=True)

    metric_columns = [
        f"MAP@{settings['top_k']}",
        f"nDCG@{settings['top_k']}",
        "Precision@10",
        f"Recall@{settings['top_k']}",
    ]
    chart_data = frame.set_index("Model")[metric_columns]
    st.subheader("Quality comparison")
    st.bar_chart(chart_data)

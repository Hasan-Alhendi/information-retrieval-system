"""Streamlit metrics table component."""

import pandas as pd
import streamlit as st

from app.domain.models.evaluation_result import EvaluationResult


def render_metrics_table(result: EvaluationResult) -> None:
    """Render evaluation metrics as a table."""
    data = {
        "Metric": ["MAP", "Recall", "Precision@10", "nDCG", "Evaluated Queries"],
        "Value": [
            result.map_score,
            result.recall,
            result.precision_at_10,
            result.ndcg,
            result.evaluated_queries,
        ],
    }
    st.dataframe(pd.DataFrame(data), use_container_width=True)

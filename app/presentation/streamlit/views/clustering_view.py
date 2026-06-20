"""Streamlit clustering view."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.infrastructure.clustering.category_profiles import get_category_profiles
from app.infrastructure.clustering.clusterer import DocumentClusterer
from app.infrastructure.datasets.dataset_registry import SUPPORTED_DATASETS
from app.infrastructure.evaluation.evaluator_v2 import RetrievalEvaluatorV2


@st.cache_resource(show_spinner=False)
def _clusterer(embedding_model_name: str) -> DocumentClusterer:
    """Reuse the embedding model across Streamlit reruns."""
    return DocumentClusterer(embedding_model_name=embedding_model_name)


def render_clustering_page() -> None:
    """Render document clustering and its retrieval impact comparison."""
    st.header("Document Clustering")
    st.caption(
        "Compare normal embedding search, automatic KMeans clustering, and guided "
        "semantic categories on the same queries and relevance judgments."
    )

    dataset_name = st.selectbox(
        "Dataset",
        options=list(SUPPORTED_DATASETS),
        format_func=lambda name: SUPPORTED_DATASETS[name].display_name,
        key="clustering_dataset",
    )
    number_of_clusters = st.slider(
        "Number of automatic clusters",
        min_value=2,
        max_value=20,
        value=5,
    )
    max_docs = st.number_input(
        "Max documents",
        min_value=100,
        max_value=10000,
        value=1000,
        step=100,
        help=(
            "All compared systems use the exact same development subset and the "
            "existing embedding index when it is already available."
        ),
    )
    sample_size = st.slider(
        "Sample documents per automatic cluster",
        min_value=1,
        max_value=10,
        value=3,
    )
    embedding_model = st.text_input(
        "Embedding model",
        value="sentence-transformers/all-MiniLM-L6-v2",
        key="clustering_embedding_model",
    )

    _render_guided_categories(dataset_name)
    _render_retrieval_impact_comparison(
        dataset_name=dataset_name,
        max_docs=int(max_docs),
        number_of_clusters=number_of_clusters,
        embedding_model=embedding_model,
    )

    if not st.button("Run Automatic Clustering", type="primary", use_container_width=True):
        return

    with st.spinner(
        "Encoding and clustering documents... The first run may take a few minutes."
    ):
        result = _clusterer(embedding_model).cluster(
            dataset_name=dataset_name,
            number_of_clusters=number_of_clusters,
            max_docs=int(max_docs),
            sample_size=sample_size,
            persist_artifacts=True,
        )

    if not result["documents_count"]:
        st.warning("No documents were available for clustering.")
        return

    st.success("Automatic clustering completed and artifacts were saved.")
    st.write(
        {
            "dataset_name": result["dataset_name"],
            "documents_count": result["documents_count"],
            "number_of_clusters": result["number_of_clusters"],
            "embedding_model": result["embedding_model"],
            "clustering_algorithm": result["clustering_algorithm"],
            "artifacts_path": result.get("artifacts_path"),
        }
    )

    _render_evaluation(result)
    _render_cluster_charts(result)
    _render_cluster_details(result)


def _render_guided_categories(dataset_name: str) -> None:
    """Show the manually defined category names used for automatic assignment."""
    profiles = get_category_profiles(dataset_name)
    with st.expander("Guided semantic categories", expanded=False):
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
    number_of_clusters: int,
    embedding_model: str,
) -> None:
    """Compare baseline, automatic clustering, and guided categories."""
    st.subheader("Retrieval Quality Comparison")
    st.caption(
        "Guided Categories reuses the existing FAISS candidate vectors. It does not "
        "rebuild the document index; only the small category descriptions are encoded."
    )

    with st.form("clustering_retrieval_comparison"):
        first_row = st.columns(3)
        max_queries = int(
            first_row[0].number_input(
                "Evaluation queries",
                min_value=1,
                max_value=1000,
                value=100,
                step=10,
            )
        )
        candidate_k = int(
            first_row[1].number_input(
                "Candidates to rerank",
                min_value=10,
                max_value=1000,
                value=100,
                step=10,
            )
        )
        top_categories = int(
            first_row[2].slider(
                "Top query categories",
                min_value=1,
                max_value=5,
                value=3,
            )
        )

        second_row = st.columns(2)
        cluster_weight = float(
            second_row[0].slider(
                "Automatic cluster weight",
                min_value=0.0,
                max_value=1.0,
                value=0.2,
                step=0.05,
            )
        )
        category_weight = float(
            second_row[1].slider(
                "Guided category weight",
                min_value=0.0,
                max_value=1.0,
                value=0.25,
                step=0.05,
            )
        )
        submitted = st.form_submit_button(
            "Compare Three Methods",
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
        cluster_count=number_of_clusters,
        cluster_weight=cluster_weight,
        cluster_candidate_k=candidate_k,
        category_weight=category_weight,
        category_candidate_k=candidate_k,
        top_categories=top_categories,
    )

    try:
        with st.spinner(
            "Evaluating baseline, automatic clustering, and guided categories..."
        ):
            baseline = evaluator.evaluate(dataset_name, "embedding")
            automatic = evaluator.evaluate(dataset_name, "embedding_clustered")
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
            _evaluation_row("Automatic KMeans", automatic),
            _evaluation_row("Guided Categories", guided),
        ]
    ).set_index("Condition")
    st.dataframe(frame, use_container_width=True)

    delta_frame = pd.DataFrame(
        [
            _delta_row("Automatic - Baseline", automatic, baseline),
            _delta_row("Guided - Baseline", guided, baseline),
        ]
    ).set_index("Comparison")
    st.subheader("Change Relative to Embedding Baseline")
    st.dataframe(delta_frame, use_container_width=True)

    quality_columns = ["MAP@10", "Recall@10", "Precision@10", "nDCG@10"]
    st.subheader("Quality Metrics")
    st.bar_chart(frame[quality_columns])

    st.caption(
        "Positive quality deltas indicate an improvement. A positive time delta "
        "indicates additional latency. Guided categories use soft assignment to the "
        "best query categories rather than forcing a single hard category."
    )
    st.download_button(
        "Download three-method comparison CSV",
        data=frame.reset_index().to_csv(index=False).encode("utf-8"),
        file_name=(
            f"{dataset_name}_automatic_vs_guided_categories_dev_{max_docs}.csv"
        ),
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


def _render_evaluation(result: dict[str, object]) -> None:
    """Render quantitative internal clustering metrics."""
    evaluation = result["evaluation"]
    assert isinstance(evaluation, dict)

    silhouette = evaluation.get("silhouette_score")
    davies_bouldin = evaluation.get("davies_bouldin_index")
    inertia = evaluation.get("inertia")
    sample_size = evaluation.get("silhouette_sample_size", 0)

    st.subheader("Clustering Evaluation")
    columns = st.columns(3)
    columns[0].metric(
        "Silhouette Score",
        "N/A" if silhouette is None else f"{float(silhouette):.4f}",
        help="Higher is better. Values closer to 1 indicate better separation.",
    )
    columns[1].metric(
        "Davies-Bouldin Index",
        "N/A" if davies_bouldin is None else f"{float(davies_bouldin):.4f}",
        help="Lower is better. It compares within-cluster compactness to separation.",
    )
    columns[2].metric(
        "Inertia",
        "N/A" if inertia is None else f"{float(inertia):.2f}",
        help="Lower means points are closer to their centroids; compare only runs on the same data.",
    )
    st.caption(
        f"Silhouette was computed on {int(sample_size):,} document vectors at most "
        "to keep the evaluation practical on a laptop."
    )


def _render_cluster_charts(result: dict[str, object]) -> None:
    """Render cluster-size and PCA visualizations."""
    clusters = result["clusters"]
    projection = result["projection"]
    assert isinstance(clusters, list)
    assert isinstance(projection, list)

    st.subheader("Cluster Size Distribution")
    size_frame = pd.DataFrame(
        [
            {
                "Cluster": f"Cluster {cluster['cluster_id']}",
                "Documents": cluster["size"],
            }
            for cluster in clusters
        ]
    ).set_index("Cluster")
    st.bar_chart(size_frame)

    if projection:
        st.subheader("PCA Two-Dimensional Projection")
        st.caption(
            "PCA is used only for visualization. Clustering itself is performed "
            "on the original 384-dimensional embeddings."
        )
        projection_frame = pd.DataFrame(projection)
        st.scatter_chart(
            projection_frame,
            x="pc1",
            y="pc2",
            color="cluster",
            use_container_width=True,
        )


def _render_cluster_details(result: dict[str, object]) -> None:
    """Render top terms and representative samples for every cluster."""
    clusters = result["clusters"]
    assert isinstance(clusters, list)

    st.subheader("Cluster Details")
    for cluster in clusters:
        with st.expander(
            f"Cluster {cluster['cluster_id']} — {cluster['size']} documents",
            expanded=False,
        ):
            st.markdown("**Top terms:** " + ", ".join(cluster["top_terms"]))
            for sample in cluster["samples"]:
                st.markdown(f"**Document:** `{sample['doc_id']}`")
                st.write(sample["preview"])
                st.divider()

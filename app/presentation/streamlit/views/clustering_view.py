"""Streamlit clustering view."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.infrastructure.clustering.clusterer import DocumentClusterer
from app.infrastructure.datasets.dataset_registry import SUPPORTED_DATASETS


@st.cache_resource(show_spinner=False)
def _clusterer(embedding_model_name: str) -> DocumentClusterer:
    """Reuse the embedding model across Streamlit reruns."""
    return DocumentClusterer(embedding_model_name=embedding_model_name)


def render_clustering_page() -> None:
    """Render document clustering with quantitative and visual evaluation."""
    st.header("Document Clustering")
    st.caption(
        "Cluster a development subset using dense embeddings and evaluate the "
        "result with internal clustering metrics."
    )

    dataset_name = st.selectbox(
        "Dataset",
        options=list(SUPPORTED_DATASETS),
        format_func=lambda name: SUPPORTED_DATASETS[name].display_name,
        key="clustering_dataset",
    )
    number_of_clusters = st.slider(
        "Number of clusters",
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
            "Clustering is an independent development-subset experiment. "
            "Larger values need more CPU time and memory."
        ),
    )
    sample_size = st.slider(
        "Sample documents per cluster",
        min_value=1,
        max_value=10,
        value=3,
    )
    embedding_model = st.text_input(
        "Embedding model",
        value="sentence-transformers/all-MiniLM-L6-v2",
        key="clustering_embedding_model",
    )

    if not st.button("Run Clustering", type="primary", use_container_width=True):
        return

    with st.spinner(
        "Encoding and clustering documents... The first run may take a few minutes."
    ):
        result = _clusterer(embedding_model).cluster(
            dataset_name=dataset_name,
            number_of_clusters=number_of_clusters,
            max_docs=int(max_docs),
            sample_size=sample_size,
        )

    if not result["documents_count"]:
        st.warning("No documents were available for clustering.")
        return

    st.success("Clustering completed.")
    st.write(
        {
            "dataset_name": result["dataset_name"],
            "documents_count": result["documents_count"],
            "number_of_clusters": result["number_of_clusters"],
            "embedding_model": result["embedding_model"],
            "clustering_algorithm": result["clustering_algorithm"],
        }
    )

    _render_evaluation(result)
    _render_cluster_charts(result)
    _render_cluster_details(result)


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

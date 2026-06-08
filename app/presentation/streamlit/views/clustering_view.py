"""Streamlit clustering view."""

import streamlit as st

from app.infrastructure.clustering.clusterer import DocumentClusterer


def render_clustering_page() -> None:
    """Render document clustering page."""
    st.header("Document Clustering")
    st.caption("Cluster a development subset of the selected dataset using dense embeddings.")

    dataset_name = st.selectbox("Dataset", ["quora", "nq"], key="clustering_dataset")
    number_of_clusters = st.slider("Number of clusters", min_value=2, max_value=20, value=5)
    max_docs = st.number_input(
        "Max documents",
        min_value=100,
        max_value=10000,
        value=1000,
        step=100,
        help="Use a small number for faster clustering demos.",
    )
    sample_size = st.slider("Sample documents per cluster", min_value=1, max_value=10, value=3)
    embedding_model = st.text_input(
        "Embedding model",
        value="sentence-transformers/all-MiniLM-L6-v2",
        key="clustering_embedding_model",
    )

    if not st.button("Run Clustering", type="primary", use_container_width=True):
        return

    with st.spinner("Clustering documents... This may take a few minutes on first run."):
        clusterer = DocumentClusterer(embedding_model_name=embedding_model)
        result = clusterer.cluster(
            dataset_name=dataset_name,
            number_of_clusters=number_of_clusters,
            max_docs=int(max_docs),
            sample_size=sample_size,
        )

    st.success("Clustering completed.")
    st.write(
        {
            "dataset_name": result["dataset_name"],
            "documents_count": result["documents_count"],
            "number_of_clusters": result["number_of_clusters"],
            "embedding_model": result["embedding_model"],
        }
    )

    for cluster in result["clusters"]:
        with st.expander(
            f"Cluster {cluster['cluster_id']} — {cluster['size']} documents",
            expanded=False,
        ):
            st.markdown("**Top terms:** " + ", ".join(cluster["top_terms"]))
            for sample in cluster["samples"]:
                st.markdown(f"**Document:** `{sample['doc_id']}`")
                st.write(sample["preview"])
                st.divider()

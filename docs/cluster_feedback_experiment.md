# Cluster Feedback Reranking

The first comparison used one constant score per global cluster. That often leaves the Top-10 order unchanged because every document in the same cluster receives the same bonus.

The stronger experiment clusters the retrieved candidates for each query, chooses the candidate cluster that best matches the query and the original ranking, and then rewards documents according to their individual similarity to that cluster centroid.

This reuses the existing FAISS index. It does not download the dataset or rebuild document embeddings.

Recommended first run:

- clusters: 8
- cluster weight: 0.35
- candidates: 200
- evaluation queries: 100

Compare against the normal embedding retriever using the same queries and qrels.

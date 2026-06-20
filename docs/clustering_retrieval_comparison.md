# Retrieval Before and After Clustering

This experiment measures whether document clustering improves retrieval quality.

## Compared systems

- **Before clustering:** the normal `embedding` retriever.
- **After clustering:** `embedding_clustered`, which retrieves semantic candidates and reranks them using the similarity between the query vector and each candidate document's cluster centroid.

Both systems use the same dataset subset, embedding model, benchmark queries, qrels, and `top_k=10`.

## Run from the command line

```bash
python scripts/compare_clustering_retrieval.py \
  --dataset quora \
  --max-docs 1000 \
  --max-queries 100 \
  --top-k 10 \
  --clusters 5 \
  --cluster-weight 0.2 \
  --candidate-k 100
```

The first run builds the development embedding index and clustering artifacts. The command then evaluates the normal embedding search and the cluster-aware version.

The CSV is saved under:

```text
storage/evaluation/<dataset>_clustering_retrieval_comparison_dev_<max_docs>.csv
```

It contains three rows:

1. `before_clustering`
2. `after_clustering`
3. `delta_after_minus_before`

## Streamlit comparison

Open the **Document Clustering** page and use **Retrieval Before / After Clustering**.

Choose:

- dataset
- development document count
- number of clusters
- evaluation query count
- cluster weight
- candidate count

Then click **Compare Before vs After**. The page displays both rows, metric deltas, average query time, and a downloadable CSV.

## Metrics

- `MAP@10`: positive delta means relevant documents tend to appear earlier.
- `Recall@10`: positive delta means more known relevant documents appear in the first ten.
- `Precision@10`: positive delta means a larger fraction of the first ten is relevant.
- `nDCG@10`: positive delta means graded relevance ordering improved.
- `Average query time`: positive delta means cluster-aware retrieval is slower.

## Important limitation

Cluster-aware retrieval currently supports development subsets only. This keeps the experiment reproducible and prevents full-corpus clustering from being loaded into memory accidentally.

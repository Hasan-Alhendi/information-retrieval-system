# Retrieval Comparison with Automatic and Guided Clustering

This experiment compares three retrieval paths on the same benchmark queries and relevance judgments.

## Compared systems

1. **Embedding baseline** — normal semantic retrieval.
2. **Automatic KMeans** — global document clustering followed by cluster-aware reranking.
3. **Guided Categories** — human-readable category names are defined once, while query and document assignment is performed automatically using embedding similarity.

The guided method does not manually label every document. It selects the best categories for each query, assigns candidate documents softly to those categories, and mixes category alignment with the original embedding score.

## Dataset-specific guided categories

- **Quora:** Technology, Education, Health, Careers, Finance, Relationships, Travel, Science, Politics, Religion and Philosophy, Entertainment, and Daily Life.
- **Touché 2020:** Politics, Environment, Technology, Education, Health, Economy, Law, Ethics, Social Issues, Science, Religion, and Public Policy.

The definitions are stored in:

```text
app/infrastructure/clustering/category_profiles.py
```

## Reuse of existing indexes

`embedding_guided_categories` reuses the existing FAISS index and reconstructs only the vectors of the retrieved candidates. It does not rebuild all document embeddings and does not download the dataset again. Only the small list of category descriptions is encoded at startup.

## Run the three-way comparison

```bash
python scripts/compare_clustering_retrieval.py \
  --dataset quora \
  --max-docs 1000 \
  --max-queries 100 \
  --top-k 10 \
  --clusters 5 \
  --cluster-weight 0.2 \
  --category-weight 0.25 \
  --top-categories 3 \
  --candidate-k 100
```

The CSV is saved under:

```text
storage/evaluation/<dataset>_automatic_vs_guided_dev_<max_docs>.csv
```

## Evaluate guided categories on an existing full index

When a finalized full FAISS index already exists, compare baseline and guided categories without rebuilding it:

```bash
python scripts/evaluate_full_system.py \
  --dataset quora \
  --models embedding embedding_guided_categories \
  --max-queries 100 \
  --top-k 10 \
  --category-weight 0.25 \
  --category-candidate-k 100 \
  --top-categories 3
```

Do not pass `--max-docs` in this command. The guided retriever requires the existing full index and fails clearly when it is unavailable rather than building a new one.

## Streamlit comparison

Open **Document Clustering**, review **Guided semantic categories**, then use **Retrieval Quality Comparison**. The page shows:

- Embedding baseline
- Automatic KMeans
- Guided Categories
- deltas relative to the baseline
- quality chart
- average query time
- downloadable CSV

## Metrics

- `MAP@10`: higher means relevant documents appear earlier.
- `Recall@10`: higher means more known relevant documents appear in the first ten.
- `Precision@10`: higher means a larger fraction of the first ten is relevant.
- `nDCG@10`: higher means graded relevance ordering improved.
- `Average query time`: lower is faster.

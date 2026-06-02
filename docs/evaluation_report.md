# Evaluation Report

This document describes how retrieval models are evaluated.

## Metrics

The system reports the following IR metrics:

| Metric | Meaning |
|---|---|
| MAP | Mean Average Precision across evaluated queries |
| Recall | Fraction of relevant documents retrieved |
| Precision@10 | Precision among the top 10 results |
| nDCG | Ranking quality with graded relevance |

The metric implementation is located in:

```text
app/infrastructure/evaluation/metrics.py
```

The evaluator is located in:

```text
app/infrastructure/evaluation/evaluator.py
```

## Models Evaluated

The evaluator supports:

- TF-IDF
- BM25
- Embedding retrieval
- Hybrid Serial
- Hybrid Parallel

## Running Evaluation

Baseline evaluation:

```bash
python scripts/evaluate_all.py --dataset msmarco --models bm25 tfidf --max-docs 1000 --max-queries 10
```

Evaluation with query refinement:

```bash
python scripts/evaluate_all.py --dataset msmarco --models bm25 --max-docs 1000 --max-queries 10 --use-query-refinement
```

## Output

Evaluation results are saved as CSV files under:

```text
storage/evaluation/
```

The output columns are:

- dataset_name
- model_name
- mode
- map_score
- recall
- precision_at_10
- ndcg
- evaluated_queries

## Baseline vs Enhanced Evaluation

The project supports comparing:

| Mode | Description |
|---|---|
| baseline | Original query without refinement |
| with_query_refinement | Query after correction and expansion |

This comparison is used to show the effect of the additional query refinement feature.

## Reporting Results

For the final report, include a table like:

| Dataset | Model | Mode | MAP | Recall | P@10 | nDCG | Queries |
|---|---|---|---:|---:|---:|---:|---:|
| msmarco | bm25 | baseline | ... | ... | ... | ... | ... |
| msmarco | bm25 | with_query_refinement | ... | ... | ... | ... | ... |

Use actual generated CSV values after running the experiments locally.

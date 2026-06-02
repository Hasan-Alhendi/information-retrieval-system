# Datasets

This project uses BEIR-compatible Information Retrieval datasets.

Each dataset should contain:

- Documents
- Queries
- qrels relevance judgments

## Registered Datasets

| Dataset | Name in Code | Source | Demo Limit |
|---|---|---|---:|
| MS MARCO | `msmarco` | BEIR | configurable |
| Natural Questions | `nq` | BEIR | configurable |

The dataset registry is located in:

```text
app/infrastructure/datasets/dataset_registry.py
```

The dataset loader is located in:

```text
app/infrastructure/datasets/beir_loader.py
```

## Development Runs

For local testing, use a small document limit:

```bash
python scripts/build_indexes.py --dataset msmarco --model bm25 --max-docs 1000
```

The Streamlit interface also provides a `Max documents for demo` option.

## Evaluation Notes

Evaluation uses qrels. When a subset is selected with `--max-docs`, qrels are filtered so that only relevant documents inside the selected subset are used.

For the final report, always mention:

- Dataset name
- Number of indexed documents
- Number of evaluated queries
- Retrieval model
- Metric values

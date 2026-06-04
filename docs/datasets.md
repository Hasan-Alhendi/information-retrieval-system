# Datasets

This project uses BEIR-compatible Information Retrieval datasets.

Each dataset should contain:

- Documents
- Queries
- qrels relevance judgments

## Registered Datasets

| Dataset | Name in Code | Source | Usage |
|---|---|---|---|
| Quora | `quora` | BEIR | Primary report dataset; lighter for local experiments |
| Natural Questions | `nq` | BEIR | Primary report dataset |
| MS MARCO | `msmarco` | BEIR | Additional / backup dataset |

The primary datasets for the final report are:

```text
quora
nq
```

MS MARCO is kept in the project as an additional option because previous experiments were already executed on it.

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
python scripts/build_indexes.py --dataset quora --model bm25 --max-docs 1000
python scripts/build_indexes.py --dataset nq --model bm25 --max-docs 1000
```

The Streamlit interface also provides a `Max documents for demo` option.

## Evaluation Notes

Evaluation uses qrels. When a subset is selected with `--max-docs`, the loader selects a qrels-aware subset so development evaluation remains meaningful.

For the final report, always mention:

- Dataset name
- Number of indexed documents
- Number of evaluated queries
- Retrieval model
- Metric values

# Information Retrieval System

A clean, modular Information Retrieval system built for the IR Project 2026 requirements.

The project follows **Clean Architecture** and supports full-corpus indexing, multiple retrieval models, benchmark evaluation, FastAPI, and an interactive Streamlit interface.

## Official Datasets

The final project uses two datasets with different retrieval tasks:

- Quora (`quora`): duplicate-question retrieval.
- Webis-Touché 2020 v2 (`touche2020-v2`): argument retrieval.

Natural Questions is retained only as a legacy configuration and is not part of the final report datasets.

## Retrieval Models

- TF-IDF / Vector Space Model.
- BM25 with configurable `k1` and `b`.
- Embedding retrieval with `sentence-transformers/all-MiniLM-L6-v2` and FAISS.
- Hybrid Serial: BM25 candidates followed by dense reranking.
- Hybrid Parallel: TF-IDF, BM25, and Embedding fused with Reciprocal Rank Fusion.

## Main Features

- Dataset-specific preprocessing profiles.
- Streaming document loading.
- Disk-backed full-corpus BM25 and TF-IDF indexes.
- Incremental FAISS construction with checkpoints and resume.
- Query refinement.
- Document clustering.
- Evaluation using MAP@K, Recall@K, Precision@10, and nDCG@K.
- Search latency benchmarking.
- Full-system evaluation charts.
- FastAPI backend.
- Streamlit interface.

## Architecture

```text
app/
  domain/          Core models and interfaces
  application/     Use cases and application services
  infrastructure/  Datasets, retrieval, storage, vector stores, evaluation
  presentation/    FastAPI and Streamlit interfaces
  shared/          Constants, logging, and utilities
scripts/           Indexing, evaluation, benchmarking, and report charts
tests/             Unit tests
storage/           Local datasets, indexes, vector stores, and evaluation outputs
docs/              Architecture and project documentation
```

## Installation

Windows:

```bash
python -m venv .env
.env\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Run the API

```bash
uvicorn app.main:app --reload
```

Open:

```text
http://127.0.0.1:8000/health
http://127.0.0.1:8000/docs
```

## Run Streamlit

```bash
streamlit run app/presentation/streamlit/ui.py
```

Open:

```text
http://localhost:8501
```

The sidebar supports:

- Quora and Touché.
- Full-corpus mode or a development subset.
- All five retrieval models.
- Top K, BM25 parameters, embedding model, and query refinement.

## Validate a Dataset Before Full Indexing

```bash
python scripts/validate_full_dataset.py --dataset quora
python scripts/validate_full_dataset.py --dataset touche2020-v2
```

The validation checks document/query/qrel availability, samples real records, verifies positive qrels, reports free disk space, and prints the recommended smoke-test commands.

## Quora Full-Corpus Preparation

Run the lexical smoke test:

```bash
python scripts/smoke_test_full_index.py --dataset quora
```

Run the embedding smoke test:

```bash
python scripts/smoke_test_full_embedding.py --dataset quora --checkpoint-size 1000 --force
```

After both smoke tests succeed, build the complete lexical index:

```bash
python scripts/build_full_dataset.py --dataset quora --components lexical
```

Then build the complete embedding index:

```bash
python scripts/build_full_dataset.py --dataset quora --components embedding
```

Both commands resume automatically after interruption. Do not use `--force` when resuming.

Check both indexes together:

```bash
python scripts/full_dataset_status.py --dataset quora
```

## Touché Full-Corpus Indexes

Lexical index shared by BM25 and TF-IDF:

```bash
python scripts/build_indexes.py --dataset touche2020-v2 --model bm25 --batch-size 128
```

Full embedding index:

```bash
python scripts/build_indexes.py --dataset touche2020-v2 --model embedding --batch-size 16 --checkpoint-size 5000
```

Check progress:

```bash
python scripts/full_dataset_status.py --dataset touche2020-v2
```

## Search Examples

BM25 or TF-IDF:

```bash
python scripts/search_full.py --dataset touche2020-v2 --model bm25 --query "Should teachers get tenure?" --top-k 10
python scripts/search_full.py --dataset quora --model bm25 --query "How can I learn programming?" --top-k 10
```

Embedding:

```bash
python scripts/search_full_embedding.py --dataset touche2020-v2 --query "Should teachers get tenure?" --top-k 10
python scripts/search_full_embedding.py --dataset quora --query "How can I learn programming?" --top-k 10
```

The Streamlit interface and FastAPI `/search` endpoint use the corrected Hybrid Serial and Hybrid Parallel implementations.

## Full-System Evaluation

Evaluate all five models on the complete dataset:

```bash
python scripts/evaluate_full_system.py --dataset touche2020-v2 --max-queries 49 --top-k 10
python scripts/evaluate_full_system.py --dataset quora --max-queries 100 --top-k 10
```

Outputs are saved under:

```text
storage/evaluation/
```

Benchmark embedding and hybrid latency:

```bash
python scripts/benchmark_dense_hybrids.py --dataset touche2020-v2 --models embedding hybrid_serial hybrid_parallel --max-queries 10 --repeats 3 --top-k 10
```

Generate report charts:

```bash
python scripts/generate_evaluation_charts.py --dataset touche2020-v2
```

Charts are saved under:

```text
storage/evaluation/charts/<dataset-name>/
```

## API Search Example

Use `max_docs: null` for full-corpus mode:

```json
{
  "query": "Should teachers get tenure?",
  "dataset_name": "touche2020-v2",
  "model_name": "hybrid_serial",
  "top_k": 10,
  "max_docs": null,
  "bm25_k1": 1.5,
  "bm25_b": 0.75,
  "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
  "use_query_refinement": false
}
```

Send it to:

```text
POST /search
```

## Tests

```bash
pytest
```

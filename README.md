# Information Retrieval System

A clean, modular Information Retrieval system built for the IR Project 2026 requirements.

The project is designed using **Clean Architecture** and SOA-inspired services. It supports multiple retrieval models, evaluation pipelines, and an interactive Streamlit UI.

## Official Datasets

The system officially uses two BEIR datasets:

- Quora (`quora`)
- Natural Questions (`nq`)

## Features

- Dataset loading from BEIR-compatible datasets.
- Text preprocessing with spaCy.
- Retrieval models:
  - TF-IDF / Vector Space Model
  - BM25 with configurable `k1` and `b`
  - Embedding-based retrieval with SentenceTransformers
  - Hybrid Serial retrieval
  - Hybrid Parallel retrieval with reciprocal-rank-style fusion
- FAISS vector store for dense retrieval.
- Query refinement.
- Document clustering.
- Evaluation using MAP, Recall, Precision@10, and nDCG.
- FastAPI backend.
- Streamlit user interface.

## Architecture

```text
app/
  domain/          Core models and interfaces
  application/     Use cases and application services
  infrastructure/  Datasets, retrieval, storage, vector stores, evaluation
  presentation/    FastAPI and Streamlit interfaces
  shared/          Constants, logging, and utilities
scripts/           CLI scripts for datasets, indexing, and evaluation
tests/             Unit tests
docs/              Architecture and project documentation
storage/           Local datasets, indexes, vector stores, and evaluation outputs
```

## Preprocessing Decision

The main preprocessing pipeline uses **spaCy** because it provides a production-oriented NLP pipeline, reliable tokenization, lemmatization, and clear integration with Clean Architecture.

NLTK may be used only as a secondary helper if needed.

## Quick Start

### 1. Create and activate a virtual environment

Windows:

```bash
python -m venv .env
.env\Scripts\activate
```

macOS / Linux:

```bash
python -m venv .env
source .env/bin/activate
```

### 2. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 3. Run the API

```bash
uvicorn app.main:app --reload
```

Open:

```text
http://127.0.0.1:8000/health
http://127.0.0.1:8000/docs
```

### 4. Run the Streamlit UI

```bash
streamlit run app/presentation/streamlit/ui.py
```

Open:

```text
http://localhost:8501
```

## Build Indexes

Use a small `--max-docs` value for local demos.

```bash
python scripts/build_indexes.py --dataset quora --model bm25 --max-docs 1000
python scripts/build_indexes.py --dataset quora --model tfidf --max-docs 1000
python scripts/build_indexes.py --dataset quora --model embedding --max-docs 1000
python scripts/build_indexes.py --dataset quora --model hybrid_serial --max-docs 1000
python scripts/build_indexes.py --dataset quora --model hybrid_parallel --max-docs 1000

python scripts/build_indexes.py --dataset nq --model bm25 --max-docs 1000
python scripts/build_indexes.py --dataset nq --model tfidf --max-docs 1000
```

## Search API Example

```json
{
  "query": "how can I learn programming",
  "dataset_name": "quora",
  "model_name": "bm25",
  "top_k": 10,
  "max_docs": 1000,
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

## Evaluate Models

```bash
python scripts/evaluate_all.py --dataset quora --models bm25 tfidf --max-docs 1000 --max-queries 10
python scripts/evaluate_all.py --dataset quora --models hybrid_parallel --max-docs 1000 --max-queries 10
python scripts/evaluate_all.py --dataset nq --models bm25 tfidf --max-docs 1000 --max-queries 10
```

Evaluation output is saved under:

```text
storage/evaluation/
```

## Run Tests

```bash
pytest
```

## Current Status

Implemented:

- Clean Architecture skeleton
- FastAPI health, search, dataset, and evaluation routes
- Streamlit UI
- spaCy preprocessing
- BEIR dataset loader
- TF-IDF retriever
- BM25 retriever
- Embedding retriever
- FAISS vector store
- Hybrid Serial retriever
- Hybrid Parallel retriever
- Query refinement
- Document clustering
- Evaluation pipeline
- Basic tests

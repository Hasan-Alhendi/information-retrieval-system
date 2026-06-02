# Final Project Report

## 1. Project Overview

This project is a modular Information Retrieval system designed using Clean Architecture and SOA-inspired services.

The system supports multiple retrieval strategies, evaluation metrics, query refinement, document clustering, FastAPI endpoints, and a Streamlit user interface.

## 2. Main Objectives

- Build a clean and extensible IR system.
- Support multiple datasets with documents, queries, and qrels.
- Implement lexical, semantic, and hybrid retrieval.
- Evaluate models using standard IR metrics.
- Provide an interactive UI for search, evaluation, datasets, and clustering.

## 3. Architecture

The project is divided into four main layers:

| Layer | Responsibility |
|---|---|
| Domain | Core models and interfaces |
| Application | Use cases and orchestration services |
| Infrastructure | Technical implementations |
| Presentation | FastAPI and Streamlit interfaces |

See:

```text
docs/architecture.md
```

## 4. Retrieval Models

### TF-IDF

Sparse lexical retrieval based on term frequency and inverse document frequency.

### BM25

Probabilistic lexical retrieval with configurable parameters:

- `k1`
- `b`

The default values are:

```text
k1 = 1.5
b = 0.75
```

### Embedding Retrieval

Dense semantic retrieval using SentenceTransformers and FAISS.

### Hybrid Serial

BM25 retrieves candidate documents, then embedding retrieval is used for reranking.

### Hybrid Parallel

TF-IDF, BM25, and embedding retrieval run separately, then results are fused.

## 5. Query Refinement

The system supports optional query refinement with:

- Lightweight normalization
- Simple spelling correction
- Domain-specific expansion

This can be enabled from the Streamlit UI or API.

## 6. Document Clustering

The system supports document clustering using:

- SentenceTransformer embeddings
- KMeans clustering
- Cluster top terms
- Sample documents per cluster

## 7. Evaluation

The system evaluates retrieval using:

- MAP
- Recall
- Precision@10
- nDCG

Evaluation can be run from:

```bash
python scripts/evaluate_all.py --dataset msmarco --models bm25 tfidf --max-docs 1000 --max-queries 10
```

## 8. User Interface

The Streamlit UI provides:

- Search page
- Evaluation page
- Dataset page
- Clustering page

Run it with:

```bash
streamlit run app/presentation/streamlit/ui.py
```

## 9. API

The FastAPI backend provides:

- Health check
- Dataset listing
- Search
- Evaluation
- Indexing placeholder

Run it with:

```bash
uvicorn app.main:app --reload
```

## 10. Current Completion Status

Implemented:

- Clean Architecture structure
- FastAPI backend
- Streamlit UI
- BEIR dataset loader
- spaCy preprocessing
- TF-IDF retrieval
- BM25 retrieval
- Embedding retrieval
- FAISS vector store
- Hybrid Serial retrieval
- Hybrid Parallel retrieval
- Query refinement
- Evaluation pipeline
- Document clustering
- Documentation
- Basic tests

## 11. Future Improvements

- Add Learning-to-Rank.
- Add RAG answer generation.
- Add more advanced query expansion.
- Add more unit and integration tests.
- Add Docker deployment.

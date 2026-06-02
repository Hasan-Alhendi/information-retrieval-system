# Architecture

This project follows **Clean Architecture** principles with SOA-inspired service separation. The goal is to keep the Information Retrieval logic independent from frameworks such as FastAPI, Streamlit, scikit-learn, FAISS, and BEIR.

## High-Level Architecture

```text
User
  |
  v
Presentation Layer
  |-- FastAPI routes
  |-- Streamlit UI
  |
  v
Application Layer
  |-- Use cases
  |-- Orchestration services
  |
  v
Domain Layer
  |-- Core models
  |-- Interfaces
  |-- Exceptions
  ^
  |
Infrastructure Layer
  |-- Dataset loaders
  |-- Preprocessing
  |-- Retrieval models
  |-- Vector stores
  |-- Evaluation
  |-- Clustering
  |-- Storage
```

## Dependency Rule

Dependencies point inward:

```text
presentation -> application -> domain
infrastructure -> domain
```

The domain layer does not depend on FastAPI, Streamlit, scikit-learn, FAISS, SentenceTransformers, or BEIR. This makes the system easier to test, extend, and explain.

## Layers

### 1. Domain Layer

Located in:

```text
app/domain/
```

Responsibilities:

- Define core models such as `Document`, `Query`, `SearchResult`, and `EvaluationResult`.
- Define contracts such as `Retriever`, `Indexer`, `Preprocessor`, and `Evaluator`.
- Define domain-specific exceptions.

### 2. Application Layer

Located in:

```text
app/application/
```

Responsibilities:

- Coordinate use cases such as search, indexing, evaluation, query refinement, and clustering.
- Keep business-level orchestration separate from technical details.

### 3. Infrastructure Layer

Located in:

```text
app/infrastructure/
```

Responsibilities:

- Load BEIR datasets.
- Preprocess text using spaCy.
- Build and query TF-IDF, BM25, embedding, and hybrid retrievers.
- Store dense vectors using FAISS.
- Compute evaluation metrics.
- Perform document clustering.

### 4. Presentation Layer

Located in:

```text
app/presentation/
```

Responsibilities:

- Expose FastAPI endpoints.
- Provide a Streamlit user interface.
- Convert user input into application requests.
- Display ranked results, metrics, datasets, and clusters.

## SOA-Inspired Services

The project is organized around independent services:

| Service | Responsibility |
|---|---|
| Dataset Service | Load documents, queries, and qrels |
| Preprocessing Service | Normalize, tokenize, and lemmatize text |
| Indexing Service | Build and persist indexes |
| Retrieval Service | Run selected retrieval models |
| Ranking Service | Normalize and fuse scores |
| Vector Store Service | Store and search dense embeddings with FAISS |
| Evaluation Service | Compute MAP, Recall, Precision@10, and nDCG |
| Query Refinement Service | Correct and expand user queries |
| Clustering Service | Group documents using embeddings and KMeans |
| Presentation Service | FastAPI and Streamlit interfaces |

## Retrieval Models

The system supports:

| Model | Description |
|---|---|
| TF-IDF | Vector Space Model using sparse term weighting |
| BM25 | Probabilistic lexical retrieval with `k1` and `b` parameters |
| Embedding | Dense semantic retrieval using SentenceTransformers and FAISS |
| Hybrid Serial | BM25 candidate retrieval followed by embedding-based reranking |
| Hybrid Parallel | TF-IDF, BM25, and embedding retrieval combined with rank fusion |

## Data Flow: Search

```text
User Query
  -> Optional Query Refinement
  -> Selected Retriever
  -> Index / Vector Store
  -> Ranked SearchResult objects
  -> API or Streamlit UI
```

## Data Flow: Evaluation

```text
Dataset queries + qrels
  -> Retriever
  -> Ranked results
  -> Metrics calculator
  -> EvaluationResult
  -> CSV report / API / UI
```

## Why This Architecture?

This architecture was selected because it provides:

- Clear separation of responsibilities.
- Easier testing.
- Easier addition of new retrieval models.
- Clean mapping to IR Project 2026 requirements.
- A professional structure suitable for report and presentation.

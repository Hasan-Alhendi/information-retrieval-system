# Architecture

This project follows Clean Architecture principles and SOA-inspired service separation.

## Layers

```text
Presentation Layer
    FastAPI routes and Streamlit UI

Application Layer
    Use cases and orchestration services

Domain Layer
    Core models, interfaces, and domain exceptions

Infrastructure Layer
    Dataset loaders, preprocessors, retrievers, indexes, vector stores, and evaluators
```

## Dependency Rule

Dependencies point inward:

```text
presentation -> application -> domain
infrastructure -> domain
```

The domain layer does not depend on FastAPI, Streamlit, scikit-learn, FAISS, or dataset-specific libraries.

## Main Services

- Dataset Service
- Preprocessing Service
- Indexing Service
- Retrieval Service
- Ranking Service
- Evaluation Service
- Query Refinement Service
- Vector Store Service
- Clustering Service
- API Gateway / UI

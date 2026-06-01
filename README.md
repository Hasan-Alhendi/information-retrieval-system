# Information Retrieval System

A clean, modular Information Retrieval system built for the IR Project 2026 requirements.

The project is designed using **Clean Architecture** and SOA-inspired services. It will support multiple retrieval models, evaluation pipelines, and an interactive UI.

## Core Goals

- Support two large IR datasets with documents, queries, and qrels.
- Implement multiple retrieval models:
  - TF-IDF / Vector Space Model
  - BM25
  - Embedding-based retrieval
  - Hybrid Serial retrieval
  - Hybrid Parallel retrieval
- Provide query preprocessing and query refinement.
- Evaluate models using MAP, Recall, Precision@10, and nDCG.
- Provide FastAPI endpoints and a Streamlit user interface.
- Keep the codebase clean, testable, and maintainable.

## Architecture

```text
app/
  domain/          Core models and interfaces
  application/     Use cases and application services
  infrastructure/  External implementations: datasets, retrieval, storage, evaluation
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

## Current Status

Phase 1: Clean project skeleton and architecture setup.

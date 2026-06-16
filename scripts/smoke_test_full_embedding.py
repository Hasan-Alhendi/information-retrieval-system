"""Build and query a disposable real-data checkpointed FAISS index."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import VECTOR_STORES_DIR
from app.infrastructure.datasets.dataset_loader import DatasetLoader
from app.infrastructure.datasets.dataset_registry import SUPPORTED_DATASETS
from app.infrastructure.datasets.runtime_profiles import get_runtime_profile
from app.infrastructure.retrieval.full_embedding_index import FullEmbeddingIndexBuilder
from app.infrastructure.vector_store.incremental_faiss_store import (
    IncrementalFaissVectorStore,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test full embedding indexing.")
    parser.add_argument(
        "--dataset",
        default="touche2020-v2",
        choices=sorted(SUPPORTED_DATASETS),
    )
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--checkpoint-size", type=int, default=1000)
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    parser.add_argument("--query", default=None)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profile = get_runtime_profile(args.dataset)
    max_docs = args.max_docs or profile.smoke_documents
    batch_size = args.batch_size or profile.embedding_batch_size
    query = args.query or profile.sample_query

    if max_docs <= 0 or batch_size <= 0 or args.checkpoint_size <= 0:
        raise ValueError("Document, encode-batch, and checkpoint sizes must be positive.")

    safe_model_name = args.embedding_model.replace("/", "__")
    base_dir = (
        VECTOR_STORES_DIR
        / args.dataset
        / f"embedding_{safe_model_name}"
        / f"smoke_{max_docs}"
    )
    store = IncrementalFaissVectorStore(
        dataset_name=args.dataset,
        model_name=f"embedding_{safe_model_name}",
        base_dir=base_dir,
    )
    model = SentenceTransformer(args.embedding_model, device="cpu")

    def encode(texts: list[str]) -> np.ndarray:
        vectors = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return np.asarray(vectors, dtype="float32")

    builder = FullEmbeddingIndexBuilder(
        args.dataset,
        encoder=encode,
        vector_store=store,
        dataset_loader=DatasetLoader(),
        checkpoint_size=args.checkpoint_size,
        document_limit=max_docs,
    )

    build_start = time.perf_counter()
    document_count = builder.build(force=args.force)
    build_seconds = time.perf_counter() - build_start

    query_start = time.perf_counter()
    query_vector = encode([query])[0]
    matches = store.search(query_vector, top_k=args.top_k)
    records = store.load_records([row_index for row_index, _ in matches])
    query_ms = (time.perf_counter() - query_start) * 1000.0

    print(
        json.dumps(
            {
                **store.status(),
                "document_limit": document_count,
                "build_seconds": round(build_seconds, 3),
                "query_time_ms": round(query_ms, 3),
                "smoke_query": query,
                "batch_size": batch_size,
                "checkpoint_size": args.checkpoint_size,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    for rank, (row_index, score) in enumerate(matches, start=1):
        record = records[row_index]
        preview = record["text"][:300].replace("\n", " ")
        print(
            json.dumps(
                {
                    "rank": rank,
                    "doc_id": record["doc_id"],
                    "score": score,
                    "title": record["title"],
                    "text_preview": preview,
                },
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()

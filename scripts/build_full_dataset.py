"""Build or resume all full-corpus indexes for an official dataset."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.infrastructure.datasets.dataset_registry import SUPPORTED_DATASETS
from app.infrastructure.datasets.runtime_profiles import get_runtime_profile
from app.infrastructure.retrieval.bm25_retriever import BM25Retriever
from app.infrastructure.retrieval.disk_lexical_index import DiskLexicalIndex
from app.infrastructure.retrieval.embedding_retriever import EmbeddingRetriever


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build checkpointed full lexical and embedding indexes."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=sorted(SUPPORTED_DATASETS),
    )
    parser.add_argument(
        "--components",
        nargs="+",
        choices=("lexical", "embedding"),
        default=["lexical", "embedding"],
    )
    parser.add_argument("--lexical-batch-size", type=int, default=None)
    parser.add_argument("--embedding-batch-size", type=int, default=None)
    parser.add_argument("--checkpoint-size", type=int, default=None)
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete the selected existing index and rebuild it from zero.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profile = get_runtime_profile(args.dataset)
    lexical_batch_size = args.lexical_batch_size or profile.lexical_batch_size
    embedding_batch_size = args.embedding_batch_size or profile.embedding_batch_size
    checkpoint_size = args.checkpoint_size or profile.embedding_checkpoint_size

    disk = shutil.disk_usage(PROJECT_ROOT)
    print(
        json.dumps(
            {
                "dataset_name": args.dataset,
                "components": args.components,
                "free_gib_before_build": round(disk.free / (1024**3), 2),
                "lexical_batch_size": lexical_batch_size,
                "embedding_batch_size": embedding_batch_size,
                "checkpoint_size": checkpoint_size,
                "force": args.force,
            },
            indent=2,
        )
    )

    if "lexical" in args.components:
        print("\n=== Building/resuming full lexical index ===")
        BM25Retriever(
            max_docs=None,
            full_batch_size=lexical_batch_size,
        ).build(
            dataset_name=args.dataset,
            force=args.force,
            max_docs=None,
        )
        print(json.dumps(DiskLexicalIndex(args.dataset).status(), indent=2))

    if "embedding" in args.components:
        print("\n=== Building/resuming full embedding index ===")
        embedding = EmbeddingRetriever(
            embedding_model_name=args.embedding_model,
            max_docs=None,
            batch_size=embedding_batch_size,
            full_checkpoint_size=checkpoint_size,
        )
        embedding.build(
            dataset_name=args.dataset,
            force=args.force,
            max_docs=None,
        )
        print(json.dumps(embedding.full_status(args.dataset), indent=2))

    disk_after = shutil.disk_usage(PROJECT_ROOT)
    print(
        json.dumps(
            {
                "dataset_name": args.dataset,
                "completed_components": args.components,
                "free_gib_after_build": round(disk_after.free / (1024**3), 2),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

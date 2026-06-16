"""Show lexical and embedding full-index status for an official dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.infrastructure.datasets.dataset_registry import SUPPORTED_DATASETS
from app.infrastructure.retrieval.disk_lexical_index import DiskLexicalIndex
from app.infrastructure.retrieval.embedding_retriever import EmbeddingRetriever


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Show all full-index statuses.")
    parser.add_argument(
        "--dataset",
        required=True,
        choices=sorted(SUPPORTED_DATASETS),
    )
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lexical = DiskLexicalIndex(args.dataset).status()
    dense = EmbeddingRetriever(
        embedding_model_name=args.embedding_model,
        max_docs=None,
    ).full_status(args.dataset)
    output = {
        "dataset_name": args.dataset,
        "lexical": lexical,
        "embedding": dense,
        "ready_for_all_models": bool(
            lexical.get("finalized") and dense.get("finalized")
        ),
    }
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

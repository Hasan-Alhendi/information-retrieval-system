"""Show progress for a full checkpointed embedding index."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.infrastructure.retrieval.embedding_retriever import EmbeddingRetriever


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Show full embedding index status.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    retriever = EmbeddingRetriever(
        embedding_model_name=args.embedding_model,
        max_docs=None,
    )
    print(json.dumps(retriever.full_status(args.dataset), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

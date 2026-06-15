"""Search a finalized full-corpus embedding index."""

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
    parser = argparse.ArgumentParser(description="Search a full FAISS embedding index.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    parser.add_argument("--preview-chars", type=int, default=500)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    retriever = EmbeddingRetriever(
        embedding_model_name=args.embedding_model,
        max_docs=None,
        batch_size=args.batch_size,
    )
    results = retriever.search(
        query=args.query,
        dataset_name=args.dataset,
        top_k=args.top_k,
    )
    print(
        json.dumps(
            [
                {
                    "rank": result.rank,
                    "doc_id": result.doc_id,
                    "score": result.score,
                    "title": result.title,
                    "text_preview": _preview(result.text, args.preview_chars),
                    "metadata": result.metadata,
                }
                for result in results
            ],
            indent=2,
            ensure_ascii=False,
        )
    )


def _preview(text: str | None, limit: int) -> str | None:
    if text is None or len(text) <= limit:
        return text
    return text[: max(50, limit)].rstrip() + "..."


if __name__ == "__main__":
    main()

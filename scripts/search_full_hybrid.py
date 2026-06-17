"""Search full-corpus hybrid retrieval models."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.application.services.retriever_factory import create_retriever


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Search a finalized full-corpus hybrid retrieval model."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--model",
        required=True,
        choices=("hybrid_serial", "hybrid_parallel"),
    )
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--bm25-k1", type=float, default=1.5)
    parser.add_argument("--bm25-b", type=float, default=0.75)
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    parser.add_argument("--preview-chars", type=int, default=500)
    parser.add_argument(
        "--full-text",
        action="store_true",
        help="Print complete document text instead of a preview.",
    )
    return parser.parse_args()


def main() -> None:
    """Prepare and search the selected full-corpus hybrid retriever."""
    args = parse_args()
    if args.top_k <= 0:
        raise ValueError("top-k must be positive.")

    retriever = create_retriever(
        args.model,
        max_docs=None,
        bm25_k1=args.bm25_k1,
        bm25_b=args.bm25_b,
        embedding_model=args.embedding_model,
    )
    if hasattr(retriever, "prepare"):
        retriever.prepare(args.dataset)

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
                    "text": _display_text(
                        result.text,
                        full_text=args.full_text,
                        preview_chars=max(50, args.preview_chars),
                    ),
                    "metadata": result.metadata,
                }
                for result in results
            ],
            indent=2,
            ensure_ascii=False,
        )
    )


def _display_text(
    text: str | None,
    *,
    full_text: bool,
    preview_chars: int,
) -> str | None:
    if text is None or full_text or len(text) <= preview_chars:
        return text
    return text[:preview_chars].rstrip() + "..."


if __name__ == "__main__":
    main()

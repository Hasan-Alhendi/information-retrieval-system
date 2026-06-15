"""Search a finalized full-corpus BM25 or TF-IDF index."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.infrastructure.retrieval.bm25_retriever import BM25Retriever
from app.infrastructure.retrieval.tfidf_retriever import TFIDFRetriever


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search a full disk-backed lexical index.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", choices=("bm25", "tfidf"), required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--bm25-k1", type=float, default=1.5)
    parser.add_argument("--bm25-b", type=float, default=0.75)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.model == "bm25":
        retriever = BM25Retriever(k1=args.bm25_k1, b=args.bm25_b, max_docs=None)
    else:
        retriever = TFIDFRetriever(max_docs=None)

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
                    "text": result.text,
                    "metadata": result.metadata,
                }
                for result in results
            ],
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

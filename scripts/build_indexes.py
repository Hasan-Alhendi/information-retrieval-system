"""Build retrieval indexes."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.infrastructure.retrieval.bm25_retriever import BM25Retriever
from app.infrastructure.retrieval.embedding_retriever import EmbeddingRetriever
from app.infrastructure.retrieval.hybrid_parallel import HybridParallelRetriever
from app.infrastructure.retrieval.hybrid_serial import HybridSerialRetriever
from app.infrastructure.retrieval.tfidf_retriever import TFIDFRetriever


RETRIEVER_BUILDERS = {
    "bm25": BM25Retriever,
    "embedding": EmbeddingRetriever,
    "hybrid_parallel": HybridParallelRetriever,
    "hybrid_serial": HybridSerialRetriever,
    "tfidf": TFIDFRetriever,
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Build retrieval indexes.")
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset name, e.g. quora or touche2020-v2.",
    )
    parser.add_argument(
        "--model",
        default="tfidf",
        choices=sorted(RETRIEVER_BUILDERS),
        help="Retrieval model index to build.",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help=(
            "Development document limit. Omit it for checkpointed full-corpus "
            "lexical or embedding indexes."
        ),
    )
    parser.add_argument("--force", action="store_true", help="Rebuild even if index exists.")
    parser.add_argument("--bm25-k1", type=float, default=1.5, help="BM25 k1 parameter.")
    parser.add_argument("--bm25-b", type=float, default=0.75, help="BM25 b parameter.")
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer model name for embedding indexes.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help=(
            "spaCy batch size for lexical indexes or SentenceTransformer encode "
            "batch size for embedding indexes. Use 16 on an 8 GB CPU laptop."
        ),
    )
    parser.add_argument(
        "--checkpoint-size",
        type=int,
        default=5000,
        help=(
            "Documents encoded before each durable full-embedding checkpoint. "
            "This does not change the neural-model batch size."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Build retrieval indexes."""
    args = parse_args()
    if args.model == "bm25":
        retriever = BM25Retriever(
            k1=args.bm25_k1,
            b=args.bm25_b,
            max_docs=args.max_docs,
            full_batch_size=args.batch_size,
        )
    elif args.model == "tfidf":
        retriever = TFIDFRetriever(
            max_docs=args.max_docs,
            full_batch_size=args.batch_size,
        )
    elif args.model == "embedding":
        retriever = EmbeddingRetriever(
            embedding_model_name=args.embedding_model,
            max_docs=args.max_docs,
            batch_size=args.batch_size,
            full_checkpoint_size=args.checkpoint_size,
        )
    elif args.model in {"hybrid_serial", "hybrid_parallel"}:
        embedding = EmbeddingRetriever(
            embedding_model_name=args.embedding_model,
            max_docs=args.max_docs,
            batch_size=args.batch_size,
            full_checkpoint_size=args.checkpoint_size,
        )
        bm25 = BM25Retriever(
            k1=args.bm25_k1,
            b=args.bm25_b,
            max_docs=args.max_docs,
            full_batch_size=args.batch_size,
        )
        if args.model == "hybrid_serial":
            retriever = HybridSerialRetriever(
                bm25_retriever=bm25,
                embedding_retriever=embedding,
                max_docs=args.max_docs,
            )
        else:
            retriever = HybridParallelRetriever(
                tfidf_retriever=TFIDFRetriever(
                    max_docs=args.max_docs,
                    full_batch_size=args.batch_size,
                ),
                bm25_retriever=bm25,
                embedding_retriever=embedding,
                max_docs=args.max_docs,
            )
    else:
        retriever = RETRIEVER_BUILDERS[args.model](max_docs=args.max_docs)

    retriever.build(dataset_name=args.dataset, force=args.force, max_docs=args.max_docs)
    suffix = f" with max_docs={args.max_docs}" if args.max_docs else " (full corpus)"
    print(f"Built {args.model} index for dataset: {args.dataset}{suffix}")


if __name__ == "__main__":
    main()

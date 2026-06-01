"""Build retrieval indexes."""

import argparse

from app.infrastructure.retrieval.bm25_retriever import BM25Retriever
from app.infrastructure.retrieval.embedding_retriever import EmbeddingRetriever
from app.infrastructure.retrieval.tfidf_retriever import TFIDFRetriever


RETRIEVER_BUILDERS = {
    "bm25": BM25Retriever,
    "embedding": EmbeddingRetriever,
    "tfidf": TFIDFRetriever,
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Build retrieval indexes.")
    parser.add_argument("--dataset", required=True, help="Dataset name, e.g. msmarco or nq.")
    parser.add_argument(
        "--model",
        default="tfidf",
        choices=sorted(RETRIEVER_BUILDERS),
        help="Retrieval model index to build.",
    )
    parser.add_argument("--max-docs", type=int, default=None, help="Development document limit.")
    parser.add_argument("--force", action="store_true", help="Rebuild even if index exists.")
    parser.add_argument("--bm25-k1", type=float, default=1.5, help="BM25 k1 parameter.")
    parser.add_argument("--bm25-b", type=float, default=0.75, help="BM25 b parameter.")
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer model name for embedding indexes.",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Embedding batch size.")
    return parser.parse_args()


def main() -> None:
    """Build retrieval indexes."""
    args = parse_args()
    if args.model == "bm25":
        retriever = BM25Retriever(k1=args.bm25_k1, b=args.bm25_b, max_docs=args.max_docs)
    elif args.model == "embedding":
        retriever = EmbeddingRetriever(
            embedding_model_name=args.embedding_model,
            max_docs=args.max_docs,
            batch_size=args.batch_size,
        )
    else:
        retriever = RETRIEVER_BUILDERS[args.model](max_docs=args.max_docs)

    retriever.build(dataset_name=args.dataset, force=args.force, max_docs=args.max_docs)
    suffix = f" with max_docs={args.max_docs}" if args.max_docs else ""
    print(f"Built {args.model} index for dataset: {args.dataset}{suffix}")


if __name__ == "__main__":
    main()

"""Build retrieval indexes."""

import argparse

from app.infrastructure.retrieval.tfidf_retriever import TFIDFRetriever


RETRIEVER_BUILDERS = {
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
    parser.add_argument("--force", action="store_true", help="Rebuild even if index exists.")
    return parser.parse_args()


def main() -> None:
    """Build retrieval indexes."""
    args = parse_args()
    retriever = RETRIEVER_BUILDERS[args.model]()
    retriever.build(dataset_name=args.dataset, force=args.force)
    print(f"Built {args.model} index for dataset: {args.dataset}")


if __name__ == "__main__":
    main()

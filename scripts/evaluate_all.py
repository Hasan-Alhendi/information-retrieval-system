"""Evaluate retrieval models."""

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import EVALUATION_DIR, SUPPORTED_RETRIEVAL_MODELS
from app.infrastructure.evaluation.evaluator import RetrievalEvaluator


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate retrieval models.")
    parser.add_argument("--dataset", required=True, help="Dataset name, e.g. msmarco or nq.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(SUPPORTED_RETRIEVAL_MODELS),
        choices=list(SUPPORTED_RETRIEVAL_MODELS),
        help="Retrieval models to evaluate.",
    )
    parser.add_argument("--max-docs", type=int, default=None, help="Development document limit.")
    parser.add_argument("--max-queries", type=int, default=25, help="Limit number of queries.")
    parser.add_argument("--top-k", type=int, default=10, help="Retrieved results per query.")
    parser.add_argument("--bm25-k1", type=float, default=1.5, help="BM25 k1 parameter.")
    parser.add_argument("--bm25-b", type=float, default=0.75, help="BM25 b parameter.")
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer model name for embedding-based evaluation.",
    )
    parser.add_argument(
        "--use-query-refinement",
        action="store_true",
        help="Evaluate after applying query refinement.",
    )
    return parser.parse_args()


def main() -> None:
    """Evaluate selected retrieval models."""
    args = parse_args()
    evaluator = RetrievalEvaluator(
        max_docs=args.max_docs,
        top_k=args.top_k,
        max_queries=args.max_queries,
        bm25_k1=args.bm25_k1,
        bm25_b=args.bm25_b,
        embedding_model=args.embedding_model,
        use_query_refinement=args.use_query_refinement,
    )

    results = []
    mode = "with_query_refinement" if args.use_query_refinement else "baseline"
    for model_name in args.models:
        result = evaluator.evaluate(dataset_name=args.dataset, model_name=model_name)
        results.append(result)
        print(
            f"{model_name} ({mode}): "
            f"MAP={result.map_score:.4f}, "
            f"Recall={result.recall:.4f}, "
            f"P@10={result.precision_at_10:.4f}, "
            f"nDCG={result.ndcg:.4f}, "
            f"queries={result.evaluated_queries}"
        )

    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"dev_{args.max_docs}" if args.max_docs else "full"
    output_path = EVALUATION_DIR / f"{args.dataset}_{suffix}_{mode}_evaluation.csv"
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "dataset_name",
                "model_name",
                "mode",
                "map_score",
                "recall",
                "precision_at_10",
                "ndcg",
                "evaluated_queries",
            ],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "dataset_name": result.dataset_name,
                    "model_name": result.model_name,
                    "mode": mode,
                    "map_score": result.map_score,
                    "recall": result.recall,
                    "precision_at_10": result.precision_at_10,
                    "ndcg": result.ndcg,
                    "evaluated_queries": result.evaluated_queries,
                }
            )
    print(f"Saved evaluation results to: {output_path}")


if __name__ == "__main__":
    main()

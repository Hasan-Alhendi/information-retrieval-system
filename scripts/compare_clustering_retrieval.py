"""Compare embedding retrieval before and after cluster-aware reranking."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import EVALUATION_DIR
from app.infrastructure.evaluation.evaluator_v2 import RetrievalEvaluatorV2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the same embedding search before and after cluster-aware reranking."
        )
    )
    parser.add_argument("--dataset", default="quora")
    parser.add_argument("--max-docs", type=int, default=1000)
    parser.add_argument("--max-queries", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--clusters", type=int, default=5)
    parser.add_argument("--cluster-weight", type=float, default=0.2)
    parser.add_argument("--candidate-k", type=int, default=100)
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_docs < 2:
        raise ValueError("max-docs must be at least 2.")
    if args.clusters < 2 or args.clusters >= args.max_docs:
        raise ValueError("clusters must be at least 2 and smaller than max-docs.")
    if not 0.0 <= args.cluster_weight <= 1.0:
        raise ValueError("cluster-weight must be between 0 and 1.")

    evaluator = RetrievalEvaluatorV2(
        max_docs=args.max_docs,
        top_k=args.top_k,
        max_queries=args.max_queries,
        embedding_model=args.embedding_model,
        cluster_count=args.clusters,
        cluster_weight=args.cluster_weight,
        cluster_candidate_k=args.candidate_k,
    )

    before = evaluator.evaluate(args.dataset, "embedding")
    after = evaluator.evaluate(args.dataset, "embedding_clustered")

    before_row = _row(
        condition="before_clustering",
        result=before,
        args=args,
    )
    after_row = _row(
        condition="after_clustering",
        result=after,
        args=args,
    )
    delta_row = {
        "condition": "delta_after_minus_before",
        "model_name": "difference",
        "dataset_name": args.dataset,
        "index_scope": f"development_{args.max_docs}",
        "top_k": args.top_k,
        "number_of_clusters": args.clusters,
        "cluster_weight": args.cluster_weight,
        "candidate_k": args.candidate_k,
        "map_at_k": after.map_score - before.map_score,
        "recall_at_k": after.recall - before.recall,
        "precision_at_10": after.precision_at_10 - before.precision_at_10,
        "ndcg_at_k": after.ndcg - before.ndcg,
        "average_query_time_ms": (
            after.average_query_time_ms - before.average_query_time_ms
        ),
        "evaluated_queries": before.evaluated_queries,
    }

    rows = [before_row, after_row, delta_row]
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    output_path = EVALUATION_DIR / (
        f"{args.dataset}_clustering_retrieval_comparison_dev_{args.max_docs}.csv"
    )
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(before_row))
        writer.writeheader()
        writer.writerows(rows)

    _print_result("Before clustering", before, args.top_k)
    _print_result("After clustering", after, args.top_k)
    print(
        "Delta (after - before): "
        f"MAP={delta_row['map_at_k']:+.4f}, "
        f"Recall={delta_row['recall_at_k']:+.4f}, "
        f"P@10={delta_row['precision_at_10']:+.4f}, "
        f"nDCG={delta_row['ndcg_at_k']:+.4f}, "
        f"latency={delta_row['average_query_time_ms']:+.2f} ms"
    )
    print(f"Saved comparison to: {output_path}")


def _row(*, condition: str, result, args: argparse.Namespace) -> dict[str, object]:
    return {
        "condition": condition,
        "model_name": result.model_name,
        "dataset_name": result.dataset_name,
        "index_scope": f"development_{args.max_docs}",
        "top_k": args.top_k,
        "number_of_clusters": args.clusters,
        "cluster_weight": args.cluster_weight,
        "candidate_k": args.candidate_k,
        "map_at_k": result.map_score,
        "recall_at_k": result.recall,
        "precision_at_10": result.precision_at_10,
        "ndcg_at_k": result.ndcg,
        "average_query_time_ms": result.average_query_time_ms,
        "evaluated_queries": result.evaluated_queries,
    }


def _print_result(label: str, result, top_k: int) -> None:
    print(
        f"{label}: "
        f"MAP@{top_k}={result.map_score:.4f}, "
        f"Recall@{top_k}={result.recall:.4f}, "
        f"P@10={result.precision_at_10:.4f}, "
        f"nDCG@{top_k}={result.ndcg:.4f}, "
        f"avg_time={result.average_query_time_ms:.2f} ms, "
        f"queries={result.evaluated_queries}"
    )


if __name__ == "__main__":
    main()

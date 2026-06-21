"""Compare embedding retrieval with guided semantic category reranking."""

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
        description="Compare normal embedding retrieval with guided category reranking."
    )
    parser.add_argument("--dataset", default="quora")
    parser.add_argument("--max-docs", type=int, default=1000)
    parser.add_argument("--max-queries", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--category-weight", type=float, default=0.25)
    parser.add_argument("--top-categories", type=int, default=3)
    parser.add_argument("--candidate-k", type=int, default=100)
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _validate_args(args)

    evaluator = RetrievalEvaluatorV2(
        max_docs=args.max_docs,
        top_k=args.top_k,
        max_queries=args.max_queries,
        embedding_model=args.embedding_model,
        category_weight=args.category_weight,
        category_candidate_k=args.candidate_k,
        top_categories=args.top_categories,
    )

    baseline = evaluator.evaluate(args.dataset, "embedding")
    guided = evaluator.evaluate(args.dataset, "embedding_guided_categories")

    baseline_row = _row("embedding_baseline", baseline, args)
    guided_row = _row("guided_categories", guided, args)
    guided_delta = _delta_row(
        "guided_minus_baseline",
        guided,
        baseline,
        args,
    )

    rows = [baseline_row, guided_row, guided_delta]
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    output_path = EVALUATION_DIR / (
        f"{args.dataset}_guided_categories_dev_{args.max_docs}.csv"
    )
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(baseline_row))
        writer.writeheader()
        writer.writerows(rows)

    _print_result("Embedding baseline", baseline, args.top_k)
    _print_result("Guided Categories", guided, args.top_k)
    _print_delta("Guided - baseline", guided_delta)
    print(f"Saved comparison to: {output_path}")


def _row(condition: str, result, args: argparse.Namespace) -> dict[str, object]:
    return {
        "condition": condition,
        "model_name": result.model_name,
        "dataset_name": result.dataset_name,
        "index_scope": f"development_{args.max_docs}",
        "top_k": args.top_k,
        "category_weight": args.category_weight,
        "top_categories": args.top_categories,
        "candidate_k": args.candidate_k,
        "map_at_k": result.map_score,
        "recall_at_k": result.recall,
        "precision_at_10": result.precision_at_10,
        "ndcg_at_k": result.ndcg,
        "average_query_time_ms": result.average_query_time_ms,
        "evaluated_queries": result.evaluated_queries,
    }


def _delta_row(
    condition: str,
    result,
    baseline,
    args: argparse.Namespace,
) -> dict[str, object]:
    row = _row(condition, result, args)
    row["model_name"] = "difference"
    row["map_at_k"] = result.map_score - baseline.map_score
    row["recall_at_k"] = result.recall - baseline.recall
    row["precision_at_10"] = result.precision_at_10 - baseline.precision_at_10
    row["ndcg_at_k"] = result.ndcg - baseline.ndcg
    row["average_query_time_ms"] = (
        result.average_query_time_ms - baseline.average_query_time_ms
    )
    return row


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


def _print_delta(label: str, row: dict[str, object]) -> None:
    print(
        f"{label}: "
        f"MAP={float(row['map_at_k']):+.4f}, "
        f"Recall={float(row['recall_at_k']):+.4f}, "
        f"P@10={float(row['precision_at_10']):+.4f}, "
        f"nDCG={float(row['ndcg_at_k']):+.4f}, "
        f"latency={float(row['average_query_time_ms']):+.2f} ms"
    )


def _validate_args(args: argparse.Namespace) -> None:
    if args.max_docs < 2:
        raise ValueError("max-docs must be at least 2.")
    if args.top_categories < 1:
        raise ValueError("top-categories must be positive.")
    if args.candidate_k < args.top_k:
        raise ValueError("candidate-k must be greater than or equal to top-k.")
    if not 0.0 <= args.category_weight <= 1.0:
        raise ValueError("category-weight must be between 0 and 1.")


if __name__ == "__main__":
    main()

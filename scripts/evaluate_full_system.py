"""Evaluate all retrieval models and save one unified full-corpus CSV."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.application.services.retriever_factory import SUPPORTED_MODELS
from app.config import EVALUATION_DIR
from app.infrastructure.evaluation.evaluator_v2 import RetrievalEvaluatorV2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the complete retrieval system.")
    parser.add_argument("--dataset", default="touche2020-v2")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=SUPPORTED_MODELS,
        default=list(SUPPORTED_MODELS),
    )
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--bm25-k1", type=float, default=1.5)
    parser.add_argument("--bm25-b", type=float, default=0.75)
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluator = RetrievalEvaluatorV2(
        max_docs=None,
        top_k=args.top_k,
        max_queries=args.max_queries,
        bm25_k1=args.bm25_k1,
        bm25_b=args.bm25_b,
        embedding_model=args.embedding_model,
    )

    rows: list[dict[str, object]] = []
    for model_name in args.models:
        result = evaluator.evaluate(args.dataset, model_name)
        row = {
            "dataset_name": result.dataset_name,
            "model_name": result.model_name,
            "index_scope": "full",
            "top_k": args.top_k,
            "map_at_k": result.map_score,
            "recall_at_k": result.recall,
            "precision_at_10": result.precision_at_10,
            "ndcg_at_k": result.ndcg,
            "evaluated_queries": result.evaluated_queries,
        }
        rows.append(row)
        print(
            f"{model_name}: MAP@{args.top_k}={result.map_score:.4f}, "
            f"Recall@{args.top_k}={result.recall:.4f}, "
            f"P@10={result.precision_at_10:.4f}, "
            f"nDCG@{args.top_k}={result.ndcg:.4f}, "
            f"queries={result.evaluated_queries}"
        )

    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    output_path = EVALUATION_DIR / f"{args.dataset}_full_system_evaluation.csv"
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved unified evaluation results to: {output_path}")


if __name__ == "__main__":
    main()

"""Compare full-corpus lexical quality and latency across pruning thresholds."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import EVALUATION_DIR
from app.infrastructure.datasets.dataset_loader import DatasetLoader
from app.infrastructure.evaluation.metrics import (
    average_precision,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from app.infrastructure.retrieval.optimized_disk_lexical_index import (
    OptimizedDiskLexicalIndex,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare BM25 and TF-IDF on a finalized full index using multiple "
            "common-term pruning thresholds."
        )
    )
    parser.add_argument("--dataset", default="touche2020-v2")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=("bm25", "tfidf"),
        default=["bm25", "tfidf"],
    )
    parser.add_argument(
        "--max-df-ratios",
        nargs="+",
        type=float,
        default=[1.0, 0.5, 0.3],
        help=(
            "Corpus-frequency thresholds to compare. 1.0 keeps every term; "
            "smaller values may prune very common terms."
        ),
    )
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--bm25-k1", type=float, default=1.5)
    parser.add_argument("--bm25-b", type=float, default=0.75)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _validate_args(args)

    loader = DatasetLoader()
    queries, qrels = loader.load_queries_qrels(args.dataset)
    query_items = [
        (query_id, queries[query_id])
        for query_id in qrels
        if query_id in queries
    ]
    if args.max_queries is not None:
        query_items = query_items[: args.max_queries]
    if not query_items:
        raise RuntimeError("No evaluation queries were found.")

    rows: list[dict[str, Any]] = []
    for ratio in args.max_df_ratios:
        index = OptimizedDiskLexicalIndex(args.dataset, max_df_ratio=ratio)
        if not index.exists():
            raise RuntimeError(
                f"Full index for '{args.dataset}' is not ready. Build it first."
            )

        for model_name in args.models:
            metrics = _evaluate_configuration(
                index=index,
                model_name=model_name,
                query_items=query_items,
                qrels=qrels,
                top_k=args.top_k,
                bm25_k1=args.bm25_k1,
                bm25_b=args.bm25_b,
            )
            row = {
                "dataset_name": args.dataset,
                "model_name": model_name,
                "max_df_ratio": ratio,
                "top_k": args.top_k,
                "evaluated_queries": len(query_items),
                **metrics,
            }
            rows.append(row)
            print(
                f"{model_name} max_df_ratio={ratio:.2f}: "
                f"MAP@{args.top_k}={metrics['map_score']:.4f}, "
                f"Recall@{args.top_k}={metrics['recall']:.4f}, "
                f"P@10={metrics['precision_at_10']:.4f}, "
                f"nDCG@{args.top_k}={metrics['ndcg']:.4f}, "
                f"avg_latency={metrics['average_latency_ms']:.2f} ms, "
                f"median_latency={metrics['median_latency_ms']:.2f} ms"
            )

    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = EVALUATION_DIR / f"{args.dataset}_pruning_threshold_comparison.csv"
    json_path = EVALUATION_DIR / f"{args.dataset}_pruning_threshold_comparison.json"

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(rows, file, indent=2, ensure_ascii=False)

    print(f"Saved comparison CSV to: {csv_path}")
    print(f"Saved comparison JSON to: {json_path}")


def _evaluate_configuration(
    *,
    index: OptimizedDiskLexicalIndex,
    model_name: str,
    query_items: list[tuple[str, str]],
    qrels: dict[str, dict[str, int]],
    top_k: int,
    bm25_k1: float,
    bm25_b: float,
) -> dict[str, float]:
    map_scores: list[float] = []
    recall_scores: list[float] = []
    precision_scores: list[float] = []
    ndcg_scores: list[float] = []
    latencies: list[float] = []

    for query_id, query_text in query_items:
        relevance_scores = {
            doc_id: float(score)
            for doc_id, score in qrels[query_id].items()
        }
        relevant_docs = {
            doc_id
            for doc_id, score in relevance_scores.items()
            if score > 0
        }

        start = time.perf_counter()
        if model_name == "bm25":
            results = index.search_bm25(
                query_text,
                top_k=top_k,
                k1=bm25_k1,
                b=bm25_b,
            )
        else:
            results = index.search_tfidf(query_text, top_k=top_k)
        wall_latency_ms = (time.perf_counter() - start) * 1000.0

        retrieved_doc_ids = [result.doc_id for result in results]
        internal_latency_ms = (
            float(results[0].metadata.get("query_time_ms", wall_latency_ms))
            if results
            else wall_latency_ms
        )
        latencies.append(internal_latency_ms)
        map_scores.append(average_precision(retrieved_doc_ids, relevant_docs))
        recall_scores.append(recall_at_k(retrieved_doc_ids, relevant_docs, k=top_k))
        precision_scores.append(precision_at_k(retrieved_doc_ids, relevant_docs, k=10))
        ndcg_scores.append(ndcg_at_k(retrieved_doc_ids, relevance_scores, k=top_k))

    return {
        "map_score": _mean(map_scores),
        "recall": _mean(recall_scores),
        "precision_at_10": _mean(precision_scores),
        "ndcg": _mean(ndcg_scores),
        "average_latency_ms": statistics.fmean(latencies),
        "median_latency_ms": statistics.median(latencies),
        "minimum_latency_ms": min(latencies),
        "maximum_latency_ms": max(latencies),
    }


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _validate_args(args: argparse.Namespace) -> None:
    if args.top_k <= 0:
        raise ValueError("top-k must be positive.")
    if args.max_queries is not None and args.max_queries <= 0:
        raise ValueError("max-queries must be positive when supplied.")
    invalid = [ratio for ratio in args.max_df_ratios if ratio <= 0 or ratio > 1]
    if invalid:
        raise ValueError(
            "Each max-df-ratio must be in the range (0, 1]. "
            f"Invalid values: {invalid}"
        )


if __name__ == "__main__":
    main()

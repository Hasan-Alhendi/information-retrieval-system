"""Benchmark full-corpus BM25 and TF-IDF latency inside one Python process."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import EVALUATION_DIR
from app.infrastructure.datasets.dataset_loader import DatasetLoader
from app.infrastructure.retrieval.optimized_disk_lexical_index import (
    OptimizedDiskLexicalIndex,
)

SearchFunction = Callable[[str], list[Any]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark finalized full-corpus lexical indexes."
    )
    parser.add_argument("--dataset", default="touche2020-v2")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=("bm25", "tfidf"),
        default=["bm25", "tfidf"],
    )
    parser.add_argument("--max-queries", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--bm25-k1", type=float, default=1.5)
    parser.add_argument("--bm25-b", type=float, default=0.75)
    parser.add_argument(
        "--max-df-ratio",
        type=float,
        default=0.30,
        help="Prune query terms appearing in more than this corpus fraction.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_queries <= 0 or args.repeats <= 0 or args.warmups < 0:
        raise ValueError("max-queries and repeats must be positive; warmups cannot be negative.")

    index = OptimizedDiskLexicalIndex(
        args.dataset,
        max_df_ratio=args.max_df_ratio,
    )
    if not index.exists():
        raise RuntimeError(
            f"Full index for '{args.dataset}' is not ready. Build it before benchmarking."
        )

    queries, qrels = DatasetLoader().load_queries_qrels(args.dataset)
    query_items = [
        (query_id, queries[query_id])
        for query_id in qrels
        if query_id in queries
    ][: args.max_queries]
    if not query_items:
        raise RuntimeError("No benchmark queries were found.")

    records: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []

    for model_name in args.models:
        search = _search_function(index, model_name, args)

        for warmup_index in range(args.warmups):
            _, warmup_query = query_items[warmup_index % len(query_items)]
            search(warmup_query)

        model_latencies: list[float] = []
        model_wall_latencies: list[float] = []
        for query_id, query_text in query_items:
            for repetition in range(1, args.repeats + 1):
                wall_start = time.perf_counter()
                results = search(query_text)
                wall_ms = (time.perf_counter() - wall_start) * 1000.0
                internal_ms = (
                    float(results[0].metadata.get("query_time_ms", wall_ms))
                    if results
                    else wall_ms
                )
                model_latencies.append(internal_ms)
                model_wall_latencies.append(wall_ms)
                metadata = results[0].metadata if results else {}
                records.append(
                    {
                        "dataset_name": args.dataset,
                        "model_name": model_name,
                        "query_id": query_id,
                        "query_text": query_text,
                        "repetition": repetition,
                        "internal_latency_ms": round(internal_ms, 3),
                        "wall_latency_ms": round(wall_ms, 3),
                        "results_count": len(results),
                        "query_terms_used": " ".join(
                            metadata.get("query_terms_used", [])
                        ),
                        "pruned_terms": " ".join(metadata.get("pruned_terms", [])),
                    }
                )
                print(
                    f"{model_name} query={query_id} repetition={repetition}: "
                    f"internal={internal_ms:.2f} ms, wall={wall_ms:.2f} ms"
                )

        summary = _summarize(
            dataset_name=args.dataset,
            model_name=model_name,
            latencies=model_latencies,
            wall_latencies=model_wall_latencies,
            query_count=len(query_items),
            repeats=args.repeats,
            max_df_ratio=args.max_df_ratio,
        )
        summaries.append(summary)
        print(json.dumps(summary, indent=2, ensure_ascii=False))

    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = EVALUATION_DIR / f"{args.dataset}_full_search_latency.csv"
    json_path = EVALUATION_DIR / f"{args.dataset}_full_search_latency_summary.json"

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(summaries, file, indent=2, ensure_ascii=False)

    print(f"Saved detailed latency records to: {csv_path}")
    print(f"Saved latency summary to: {json_path}")


def _search_function(
    index: OptimizedDiskLexicalIndex,
    model_name: str,
    args: argparse.Namespace,
) -> SearchFunction:
    if model_name == "bm25":
        return lambda query: index.search_bm25(
            query,
            top_k=args.top_k,
            k1=args.bm25_k1,
            b=args.bm25_b,
        )
    return lambda query: index.search_tfidf(query, top_k=args.top_k)


def _summarize(
    *,
    dataset_name: str,
    model_name: str,
    latencies: list[float],
    wall_latencies: list[float],
    query_count: int,
    repeats: int,
    max_df_ratio: float,
) -> dict[str, Any]:
    ordered = sorted(latencies)
    return {
        "dataset_name": dataset_name,
        "model_name": model_name,
        "query_count": query_count,
        "repeats_per_query": repeats,
        "measurements": len(latencies),
        "max_df_ratio": max_df_ratio,
        "average_ms": round(statistics.fmean(latencies), 3),
        "median_p50_ms": round(statistics.median(latencies), 3),
        "p95_ms": round(_percentile(ordered, 0.95), 3),
        "minimum_ms": round(min(latencies), 3),
        "maximum_ms": round(max(latencies), 3),
        "average_wall_ms": round(statistics.fmean(wall_latencies), 3),
    }


def _percentile(ordered_values: list[float], fraction: float) -> float:
    if not ordered_values:
        return 0.0
    index = max(0, math.ceil(fraction * len(ordered_values)) - 1)
    return ordered_values[index]


if __name__ == "__main__":
    main()

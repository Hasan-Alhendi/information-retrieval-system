"""Benchmark warmed embedding and corrected hybrid retrieval in one process."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import EVALUATION_DIR
from app.infrastructure.datasets.dataset_loader import DatasetLoader
from app.infrastructure.retrieval.bm25_retriever import BM25Retriever
from app.infrastructure.retrieval.embedding_retriever import EmbeddingRetriever
from app.infrastructure.retrieval.hybrid_serial_v2 import HybridSerialRetrieverV2
from app.infrastructure.retrieval.rrf_retriever import ReciprocalRankFusionRetriever
from app.infrastructure.retrieval.tfidf_retriever import TFIDFRetriever


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark dense and hybrid search.")
    parser.add_argument("--dataset", default="touche2020-v2")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=("embedding", "hybrid_serial", "hybrid_parallel"),
        default=["embedding", "hybrid_serial", "hybrid_parallel"],
    )
    parser.add_argument("--max-queries", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    queries, qrels = DatasetLoader().load_queries_qrels(args.dataset)
    query_items = [
        (query_id, queries[query_id])
        for query_id in qrels
        if query_id in queries
    ][: args.max_queries]

    embedding = EmbeddingRetriever(max_docs=None, batch_size=16)
    models = {
        "embedding": embedding,
        "hybrid_serial": HybridSerialRetrieverV2(
            bm25_retriever=BM25Retriever(max_docs=None),
            embedding_retriever=embedding,
            max_docs=None,
        ),
        "hybrid_parallel": ReciprocalRankFusionRetriever(
            tfidf_retriever=TFIDFRetriever(max_docs=None),
            bm25_retriever=BM25Retriever(max_docs=None),
            embedding_retriever=embedding,
            max_docs=None,
        ),
    }

    rows, summaries = [], []
    for model_name in args.models:
        retriever = models[model_name]
        if hasattr(retriever, "prepare"):
            retriever.prepare(args.dataset)
        else:
            retriever.search("initialization", args.dataset, top_k=1)

        latencies = []
        for query_id, query_text in query_items:
            for repetition in range(1, args.repeats + 1):
                started = time.perf_counter()
                results = retriever.search(
                    query_text,
                    args.dataset,
                    top_k=args.top_k,
                )
                wall_ms = (time.perf_counter() - started) * 1000.0
                internal_ms = (
                    float(results[0].metadata.get("query_time_ms", wall_ms))
                    if results
                    else wall_ms
                )
                latencies.append(internal_ms)
                rows.append(
                    {
                        "dataset_name": args.dataset,
                        "model_name": model_name,
                        "query_id": query_id,
                        "repetition": repetition,
                        "internal_latency_ms": round(internal_ms, 3),
                        "wall_latency_ms": round(wall_ms, 3),
                    }
                )
                print(
                    f"{model_name} query={query_id} repetition={repetition}: "
                    f"{internal_ms:.2f} ms"
                )

        ordered = sorted(latencies)
        summary = {
            "dataset_name": args.dataset,
            "model_name": model_name,
            "measurements": len(latencies),
            "average_ms": round(statistics.fmean(latencies), 3),
            "median_p50_ms": round(statistics.median(latencies), 3),
            "p95_ms": round(_percentile(ordered, 0.95), 3),
            "minimum_ms": round(min(latencies), 3),
            "maximum_ms": round(max(latencies), 3),
        }
        summaries.append(summary)
        print(json.dumps(summary, indent=2, ensure_ascii=False))

    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = EVALUATION_DIR / f"{args.dataset}_dense_hybrid_latency.csv"
    json_path = EVALUATION_DIR / f"{args.dataset}_dense_hybrid_latency_summary.json"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(summaries, file, indent=2, ensure_ascii=False)
    print(f"Saved detailed latency records to: {csv_path}")
    print(f"Saved latency summary to: {json_path}")


def _percentile(values, fraction):
    index = max(0, math.ceil(fraction * len(values)) - 1)
    return values[index]


if __name__ == "__main__":
    main()

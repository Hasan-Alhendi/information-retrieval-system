"""Generate report-ready charts from full-system evaluation outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import EVALUATION_DIR

MODEL_LABELS = {
    "bm25": "BM25",
    "tfidf": "TF-IDF",
    "embedding": "Embedding",
    "hybrid_serial": "Hybrid Serial",
    "hybrid_parallel": "Hybrid Parallel",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate evaluation report charts.")
    parser.add_argument("--dataset", default="touche2020-v2")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    quality_path = EVALUATION_DIR / f"{args.dataset}_full_system_evaluation.csv"
    if not quality_path.exists():
        raise FileNotFoundError(
            f"Missing {quality_path}. Run scripts/evaluate_full_system.py first."
        )

    output_dir = EVALUATION_DIR / "charts" / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    quality = pd.read_csv(quality_path)
    quality["model_label"] = quality["model_name"].map(MODEL_LABELS)

    _single_metric_chart(
        quality,
        metric="map_at_k",
        title="MAP@10 by Retrieval Model",
        ylabel="MAP@10",
        output_path=output_dir / "map_at_10.png",
    )
    _single_metric_chart(
        quality,
        metric="ndcg_at_k",
        title="nDCG@10 by Retrieval Model",
        ylabel="nDCG@10",
        output_path=output_dir / "ndcg_at_10.png",
    )
    _precision_recall_chart(
        quality,
        output_path=output_dir / "precision_recall_at_10.png",
    )

    latency = _load_latency(args.dataset)
    if not latency.empty:
        _latency_chart(latency, output_path=output_dir / "average_latency_ms.png")

    pruning_path = EVALUATION_DIR / f"{args.dataset}_pruning_threshold_comparison.csv"
    if pruning_path.exists():
        _pruning_chart(
            pd.read_csv(pruning_path),
            output_path=output_dir / "pruning_quality_latency_tradeoff.png",
        )

    print(f"Saved evaluation charts to: {output_dir}")


def _single_metric_chart(
    frame: pd.DataFrame,
    *,
    metric: str,
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.bar(frame["model_label"], frame[metric])
    axis.set_title(title)
    axis.set_xlabel("Model")
    axis.set_ylabel(ylabel)
    axis.tick_params(axis="x", rotation=20)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def _precision_recall_chart(frame: pd.DataFrame, *, output_path: Path) -> None:
    chart = frame.set_index("model_label")[["precision_at_10", "recall_at_k"]]
    axis = chart.plot(kind="bar", figsize=(10, 5))
    axis.set_title("Precision@10 and Recall@10")
    axis.set_xlabel("Model")
    axis.set_ylabel("Score")
    axis.tick_params(axis="x", rotation=20)
    axis.grid(axis="y", alpha=0.25)
    axis.figure.tight_layout()
    axis.figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(axis.figure)


def _load_latency(dataset_name: str) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for filename in (
        f"{dataset_name}_full_search_latency_summary.json",
        f"{dataset_name}_dense_hybrid_latency_summary.json",
    ):
        path = EVALUATION_DIR / filename
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
        records.extend(payload if isinstance(payload, list) else [payload])

    if not records:
        return pd.DataFrame()
    frame = pd.DataFrame(records)
    frame = frame.drop_duplicates(subset=["model_name"], keep="last")
    frame["model_label"] = frame["model_name"].map(MODEL_LABELS)
    return frame


def _latency_chart(frame: pd.DataFrame, *, output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.bar(frame["model_label"], frame["average_ms"])
    axis.set_title("Average Search Latency")
    axis.set_xlabel("Model")
    axis.set_ylabel("Milliseconds")
    axis.tick_params(axis="x", rotation=20)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def _pruning_chart(frame: pd.DataFrame, *, output_path: Path) -> None:
    bm25 = frame[frame["model_name"] == "bm25"].sort_values("max_df_ratio")
    if bm25.empty:
        return
    figure, quality_axis = plt.subplots(figsize=(9, 5))
    quality_axis.plot(bm25["max_df_ratio"], bm25["map_score"], marker="o", label="MAP@10")
    quality_axis.plot(bm25["max_df_ratio"], bm25["ndcg"], marker="o", label="nDCG@10")
    quality_axis.set_xlabel("max_df_ratio")
    quality_axis.set_ylabel("Quality score")
    quality_axis.set_title("BM25 Pruning: Quality and Latency Trade-off")
    quality_axis.grid(alpha=0.25)
    quality_axis.legend(loc="upper left")

    latency_axis = quality_axis.twinx()
    latency_axis.plot(
        bm25["max_df_ratio"],
        bm25["average_latency_ms"],
        marker="s",
        linestyle="--",
        label="Average latency",
    )
    latency_axis.set_ylabel("Average latency (ms)")
    latency_axis.legend(loc="upper right")
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()

"""Evaluate document clustering independently and save report-ready outputs."""

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
from app.infrastructure.clustering.clusterer import (
    DEFAULT_CLUSTERING_EMBEDDING_MODEL,
    DocumentClusterer,
)
from app.infrastructure.datasets.dataset_registry import SUPPORTED_DATASETS


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Cluster a development subset and save metrics and charts."
    )
    parser.add_argument(
        "--dataset",
        default="quora",
        choices=sorted(SUPPORTED_DATASETS),
    )
    parser.add_argument("--max-docs", type=int, default=1000)
    parser.add_argument("--clusters", type=int, default=5)
    parser.add_argument("--sample-documents", type=int, default=3)
    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_CLUSTERING_EMBEDDING_MODEL,
    )
    return parser.parse_args()


def main() -> None:
    """Run clustering and persist quantitative and visual evaluation outputs."""
    args = parse_args()
    if args.max_docs < 2:
        raise ValueError("max-docs must be at least 2.")
    if args.clusters < 2:
        raise ValueError("clusters must be at least 2.")
    if args.clusters >= args.max_docs:
        raise ValueError("clusters must be smaller than max-docs.")

    result = DocumentClusterer(
        embedding_model_name=args.embedding_model,
    ).cluster(
        dataset_name=args.dataset,
        number_of_clusters=args.clusters,
        max_docs=args.max_docs,
        sample_size=args.sample_documents,
    )

    output_dir = EVALUATION_DIR / "clustering" / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / (
        f"clustering_{args.max_docs}_docs_{args.clusters}_clusters.json"
    )
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2, ensure_ascii=False)

    _save_cluster_sizes(result, output_dir / "cluster_sizes.png")
    _save_pca_projection(result, output_dir / "pca_projection.png")

    evaluation = result["evaluation"]
    print(
        json.dumps(
            {
                "dataset_name": result["dataset_name"],
                "documents_count": result["documents_count"],
                "number_of_clusters": result["number_of_clusters"],
                "embedding_model": result["embedding_model"],
                "clustering_algorithm": result["clustering_algorithm"],
                "evaluation": evaluation,
                "output_dir": str(output_dir),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def _save_cluster_sizes(result: dict[str, object], output_path: Path) -> None:
    clusters = result["clusters"]
    assert isinstance(clusters, list)
    frame = pd.DataFrame(
        [
            {
                "cluster": f"Cluster {cluster['cluster_id']}",
                "documents": cluster["size"],
            }
            for cluster in clusters
        ]
    )

    figure, axis = plt.subplots(figsize=(9, 5))
    axis.bar(frame["cluster"], frame["documents"])
    axis.set_title("Cluster Size Distribution")
    axis.set_xlabel("Cluster")
    axis.set_ylabel("Documents")
    axis.tick_params(axis="x", rotation=20)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def _save_pca_projection(result: dict[str, object], output_path: Path) -> None:
    projection = result["projection"]
    assert isinstance(projection, list)
    if not projection:
        return

    frame = pd.DataFrame(projection)
    figure, axis = plt.subplots(figsize=(9, 6))
    for cluster_name, group in frame.groupby("cluster"):
        axis.scatter(
            group["pc1"],
            group["pc2"],
            label=f"Cluster {cluster_name}",
            alpha=0.65,
            s=18,
        )
    axis.set_title("PCA Projection of Document Clusters")
    axis.set_xlabel("Principal Component 1")
    axis.set_ylabel("Principal Component 2")
    axis.grid(alpha=0.2)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()

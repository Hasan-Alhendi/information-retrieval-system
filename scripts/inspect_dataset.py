"""Inspect dataset metadata and optional samples without building indexes."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.infrastructure.datasets.dataset_registry import (
    EXPERIMENTAL_DATASETS,
    get_dataset_config,
)
from app.infrastructure.datasets.ir_datasets_inspector import IrDatasetsInspector


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Inspect an experimental ir_datasets dataset before full integration."
    )
    parser.add_argument(
        "--dataset",
        default="touche2020-v2",
        choices=sorted(EXPERIMENTAL_DATASETS),
        help="Experimental dataset registry name.",
    )
    parser.add_argument("--sample-queries", type=int, default=3)
    parser.add_argument("--sample-qrels", type=int, default=5)
    parser.add_argument(
        "--sample-documents",
        type=int,
        default=0,
        help=(
            "Number of documents to sample. A value greater than zero may trigger "
            "downloading the corpus, so metadata-only inspection uses zero."
        ),
    )
    parser.add_argument(
        "--include-nonrelevant-qrels",
        action="store_true",
        help="Include qrels with relevance zero in the displayed sample.",
    )
    return parser.parse_args()


def main() -> None:
    """Inspect the requested dataset and print JSON output."""
    args = parse_args()
    config = get_dataset_config(args.dataset, include_experimental=True)
    inspection = IrDatasetsInspector().inspect(
        config,
        sample_queries=max(0, args.sample_queries),
        sample_documents=max(0, args.sample_documents),
        sample_qrels=max(0, args.sample_qrels),
        positive_qrels_only=not args.include_nonrelevant_qrels,
    )
    print(json.dumps(asdict(inspection), indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()

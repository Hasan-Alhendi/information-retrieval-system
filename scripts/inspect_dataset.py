"""Inspect an ir_datasets benchmark without building indexes."""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.infrastructure.datasets.dataset_registry import get_dataset_config
from app.infrastructure.datasets.ir_datasets_inspector import IrDatasetsInspector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect dataset metadata and samples.")
    parser.add_argument("--dataset", default="touche2020-v2")
    parser.add_argument("--sample-queries", type=int, default=3)
    parser.add_argument("--sample-qrels", type=int, default=5)
    parser.add_argument("--sample-documents", type=int, default=0)
    parser.add_argument("--include-nonrelevant-qrels", action="store_true")
    return parser.parse_args()


def main() -> None:
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

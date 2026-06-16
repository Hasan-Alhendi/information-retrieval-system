"""Validate an official dataset before starting full-corpus indexing."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.infrastructure.datasets.dataset_loader import DatasetLoader
from app.infrastructure.datasets.dataset_registry import SUPPORTED_DATASETS
from app.infrastructure.datasets.ir_datasets_inspector import IrDatasetsInspector
from app.infrastructure.datasets.runtime_profiles import get_runtime_profile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate counts, schemas, samples, qrels, and disk space."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=sorted(SUPPORTED_DATASETS),
    )
    parser.add_argument("--sample-documents", type=int, default=3)
    parser.add_argument("--sample-queries", type=int, default=3)
    parser.add_argument("--sample-qrels", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = SUPPORTED_DATASETS[args.dataset]
    runtime = get_runtime_profile(args.dataset)
    inspection = IrDatasetsInspector().inspect(
        config,
        sample_queries=max(1, args.sample_queries),
        sample_documents=max(1, args.sample_documents),
        sample_qrels=max(1, args.sample_qrels),
        positive_qrels_only=True,
    )

    queries, qrels = DatasetLoader().load_queries_qrels(args.dataset)
    sample_documents = inspection.sample_documents
    checks = {
        "has_external_id": bool(config.external_id),
        "documents_count_known": bool(inspection.documents_count),
        "queries_count_known": bool(inspection.queries_count),
        "qrels_count_known": bool(inspection.qrels_count),
        "sample_documents_available": bool(sample_documents),
        "sample_queries_available": bool(inspection.sample_queries),
        "positive_qrels_available": bool(inspection.sample_qrels),
        "query_ids_overlap_qrels": bool(set(queries).intersection(qrels)),
        "documents_have_text": all(
            bool(str(document.get("text", "")).strip())
            for document in sample_documents
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    disk = shutil.disk_usage(PROJECT_ROOT)

    output = {
        "dataset": asdict(config),
        "runtime_profile": asdict(runtime),
        "inspection": asdict(inspection),
        "validation_checks": checks,
        "validation_status": "passed" if not failed else "failed",
        "failed_checks": failed,
        "disk_space": {
            "project_root": str(PROJECT_ROOT),
            "free_bytes": disk.free,
            "free_gib": round(disk.free / (1024**3), 2),
        },
        "next_commands": {
            "lexical_smoke": (
                f"python scripts/smoke_test_full_index.py --dataset {args.dataset} "
                f"--max-docs {runtime.smoke_documents} "
                f"--batch-size {runtime.lexical_batch_size}"
            ),
            "embedding_smoke": (
                f"python scripts/smoke_test_full_embedding.py --dataset {args.dataset} "
                f"--max-docs {runtime.smoke_documents} "
                f"--batch-size {runtime.embedding_batch_size} "
                f"--checkpoint-size 1000 --force"
            ),
        },
    }
    print(json.dumps(output, indent=2, ensure_ascii=False, default=str))

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

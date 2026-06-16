"""Repair a derived SQLite postings index and resume lexical finalization."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.infrastructure.datasets.dataset_registry import SUPPORTED_DATASETS
from app.infrastructure.retrieval.resilient_disk_lexical_index import (
    ResilientOptimizedDiskLexicalIndex,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Back up a full lexical SQLite database, rebuild its derived postings "
            "index, and resume chunked TF-IDF norm finalization."
        )
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=sorted(SUPPORTED_DATASETS),
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--diagnose-only",
        action="store_true",
        help="Run quick_check without changing the database.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    index = ResilientOptimizedDiskLexicalIndex(
        args.dataset,
        batch_size=args.batch_size,
    )

    diagnosis = index.quick_check()
    print(
        json.dumps(
            {
                "dataset_name": args.dataset,
                "database_path": str(index.database_path),
                "quick_check": diagnosis,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    if args.diagnose_only:
        return

    if diagnosis["ok"]:
        print("Database quick_check passed; no secondary-index repair is required.")
    else:
        print("Creating a recovery backup and rebuilding idx_postings_document...")
        repair = index.repair_secondary_index(create_backup=True)
        print(json.dumps(repair, indent=2, ensure_ascii=False))
        if not repair["ok_after_repair"]:
            raise SystemExit(
                "Repair did not restore SQLite quick_check. Do not use --force; "
                "keep the recovery backup and inspect the reported error."
            )

    print("Resuming the existing build without re-indexing completed documents...")
    index.build(force=False)
    print(json.dumps(index.status(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

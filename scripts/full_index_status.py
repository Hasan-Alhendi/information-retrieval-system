"""Show status for a full disk-backed lexical index."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.infrastructure.retrieval.disk_lexical_index import DiskLexicalIndex


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Show full lexical index status.")
    parser.add_argument("--dataset", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    status = DiskLexicalIndex(args.dataset).status()
    print(json.dumps(status, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

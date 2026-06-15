"""Build a small real-data SQLite index to validate the full-corpus pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.infrastructure.datasets.dataset_loader import DatasetLoader
from app.infrastructure.retrieval.disk_lexical_index import DiskLexicalIndex
from app.infrastructure.storage.index_store import get_index_dir


class LimitedDatasetLoader(DatasetLoader):
    """Limit the real streaming source only for a disposable smoke test."""

    def __init__(self, limit: int) -> None:
        super().__init__()
        self.limit = limit

    def iter_documents(self, dataset_name: str, max_docs: int | None = None):
        del max_docs
        yield from super().iter_documents(dataset_name, max_docs=self.limit)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test the full disk index pipeline.")
    parser.add_argument("--dataset", default="touche2020-v2")
    parser.add_argument("--max-docs", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--query",
        default="Should teachers get tenure?",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    database_path = (
        get_index_dir(args.dataset, "disk_lexical")
        / "full"
        / f"smoke_{args.max_docs}.sqlite3"
    )
    index = DiskLexicalIndex(
        args.dataset,
        dataset_loader=LimitedDatasetLoader(args.max_docs),
        batch_size=args.batch_size,
        database_path=database_path,
    )
    index.build(force=True)

    print(json.dumps(index.status(), indent=2, ensure_ascii=False))
    for model_name, results in (
        ("bm25", index.search_bm25(args.query, top_k=3)),
        ("tfidf", index.search_tfidf(args.query, top_k=3)),
    ):
        print(f"\n{model_name.upper()} RESULTS")
        for result in results:
            print(
                json.dumps(
                    {
                        "rank": result.rank,
                        "doc_id": result.doc_id,
                        "score": result.score,
                        "query_time_ms": result.metadata.get("query_time_ms"),
                        "text_preview": result.text[:200],
                    },
                    ensure_ascii=False,
                )
            )


if __name__ == "__main__":
    main()

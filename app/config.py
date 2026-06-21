"""Central application configuration."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STORAGE_DIR = PROJECT_ROOT / "storage"
DATASETS_DIR = STORAGE_DIR / "datasets"
INDEXES_DIR = STORAGE_DIR / "indexes"
VECTOR_STORES_DIR = STORAGE_DIR / "vector_stores"
EVALUATION_DIR = STORAGE_DIR / "evaluation"

DEFAULT_TOP_K = 10
DEFAULT_BM25_K1 = 1.5
DEFAULT_BM25_B = 0.75

SUPPORTED_RETRIEVAL_MODELS = (
    "tfidf",
    "bm25",
    "embedding",
    "embedding_guided_categories",
    "hybrid_serial",
    "hybrid_parallel",
)

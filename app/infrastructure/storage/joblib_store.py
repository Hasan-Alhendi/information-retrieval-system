"""Joblib-based persistence helpers."""

from pathlib import Path
from typing import Any

import joblib


def save_object(value: Any, path: Path) -> None:
    """Persist a Python object using joblib."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(value, path)


def load_object(path: Path) -> Any:
    """Load a Python object using joblib."""
    return joblib.load(path)

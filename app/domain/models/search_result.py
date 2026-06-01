"""Search result domain model."""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SearchResult:
    """Represents a ranked retrieval result."""

    doc_id: str
    score: float
    rank: int
    text: str | None = None
    title: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

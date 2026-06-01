"""Query domain model."""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Query:
    """Represents a user or benchmark query."""

    query_id: str | None
    text: str
    dataset_name: str
    metadata: dict[str, Any] = field(default_factory=dict)

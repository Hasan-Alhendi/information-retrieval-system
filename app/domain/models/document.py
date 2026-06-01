"""Document domain model."""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Document:
    """Represents a searchable document."""

    doc_id: str
    text: str
    title: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

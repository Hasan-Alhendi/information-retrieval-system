"""API schemas."""

from typing import Any

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    """Search request payload."""

    query: str = Field(..., min_length=1)
    dataset_name: str
    model_name: str
    top_k: int = Field(default=10, ge=1, le=100)
    max_docs: int | None = Field(
        default=None,
        ge=1,
        description="Leave null to use finalized full-corpus indexes.",
    )
    bm25_k1: float = Field(default=1.5, gt=0)
    bm25_b: float = Field(default=0.75, ge=0, le=1)
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    use_query_refinement: bool = False


class SearchResultResponse(BaseModel):
    """Search result response item."""

    doc_id: str
    rank: int
    score: float
    title: str | None = None
    text: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class QueryRefinementResponse(BaseModel):
    """Query refinement details."""

    original_query: str
    refined_query: str
    corrections: dict[str, str]
    expansions: dict[str, list[str]]


class SearchResponse(BaseModel):
    """Search response payload."""

    dataset_name: str
    model_name: str
    index_scope: str
    search_time_ms: float | None = None
    results: list[SearchResultResponse]
    query_refinement: QueryRefinementResponse | None = None

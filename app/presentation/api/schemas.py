"""API schemas."""

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    """Search request payload."""

    query: str = Field(..., min_length=1)
    dataset_name: str
    model_name: str
    top_k: int = Field(default=10, ge=1, le=100)
    max_docs: int | None = Field(default=None, ge=1)
    bm25_k1: float = Field(default=1.5, gt=0)
    bm25_b: float = Field(default=0.75, ge=0, le=1)


class SearchResultResponse(BaseModel):
    """Search result response item."""

    doc_id: str
    rank: int
    score: float
    title: str | None = None
    text: str | None = None


class SearchResponse(BaseModel):
    """Search response payload."""

    results: list[SearchResultResponse]

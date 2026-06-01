"""Search API routes."""

from fastapi import APIRouter

from app.presentation.api.schemas import SearchRequest, SearchResponse

router = APIRouter(prefix="/search", tags=["search"])


@router.post("", response_model=SearchResponse)
def search_documents(request: SearchRequest) -> SearchResponse:
    """Search documents using the selected retrieval model."""
    _ = request
    return SearchResponse(results=[])

"""Search API routes."""

from fastapi import APIRouter, HTTPException

from app.infrastructure.retrieval.tfidf_retriever import TFIDFRetriever
from app.presentation.api.schemas import SearchRequest, SearchResponse, SearchResultResponse

router = APIRouter(prefix="/search", tags=["search"])

RETRIEVERS = {
    "tfidf": TFIDFRetriever,
}


@router.post("", response_model=SearchResponse)
def search_documents(request: SearchRequest) -> SearchResponse:
    """Search documents using the selected retrieval model."""
    retriever_cls = RETRIEVERS.get(request.model_name)
    if retriever_cls is None:
        raise HTTPException(status_code=400, detail=f"Unsupported model: {request.model_name}")

    retriever = retriever_cls()
    results = retriever.search(
        query=request.query,
        dataset_name=request.dataset_name,
        top_k=request.top_k,
    )
    return SearchResponse(
        results=[
            SearchResultResponse(
                doc_id=result.doc_id,
                rank=result.rank,
                score=result.score,
                title=result.title,
                text=result.text,
            )
            for result in results
        ]
    )

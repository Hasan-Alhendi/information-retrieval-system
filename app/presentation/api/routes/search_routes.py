"""Search API routes."""

from fastapi import APIRouter, HTTPException

from app.application.services.query_refinement_service import QueryRefinementService
from app.application.services.retriever_factory import create_retriever
from app.presentation.api.schemas import (
    QueryRefinementResponse,
    SearchRequest,
    SearchResponse,
    SearchResultResponse,
)

router = APIRouter(prefix="/search", tags=["search"])


@router.post("", response_model=SearchResponse)
def search_documents(request: SearchRequest) -> SearchResponse:
    """Search a full corpus or development subset with the selected model."""
    try:
        retriever = create_retriever(
            request.model_name,
            max_docs=request.max_docs,
            bm25_k1=request.bm25_k1,
            bm25_b=request.bm25_b,
            embedding_model=request.embedding_model,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    query = request.query
    refinement_response = None
    if request.use_query_refinement:
        refinement = QueryRefinementService().refine(request.query)
        query = refinement.refined_query
        refinement_response = QueryRefinementResponse(
            original_query=refinement.original_query,
            refined_query=refinement.refined_query,
            corrections=refinement.corrections,
            expansions=refinement.expansions,
        )

    try:
        if request.max_docs is None and hasattr(retriever, "prepare"):
            retriever.prepare(request.dataset_name)
        results = retriever.search(
            query=query,
            dataset_name=request.dataset_name,
            top_k=request.top_k,
        )
    except (RuntimeError, ValueError, OSError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    search_time_ms = (
        float(results[0].metadata.get("query_time_ms"))
        if results and results[0].metadata.get("query_time_ms") is not None
        else None
    )
    return SearchResponse(
        dataset_name=request.dataset_name,
        model_name=request.model_name,
        index_scope="full" if request.max_docs is None else "development",
        search_time_ms=search_time_ms,
        query_refinement=refinement_response,
        results=[
            SearchResultResponse(
                doc_id=result.doc_id,
                rank=result.rank,
                score=result.score,
                title=result.title,
                text=result.text,
                metadata=result.metadata,
            )
            for result in results
        ],
    )

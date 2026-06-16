"""Search API routes."""

from functools import lru_cache

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


@lru_cache(maxsize=32)
def _cached_retriever(
    model_name: str,
    dataset_name: str,
    max_docs: int | None,
    bm25_k1: float,
    bm25_b: float,
    embedding_model: str,
):
    """Keep heavy models and indexes loaded across API requests."""
    retriever = create_retriever(
        model_name,
        max_docs=max_docs,
        bm25_k1=bm25_k1,
        bm25_b=bm25_b,
        embedding_model=embedding_model,
    )
    if max_docs is None and hasattr(retriever, "prepare"):
        retriever.prepare(dataset_name)
    return retriever


@router.post("", response_model=SearchResponse)
def search_documents(request: SearchRequest) -> SearchResponse:
    """Search a full corpus or development subset with the selected model."""
    try:
        retriever = _cached_retriever(
            request.model_name,
            request.dataset_name,
            request.max_docs,
            request.bm25_k1,
            request.bm25_b,
            request.embedding_model,
        )
    except (RuntimeError, ValueError, OSError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

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

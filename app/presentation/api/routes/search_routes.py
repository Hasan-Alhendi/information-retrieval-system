"""Search API routes."""

from fastapi import APIRouter, HTTPException

from app.infrastructure.retrieval.bm25_retriever import BM25Retriever
from app.infrastructure.retrieval.embedding_retriever import EmbeddingRetriever
from app.infrastructure.retrieval.hybrid_parallel import HybridParallelRetriever
from app.infrastructure.retrieval.hybrid_serial import HybridSerialRetriever
from app.infrastructure.retrieval.tfidf_retriever import TFIDFRetriever
from app.presentation.api.schemas import SearchRequest, SearchResponse, SearchResultResponse

router = APIRouter(prefix="/search", tags=["search"])


def _create_retriever(request: SearchRequest):
    if request.model_name == "tfidf":
        return TFIDFRetriever(max_docs=request.max_docs)
    if request.model_name == "bm25":
        return BM25Retriever(
            k1=request.bm25_k1,
            b=request.bm25_b,
            max_docs=request.max_docs,
        )
    if request.model_name == "embedding":
        return EmbeddingRetriever(
            embedding_model_name=request.embedding_model,
            max_docs=request.max_docs,
        )
    if request.model_name == "hybrid_serial":
        return HybridSerialRetriever(
            bm25_retriever=BM25Retriever(
                k1=request.bm25_k1,
                b=request.bm25_b,
                max_docs=request.max_docs,
            ),
            embedding_retriever=EmbeddingRetriever(
                embedding_model_name=request.embedding_model,
                max_docs=request.max_docs,
            ),
            max_docs=request.max_docs,
        )
    if request.model_name == "hybrid_parallel":
        return HybridParallelRetriever(
            tfidf_retriever=TFIDFRetriever(max_docs=request.max_docs),
            bm25_retriever=BM25Retriever(
                k1=request.bm25_k1,
                b=request.bm25_b,
                max_docs=request.max_docs,
            ),
            embedding_retriever=EmbeddingRetriever(
                embedding_model_name=request.embedding_model,
                max_docs=request.max_docs,
            ),
            max_docs=request.max_docs,
        )
    raise HTTPException(status_code=400, detail=f"Unsupported model: {request.model_name}")


@router.post("", response_model=SearchResponse)
def search_documents(request: SearchRequest) -> SearchResponse:
    """Search documents using the selected retrieval model."""
    retriever = _create_retriever(request)
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

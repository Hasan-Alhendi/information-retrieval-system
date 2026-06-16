"""FastAPI application entry point."""

from fastapi import FastAPI

from app.presentation.api.routes.dataset_routes import router as dataset_router
from app.presentation.api.routes.evaluation_routes import router as evaluation_router
from app.presentation.api.routes.indexing_routes import router as indexing_router
from app.presentation.api.routes.search_routes import router as search_router

app = FastAPI(
    title="Information Retrieval System",
    description=(
        "Full-corpus retrieval for Quora and Webis-Touché 2020 v2 with "
        "BM25, TF-IDF, Embedding, Hybrid Serial, and Hybrid Parallel models."
    ),
    version="1.0.0",
)

app.include_router(dataset_router)
app.include_router(evaluation_router)
app.include_router(indexing_router)
app.include_router(search_router)


@app.get("/health", tags=["system"])
def health_check() -> dict[str, str]:
    """Return API health status."""
    return {"status": "ok", "version": app.version}

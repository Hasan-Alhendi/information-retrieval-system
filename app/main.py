"""FastAPI application entry point."""

from fastapi import FastAPI

app = FastAPI(
    title="Information Retrieval System",
    description="Clean Architecture IR system with multiple retrieval models.",
    version="0.1.0",
)


@app.get("/health", tags=["system"])
def health_check() -> dict[str, str]:
    """Return API health status."""
    return {"status": "ok"}

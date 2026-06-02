"""Application query refinement service."""

from app.infrastructure.query_refinement.query_refiner import QueryRefinementResult, QueryRefiner


class QueryRefinementService:
    """Coordinates query refinement strategies."""

    def __init__(self, query_refiner: QueryRefiner | None = None) -> None:
        self._query_refiner = query_refiner or QueryRefiner()

    def refine(self, query: str) -> QueryRefinementResult:
        """Return a refined query with explanation details."""
        return self._query_refiner.refine(query)

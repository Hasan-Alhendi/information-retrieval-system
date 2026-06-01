"""Application query refinement service."""


class QueryRefinementService:
    """Coordinates query refinement strategies."""

    def refine(self, query: str) -> str:
        """Return a refined query.

        The initial implementation trims whitespace. Expansion and correction will be added later.
        """
        return query.strip()

"""Rule-based query refinement implementation."""

from dataclasses import dataclass

from app.infrastructure.preprocessing.text_normalizer import normalize_text

COMMON_CORRECTIONS = {
    "retrival": "retrieval",
    "retreival": "retrieval",
    "infomation": "information",
    "informtion": "information",
    "rankng": "ranking",
    "documnt": "document",
    "documnet": "document",
    "serach": "search",
    "querry": "query",
}

DOMAIN_EXPANSIONS = {
    "retrieval": ["search", "ranking"],
    "ranking": ["retrieval", "relevance"],
    "query": ["question", "search"],
    "document": ["text", "passage"],
    "semantic": ["embedding", "meaning"],
    "embedding": ["semantic", "vector"],
    "vector": ["embedding", "representation"],
    "evaluation": ["metrics", "benchmark"],
    "relevance": ["ranking", "qrels"],
}


@dataclass(frozen=True)
class QueryRefinementResult:
    """Represents a refined query and the applied changes."""

    original_query: str
    refined_query: str
    corrections: dict[str, str]
    expansions: dict[str, list[str]]


class QueryRefiner:
    """Simple, explainable query refinement strategy.

    The strategy applies lightweight normalization, a small domain-specific spelling
    correction dictionary, and optional IR-domain term expansion.
    """

    def __init__(
        self,
        enable_correction: bool = True,
        enable_expansion: bool = True,
        max_expansions_per_term: int = 2,
    ) -> None:
        self.enable_correction = enable_correction
        self.enable_expansion = enable_expansion
        self.max_expansions_per_term = max_expansions_per_term

    def refine(self, query: str) -> QueryRefinementResult:
        """Return a refined query and details about the applied changes."""
        original_query = query
        normalized_query = normalize_text(query).lower()
        tokens = normalized_query.split()

        corrections: dict[str, str] = {}
        corrected_tokens: list[str] = []
        for token in tokens:
            corrected = COMMON_CORRECTIONS.get(token, token) if self.enable_correction else token
            if corrected != token:
                corrections[token] = corrected
            corrected_tokens.append(corrected)

        expansions: dict[str, list[str]] = {}
        expanded_tokens = list(corrected_tokens)
        if self.enable_expansion:
            existing_terms = set(corrected_tokens)
            for token in corrected_tokens:
                related_terms = DOMAIN_EXPANSIONS.get(token, [])[: self.max_expansions_per_term]
                terms_to_add = [term for term in related_terms if term not in existing_terms]
                if terms_to_add:
                    expansions[token] = terms_to_add
                    expanded_tokens.extend(terms_to_add)
                    existing_terms.update(terms_to_add)

        refined_query = " ".join(expanded_tokens).strip()
        return QueryRefinementResult(
            original_query=original_query,
            refined_query=refined_query,
            corrections=corrections,
            expansions=expansions,
        )

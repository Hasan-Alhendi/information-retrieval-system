"""Guided semantic category reranking for Quora and Touché."""

from __future__ import annotations

import time

import numpy as np

from app.domain.models.search_result import SearchResult
from app.infrastructure.clustering.category_profiles import (
    CategoryProfile,
    get_category_profiles,
)
from app.infrastructure.retrieval.embedding_retriever import EmbeddingRetriever


class GuidedCategoryRetriever:
    """Rerank embedding candidates using automatic soft category assignment."""

    model_name = "embedding_guided_categories"

    def __init__(
        self,
        *,
        embedding_retriever: EmbeddingRetriever,
        category_weight: float = 0.25,
        candidate_k: int = 100,
        top_categories: int = 3,
    ) -> None:
        if not 0.0 <= category_weight <= 1.0:
            raise ValueError("category_weight must be between 0 and 1.")
        if candidate_k < 1:
            raise ValueError("candidate_k must be positive.")
        if top_categories < 1:
            raise ValueError("top_categories must be positive.")

        self._embedding = embedding_retriever
        self.category_weight = category_weight
        self.candidate_k = candidate_k
        self.top_categories = top_categories
        self._profiles: dict[str, tuple[CategoryProfile, ...]] = {}
        self._category_vectors: dict[str, np.ndarray] = {}
        self._prepared_datasets: set[str] = set()

    def build(
        self,
        dataset_name: str,
        force: bool = False,
        max_docs: int | None = None,
    ) -> None:
        """Reuse the normal embedding index; no separate document index is built."""
        self._embedding.build(
            dataset_name=dataset_name,
            force=force,
            max_docs=max_docs,
        )

    def prepare(self, dataset_name: str) -> None:
        """Warm the model and encode each dataset's category list only once."""
        if dataset_name in self._prepared_datasets:
            return
        self._embedding.prepare(dataset_name)
        self._ensure_categories(dataset_name)
        self._prepared_datasets.add(dataset_name)

    def search(
        self,
        query: str,
        dataset_name: str,
        top_k: int = 10,
    ) -> list[SearchResult]:
        """Retrieve candidates and rerank them by category alignment."""
        if top_k <= 0:
            return []

        started = time.perf_counter()
        self.prepare(dataset_name)
        candidate_count = max(top_k, self.candidate_k)
        candidates = self._embedding.search(
            query=query,
            dataset_name=dataset_name,
            top_k=candidate_count,
        )
        if not candidates:
            return []

        row_indexes = [
            int(candidate.metadata["row_index"])
            for candidate in candidates
            if "row_index" in candidate.metadata
        ]
        if len(row_indexes) != len(candidates):
            candidate_vectors = self._embedding.encode_texts(
                [candidate.text for candidate in candidates]
            )
        else:
            candidate_vectors = self._embedding.load_candidate_vectors(
                dataset_name,
                row_indexes,
            )

        query_vector = self._embedding.encode_query(query)
        profiles = self._profiles[dataset_name]
        category_vectors = self._category_vectors[dataset_name]
        selected_indexes, selected_weights, query_category_scores = _select_categories(
            query_vector=query_vector,
            category_vectors=category_vectors,
            top_categories=self.top_categories,
        )

        selected_category_vectors = category_vectors[selected_indexes]
        document_category_scores = np.clip(
            (candidate_vectors @ selected_category_vectors.T + 1.0) / 2.0,
            0.0,
            1.0,
        )
        category_alignment = document_category_scores @ selected_weights

        base_scores = _min_max_normalize(
            np.asarray([candidate.score for candidate in candidates], dtype="float32")
        )
        final_scores = (
            (1.0 - self.category_weight) * base_scores
            + self.category_weight * category_alignment
        )

        order = np.argsort(final_scores)[::-1][:top_k]
        latency_ms = round((time.perf_counter() - started) * 1000.0, 3)
        selected_categories = [
            {
                "name": profiles[int(index)].name,
                "label": profiles[int(index)].label,
                "query_score": float(query_category_scores[int(index)]),
                "weight": float(selected_weights[position]),
            }
            for position, index in enumerate(selected_indexes)
        ]

        results: list[SearchResult] = []
        for rank, candidate_index in enumerate(order, start=1):
            candidate = candidates[int(candidate_index)]
            best_local_index = int(
                np.argmax(document_category_scores[int(candidate_index)])
            )
            best_category_index = int(selected_indexes[best_local_index])
            best_profile = profiles[best_category_index]
            results.append(
                SearchResult(
                    doc_id=candidate.doc_id,
                    score=float(final_scores[int(candidate_index)]),
                    rank=rank,
                    text=candidate.text,
                    title=candidate.title,
                    metadata={
                        **candidate.metadata,
                        "model": self.model_name,
                        "pipeline": "embedding_then_guided_category_reranking",
                        "base_embedding_score": float(candidate.score),
                        "normalized_base_score": float(base_scores[int(candidate_index)]),
                        "category_alignment_score": float(
                            category_alignment[int(candidate_index)]
                        ),
                        "best_category": best_profile.name,
                        "best_category_label": best_profile.label,
                        "selected_query_categories": selected_categories,
                        "category_weight": self.category_weight,
                        "top_categories": len(selected_indexes),
                        "candidate_k": candidate_count,
                        "query_time_ms": latency_ms,
                    },
                )
            )
        return results

    def _ensure_categories(self, dataset_name: str) -> None:
        if dataset_name in self._category_vectors:
            return
        profiles = get_category_profiles(dataset_name)
        descriptions = [
            f"{profile.label}. {profile.description}"
            for profile in profiles
        ]
        self._profiles[dataset_name] = profiles
        self._category_vectors[dataset_name] = self._embedding.encode_texts(descriptions)


def _select_categories(
    *,
    query_vector: np.ndarray,
    category_vectors: np.ndarray,
    top_categories: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select the query's strongest categories and return soft assignment weights."""
    similarities = np.asarray(category_vectors @ query_vector, dtype="float32")
    scores = np.clip((similarities + 1.0) / 2.0, 0.0, 1.0)
    count = min(top_categories, len(scores))
    selected_indexes = np.argsort(scores)[::-1][:count]
    selected_scores = scores[selected_indexes]
    shifted = selected_scores - np.max(selected_scores)
    exponentials = np.exp(shifted / 0.10)
    weights = exponentials / np.maximum(np.sum(exponentials), 1e-12)
    return selected_indexes, weights.astype("float32"), scores


def _min_max_normalize(values: np.ndarray) -> np.ndarray:
    """Normalize candidate scores to [0, 1]."""
    if values.size == 0:
        return values
    lowest = float(np.min(values))
    highest = float(np.max(values))
    if highest == lowest:
        return np.ones_like(values, dtype="float32")
    return ((values - lowest) / (highest - lowest)).astype("float32")

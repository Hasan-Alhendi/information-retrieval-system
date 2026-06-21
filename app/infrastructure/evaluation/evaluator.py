"""Retrieval evaluator implementation."""

import time
from collections.abc import Callable
from typing import Any

from app.application.services.query_refinement_service import QueryRefinementService
from app.config import DEFAULT_TOP_K
from app.domain.models.evaluation_result import EvaluationResult
from app.infrastructure.datasets.dataset_loader import DatasetLoader
from app.infrastructure.datasets.dataset_registry import get_dataset_config
from app.infrastructure.evaluation.metrics import (
    average_precision,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from app.infrastructure.retrieval.bm25_retriever import BM25Retriever
from app.infrastructure.retrieval.embedding_retriever import EmbeddingRetriever
from app.infrastructure.retrieval.guided_category_retriever import GuidedCategoryRetriever
from app.infrastructure.retrieval.hybrid_parallel import HybridParallelRetriever
from app.infrastructure.retrieval.hybrid_serial import HybridSerialRetriever
from app.infrastructure.retrieval.tfidf_retriever import TFIDFRetriever

RetrieverFactory = Callable[[], Any]


class RetrievalEvaluator:
    """Evaluates retrieval models using benchmark qrels."""

    def __init__(
        self,
        dataset_loader: DatasetLoader | None = None,
        max_docs: int | None = None,
        top_k: int = DEFAULT_TOP_K,
        max_queries: int | None = None,
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        use_query_refinement: bool = False,
        category_weight: float = 0.25,
        category_candidate_k: int = 100,
        top_categories: int = 3,
    ) -> None:
        self._dataset_loader = dataset_loader or DatasetLoader()
        self.max_docs = max_docs
        self.top_k = top_k
        self.max_queries = max_queries
        self.bm25_k1 = bm25_k1
        self.bm25_b = bm25_b
        self.embedding_model = embedding_model
        self.use_query_refinement = use_query_refinement
        self.category_weight = category_weight
        self.category_candidate_k = category_candidate_k
        self.top_categories = top_categories
        self._query_refinement_service = QueryRefinementService()

    def evaluate(self, dataset_name: str, model_name: str) -> EvaluationResult:
        """Evaluate a retrieval model on a dataset."""
        config = get_dataset_config(dataset_name, include_experimental=True)
        if self.max_docs is None and config.external_id:
            queries, qrels = self._dataset_loader.load_queries_qrels(dataset_name)
        else:
            _, _, queries, qrels = self._dataset_loader.prepare_dataset(
                dataset_name,
                max_docs=self.max_docs,
            )

        retriever = self._create_retriever(model_name)
        retriever.build(dataset_name=dataset_name, force=False, max_docs=self.max_docs)
        if hasattr(retriever, "prepare"):
            retriever.prepare(dataset_name)

        query_items = [(query_id, queries[query_id]) for query_id in qrels if query_id in queries]
        if self.max_queries is not None:
            query_items = query_items[: self.max_queries]

        if not query_items:
            return EvaluationResult(
                dataset_name=dataset_name,
                model_name=model_name,
                map_score=0.0,
                recall=0.0,
                precision_at_10=0.0,
                ndcg=0.0,
                evaluated_queries=0,
                average_query_time_ms=0.0,
            )

        map_scores: list[float] = []
        recall_scores: list[float] = []
        precision_scores: list[float] = []
        ndcg_scores: list[float] = []
        query_times_ms: list[float] = []

        for query_id, query_text in query_items:
            relevance_scores = {doc_id: float(score) for doc_id, score in qrels[query_id].items()}
            relevant_docs = {doc_id for doc_id, score in relevance_scores.items() if score > 0}
            active_query = self._prepare_query(query_text)

            started = time.perf_counter()
            results = retriever.search(
                query=active_query,
                dataset_name=dataset_name,
                top_k=self.top_k,
            )
            wall_time_ms = (time.perf_counter() - started) * 1000.0
            internal_time_ms = (
                float(results[0].metadata.get("query_time_ms", wall_time_ms))
                if results
                else wall_time_ms
            )
            query_times_ms.append(internal_time_ms)
            retrieved_doc_ids = [result.doc_id for result in results]

            map_scores.append(average_precision(retrieved_doc_ids, relevant_docs))
            recall_scores.append(recall_at_k(retrieved_doc_ids, relevant_docs, k=self.top_k))
            precision_scores.append(precision_at_k(retrieved_doc_ids, relevant_docs, k=10))
            ndcg_scores.append(ndcg_at_k(retrieved_doc_ids, relevance_scores, k=self.top_k))

        evaluated_queries = len(query_items)
        return EvaluationResult(
            dataset_name=dataset_name,
            model_name=model_name,
            map_score=_mean(map_scores),
            recall=_mean(recall_scores),
            precision_at_10=_mean(precision_scores),
            ndcg=_mean(ndcg_scores),
            evaluated_queries=evaluated_queries,
            average_query_time_ms=_mean(query_times_ms),
        )

    def _prepare_query(self, query_text: str) -> str:
        if not self.use_query_refinement:
            return query_text
        return self._query_refinement_service.refine(query_text).refined_query

    def _create_retriever(self, model_name: str):
        if model_name == "tfidf":
            return TFIDFRetriever(max_docs=self.max_docs)
        if model_name == "bm25":
            return BM25Retriever(k1=self.bm25_k1, b=self.bm25_b, max_docs=self.max_docs)
        if model_name == "embedding":
            return EmbeddingRetriever(
                embedding_model_name=self.embedding_model,
                max_docs=self.max_docs,
            )
        if model_name == "embedding_guided_categories":
            embedding = EmbeddingRetriever(
                embedding_model_name=self.embedding_model,
                max_docs=self.max_docs,
            )
            return GuidedCategoryRetriever(
                embedding_retriever=embedding,
                category_weight=self.category_weight,
                candidate_k=self.category_candidate_k,
                top_categories=self.top_categories,
            )
        if model_name == "hybrid_serial":
            return HybridSerialRetriever(
                bm25_retriever=BM25Retriever(
                    k1=self.bm25_k1,
                    b=self.bm25_b,
                    max_docs=self.max_docs,
                ),
                embedding_retriever=EmbeddingRetriever(
                    embedding_model_name=self.embedding_model,
                    max_docs=self.max_docs,
                ),
                max_docs=self.max_docs,
            )
        if model_name == "hybrid_parallel":
            return HybridParallelRetriever(
                tfidf_retriever=TFIDFRetriever(max_docs=self.max_docs),
                bm25_retriever=BM25Retriever(
                    k1=self.bm25_k1,
                    b=self.bm25_b,
                    max_docs=self.max_docs,
                ),
                embedding_retriever=EmbeddingRetriever(
                    embedding_model_name=self.embedding_model,
                    max_docs=self.max_docs,
                ),
                max_docs=self.max_docs,
            )
        raise ValueError(f"Unsupported retrieval model: {model_name}")


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)

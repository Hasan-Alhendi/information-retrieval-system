"""Evaluator wired to corrected hybrid retrievers."""

from app.infrastructure.evaluation.evaluator import RetrievalEvaluator
from app.infrastructure.retrieval.bm25_retriever import BM25Retriever
from app.infrastructure.retrieval.embedding_retriever import EmbeddingRetriever
from app.infrastructure.retrieval.hybrid_serial import HybridSerialRetriever
from app.infrastructure.retrieval.rrf_retriever import ReciprocalRankFusionRetriever
from app.infrastructure.retrieval.tfidf_retriever import TFIDFRetriever


class RetrievalEvaluatorV2(RetrievalEvaluator):
    """Use corrected serial reranking and latency-aware rank fusion."""

    def _create_retriever(self, model_name: str):
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
            return ReciprocalRankFusionRetriever(
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
        return super()._create_retriever(model_name)

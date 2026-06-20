"""Compatibility import for the consolidated Hybrid Serial retriever."""

from app.infrastructure.retrieval.hybrid_serial import HybridSerialRetriever

HybridSerialRetrieverV2 = HybridSerialRetriever

__all__ = ["HybridSerialRetrieverV2"]

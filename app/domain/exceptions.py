"""Domain-specific exceptions."""


class IRSystemError(Exception):
    """Base exception for the IR system."""


class DatasetNotFoundError(IRSystemError):
    """Raised when a requested dataset is not registered or unavailable."""


class IndexNotFoundError(IRSystemError):
    """Raised when a required index is missing."""


class UnsupportedRetrievalModelError(IRSystemError):
    """Raised when a retrieval model is not supported."""

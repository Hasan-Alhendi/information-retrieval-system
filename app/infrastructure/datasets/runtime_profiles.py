"""Safe local runtime defaults for the official datasets."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetRuntimeProfile:
    """Recommended smoke-test and full-index parameters for one dataset."""

    sample_query: str
    smoke_documents: int
    lexical_batch_size: int
    embedding_batch_size: int
    embedding_checkpoint_size: int


RUNTIME_PROFILES: dict[str, DatasetRuntimeProfile] = {
    "quora": DatasetRuntimeProfile(
        sample_query="How can I learn programming?",
        smoke_documents=5000,
        lexical_batch_size=256,
        embedding_batch_size=32,
        embedding_checkpoint_size=5000,
    ),
    "touche2020-v2": DatasetRuntimeProfile(
        sample_query="Should teachers get tenure?",
        smoke_documents=5000,
        lexical_batch_size=128,
        embedding_batch_size=16,
        embedding_checkpoint_size=5000,
    ),
}


def get_runtime_profile(dataset_name: str) -> DatasetRuntimeProfile:
    """Return safe local defaults for an official dataset."""
    try:
        return RUNTIME_PROFILES[dataset_name]
    except KeyError as exc:
        supported = ", ".join(RUNTIME_PROFILES)
        raise ValueError(
            f"No runtime profile for '{dataset_name}'. Supported: {supported}"
        ) from exc

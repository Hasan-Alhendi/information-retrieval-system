"""Dataset registry definitions."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetConfig:
    """Configuration for a supported dataset."""

    name: str
    display_name: str
    source: str
    document_limit: int | None = None


SUPPORTED_DATASETS: dict[str, DatasetConfig] = {
    "msmarco": DatasetConfig(
        name="msmarco",
        display_name="MS MARCO",
        source="beir",
        document_limit=250_000,
    ),
    "nq": DatasetConfig(
        name="nq",
        display_name="Natural Questions",
        source="beir",
        document_limit=250_000,
    ),
}


def get_dataset_config(dataset_name: str) -> DatasetConfig:
    """Return dataset configuration by name."""
    try:
        return SUPPORTED_DATASETS[dataset_name]
    except KeyError as exc:
        supported = ", ".join(SUPPORTED_DATASETS)
        raise ValueError(f"Unsupported dataset '{dataset_name}'. Supported: {supported}") from exc

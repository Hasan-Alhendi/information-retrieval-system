"""Dataset registry definitions."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetConfig:
    """Configuration for a supported dataset."""

    name: str
    display_name: str
    source: str
    document_limit: int | None = None
    external_id: str | None = None
    task_type: str | None = None
    processing_profile: str = "default"


SUPPORTED_DATASETS: dict[str, DatasetConfig] = {
    "quora": DatasetConfig(
        name="quora",
        display_name="Quora",
        source="beir",
        document_limit=250_000,
        task_type="duplicate-question retrieval",
        processing_profile="question",
    ),
    "nq": DatasetConfig(
        name="nq",
        display_name="Natural Questions",
        source="beir",
        document_limit=250_000,
        task_type="question-answer passage retrieval",
        processing_profile="passage",
    ),
}


EXPERIMENTAL_DATASETS: dict[str, DatasetConfig] = {
    "touche2020-v2": DatasetConfig(
        name="touche2020-v2",
        display_name="Webis-Touché 2020 v2",
        source="ir_datasets",
        external_id="beir/webis-touche2020/v2",
        task_type="argument retrieval",
        processing_profile="argument",
    ),
}


PRIMARY_REPORT_DATASETS = ("quora", "nq")


def get_dataset_config(dataset_name: str, *, include_experimental: bool = False) -> DatasetConfig:
    """Return dataset configuration by name."""
    datasets = dict(SUPPORTED_DATASETS)
    if include_experimental:
        datasets.update(EXPERIMENTAL_DATASETS)

    try:
        return datasets[dataset_name]
    except KeyError as exc:
        supported = ", ".join(datasets)
        raise ValueError(f"Unsupported dataset '{dataset_name}'. Supported: {supported}") from exc

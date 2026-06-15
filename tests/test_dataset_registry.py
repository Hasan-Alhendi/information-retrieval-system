"""Dataset registry tests."""

from app.infrastructure.datasets.dataset_registry import (
    EXPERIMENTAL_DATASETS,
    SUPPORTED_DATASETS,
    get_dataset_config,
)


def test_touche_is_experimental_until_validation() -> None:
    assert "touche2020-v2" in EXPERIMENTAL_DATASETS
    assert "touche2020-v2" not in SUPPORTED_DATASETS


def test_touche_external_id_and_profile() -> None:
    config = get_dataset_config("touche2020-v2", include_experimental=True)

    assert config.external_id == "beir/webis-touche2020/v2"
    assert config.task_type == "argument retrieval"
    assert config.processing_profile == "argument"

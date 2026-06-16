"""Dataset registry tests."""

from app.infrastructure.datasets.dataset_registry import (
    EXPERIMENTAL_DATASETS,
    PRIMARY_REPORT_DATASETS,
    SUPPORTED_DATASETS,
    get_dataset_config,
)


def test_touche_is_an_official_dataset() -> None:
    assert "touche2020-v2" in SUPPORTED_DATASETS
    assert "touche2020-v2" not in EXPERIMENTAL_DATASETS
    assert PRIMARY_REPORT_DATASETS == ("quora", "touche2020-v2")


def test_touche_external_id_and_profile() -> None:
    config = get_dataset_config("touche2020-v2")

    assert config.external_id == "beir/webis-touche2020/v2"
    assert config.task_type == "argument retrieval"
    assert config.processing_profile == "argument"


def test_nq_is_retained_only_as_legacy_configuration() -> None:
    assert "nq" in EXPERIMENTAL_DATASETS
    assert "nq" not in SUPPORTED_DATASETS

from pathlib import Path
import tomli
from pydantic import ValidationError
import pytest

from ez_openmmlab.utils.toml_config import (
    UserConfig,
    ModelSection,
    DataSection,
    TrainingSection,
    save_user_config,
    load_user_config,
)


@pytest.fixture
def sample_config() -> UserConfig:
    """Returns a sample, valid UserConfig object."""
    return UserConfig(
        model=ModelSection(name="rtmdet_tiny", num_classes=2, load_from=None),
        data=DataSection(
            root="tests/dummy_data/crayfish",
            train_ann="annotations/train.json",
            train_img="images/",
            val_ann="annotations/val.json",
            val_img="images/",
            classes=["crayfish", "scallop"],
        ),
        training=TrainingSection(
            epochs=2,
            batch_size=4,
            learning_rate=0.0001,
            device="cpu",
            work_dir="output_test",
        ),
    )


def test_save_and_load_roundtrip(tmp_path: Path, sample_config: UserConfig):
    """
    Tests that saving a config and loading it back results in the same object.
    """
    config_path = tmp_path / "config.toml"

    # Save and reload
    save_user_config(sample_config, config_path)
    loaded_config = load_user_config(config_path)

    # Should be identical
    assert loaded_config == sample_config


def test_generated_toml_content(tmp_path: Path, sample_config: UserConfig):
    """
    Verifies that the generated TOML has the correct structure and values.
    """
    config_path = tmp_path / "config.toml"
    save_user_config(sample_config, config_path)

    with open(config_path, "rb") as f:
        data = tomli.load(f)

    # Check sections
    assert "model" in data
    assert "data" in data
    assert "training" in data

    # Check specific values
    assert data["model"]["name"] == "rtmdet_tiny"
    assert data["model"]["num_classes"] == 2
    assert "load_from" not in data["model"]  # None should be excluded

    assert data["data"]["root"] == "tests/dummy_data/crayfish"
    assert data["data"]["classes"] == ["crayfish", "scallop"]

    assert data["training"]["epochs"] == 2
    assert data["training"]["device"] == "cpu"
    assert data["training"]["work_dir"] == "output_test"


def test_missing_required_field_raises_error(tmp_path: Path):
    """
    Ensures that loading a TOML with missing required fields raises a Pydantic error.
    """
    bad_toml_content = """
    [model]
    name = "rtmdet_tiny"
    # num_classes is missing

    [data]
    root = "some/path"

    [training]
    epochs = 10
    """
    config_path = tmp_path / "bad_config.toml"
    config_path.write_text(bad_toml_content)

    with pytest.raises(ValidationError):
        load_user_config(config_path)


def test_incorrect_type_raises_error(tmp_path: Path):
    """
    Ensures that loading a TOML with incorrect data types raises a Pydantic error.
    """
    bad_toml_content = """
    [model]
    name = "rtmdet_tiny"
    num_classes = "eighty"  # Should be an integer

    [data]
    root = "some/path"

    [training]
    epochs = 10
    """
    config_path = tmp_path / "bad_config.toml"
    config_path.write_text(bad_toml_content)

    with pytest.raises(ValidationError):
        load_user_config(config_path)

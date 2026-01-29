import pytest
from pathlib import Path
from pydantic import ValidationError
from ez_mmdetection.utils.toml_config import UserConfig, TrainingSection, ModelSection, DataSection
from ez_mmdetection.schemas.dataset import DatasetConfig, SplitConfig
from ez_mmdetection.schemas.model import ModelName

def test_training_section_defaults():
    """Test TrainingSection default values."""
    ts = TrainingSection()
    assert ts.epochs == 100
    assert ts.batch_size == 8
    assert ts.num_workers == 2
    assert ts.learning_rate == 0.001
    assert ts.device == "cuda"
    assert ts.work_dir == "./runs/train"
    assert ts.log_level == "INFO"
    assert ts.enable_tensorboard is True
    assert ts.amp is True

def test_training_section_constraints():
    """Test TrainingSection validation constraints."""
    with pytest.raises(ValidationError):
        TrainingSection(epochs=0)
    with pytest.raises(ValidationError):
        TrainingSection(batch_size=-1)
    with pytest.raises(ValidationError):
        TrainingSection(num_workers=-1)
    with pytest.raises(ValidationError):
        TrainingSection(learning_rate=-0.1)

def test_user_config_validation():
    """Test UserConfig validation with valid and invalid data."""
    valid_data = {
        "model": {"name": "rtmdet_tiny", "num_classes": 80},
        "data": {"root": "data/coco"},
        "training": {"epochs": 50}
    }
    config = UserConfig(**valid_data)
    assert config.model.name == ModelName.RTM_DET_TINY
    assert config.training.epochs == 50

    invalid_data = {
        "model": {"name": "invalid_model", "num_classes": 80},
        "data": {"root": "data/coco"},
        "training": {"epochs": 50}
    }
    with pytest.raises(ValidationError):
        UserConfig(**invalid_data)

def test_dataset_config_validation():
    """Test DatasetConfig validation."""
    valid_data = {
        "data_root": "datasets/my_data",
        "train": {"ann_file": "train.json", "img_dir": "train/"},
        "val": {"ann_file": "val.json", "img_dir": "val/"}
    }
    ds = DatasetConfig(**valid_data)
    assert ds.data_root == Path("datasets/my_data")
    assert ds.train.ann_file == "train.json"

    invalid_data = {
        "data_root": "datasets/my_data",
        "train": {"ann_file": "train.json"} # Missing img_dir
    }
    with pytest.raises(ValidationError):
        DatasetConfig(**invalid_data)

def test_model_name_enum():
    """Test ModelName enum values."""
    assert ModelName.RTM_DET_TINY == "rtmdet_tiny"
    assert "rtmdet_tiny" in [m.value for m in ModelName]
    with pytest.raises(ValueError):
        ModelName("invalid_model")

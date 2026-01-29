import pytest
from unittest.mock import MagicMock
from mmengine.config import Config, ConfigDict
from ez_openmmlab.core.handlers import DataloaderHandler, RuntimeHandler
from ez_openmmlab.utils.toml_config import UserConfig, DataSection, TrainingSection, ModelSection

@pytest.fixture
def mock_user_config():
    return UserConfig(
        model=ModelSection(name="rtmdet_tiny", num_classes=2),
        data=DataSection(
            root="data/coco",
            train_ann="annotations/train.json",
            train_img="train2017",
            val_ann="annotations/val.json",
            val_img="val2017",
            classes=["cat", "dog"]
        ),
        training=TrainingSection(
            epochs=10,
            batch_size=4,
            num_workers=2,
            learning_rate=0.01,
            amp=True,
            enable_tensorboard=True
        )
    )

def test_dataloader_handler_applies_paths_and_params(mock_user_config):
    """Test that DataloaderHandler correctly sets paths and parameters."""
    cfg = Config(dict(
        train_dataloader=dict(dataset=dict()),
        val_dataloader=dict(dataset=dict())
    ))
    
    handler = DataloaderHandler()
    handler.apply(cfg, mock_user_config)
    
    assert cfg.data_root == "data/coco"
    assert cfg.train_dataloader.num_workers == 2
    assert cfg.train_dataloader.dataset.ann_file == "data/coco/annotations/train.json"
    assert cfg.train_dataloader.dataset.metainfo.classes == ["cat", "dog"]

def test_dataloader_handler_no_classes(mock_user_config):
    """Test handler when classes list is None."""
    mock_user_config.data.classes = None
    cfg = Config(dict(
        train_dataloader=dict(dataset=dict())
    ))
    
    handler = DataloaderHandler()
    handler.apply(cfg, mock_user_config)
    
    assert "metainfo" not in cfg.train_dataloader.dataset
    assert "metainfo" not in cfg

def test_runtime_handler_applies_settings(mock_user_config):
    """Test that RuntimeHandler correctly sets optimizer, visualizer, and runtime params."""
    cfg = Config(dict(
        train_cfg=dict(),
        optim_wrapper=dict(optimizer=dict())
    ))
    
    handler = RuntimeHandler()
    handler.apply(cfg, mock_user_config)
    
    assert cfg.work_dir == "./runs/train"
    assert cfg.train_cfg.max_epochs == 10
    assert cfg.optim_wrapper.type == "AmpOptimWrapper"
    
    # Check TensorBoard
    backends = [b['type'] for b in cfg.visualizer.vis_backends]
    assert "TensorboardVisBackend" in backends

def test_runtime_handler_disabled_features(mock_user_config):
    """Test handler when AMP and TensorBoard are disabled."""
    mock_user_config.training.amp = False
    mock_user_config.training.enable_tensorboard = False
    cfg = Config(dict(
        train_cfg=dict(),
        optim_wrapper=dict(optimizer=dict())
    ))
    
    handler = RuntimeHandler()
    handler.apply(cfg, mock_user_config)
    
    assert cfg.optim_wrapper.type == "OptimWrapper"
    assert "visualizer" not in cfg

def test_runtime_handler_missing_optim_wrapper(mock_user_config):
    """Test that handler doesn't crash if optim_wrapper is missing."""
    cfg = Config(dict(train_cfg=dict()))
    
    handler = RuntimeHandler()
    handler.apply(cfg, mock_user_config) # Should not raise
    
    assert cfg.work_dir == "./runs/train"
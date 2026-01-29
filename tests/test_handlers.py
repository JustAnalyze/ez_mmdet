import pytest
from unittest.mock import MagicMock
from mmengine.config import Config
from ez_mmdetection.core.handlers import DataloaderHandler, RuntimeHandler
from ez_mmdetection.utils.toml_config import UserConfig, DataSection, TrainingSection, ModelSection

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
            amp=True
        )
    )

def test_dataloader_handler_applies_paths_and_params(mock_user_config):
    """Test that DataloaderHandler correctly sets paths and parameters."""
    # Setup mock config structure
    cfg = Config()
    cfg.train_dataloader = MagicMock()
    cfg.val_dataloader = MagicMock()
    cfg.test_dataloader = MagicMock()
    
    # Initialize handler and apply
    handler = DataloaderHandler()
    handler.apply(cfg, mock_user_config)
    
    # Assert global data_root
    assert cfg.data_root == "data/coco"
    
    # Assert train dataloader
    train_dl = cfg.train_dataloader
    assert train_dl.dataset.data_root == ""
    assert train_dl.batch_size == 4
    assert train_dl.num_workers == 2
    assert train_dl.dataset.ann_file == "data/coco/annotations/train.json"
    assert train_dl.dataset.data_prefix == {"img": "data/coco/train2017"}
    assert train_dl.dataset.metainfo == {"classes": ["cat", "dog"]}
    
    # Assert val dataloader
    val_dl = cfg.val_dataloader
    assert val_dl.dataset.data_root == ""
    assert val_dl.batch_size == 4
    assert val_dl.num_workers == 2
    assert val_dl.dataset.ann_file == "data/coco/annotations/val.json"
    assert val_dl.dataset.data_prefix == {"img": "data/coco/val2017"}
    assert val_dl.dataset.metainfo == {"classes": ["cat", "dog"]}

def test_dataloader_handler_handles_missing_dataloaders(mock_user_config):
    """Test that handler doesn't crash if dataloaders are missing from config."""
    cfg = Config()
    # No dataloaders defined
    
    handler = DataloaderHandler()
    handler.apply(cfg, mock_user_config)
    
    assert cfg.data_root == "data/coco"

def test_runtime_handler_applies_settings(mock_user_config):
    """Test that RuntimeHandler correctly sets optimizer, visualizer, and runtime params."""
    cfg = Config()
    cfg.train_cfg = MagicMock()
    cfg.optim_wrapper = MagicMock()
    # Mocking dictionary access for optim_wrapper
    cfg.optim_wrapper.optimizer = MagicMock()
    
    handler = RuntimeHandler()
    handler.apply(cfg, mock_user_config)
    
    assert cfg.work_dir == "./runs/train"
    assert cfg.train_cfg.max_epochs == 10
    assert cfg.log_level == "INFO"
    assert cfg.optim_wrapper.optimizer.lr == 0.01
    
    # Check AMP
    assert cfg.optim_wrapper.type == "AmpOptimWrapper"
    assert cfg.optim_wrapper.loss_scale == "dynamic"
    
    # Check TensorBoard
    assert hasattr(cfg, "visualizer")
    backends = [b['type'] for b in cfg.visualizer['vis_backends']]
    assert "LocalVisBackend" in backends
    assert "TensorboardVisBackend" in backends
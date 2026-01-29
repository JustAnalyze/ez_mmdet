import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from ez_mmdetection.core.config_loader import ConfigLoader
from ez_mmdetection.utils.download import ensure_model_checkpoint, MODEL_URLS
from ez_mmdetection.schemas.model import ModelName

@pytest.fixture
def mock_config_root(tmp_path):
    """Creates a mock mmdetection/configs directory structure."""
    config_root = tmp_path / "libs" / "mmdetection" / "configs"
    config_root.mkdir(parents=True)
    return config_root

def test_config_loader_init_validation(tmp_path):
    """Test that ConfigLoader validates the config root existence."""
    with patch("pathlib.Path.cwd", return_value=tmp_path):
        # Should raise FileNotFoundError if libs/mmdetection/configs doesn't exist
        with pytest.raises(FileNotFoundError):
            ConfigLoader()

def test_config_loader_get_config_path_success(mock_config_root, tmp_path):
    """Test successful config path resolution."""
    # Create a dummy config file
    config_file = mock_config_root / "rtmdet" / "rtmdet_tiny_8xb32-300e_coco.py"
    config_file.parent.mkdir(parents=True)
    config_file.touch()

    with patch("pathlib.Path.cwd", return_value=tmp_path):
        loader = ConfigLoader()
        path = loader.get_config_path("rtmdet_tiny")
        assert path == config_file

def test_config_loader_invalid_model(mock_config_root, tmp_path):
    """Test ConfigLoader with an unsupported model name."""
    with patch("pathlib.Path.cwd", return_value=tmp_path):
        loader = ConfigLoader()
        with pytest.raises(ValueError, match="not found in internal map"):
            loader.get_config_path("invalid_model")

def test_config_loader_missing_file(mock_config_root, tmp_path):
    """Test ConfigLoader when the mapped file is missing on disk."""
    # We DON'T create the file here
    with patch("pathlib.Path.cwd", return_value=tmp_path):
        loader = ConfigLoader()
        with pytest.raises(FileNotFoundError, match="not found at"):
            loader.get_config_path("rtmdet_tiny")

@patch("ez_mmdetection.utils.download.download_checkpoint")
def test_ensure_model_checkpoint_existing(mock_download, tmp_path):
    """Test that ensure_model_checkpoint returns path if file exists."""
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    checkpoint_file = checkpoint_dir / "rtmdet_tiny.pth"
    checkpoint_file.touch()

    with patch("pathlib.Path.cwd", return_value=tmp_path):
        path = ensure_model_checkpoint("rtmdet_tiny")
        assert path == checkpoint_file
        mock_download.assert_not_called()

@patch("ez_mmdetection.utils.download.download_checkpoint")
def test_ensure_model_checkpoint_download_trigger(mock_download, tmp_path):
    """Test that ensure_model_checkpoint triggers download if file is missing."""
    with patch("pathlib.Path.cwd", return_value=tmp_path):
        path = ensure_model_checkpoint("rtmdet_tiny")
        expected_path = tmp_path / "checkpoints" / "rtmdet_tiny.pth"
        assert path == expected_path
        mock_download.assert_called_once_with(MODEL_URLS["rtmdet_tiny"], expected_path)

def test_ensure_model_checkpoint_missing_url(tmp_path):
    """Test ensure_model_checkpoint when model has no URL and path is missing."""
    with patch("pathlib.Path.cwd", return_value=tmp_path):
        # Case 1: Custom path provided, file missing -> raise FileNotFoundError
        with pytest.raises(FileNotFoundError):
            ensure_model_checkpoint("unknown_model", checkpoint_path="missing.pth")
        
        # Case 2: No path provided, unknown model -> return path but log warning (non-fatal)
        path = ensure_model_checkpoint("unknown_model")
        assert path == tmp_path / "checkpoints" / "unknown_model.pth"

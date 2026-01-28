import pytest
from typer.testing import CliRunner
from ez_mmdetection.cli import app

runner = CliRunner()

def test_cli_help():
    """Test that the CLI provides help output."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Usage:" in result.stdout

def test_train_command_requires_dataset_config():
    """Test that the train command fails if dataset_config_path is missing."""
    result = runner.invoke(app, ["train"])
    assert result.exit_code != 0
    assert "Missing argument 'DATASET_CONFIG_PATH'" in result.output

from unittest.mock import MagicMock, patch

def test_train_command_calls_detector_train(tmp_path):
    """Test that the train command initializes RTMDet and calls its train method."""
    dataset_config = tmp_path / "dataset.toml"
    dataset_config.write_text("root = '.'")
    
    with patch("ez_mmdetection.cli.RTMDet") as mock_detector_cls:
        mock_detector_instance = MagicMock()
        mock_detector_cls.return_value = mock_detector_instance
        
        result = runner.invoke(app, ["train", str(dataset_config), "--epochs", "10", "--batch-size", "4"])
        
        assert result.exit_code == 0
        mock_detector_cls.assert_called_once()
        mock_detector_instance.train.assert_called_once()
        _, kwargs = mock_detector_instance.train.call_args
        assert kwargs["dataset_config_path"] == dataset_config
        assert kwargs["epochs"] == 10
        assert kwargs["batch_size"] == 4

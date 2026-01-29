import pytest
from typer.testing import CliRunner
from ez_openmmlab.cli import app
from unittest.mock import MagicMock, patch

runner = CliRunner()

def test_cli_help():
    """Test that the CLI provides help output."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Usage:" in result.stdout

def test_train_command_requires_model_name():
    """Test that the train command fails if model_name is missing."""
    result = runner.invoke(app, ["train"])
    assert result.exit_code != 0
    assert "Missing argument" in result.output
    assert "model_name" in result.output.lower()

def test_train_command_requires_dataset_config():
    """Test that the train command fails if dataset_config_path is missing."""
    result = runner.invoke(app, ["train", "rtmdet_tiny"])
    assert result.exit_code != 0
    assert "Missing argument 'DATASET_CONFIG_PATH'" in result.output

def test_predict_command_calls_detector_predict(tmp_path):
    """Test that the predict command initializes RTMDet and calls its predict method."""
    checkpoint = tmp_path / "best.pth"
    checkpoint.touch()
    image = tmp_path / "demo.jpg"
    image.touch()
    
    with patch("ez_openmmlab.cli.RTMDet") as mock_detector_cls:
        mock_detector_instance = MagicMock()
        mock_detector_cls.return_value = mock_detector_instance
        
        result = runner.invoke(app, ["predict", "rtmdet_tiny", str(checkpoint), str(image), "--out-dir", "output"])
        
        assert result.exit_code == 0
        mock_detector_cls.assert_called_once()
        mock_detector_instance.predict.assert_called_once()
        _, kwargs = mock_detector_instance.predict.call_args
        assert kwargs["checkpoint_path"] == checkpoint
        assert kwargs["image_path"] == image
        assert kwargs["out_dir"] == "output"

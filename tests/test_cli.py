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

def test_predict_command_requires_arguments():
    """Test that the predict command fails if required arguments are missing."""
    result = runner.invoke(app, ["predict"])
    assert result.exit_code != 0
    assert "Missing argument" in result.output

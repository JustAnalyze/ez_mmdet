import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from ez_openmmlab import RTMDet
from ez_openmmlab.schemas.model import ModelName

@patch("ez_openmmlab.core.detection.DetInferencer")
@patch("ez_openmmlab.core.base.ensure_model_checkpoint")
def test_predict_handles_none_out_dir(mock_ensure, mock_inferencer_cls):
    """
    Regression test: Verifies that predict() handles out_dir=None correctly
    by passing an empty string to DetInferencer (preventing TypeError).
    """
    mock_ensure.return_value = Path("dummy.pth")
    mock_inferencer_instance = MagicMock()
    mock_inferencer_instance.return_value = {"predictions": []}
    mock_inferencer_cls.return_value = mock_inferencer_instance
    
    detector = RTMDet(ModelName.RTM_DET_TINY)
    
    # Call with out_dir=None (default)
    detector.predict(image_path="dummy.jpg")
    
    # Verify DetInferencer was called with out_dir=""
    mock_inferencer_instance.assert_called_once()
    args, kwargs = mock_inferencer_instance.call_args
    assert kwargs["out_dir"] == ""

@patch("ez_openmmlab.core.detection.DetInferencer")
@patch("ez_openmmlab.core.base.ensure_model_checkpoint")
def test_predict_resolves_config_path(mock_ensure, mock_inferencer_cls):
    """
    Regression test: Verifies that predict() resolves model name to full path
    instead of passing the short name to DetInferencer.
    """
    mock_ensure.return_value = Path("dummy.pth")
    mock_inferencer_instance = MagicMock()
    mock_inferencer_instance.return_value = {"predictions": []}
    mock_inferencer_cls.return_value = mock_inferencer_instance
    
    detector = RTMDet(ModelName.RTM_DET_TINY)
    detector.predict(image_path="dummy.jpg")
    
    # Verify DetInferencer was initialized with a full path (.py)
    mock_inferencer_cls.assert_called_once()
    _, kwargs = mock_inferencer_cls.call_args
    model_arg = kwargs["model"]
    assert model_arg.endswith(".py")
    assert "libs/mmdetection/configs" in model_arg
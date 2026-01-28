def test_predict_initializes_inferencer_and_calls_it():
    # ... (existing test code)
    pass # I'll actually rewrite the file to keep it clean

import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
from ez_mmdetection import RTMDet
from ez_mmdetection.schemas.inference import InferenceResult

def test_predict_initializes_inferencer_and_calls_it():
    """
    Test that predict() initializes DetInferencer with correct params
    and calls it with the provided image.
    """
    model_name = "rtmdet_tiny"
    checkpoint_path = "checkpoints/best.pth"
    image_path = "demo.jpg"
    
    # Mock result
    mock_result = {"predictions": [{"labels": [0], "scores": [0.9], "bboxes": [[10, 10, 100, 100]]}]}
    
    with patch("ez_mmdetection.core.base.DetInferencer") as mock_inferencer_cls:
        # Configure mock inferencer instance
        mock_inferencer_instance = MagicMock()
        mock_inferencer_instance.return_value = mock_result
        mock_inferencer_cls.return_value = mock_inferencer_instance
        
        detector = RTMDet(model_name=model_name)
        
        results = detector.predict(image_path=image_path, checkpoint_path=checkpoint_path)
        
        # Verify DetInferencer was initialized correctly
        mock_inferencer_cls.assert_called_once()
        _, kwargs = mock_inferencer_cls.call_args
        assert kwargs["model"] == model_name
        assert kwargs["weights"] == checkpoint_path
        
        # Verify inferencer was called with the image
        mock_inferencer_instance.assert_called_once_with(str(image_path), out_dir=None, show=False)
        
        # Verify results is an InferenceResult object
        assert isinstance(results, InferenceResult)
        assert len(results.predictions) == 1
        assert results.predictions[0].label == 0
        assert results.predictions[0].score == 0.9
        assert results.predictions[0].bbox == [10, 10, 100, 100]


def test_predict_with_out_dir_creates_directory(tmp_path):
    """
    Test that predict() passes out_dir to the inferencer.
    """
    model_name = "rtmdet_tiny"
    checkpoint_path = "checkpoints/best.pth"
    image_path = "demo.jpg"
    out_dir = tmp_path / "results"
    
    with patch("ez_mmdetection.core.base.DetInferencer") as mock_inferencer_cls:
        mock_inferencer_instance = MagicMock()
        mock_inferencer_instance.return_value = {"predictions": []}
        mock_inferencer_cls.return_value = mock_inferencer_instance
        
        detector = RTMDet(model_name=model_name)
        detector.predict(image_path=image_path, checkpoint_path=checkpoint_path, out_dir=str(out_dir))
        
        # Verify inferencer was called with the out_dir
        mock_inferencer_instance.assert_called_once_with(str(image_path), out_dir=str(out_dir), show=False)


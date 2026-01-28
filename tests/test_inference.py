import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
from ez_mmdetection import RTMDet

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
        
        # This should fail because predict() is not implemented
        results = detector.predict(image_path=image_path, checkpoint_path=checkpoint_path)
        
        # Verify DetInferencer was initialized correctly
        # DetInferencer(model=model_name, weights=checkpoint_path, device=...)
        mock_inferencer_cls.assert_called_once()
        args, kwargs = mock_inferencer_cls.call_args
        assert kwargs["model"] == model_name
        assert kwargs["weights"] == checkpoint_path
        
        # Verify inferencer was called with the image
        mock_inferencer_instance.assert_called_once_with(image_path, out_dir=None, show=False)
        
        # Verify results
        assert results == mock_result

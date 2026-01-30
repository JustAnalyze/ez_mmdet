import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
from ez_openmmlab import RTMDet
from ez_openmmlab.schemas.inference import InferenceResult


@patch("ez_openmmlab.core.detection.ensure_model_checkpoint")
def test_predict_initializes_inferencer_and_calls_it(mock_ensure):
    """
    Test that predict() initializes DetInferencer with correct params
    and calls it with the provided image.
    """
    model_name = "rtmdet_tiny"
    checkpoint_path = "checkpoints/best.pth"
    image_path = "demo.jpg"
    mock_ensure.return_value = Path(checkpoint_path)

    # Mock result
    mock_result = {
        "predictions": [
            {"labels": [0], "scores": [0.9], "bboxes": [[10, 10, 100, 100]]}
        ]
    }

    with patch("ez_openmmlab.core.detection.DetInferencer") as mock_inferencer_cls:
        # Configure mock inferencer instance
        mock_inferencer_instance = MagicMock()
        mock_inferencer_instance.return_value = mock_result
        mock_inferencer_cls.return_value = mock_inferencer_instance

        # Initialize with checkpoint
        detector = RTMDet(model_name=model_name, checkpoint_path=checkpoint_path)

        # Predict WITHOUT checkpoint_path
        results = detector.predict(image_path=image_path, confidence=0.5)

        # Verify DetInferencer was initialized correctly
        mock_inferencer_cls.assert_called_once()
        _, kwargs = mock_inferencer_cls.call_args

        # We expect the full path here now
        expected_config = str(
            Path.cwd()
            / "libs/mmdetection/configs/rtmdet/rtmdet_tiny_8xb32-300e_coco.py"
        )
        assert kwargs["model"] == expected_config
        assert kwargs["weights"] == checkpoint_path

        # Verify inferencer was called with the image and threshold
        mock_inferencer_instance.assert_called_once_with(
            str(image_path), out_dir="", show=False, pred_score_thr=0.5
        )

        # Verify results is an InferenceResult object
        assert isinstance(results, InferenceResult)


@patch("ez_openmmlab.core.detection.ensure_model_checkpoint")
def test_predict_with_out_dir_creates_directory(mock_ensure, tmp_path):
    """
    Test that predict() passes out_dir to the inferencer.
    """
    model_name = "rtmdet_tiny"
    checkpoint_path = "checkpoints/best.pth"
    image_path = "demo.jpg"
    out_dir = tmp_path / "results"
    mock_ensure.return_value = Path(checkpoint_path)

    with patch("ez_openmmlab.core.detection.DetInferencer") as mock_inferencer_cls:
        mock_inferencer_instance = MagicMock()
        mock_inferencer_instance.return_value = {"predictions": []}
        mock_inferencer_cls.return_value = mock_inferencer_instance

        detector = RTMDet(model_name=model_name, checkpoint_path=checkpoint_path)
        detector.predict(image_path=image_path, out_dir=str(out_dir))

        # Verify inferencer was called with the out_dir
        mock_inferencer_instance.assert_called_once_with(
            str(image_path), out_dir=str(out_dir), show=False, pred_score_thr=0.5
        )


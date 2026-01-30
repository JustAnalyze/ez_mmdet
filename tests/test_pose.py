import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
from ez_openmmlab.models.mmpose import RTMPose
from ez_openmmlab.schemas.inference import PoseInferenceResult, PosePrediction
from ez_openmmlab.schemas.model import ModelName

@patch("pathlib.Path.exists")
@patch("ez_openmmlab.engines.mmpose.MMPoseInferencer")
@patch("ez_openmmlab.core.base.ensure_model_checkpoint")
def test_rtmpose_predict_converts_results(mock_ensure, mock_inferencer_cls, mock_exists):
    """Verifies that RTMPose correctly calls MMPoseInferencer and converts results."""
    mock_ensure.return_value = Path("dummy.pth")
    mock_exists.return_value = True
    
    # Mock raw MMPose result
    # MMPose 1.x inferencer returns results in a specific format
    raw_result = {
        "predictions": [[
            {
                "keypoints": [[100, 100], [200, 200]],
                "keypoint_scores": [0.9, 0.8],
                "bbox": [MagicMock(tolist=lambda: [0, 0, 50, 50])],
                "score": 0.95
            }
        ]]
    }
    
    mock_inferencer_instance = MagicMock()
    mock_inferencer_instance.return_value = iter([raw_result])
    mock_inferencer_cls.return_value = mock_inferencer_instance
    
    model = RTMPose(ModelName.RTM_POSE_TINY)
    result = model.predict("dummy.jpg", device="cpu", bbox_thr=0.4, kpt_thr=0.4)
    
    # Verify inferencer was called with correct thresholds
    mock_inferencer_instance.assert_called_once()
    _, kwargs = mock_inferencer_instance.call_args
    assert kwargs["bbox_thr"] == 0.4
    assert kwargs["kpt_thr"] == 0.4
    
    assert isinstance(result, PoseInferenceResult)
    assert len(result.predictions) == 1
    assert result.predictions[0].score == 0.95
    assert result.predictions[0].keypoints == [[100, 100], [200, 200]]
    assert result.predictions[0].keypoint_scores == [0.9, 0.8]

def test_pose_schema_validation():
    """Test the PoseInferenceResult schema directly."""
    raw_preds = [
        {
            "keypoints": [[10, 10]],
            "keypoint_scores": [1.0],
            "score": 0.9
        }
    ]
    result = PoseInferenceResult.from_mmpose(raw_preds)
    assert len(result.predictions) == 1
    assert result.predictions[0].score == 0.9
    assert result.predictions[0].keypoints == [[10, 10]]

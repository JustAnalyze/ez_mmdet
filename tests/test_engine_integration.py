import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from ez_openmmlab import RTMDet
from ez_openmmlab.schemas.model import ModelName

@pytest.fixture
def dummy_dataset_config(tmp_path):
    """Creates a dummy dataset.toml for integration testing."""
    config_path = tmp_path / "dataset.toml"
    config_path.write_text("""
    data_root = "./dummy_data"
    classes = ["crayfish", "lobster"]
    
    [train]
    ann_file = "annotations/train.json"
    img_dir = "train2017"

    [val]
    ann_file = "annotations/val.json"
    img_dir = "val2017"
    """)
    return config_path

@patch("ez_openmmlab.core.base.Runner")
@patch("ez_openmmlab.core.base.ensure_model_checkpoint")
def test_train_orchestration_and_artifact_creation(mock_ensure, mock_runner, dummy_dataset_config, tmp_path):
    """
    Integration test for EZMMDetector.train:
    Verifies that handlers are called and user_config.toml is created.
    """
    mock_ensure.return_value = Path("dummy.pth")
    work_dir = tmp_path / "runs" / "test_run"
    
    detector = RTMDet(ModelName.RTM_DET_TINY)
    
    # We expect this to run through base config loading, handlers, and save user_config.toml
    # It will stop at mock_runner.from_cfg
    detector.train(
        dataset_config_path=dummy_dataset_config,
        epochs=1,
        work_dir=str(work_dir)
    )
    
    # 1. Verify user_config.toml exists
    saved_config = work_dir / "user_config.toml"
    assert saved_config.exists()
    
    # 2. Verify Runner was called with modified config
    mock_runner.from_cfg.assert_called_once()
    cfg = mock_runner.from_cfg.call_args[0][0]
    
    # Verify some key config overrides were applied (Integration between base.py and handlers)
from ez_openmmlab.schemas.inference import InferenceResult

@patch("ez_openmmlab.engines.mmdet.DetInferencer")
@patch("ez_openmmlab.core.base.ensure_model_checkpoint")
def test_predict_result_conversion(mock_ensure, mock_inferencer_cls):
    """Verifies that predict() returns a structured InferenceResult."""
    mock_ensure.return_value = Path("dummy.pth")
    
    # Mock raw result from DetInferencer
    raw_result = {"predictions": [{"labels": [1], "scores": [0.85], "bboxes": [[0, 0, 10, 10]]}]}
    mock_inferencer_instance = MagicMock()
    mock_inferencer_instance.return_value = raw_result
    mock_inferencer_cls.return_value = mock_inferencer_instance
    
    detector = RTMDet(ModelName.RTM_DET_TINY)
    result = detector.predict(image_path="dummy.jpg")
    
    assert isinstance(result, InferenceResult)
    assert len(result.predictions) == 1
    assert result.predictions[0].label == 1
    assert result.predictions[0].score == 0.85
    assert result.predictions[0].bbox == [0, 0, 10, 10]

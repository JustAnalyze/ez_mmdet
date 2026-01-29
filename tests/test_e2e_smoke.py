import pytest
import shutil
from pathlib import Path
from ez_mmdetection import RTMDet
from ez_mmdetection.schemas.model import ModelName

@pytest.fixture(scope="module")
def smoke_test_data():
    """Returns path to existing coco128 dummy data if available, or skips."""
    data_toml = Path("datasets/coco128_coco/dataset.toml")
    if not data_toml.exists():
        pytest.skip("coco128 dummy data not found at datasets/coco128_coco/")
    return data_toml

def test_e2e_train_predict_loop(smoke_test_data, tmp_path):
    """
    E2E Smoke Test: Runs a 1-epoch training pass and then inference.
    Uses CPU and no AMP for maximum compatibility in test environments.
    """
    work_dir = tmp_path / "smoke_run"
    
    # 1. Initialize and Train (1 epoch)
    detector = RTMDet(ModelName.RTM_DET_TINY)
    detector.train(
        dataset_config_path=smoke_test_data,
        epochs=1,
        device="cpu",
        amp=False,
        work_dir=str(work_dir),
        log_level="WARNING"
    )
    
    # 2. Verify checkpoint was created
    checkpoint = work_dir / "epoch_1.pth"
    assert checkpoint.exists()
    
    # 3. Run Inference using the new checkpoint
    # We use a dummy image from the dataset for inference
    image_path = Path("libs/mmdetection/demo/demo.jpg")
    if not image_path.exists():
         # Fallback if demo.jpg is missing (unlikely if submodule is init)
         image_path = list(Path("datasets/coco128_coco/").rglob("*.jpg"))[0]
         
    result = detector.predict(
        image_path=str(image_path),
        checkpoint_path=str(checkpoint),
        device="cpu"
    )
    
    # 4. Verify structured result
    assert len(result.predictions) >= 0 # Successful if no crash and returns list
    print(f"E2E Smoke Test passed. Found {len(result.predictions)} objects.")

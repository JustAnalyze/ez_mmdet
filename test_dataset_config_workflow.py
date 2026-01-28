import datetime
import sys
import tempfile
from pathlib import Path

from loguru import logger

# Add the src directory to the Python path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from ez_mmdetection import RTMDet


def main():
    """Demonstrates the new dataset-config based training workflow using the RTMDet concrete class."""
    logger.remove()
    logger.add(sys.stdout, level="INFO")
    logger.info("Loguru logger configured.")

    logger.info("\n--- Demonstrating Dataset Config Workflow with RTMDet ---")

    # Define a unique work_dir within the 'runs' folder
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_work_dir = Path(f"./runs/dataset_config_test_{timestamp}")

    # Create a temporary directory for the dummy dataset config
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        dataset_config_path = tmp_path / "dataset.toml"

        # Create a dummy dataset.toml content
        dataset_content = """
        data_root = "./dummy_data"
        classes = ["crayfish", "lobster"]
        
        [train]
        ann_file = "annotations/train.json"
        img_dir = "train2017"

        [val]
        ann_file = "annotations/val.json"
        img_dir = "val2017"
        """
        dataset_config_path.write_text(dataset_content)
        logger.info(f"Created temporary dataset config at: {dataset_config_path}")

        try:
            logger.info("1. Initializing RTMDet.")
            detector = RTMDet(model_name="rtmdet_tiny")

            logger.info(
                f"2. Calling train() with log_level='WARNING'. Output work_dir: {output_work_dir}"
            )
            detector.train(
                dataset_config_path=dataset_config_path,
                epochs=1,
                batch_size=2,
                device="cpu",
                work_dir=str(output_work_dir),
                learning_rate=0.0001,
                log_level="WARNING",
            )

        except Exception as e:
            logger.warning(f"Caught expected exception during training attempt: {e}")
            logger.info(
                "This is expected as we are not running a full MMDetection environment."
            )

    # Verify the generated user_config.toml exists
    saved_config_path = output_work_dir / "user_config.toml"
    if saved_config_path.exists():
        logger.info(f"SUCCESS: user_config.toml found at: {saved_config_path}")
        logger.info("Content of the saved user_config.toml:")
        logger.info("----------------------------------")
        logger.info(saved_config_path.read_text())
        logger.info("----------------------------------")
    else:
        logger.error(f"FAILURE: user_config.toml NOT found at: {saved_config_path}")


if __name__ == "__main__":
    main()

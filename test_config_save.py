import sys
from pathlib import Path
import tempfile
import datetime

from loguru import logger

# Add the src directory to the Python path
project_root = Path(__file__).parent
src_path = project_root / 'src'
sys.path.insert(0, str(src_path))

from ez_mmdetection.core.engine import EZDetector
from ez_mmdetection.utils.toml_config import (
    UserConfig,
    ModelSection,
    DataSection,
    TrainingSection,
    save_user_config,
)


def main():
    """
    Demonstrates saving of the UserConfig to a TOML file during a parameter-based training call.
    The generated config will be in the 'runs/' directory for inspection.
    """
    logger.remove()
    logger.add(sys.stdout, level="INFO")
    logger.info("Loguru logger configured.")

    logger.info("\n--- Demonstrating UserConfig Saving During Parameter-Based Training ---")

    # Define a unique work_dir within the 'runs' folder
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_work_dir = Path(f"./runs/manual_config_save_test_{timestamp}")

    try:
        logger.info("1. Initializing EZDetector.")
        sample_classes = ["crayfish", "lobster"]
        detector = EZDetector(model_name="rtmdet_tiny", classes=sample_classes)
        
        logger.info(f"2. Calling train() with parameters. Output work_dir: {output_work_dir}")
        # This call will construct UserConfig and save it to work_dir
        detector.train(
            data_root="./dummy_data", # Placeholder, won't be used in this test run
            epochs=1,
            batch_size=2,
            device="cpu",
            work_dir=str(output_work_dir),
            learning_rate=0.0001,
        )

    except Exception as e:
        logger.warning(f"Caught expected exception during training attempt: {e}")
        logger.info("This is expected as we are not running a full MMDetection environment. "
                    "The goal is to verify config file creation.")

    logger.info(f"3. Checking for saved user_config.toml in {output_work_dir}")
    saved_config_path = output_work_dir / "user_config.toml"

    if saved_config_path.exists():
        logger.info(f"SUCCESS: user_config.toml found at: {saved_config_path}")
        logger.info("Content of the saved config.toml:")
        logger.info("----------------------------------")
        logger.info(saved_config_path.read_text())
        logger.info("----------------------------------")
    else:
        logger.error(f"FAILURE: user_config.toml NOT found at: {saved_config_path}")


if __name__ == "__main__":
    main()

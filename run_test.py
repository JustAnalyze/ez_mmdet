import sys
from pathlib import Path
import tempfile

from loguru import logger

# Add the src directory to the Python path
project_root = Path(__file__).parent
src_path = project_root / 'src'
sys.path.insert(0, str(src_path))

# Must be imported after sys.path is updated
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
    Initializes logging and demonstrates the new config-file-based training.
    """
    logger.remove()
    logger.add(sys.stdout, level="INFO")
    logger.info("Loguru logger configured.")

    logger.info("\n--- Demonstrating Config-File Based Training Call (will fail) ---")
    
    # Create a temporary directory for the test run
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_dir = Path(tmpdir)
        work_dir = temp_dir / "experiment_1"
        data_root = temp_dir / "dummy_data"
        data_root.mkdir()

        # 1. Create a UserConfig object programmatically
        logger.info("1. Creating UserConfig object...")
        sample_classes = ["cat", "dog"]
        user_config = UserConfig(
            model=ModelSection(
                name="rtmdet_tiny",
                num_classes=len(sample_classes),
            ),
            data=DataSection(
                root=str(data_root),
                classes=sample_classes,
                # Using default ann/img paths for this test
            ),
            training=TrainingSection(
                epochs=1, # Keep it short for a test
                batch_size=2,
                device="cpu", # Assume CPU for testing
                work_dir=str(work_dir),
            ),
        )

        # 2. Save it to a TOML file
        config_path = temp_dir / "test_config.toml"
        logger.info(f"2. Saving config to: {config_path}")
        save_user_config(user_config, config_path)

        # 3. Instantiate detector and run training from the file
        try:
            logger.info("3. Initializing EZDetector and calling train(config_file=...).")
            # Initialize with the class list
            detector = EZDetector(model_name="rtmdet_tiny", classes=sample_classes)
            
            # This will fail at runner.train(), but will show all logs leading up to it.
            detector.train(config_file=config_path)

        except Exception as e:
            # We expect an error because we don't have a real dataset or a full mmdet setup
            # For this test, we are happy if it gets past the config loading and override stage.
            # A real integration test would require mocking the Runner.
            logger.opt(exception=True).warning(f"Caught expected exception during training: {e}")
            logger.info("This is expected as we are not running a full MMDetection environment.")

        # 4. Verify that the user_config.toml was created in the work_dir
        logger.info("4. Verifying that user_config.toml was created in the work_dir...")
        expected_output_config = work_dir / "user_config.toml"
        if expected_output_config.exists():
            logger.info(f"SUCCESS: Found config file at: {expected_output_config}")
        else:
            logger.error(f"FAILURE: Did not find config file at: {expected_output_config}")


if __name__ == "__main__":
    main()



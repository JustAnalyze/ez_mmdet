import sys
from pathlib import Path

from loguru import logger

# Add the src directory to the Python path
# This is a common pattern for running scripts in a project with a src-layout
# In a real application, you would install the package instead.
project_root = Path(__file__).parent
src_path = project_root / 'src'
sys.path.insert(0, str(src_path))


def main():
    """
    Initializes logging and demonstrates the EZDetector initialization.
    """
    # Configure loguru to intercept standard logging calls and set level
    logger.remove()  # Remove default handler to avoid duplicate outputs
    logger.add(sys.stdout, level="INFO")
    logger.info("Loguru logger configured.")

    # This import will fail until the project is properly structured or installed
    try:
        from ez_mmdetection.core.engine import EZDetector
        from ez_mmdetection.core.config_loader import get_config_file
    except ImportError as e:
        logger.error(f"Failed to import ez_mmdetection modules: {e}")
        logger.error("Please ensure you are running this from the project root and the 'src' directory is in the python path.")
        return

    logger.info("--- Demonstrating EZDetector Initialization ---")
    try:
        # This will print the init logs from EZDetector
        detector = EZDetector(model_name="rtmdet_tiny", num_classes=80)
    except Exception as e:
        logger.opt(exception=True).error(f"An error occurred during EZDetector initialization: {e}")


    logger.info("\n--- Demonstrating ConfigLoader ---")
    try:
        # This will trigger all the logs in ConfigLoader
        # Note: This is already called when config_loader is imported,
        # but we call it again to explicitly show the logs.
        config_path = get_config_file("rtmdet_tiny")
        logger.info(f"Successfully retrieved config path: {config_path}")
    except Exception as e:
        logger.opt(exception=True).error(f"An error occurred during config loading: {e}")


    logger.info("\n--- Demonstrating Training Call (will fail) ---")
    try:
        # This will fail at _load_base_config, but will show all logs leading up to it.
        # We need a dummy data_root for Pydantic validation to pass.
        dummy_data_dir = project_root / 'dummy_data'
        dummy_data_dir.mkdir(exist_ok=True)
        detector.train(data_root=dummy_data_dir, epochs=1)
    except NotImplementedError as e:
        logger.warning(f"Caught expected error: {e}")
    except Exception as e:
        logger.opt(exception=True).error(f"An unexpected error occurred during train call: {e}")


if __name__ == "__main__":
    main()


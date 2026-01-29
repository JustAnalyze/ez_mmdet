from pathlib import Path
from loguru import logger
from ez_openmmlab.schemas.model import ModelName

class ConfigLoader:
    """Resolves model names to absolute paths of official MMDetection config files."""

    def __init__(self):
        # Assumes the script is running from the project root where 'libs/mmdetection' exists
        self._project_root = Path.cwd()
        self._config_root = self._project_root / "libs" / "mmdetection" / "configs"
        logger.info(
            f"Initializing ConfigLoader. Config root: {self._config_root}"
        )
        self._validate_root()

    def _validate_root(self):
        """Ensures the config root exists."""
        if not self._config_root.exists():
            logger.error(
                f"Could not find local mmdetection configs at: {self._config_root}"
            )
            raise FileNotFoundError(
                f"Could not find local mmdetection configs at: {self._config_root}\n"
                "Ensure you are running from the project root and 'libs/mmdetection' is initialized."
            )

    def get_config_path(self, model_name: str) -> Path:
        """Resolves a model name to its absolute config path."""
        try:
            model = ModelName(model_name)
            rel_path = model.config_path
        except ValueError:
            logger.error(f"Model '{model_name}' is not supported or recognized.")
            supported = ", ".join([m.value for m in ModelName])
            raise ValueError(
                f"Model '{model_name}' not found in internal map.\n"
                f"Currently supported models: {supported}"
            )

        config_path = self._config_root / rel_path
        
        if not config_path.exists():
            logger.error(f"Config file for '{model_name}' missing at: {config_path}")
            raise FileNotFoundError(
                f"Config file for '{model_name}' not found at: {config_path}\n"
                "Please verify the MMDetection submodule is correctly initialized."
            )

        logger.info(f"Resolved model '{model_name}' to: {config_path}")
        return config_path


# Global instance
_LOADER = ConfigLoader()


def get_config_file(model_name: str) -> Path:
    return _LOADER.get_config_path(model_name)
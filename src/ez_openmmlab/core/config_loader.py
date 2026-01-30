from pathlib import Path
from loguru import logger
from ez_openmmlab.schemas.model import ModelName

class ConfigLoader:
    """Resolves model names to absolute paths of official OpenMMLab config files."""

    def __init__(self):
        # Assumes the script is running from the project root
        self._project_root = Path.cwd()
        self._mmdet_config_root = self._project_root / "libs" / "mmdetection" / "configs"
        self._mmpose_config_root = self._project_root / "libs" / "mmpose" / "configs"
        
        logger.info(
            f"Initializing ConfigLoader. Project root: {self._project_root}"
        )
        self._validate_root()

    def _validate_root(self):
        """Ensures at least one config root exists."""
        if not self._mmdet_config_root.exists() and not self._mmpose_config_root.exists():
            logger.error(
                f"Could not find local OpenMMLab configs."
            )
            raise FileNotFoundError(
                "Could not find local mmdetection or mmpose configs.\n"
                "Ensure submodules are initialized."
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

        # Determine which library root to use
        if "rtmpose" in model_name:
            config_root = self._mmpose_config_root
        else:
            config_root = self._mmdet_config_root

        config_path = config_root / rel_path
        
        if not config_path.exists():
            logger.error(f"Config file for '{model_name}' missing at: {config_path}")
            raise FileNotFoundError(
                f"Config file for '{model_name}' not found at: {config_path}\n"
                "Please verify the appropriate OpenMMLab submodule is correctly initialized."
            )

        logger.info(f"Resolved model '{model_name}' to: {config_path}")
        return config_path


# Global instance
_LOADER = ConfigLoader()


def get_config_file(model_name: str) -> Path:
    return _LOADER.get_config_path(model_name)

from pathlib import Path
from typing import Dict

from loguru import logger

from ez_mmdetection.schemas.model import ModelName

# Strict mapping of supported model names to their relative config paths within libs/mmdetection/configs/
MODEL_CONFIG_MAP: Dict[str, str] = {
    # Bounding Box detection
    ModelName.RTM_DET_TINY.value: "rtmdet/rtmdet_tiny_8xb32-300e_coco.py",
    ModelName.RTM_DET_S.value: "rtmdet/rtmdet_s_8xb32-300e_coco.py",
    ModelName.RTM_DET_M.value: "rtmdet/rtmdet_m_8xb32-300e_coco.py",
    ModelName.RTM_DET_L.value: "rtmdet/rtmdet_l_8xb32-300e_coco.py",
    ModelName.RTM_DET_X.value: "rtmdet/rtmdet_x_8xb32-300e_coco.py",
    # Instance Segmentation
    ModelName.RTM_DET_INS_TINY.value: "rtmdet/rtmdet-ins_tiny_8xb32-300e_coco.py",
    ModelName.RTM_DET_INS_S.value: "rtmdet/rtmdet-ins_s_8xb32-300e_coco.py",
    ModelName.RTM_DET_INS_M.value: "rtmdet/rtmdet-ins_m_8xb32-300e_coco.py",
    ModelName.RTM_DET_INS_L.value: "rtmdet/rtmdet-ins_l_8xb32-300e_coco.py",
    ModelName.RTM_DET_INS_X.value: "rtmdet/rtmdet-ins_x_8xb16-300e_coco.py",
}


class ConfigLoader:
    """Resolves model names to absolute paths of official MMDetection config files."""

    def __init__(self):
        # Assumes the script is running from the project root where 'libs/mmdetection' exists
        self._project_root = Path.cwd()
        self._config_root = self._project_root / "libs" / "mmdetection" / "configs"
        logger.info(f"Initializing ConfigLoader. Config root: {self._config_root}")
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
        rel_path = MODEL_CONFIG_MAP.get(model_name)

        if not rel_path:
            logger.error(f"Model '{model_name}' is not supported or recognized.")
            supported = ", ".join(list(MODEL_CONFIG_MAP.keys()))
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


# Global instance
_LOADER = ConfigLoader()


def get_config_file(model_name: str) -> Path:
    return _LOADER.get_config_path(model_name)

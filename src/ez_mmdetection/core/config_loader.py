from pathlib import Path
from typing import Dict

from loguru import logger


class ConfigLoader:
    """Finds MMDetection config files within the local frozen mmdetection repo."""

    def __init__(self):
        # Assumes the script is running from the project root where 'mmdetection' folder exists
        self._project_root = Path.cwd()
        self._config_root = self._project_root / "libs" / "mmdetection" / "configs"
        self._config_map: Dict[str, Path] = {}
        self._scanned = False
        logger.info(
            f"Initializing ConfigLoader. Project root: {self._project_root}, Config root: {self._config_root}"
        )

        self._validate_root()

    def _validate_root(self):
        """Ensures the frozen repo exists."""
        if not self._config_root.exists():
            logger.error(
                f"Could not find local mmdetection configs at: {self._config_root}"
            )
            raise FileNotFoundError(
                f"Could not find local mmdetection configs at: {self._config_root}\n"
                "Ensure you are running from the project root and 'mmdetection' is cloned there."
            )
        logger.info(f"MMDetection config root validated: {self._config_root}")

    def get_config_path(self, model_name: str) -> Path:
        """Finds the config path for a given model name (e.g., 'rtmdet_tiny')."""
        logger.info(f"Attempting to get config path for model: '{model_name}'")
        if not self._scanned:
            self._scan_configs()

        # 1. Exact match
        if model_name in self._config_map:
            match = self._config_map[model_name]
            logger.info(f"Exact match found for '{model_name}': {match}")
            return match

        # 2. Fuzzy match
        matches = [p for name, p in self._config_map.items() if model_name in name]
        logger.debug(f"Found {len(matches)} fuzzy matches for '{model_name}'.")

        if not matches:
            logger.error(f"Model '{model_name}' not found in {self._config_root}.")
            raise ValueError(
                f"Model '{model_name}' not found in {self._config_root}.\n"
                "Available models (partial list): "
                + ", ".join(list(self._config_map.keys())[:5])
            )

        # Prefer short names (base configs) and 'coco'
        best_match = sorted(matches, key=lambda p: (len(str(p)), "coco" not in str(p)))[
            0
        ]
        logger.info(f"Best fuzzy match for '{model_name}' is: {best_match}")
        return best_match

    def _scan_configs(self) -> None:
        """Indexes all .py config files in the local directory."""
        logger.info(f"Scanning for config files in {self._config_root}...")
        for path in self._config_root.rglob("*.py"):
            # Skip base configs and other non-model files
            if "_base_" in path.parts:
                continue

            logger.debug(f"Found config: {path.stem} -> {path}")
            self._config_map[path.stem] = path

        self._scanned = True
        logger.info(
            f"Finished scanning. Found {len(self._config_map)} model config files."
        )


# Global instance
_LOADER = ConfigLoader()


def get_config_file(model_name: str) -> Path:
    return _LOADER.get_config_path(model_name)

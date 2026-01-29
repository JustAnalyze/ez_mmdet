from loguru import logger

from ez_openmmlab.core.detection import EZMMDetector
from ez_openmmlab.utils.toml_config import UserConfig


class RTMDet(EZMMDetector):
    """RTMDet implementation of the EZDetector.

    Handles the specific configuration for RTMDet models (one-stage detectors).
    """

    def _configure_model_specifics(self, config: UserConfig) -> None:
        """Overrides the bbox_head num_classes for RTMDet."""
        if not self._cfg:
            raise RuntimeError("Config not loaded before configuring specifics.")

        model_cfg = config.model
        logger.info(
            f"[{self.__class__.__name__}] Overriding 'model.bbox_head.num_classes' to {model_cfg.num_classes}"
        )
        # Validation: Ensure the model has a bbox_head
        if not hasattr(self._cfg.model, "bbox_head"):
            raise ValueError(
                f"The loaded config for '{self.model_name}' does not have a 'bbox_head'. "
                "This class (RTMDet) expects a one-stage detector structure."
            )

        # Apply override
        bbox_head = self._cfg.model.bbox_head
        if isinstance(bbox_head, list):
            for head in bbox_head:
                head.num_classes = model_cfg.num_classes
        else:
            bbox_head.num_classes = model_cfg.num_classes

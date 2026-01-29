from pathlib import Path
from typing import Optional, Union

from loguru import logger
from mmdet.apis import DetInferencer
from mmdet.utils import register_all_modules

from ez_openmmlab.core.base import EZMMLab
from ez_openmmlab.core.config_loader import get_config_file
from ez_openmmlab.schemas.inference import InferenceResult
from ez_openmmlab.utils.download import ensure_model_checkpoint

# Force registration of MMDet modules
register_all_modules()

class EZMMDetector(EZMMLab):
    """Abstract base class for training and inference using MMDetection."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._inferencer: Optional[DetInferencer] = None

    def predict(
        self,
        image_path: Union[str, Path],
        checkpoint_path: Optional[Union[str, Path]] = None,
        device: str = "cuda",
        out_dir: Optional[str] = None,
        show: bool = False,
    ) -> InferenceResult:
        """Performs object detection on an image."""
        target_checkpoint = self.checkpoint_path
        if checkpoint_path:
            target_checkpoint = ensure_model_checkpoint(
                self.model_name, checkpoint_path
            )

        if self._inferencer is None:
            config_path = get_config_file(self.model_name)
            logger.info(
                f"Initializing inferencer for model: {self.model_name} (using config: {config_path})"
            )
            self._inferencer = DetInferencer(
                model=str(config_path),
                weights=str(target_checkpoint),
                device=device,
            )

        logger.info(f"Running inference on: {image_path}")
        results = self._inferencer(
            str(image_path), out_dir=out_dir or "", show=show
        )
        return InferenceResult.from_mmdet(results)

    def _configure_model_specifics(self, config):
        """Detection specific overrides are handled by subclasses."""
        pass

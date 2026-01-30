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

    _inferencer: Optional[DetInferencer]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._inferencer = None

    def predict(
        self,
        image_path: Union[str, Path],
        confidence: float = 0.5,
        device: str = "cuda",
        out_dir: Optional[str] = None,
        show: bool = False,
    ) -> InferenceResult:
        """Runs object detection on an image and returns structured results.

        Args:
            image_path: Path to the input image.
            confidence: Confidence threshold for filtering detections (default: 0.3).
            device: Computing device ('cuda', 'cpu').
            out_dir: Directory to save visualization images.
            show: Whether to pop up a window with the result.
        """
        if self._inferencer is None:
            config_path = get_config_file(self.model_name)
            logger.info(
                f"Initializing inferencer for model: {self.model_name} (using config: {config_path})"
            )
            self._inferencer = DetInferencer(
                model=str(config_path),
                weights=str(self.checkpoint_path),
                device=device,
            )

        logger.info(f"Running inference on: {image_path} (threshold: {confidence})")
        results = self._inferencer(
            str(image_path), out_dir=out_dir or "", show=show, pred_score_thr=confidence
        )
        return InferenceResult.from_mmdet(results)

    def _configure_model_specifics(self, config):
        """Detection specific overrides are handled by subclasses."""
        pass

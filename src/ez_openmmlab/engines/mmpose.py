from pathlib import Path
from typing import Optional, Union, List

from loguru import logger
from mmpose.apis import MMPoseInferencer

from ez_openmmlab.core.base import EZMMLab
from ez_openmmlab.core.config_loader import get_config_file
from ez_openmmlab.schemas.inference import PoseInferenceResult
from ez_openmmlab.schemas.model import ModelName
from ez_openmmlab.utils.download import ensure_model_checkpoint


class EZMMPose(EZMMLab):
    """Abstract base class for training and inference using MMPose."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._inferencer: Optional[MMPoseInferencer] = None

    def predict(
        self,
        image_path: Union[str, Path],
        det_model: Optional[str] = "rtmdet_tiny",
        det_weights: Optional[str] = None,
        device: str = "cuda",
        show: bool = False,
        out_dir: Optional[str] = None,
    ) -> PoseInferenceResult:
        """Runs pose estimation on an image and returns structured results.
        
        Args:
            image_path: Path to the input image.
            det_model: Detector model name or config path.
            det_weights: Path to detector weights.
            device: Computing device ('cuda', 'cpu').
            show: Whether to display results.
            out_dir: Directory to save visualization.
        """
        if self._inferencer is None:
            config_path = get_config_file(self.model_name)
            
            # Resolve detector config if it's a known model name
            actual_det_model = det_model
            if det_model in [m.value for m in ModelName]:
                actual_det_model = str(get_config_file(det_model))
            
            # Resolve detector weights if not provided
            actual_det_weights = det_weights
            if det_model and not det_weights:
                actual_det_weights = str(ensure_model_checkpoint(det_model))

            logger.info(
                f"Initializing pose inferencer for model: {self.model_name} (using config: {config_path})"
            )
            self._inferencer = MMPoseInferencer(
                pose2d=str(config_path),
                pose2d_weights=str(self.checkpoint_path),
                det_model=actual_det_model,
                det_weights=actual_det_weights,
                device=device,
            )

        logger.info(f"Running pose estimation on: {image_path}")

        # MMPoseInferencer returns a generator or dict based on input
        # We wrap it to get the raw results
        results = self._inferencer(str(image_path), out_dir=out_dir or "", show=show)

        # Result format: {'predictions': [[{'keypoints': [...], 'keypoint_scores': [...]}, ...]], 'visualization': [...]}
        # We need the inner predictions list
        raw_preds = []
        if isinstance(results, dict) and "predictions" in results:
            # Flatten the nested lists if necessary
            for batch in results["predictions"]:
                raw_preds.extend(batch)

        return PoseInferenceResult.from_mmpose(raw_preds)

    def _configure_model_specifics(self, config):
        """Pose specific overrides (e.g., keypoint head num_classes)."""
        if not self._cfg:
            raise RuntimeError("Config not loaded before configuring specifics.")

        # RTMPose specific overrides
        if hasattr(self._cfg.model, "head"):
            head = self._cfg.model.head
            logger.info(
                f"[{self.__class__.__name__}] Setting model.head.out_channels to {config.model.num_classes}"
            )
            # In MMPose, num_classes for pose heads is often 'out_channels' or similar
            head.out_channels = config.model.num_classes


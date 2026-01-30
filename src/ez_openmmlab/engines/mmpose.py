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
        bbox_thr: float = 0.3,
        kpt_thr: float = 0.3,
        device: str = "cuda",
        show: bool = False,
        out_dir: Optional[str] = None,
    ) -> PoseInferenceResult:
        """Runs pose estimation on an image and returns structured results.
        
        Args:
            image_path: Path to the input image.
            det_model: Detector model name or config path.
            det_weights: Path to detector weights.
            bbox_thr: Bounding box score threshold.
            kpt_thr: Keypoint score threshold.
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

        # MMPoseInferencer is a generator. We must iterate/list it to trigger 
        # the internal logic (inference, visualization, and saving).
        results_gen = self._inferencer(
            str(image_path), 
            out_dir=out_dir if out_dir else None, 
            show=show,
            bbox_thr=bbox_thr,
            kpt_thr=kpt_thr
        )
        
        # Consume generator to execute the work
        all_results = list(results_gen)

        # MMPose yields a dict for each image (or batch). 
        # In our case (single image), we take the first result.
        # Format: {'predictions': [[{'keypoints': [...], 'keypoint_scores': [...]}, ...]], 'visualization': [...]}
        raw_preds = []
        if all_results:
            first_result = all_results[0]
            if "predictions" in first_result:
                # MMPose 1.x returns nested predictions [batch][instances]
                for batch in first_result["predictions"]:
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


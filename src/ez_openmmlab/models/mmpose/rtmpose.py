from pathlib import Path
from typing import Optional, Union

from loguru import logger
from mmpose.apis import MMPoseInferencer

from ez_openmmlab.engines.mmpose import EZMMPose
from ez_openmmlab.core.config_loader import get_config_file
from ez_openmmlab.schemas.inference import PoseInferenceResult
from ez_openmmlab.schemas.model import ModelName
from ez_openmmlab.utils.download import ensure_model_checkpoint
from ez_openmmlab.utils.toml_config import UserConfig


class RTMPose(EZMMPose):
    """RTMPose implementation for fast 2D keypoint estimation.

    Supported variants: rtmpose_tiny, rtmpose_s, rtmpose_m, rtmpose_l.
    Note: RTMPose is a top-down model and requires a detector.
    """

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
        **kwargs,
    ) -> PoseInferenceResult:
        """Runs RTMPose inference with a person detector."""
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

            logger.info(f"Initializing RTMPose inferencer: {self.model_name}")
            with self.switch_to_lib_root():
                self._inferencer = MMPoseInferencer(
                    pose2d=str(config_path),
                    pose2d_weights=str(self.checkpoint_path),
                    det_model=actual_det_model,
                    det_weights=actual_det_weights,
                    device=device,
                )

        return self._execute_mmpose_inferencer(
            image_path=image_path,
            out_dir=out_dir,
            show=show,
            bbox_thr=bbox_thr,
            kpt_thr=kpt_thr,
            **kwargs,
        )

    def _configure_model_specifics(self, config: UserConfig) -> None:
        """RTMPose specific head overrides."""
        if not self._cfg:
            raise RuntimeError("Config not loaded.")

        if hasattr(self._cfg.model, "head"):
            head = self._cfg.model.head
            logger.info(
                f"[{self.__class__.__name__}] Setting model.head.out_channels to {config.model.num_classes}"
            )
            head.out_channels = config.model.num_classes

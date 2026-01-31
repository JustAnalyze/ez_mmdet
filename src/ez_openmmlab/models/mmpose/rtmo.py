from pathlib import Path
from typing import Optional, Union

from loguru import logger
from mmpose.apis import MMPoseInferencer

from ez_openmmlab.engines.mmpose import EZMMPose
from ez_openmmlab.core.config_loader import get_config_file
from ez_openmmlab.schemas.inference import PoseInferenceResult
from ez_openmmlab.utils.toml_config import UserConfig


class RTMO(EZMMPose):
    """RTMO implementation for fast bottom-up 2D multi-person pose estimation.

    Supported variants: rtmo_s, rtmo_m, rtmo_l.
    Note: RTMO is a bottom-up model and does not require a separate detector.
    """

    def predict(
        self,
        image_path: Union[str, Path],
        bbox_thr: float = 0.3,
        kpt_thr: float = 0.3,
        device: str = "cuda",
        show: bool = False,
        out_dir: Optional[str] = None,
        **kwargs,
    ) -> PoseInferenceResult:
        """Runs RTMO inference without a separate detector."""
        if self._inferencer is None:
            config_path = get_config_file(self.model_name)

            logger.info(f"Initializing RTMO inferencer: {self.model_name}")
            with self.switch_to_lib_root():
                self._inferencer = MMPoseInferencer(
                    pose2d=str(config_path),
                    pose2d_weights=str(self.checkpoint_path),
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
        """RTMO specific head overrides."""
        if not self._cfg:
            raise RuntimeError("Config not loaded.")

        if hasattr(self._cfg.model, "head"):
            head = self._cfg.model.head
            logger.info(
                f"[{self.__class__.__name__}] Setting RTMO model.head.num_keypoints to {config.model.num_classes}"
            )
            head.num_keypoints = config.model.num_classes

            if hasattr(head, "head_module_cfg"):
                logger.info(
                    f"[{self.__class__.__name__}] Setting RTMO model.head.head_module_cfg.num_classes to 1"
                )
                head.head_module_cfg.num_classes = 1


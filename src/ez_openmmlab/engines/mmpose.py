from abc import abstractmethod
from pathlib import Path
from typing import List, Optional, Union

from loguru import logger
from mmpose.apis import MMPoseInferencer

from ez_openmmlab.core.base import EZMMLab
from ez_openmmlab.schemas.inference import PoseInferenceResult
from ez_openmmlab.utils.path import get_unique_dir


class EZMMPose(EZMMLab):
    """Base engine for MMPose models.

    Provides shared utilities for interacting with MMPoseInferencer.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._inferencer: Optional[MMPoseInferencer] = None

    @abstractmethod
    def predict(self, image_path: Union[str, Path], **kwargs) -> PoseInferenceResult:
        """Subclasses must implement specific inference logic."""
        pass

    def _execute_mmpose_inferencer(
        self,
        image_path: Union[str, Path],
        out_dir: Optional[str] = None,
        show: bool = False,
        **kwargs,
    ) -> PoseInferenceResult:
        """Shared execution logic for MMPoseInferencer.

        Consumes the generator and parses results into a PoseInferenceResult.
        """
        if self._inferencer is None:
            raise RuntimeError(
                "Inferencer not initialized. Call init_inferencer first."
            )

        logger.info(f"Running pose estimation on: {image_path}")

        # MMPoseInferencer is a generator. We must iterate/list it to trigger
        # the internal logic (inference, visualization, and saving).

        actual_out_dir = str(get_unique_dir(out_dir)) if out_dir else None

        inferencer_kwargs = {
            "inputs": str(image_path),
            "show": show,
            "out_dir": actual_out_dir,
            **kwargs,
        }

        results_gen = self._inferencer(**inferencer_kwargs)

        # Consume generator to execute the work
        all_results = list(results_gen)

        # MMPose yields a dict for each image (or batch).
        # Format: {'predictions': [[{'keypoints': [...], 'keypoint_scores': [...]}, ...]], 'visualization': [...]}
        raw_preds = []
        if all_results:
            first_result = all_results[
                0
            ]  # FIX: WHAT IF WE ARE INFERENCING ON MULTIPLE IMAGE?
            if "predictions" in first_result:
                # MMPose 1.x returns nested predictions [batch][instances]
                for batch in first_result["predictions"]:
                    raw_preds.extend(batch)

        return PoseInferenceResult.from_mmpose(raw_preds)

    @abstractmethod
    def _configure_model_specifics(self, config):
        """Subclasses must implement architecture-specific overrides."""
        pass

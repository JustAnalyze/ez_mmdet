from abc import abstractmethod
from pathlib import Path
from typing import List, Optional, Union

from loguru import logger
from mmpose.apis import MMPoseInferencer

from ez_openmmlab.core.base import EZMMLab
from ez_openmmlab.schemas.inference import PoseInferenceResult


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
        
        # Explicitly handle out_dir to ensure subdirectories are set up correctly
        inferencer_kwargs = {
            "inputs": str(image_path),
            "show": show,
            **kwargs
        }
        
        if out_dir:
            # We explicitly pass these to be safe, as MMPoseInferencer's internal
            # logic for 'out_dir' might skip if the directory already exists.
            inferencer_kwargs["vis_out_dir"] = f"{out_dir}/visualizations"
            inferencer_kwargs["pred_out_dir"] = f"{out_dir}/predictions"

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

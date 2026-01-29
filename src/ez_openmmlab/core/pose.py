from ez_openmmlab.core.base import EZMMLab

class EZMMPose(EZMMLab):
    """Abstract base class for training and inference using MMPose."""

    def predict(self, image_path, *args, **kwargs):
        """Perform pose estimation."""
        raise NotImplementedError("Pose estimation predict() not yet implemented.")

    def _configure_model_specifics(self, config):
        """Pose specific overrides."""
        pass

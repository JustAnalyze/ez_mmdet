import warnings
import logging

# Suppress noisy MMLab and PyTorch warnings globally
warnings.filterwarnings("ignore", message=".*LocalVisBackend")
warnings.filterwarnings("ignore", message=".*meshgrid")
warnings.filterwarnings("ignore", message=".*bbox is out of bounds")
warnings.filterwarnings("ignore", message=".*polygon is out of bounds")

# Suppress mmengine log warnings
logging.getLogger("mmengine").setLevel(logging.ERROR)

from .models.rtmdet import RTMDet
from .core.base import EZMMLab
from .core.detection import EZMMDetector
from .core.pose import EZMMPose

__all__ = ["RTMDet", "EZMMLab", "EZMMDetector", "EZMMPose"]

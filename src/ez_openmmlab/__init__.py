import warnings
import logging
from mmengine.logging import MMLogger
from mmdet.utils import register_all_modules

# Suppress noisy MMLab and PyTorch warnings globally
warnings.filterwarnings("ignore", message=".*LocalVisBackend")
warnings.filterwarnings("ignore", message=".*meshgrid")
warnings.filterwarnings("ignore", message=".*bbox is out of bounds")
warnings.filterwarnings("ignore", message=".*polygon is out of bounds")

# Suppress mmengine log warnings
logging.getLogger("mmengine").setLevel(logging.ERROR)
# Ensure MMLogger (used by runner) is also silenced if already initialized
try:
    MMLogger.get_instance("mmengine").setLevel(logging.ERROR)
except Exception:
    pass

# Ensure MMDet modules are registered with default scope immediately
register_all_modules(init_default_scope=True)

from .models.mmdet import RTMDet
from .models.mmpose import RTMPose
from .core.base import EZMMLab
from .engines.mmdet import EZMMDetector
from .engines.mmpose import EZMMPose

__all__ = ["RTMDet", "RTMPose", "EZMMLab", "EZMMDetector", "EZMMPose"]

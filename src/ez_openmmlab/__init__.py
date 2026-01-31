import warnings

# Suppress noisy library warnings immediately on import
warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated")

from mmdet.utils import register_all_modules

# Ensure MMDet modules are registered with default scope immediately
register_all_modules(init_default_scope=True)

from .models.mmdet import RTMDet
from .models.mmpose import RTMPose, RTMO
from .core.base import EZMMLab
from .engines.mmdet import EZMMDetector
from .engines.mmpose import EZMMPose

__all__ = [
    "RTMDet",
    "RTMPose",
    "RTMO",
    "EZMMLab",
    "EZMMDetector",
    "EZMMPose",
    "mute_warnings",
]


def mute_warnings():
    """Helper to suppress MMLab and noisy library verbosity manually."""
    import warnings
    import logging
    from mmengine.logging import MMLogger

    # Suppress noisy MMLab and PyTorch warnings
    warnings.filterwarnings("ignore", message=".*LocalVisBackend")
    warnings.filterwarnings("ignore", message=".*meshgrid")
    warnings.filterwarnings("ignore", message=".*bbox is out of bounds")
    warnings.filterwarnings("ignore", message=".*polygon is out of bounds")

    # Suppress mmengine log warnings
    logging.getLogger("mmengine").setLevel(logging.ERROR)
    try:
        MMLogger.get_instance("mmengine").setLevel(logging.ERROR)
    except Exception:
        pass

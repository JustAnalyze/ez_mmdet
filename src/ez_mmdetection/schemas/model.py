from enum import Enum

class ModelName(str, Enum):
    """Currently supported model architectures in ez_mmdet."""
    
    # Bounding Box detection
    RTM_DET_TINY = "rtmdet_tiny"
    RTM_DET_S = "rtmdet_s"
    RTM_DET_M = "rtmdet_m"
    RTM_DET_L = "rtmdet_l"
    RTM_DET_X = "rtmdet_x"
    
    # Instance Segmentation
    RTM_DET_INS_TINY = "rtmdet-ins_tiny"
    RTM_DET_INS_S = "rtmdet-ins_s"
    RTM_DET_INS_M = "rtmdet-ins_m"
    RTM_DET_INS_L = "rtmdet-ins_l"
    RTM_DET_INS_X = "rtmdet-ins_x"

from ez_openmmlab.engines.mmpose import EZMMPose

class RTMO(EZMMPose):
    """RTMO implementation for fast bottom-up 2D multi-person pose estimation.
    
    Supported variants: rtmo_s, rtmo_m, rtmo_l.
    Note: RTMO is a bottom-up model and does not require a separate detector.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

from ez_openmmlab.engines.mmpose import EZMMPose

class RTMPose(EZMMPose):
    """RTMPose implementation for fast 2D keypoint estimation.
    
    Supported variants: rtmpose_tiny, rtmpose_s, rtmpose_m, rtmpose_l.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

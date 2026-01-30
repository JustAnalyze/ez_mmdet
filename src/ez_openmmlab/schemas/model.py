from enum import Enum
from typing import Dict

# --- Model Registry Data ---
# Organized by architecture family for easy maintenance

RTM_DET_CONFIGS = {
    "rtmdet_tiny": "rtmdet/rtmdet_tiny_8xb32-300e_coco.py",
    "rtmdet_s": "rtmdet/rtmdet_s_8xb32-300e_coco.py",
    "rtmdet_m": "rtmdet/rtmdet_m_8xb32-300e_coco.py",
    "rtmdet_l": "rtmdet/rtmdet_l_8xb32-300e_coco.py",
    "rtmdet_x": "rtmdet/rtmdet_x_8xb32-300e_coco.py",
}

RTM_DET_INS_CONFIGS = {
    "rtmdet-ins_tiny": "rtmdet/rtmdet-ins_tiny_8xb32-300e_coco.py",
    "rtmdet-ins_s": "rtmdet/rtmdet-ins_s_8xb32-300e_coco.py",
    "rtmdet-ins_m": "rtmdet/rtmdet-ins_m_8xb32-300e_coco.py",
    "rtmdet-ins_l": "rtmdet/rtmdet-ins_l_8xb32-300e_coco.py",
    "rtmdet-ins_x": "rtmdet/rtmdet-ins_x_8xb16-300e_coco.py",
}

RTM_POSE_CONFIGS = {
    "rtmpose_tiny": "body_2d_keypoint/rtmpose/coco/rtmpose-t_8xb256-420e_coco-256x192.py",
    "rtmpose_s": "body_2d_keypoint/rtmpose/coco/rtmpose-s_8xb256-420e_coco-256x192.py",
    "rtmpose_m": "body_2d_keypoint/rtmpose/coco/rtmpose-m_8xb256-420e_coco-256x192.py",
    "rtmpose_l": "body_2d_keypoint/rtmpose/coco/rtmpose-l_8xb256-420e_coco-256x192.py",
}

RTM_DET_URLS = {
    "rtmdet_tiny": "https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_tiny_8xb32-300e_coco/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth",
    "rtmdet_s": "https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_s_8xb32-300e_coco/rtmdet_s_8xb32-300e_coco_20220905_161602-387a891e.pth",
    "rtmdet_m": "https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_m_8xb32-300e_coco/rtmdet_m_8xb32-300e_coco_20220719_112220-229f527c.pth",
    "rtmdet_l": "https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_l_8xb32-300e_coco/rtmdet_l_8xb32-300e_coco_20220719_112030-5a0be7c4.pth",
    "rtmdet_x": "https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_x_8xb32-300e_coco/rtmdet_x_8xb32-300e_coco_20220715_230555-cc79b9ae.pth",
}

RTM_DET_INS_URLS = {
    "rtmdet-ins_tiny": "https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet-ins_tiny_8xb32-300e_coco/rtmdet-ins_tiny_8xb32-300e_coco_20221130_151727-ec670f7e.pth",
    "rtmdet-ins_s": "https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet-ins_s_8xb32-300e_coco/rtmdet-ins_s_8xb32-300e_coco_20221121_212604-fdc5d7ec.pth",
    "rtmdet-ins_m": "https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet-ins_m_8xb32-300e_coco/rtmdet-ins_m_8xb32-300e_coco_20221123_001039-6eba602e.pth",
    "rtmdet-ins_l": "https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet-ins_l_8xb32-300e_coco/rtmdet-ins_l_8xb32-300e_coco_20221124_103237-78d1d652.pth",
    "rtmdet-ins_x": "https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet-ins_x_8xb16-300e_coco/rtmdet-ins_x_8xb16-300e_coco_20221124_111313-33d4595b.pth",
}

RTM_POSE_URLS = {
    "rtmpose_tiny": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-tiny_simcc-coco_pt-aic-coco_420e-256x192-e613ba3f_20230127.pth",
    "rtmpose_s": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-s_simcc-coco_pt-aic-coco_420e-256x192-8edcf0d7_20230127.pth",
    "rtmpose_m": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-coco_pt-aic-coco_420e-256x192-d8dd5ca4_20230127.pth",
    "rtmpose_l": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-l_simcc-coco_pt-aic-coco_420e-256x192-1352a4d2_20230127.pth",
}

# Unified Master Maps
MODEL_CONFIG_MAP = {**RTM_DET_CONFIGS, **RTM_DET_INS_CONFIGS, **RTM_POSE_CONFIGS}
MODEL_URLS = {**RTM_DET_URLS, **RTM_DET_INS_URLS, **RTM_POSE_URLS}


class ModelName(str, Enum):
    """Currently supported model architectures in ez_openmmlab."""

    # Bounding Box detection
    RTM_DET_TINY = "rtmdet_tiny"
    RTM_DET_S = "rtmdet_s"
    RTM_DET_M = "rtmdet_m"
    RTM_DET_L = "rtmdet_l"
    RTM_DET_X = "rtmdet_x"

    # Instance Segmentation
    RTM_DET_INS_TINY = "rtmdet-ins_tiny"
    # Note: Value in MODEL_CONFIG_MAP uses hyphen for ins, consistent with ModelName values
    RTM_DET_INS_S = "rtmdet-ins_s"
    RTM_DET_INS_M = "rtmdet-ins_m"
    RTM_DET_INS_L = "rtmdet-ins_l"
    RTM_DET_INS_X = "rtmdet-ins_x"

    # Pose Estimation (RTMPose)
    RTM_POSE_TINY = "rtmpose_tiny"
    RTM_POSE_S = "rtmpose_s"
    RTM_POSE_M = "rtmpose_m"
    RTM_POSE_L = "rtmpose_l"

    @property
    def config_path(self) -> str:
        """Returns the relative config path for this model."""
        return MODEL_CONFIG_MAP[self.value]

    @property
    def weights_url(self) -> str:
        """Returns the official download URL for this model's weights."""
        return MODEL_URLS[self.value]


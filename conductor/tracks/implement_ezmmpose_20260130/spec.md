# Track Specification: Implement EZMMPose Support

## Overview
This track implements first-class support for pose estimation using the MMPose library. We will introduce `EZMMPose` as a core engine, reorganize the core engine structure for better intuition, and implement the RTMPose architecture.

## Objectives
- Reorganize `src/ez_openmmlab/core/` into a more intuitive structure (e.g., `src/ez_openmmlab/engines/`).
- Rename `detection.py` to `detector.py` and `pose.py` to `pose_estimator.py` (or similar) within the new structure.
- Implement the `EZMMPose` engine class.
- Support RTMPose variants (tiny, s, m, l) with automatic weight management.
- Define a strict `PoseInferenceResult` schema for structured output.
- Enable training with custom keypoint metadata via `dataset.toml`.

## Functional Requirements
- **Core Engine Reorganization:** Move `EZMMDetector` and `EZMMPose` to `src/ez_openmmlab/engines/`.
- **EZMMPose Implementation:** Inherit from `EZMMLab` and implement `predict()` using `MMPoseInferencer`.
- **Model Registry:** Add RTMPose model variants to `ModelName` enum and update `ConfigLoader`.
- **Inference Schema:** Create `PoseInferenceResult` and `PosePrediction` Pydantic models.
- **Training Workflow:** Implement metadata injection for pose datasets.

## Non-Functional Requirements
- **Consistency:** Maintain the "EZ" user experience.
- **Modular Design:** Ensure clean separation between detection and pose logic.

## Acceptance Criteria
- `EZMMPose("rtmpose-m").predict(image)` returns a structured `PoseInferenceResult`.
- Core files are moved to `src/ez_openmmlab/engines/` and imports are updated.
- RTMPose models can be trained using a `dataset.toml` with `metainfo`.
- All existing detection tests continue to pass.

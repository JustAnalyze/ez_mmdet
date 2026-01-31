# Implementation Plan: Implement EZMMPose Support

## Phase 1: Core Reorganization [checkpoint: 362cedc]

- [x] Task: Move and Rename Core Engines
  - [x] Create `src/ez_openmmlab/engines/` directory.
  - [x] Move `src/ez_openmmlab/core/detection.py` to `src/ez_openmmlab/engines/mmdet.py`.
  - [x] Move `src/ez_openmmlab/core/pose.py` to `src/ez_openmmlab/engines/mmpose.py`.
  - [x] Update all imports in the codebase to reflect these changes.
- [x] Task: Conductor - User Manual Verification 'Core Reorganization' (Protocol in workflow.md)

## Phase 2: EZMMPose Implementation [checkpoint: 856b16b]

- [x] Task: Implement Pose Schema and Registry
  - [x] Create `PoseInferenceResult` schema.
  - [x] Update `ModelName` and `ConfigLoader` with RTMPose models.
- [x] Task: Implement EZMMPose Engine
  - [x] Implement `EZMMPose.predict()` using `MMPoseInferencer`.
  - [x] Implement `_configure_model_specifics()` for pose.
- [x] Task: Conductor - User Manual Verification 'EZMMPose Implementation' (Protocol in workflow.md)

## Phase 3: Training & Verification [checkpoint: 4828539]

- [x] Task: Implement Pose Training Support
  - [x] Update dataloader handlers to support `metainfo` injection for pose datasets.
- [x] Task: E2E Verification
  - [x] Create E2E smoke test for pose estimation (using `sample_mmpose.py` as reference).
  - [x] Verify detection smoke tests still pass.
- [x] Task: Conductor - User Manual Verification 'Training & Verification' (Protocol in workflow.md)

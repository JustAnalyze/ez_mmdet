# Implementation Plan: Refactor base.py Configuration Logic

## Phase 1: Configuration Handlers [checkpoint: ca91e32]

- [x] Task: Create `BaseConfigHandler` base protocol
    - [x] Define a general base class for config handlers that accepts a `Config` and `UserConfig`.
- [x] Task: Extract Dataloader Configuration
    - [x] Write unit tests for `DataloaderHandler` (verifying it sets paths and workers correctly).
    - [x] Implement `DataloaderHandler` by moving logic from `_apply_common_overrides`.
- [x] Task: Extract Runtime Configuration (Visualizer & Optimizer)
    - [x] Write unit tests for `RuntimeHandler` (verifying TensorBoard setup and optimizer settings).
    - [x] Implement `RuntimeHandler`.
- [x] Task: Conductor - User Manual Verification 'Configuration Handlers' (Protocol in workflow.md)

## Phase 2: Refactor EZMMDetector [checkpoint: PENDING]

- [~] Task: Integrate Handlers into `EZMMDetector`
    - [~] Modify `_apply_common_overrides` to instantiate and use the new handlers.
    - [ ] Remove the old, in-lined configuration logic.
- [ ] Task: Verify Refactor
    - [ ] Run the existing full test suite to ensure no regression.
    - [ ] Verify that `user_config.toml` is still generated correctly.
- [ ] Task: Conductor - User Manual Verification 'Refactor EZMMDetector' (Protocol in workflow.md)

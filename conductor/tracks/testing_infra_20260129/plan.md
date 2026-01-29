# Implementation Plan: Comprehensive Testing Infrastructure

## Phase 1: Unit & Schema Tests [checkpoint: 789ff8c]

- [x] Task: Implement comprehensive Schema tests
    - [x] Write unit tests for `UserConfig` and `TrainingSection` (validating defaults and constraints)
    - [x] Write unit tests for `DatasetConfig` (validating TOML parsing and required fields)
    - [x] Write unit tests for `ModelName` enum validation
- [x] Task: Implement Model Management tests
    - [x] Write unit tests for `ConfigLoader` (verifying strict mapping and path resolution)
    - [x] Write unit tests for `ensure_model_checkpoint` (mocking requests and file system)
- [x] Task: Refine Handler tests
    - [x] Expand `tests/test_handlers.py` to cover edge cases for `DataloaderHandler` and `RuntimeHandler`
- [x] Task: Conductor - User Manual Verification 'Unit & Schema Tests' (Protocol in workflow.md)

## Phase 2: Integration & Regression Tests [checkpoint: 8924112]

- [x] Task: Implement Engine Integration tests
    - [x] Write integration tests for `EZMMDetector.train` (verifying handler orchestration and `user_config.toml` creation)
    - [x] Write integration tests for `EZMMDetector.predict` (verifying `InferenceResult` creation)
- [x] Task: Implement Regression tests
    - [x] Add explicit test cases for the `NoneType` visualization error fix
    - [x] Add explicit test cases for the model name path resolution fix
- [x] Task: Conductor - User Manual Verification 'Integration & Regression Tests' (Protocol in workflow.md)

## Phase 3: E2E Smoke Tests & Coverage Audit [checkpoint: PENDING]

- [x] Task: Implement E2E Smoke tests
    - [x] Create a minimal end-to-end test that runs a 1-epoch training pass on dummy data (CPU)
    - [x] Verify that the generated checkpoint can be loaded for a successful `predict()` call
- [~] Task: Final Coverage Audit
    - [~] Run the full suite with `pytest-cov` and ensure total coverage is >80%
    - [ ] Generate and store the final coverage report
- [ ] Task: Conductor - User Manual Verification 'E2E & Coverage' (Protocol in workflow.md)

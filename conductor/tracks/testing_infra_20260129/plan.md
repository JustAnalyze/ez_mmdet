# Implementation Plan: Comprehensive Testing Infrastructure

## Phase 1: Unit & Schema Tests [checkpoint: PENDING]

- [x] Task: Implement comprehensive Schema tests
    - [x] Write unit tests for `UserConfig` and `TrainingSection` (validating defaults and constraints)
    - [x] Write unit tests for `DatasetConfig` (validating TOML parsing and required fields)
    - [x] Write unit tests for `ModelName` enum validation
- [ ] Task: Implement Model Management tests
    - [ ] Write unit tests for `ConfigLoader` (verifying strict mapping and path resolution)
    - [ ] Write unit tests for `ensure_model_checkpoint` (mocking requests and file system)
- [ ] Task: Refine Handler tests
    - [ ] Expand `tests/test_handlers.py` to cover edge cases for `DataloaderHandler` and `RuntimeHandler`
- [ ] Task: Conductor - User Manual Verification 'Unit & Schema Tests' (Protocol in workflow.md)

## Phase 2: Integration & Regression Tests [checkpoint: PENDING]

- [ ] Task: Implement Engine Integration tests
    - [ ] Write integration tests for `EZMMDetector.train` (verifying handler orchestration and `user_config.toml` creation)
    - [ ] Write integration tests for `EZMMDetector.predict` (verifying `InferenceResult` creation)
- [ ] Task: Implement Regression tests
    - [ ] Add explicit test cases for the `NoneType` visualization error fix
    - [ ] Add explicit test cases for the model name path resolution fix
- [ ] Task: Conductor - User Manual Verification 'Integration & Regression Tests' (Protocol in workflow.md)

## Phase 3: E2E Smoke Tests & Coverage Audit [checkpoint: PENDING]

- [ ] Task: Implement E2E Smoke tests
    - [ ] Create a minimal end-to-end test that runs a 1-epoch training pass on dummy data (CPU)
    - [ ] Verify that the generated checkpoint can be loaded for a successful `predict()` call
- [ ] Task: Final Coverage Audit
    - [ ] Run the full suite with `pytest-cov` and ensure total coverage is >80%
    - [ ] Generate and store the final coverage report
- [ ] Task: Conductor - User Manual Verification 'E2E & Coverage' (Protocol in workflow.md)

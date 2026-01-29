# Track Specification: Comprehensive Testing Infrastructure

## Overview
This track focuses on building a robust automated testing suite for `ez_mmdet`. The goal is to ensure the codebase remains functional and reliable as new features are added, while also providing a "fail-fast" mechanism for developers and users.

## Objectives
- Establish a high-coverage test suite using `pytest`.
- Validate all core logic, from configuration parsing to model execution.
- Prevent regressions for previously resolved issues.
- Ensure the codebase is "CI-Ready" for future automation.

## Functional Requirements
- **Unit Testing:** Implement isolated tests for `DataloaderHandler`, `RuntimeHandler`, `ConfigLoader`, and `Download` utilities using mocks.
- **Schema Validation:** Comprehensive tests for `UserConfig` and `DatasetConfig` to verify strict TOML parsing and error handling.
- **Integration Testing:** Verify the interaction between `EZMMDetector` and its handlers, ensuring internal MMDetection configs are modified correctly.
- **E2E Smoke Tests:** Create a minimal end-to-end flow using dummy data and CPU to verify the full "Train -> Save -> Predict" loop.

## Non-Functional Requirements
- **Total Code Coverage:** Aim for >80% coverage across the `src/ez_mmdetection` package.
- **Execution Efficiency:** Tests should run quickly and be easily executable via `uv run pytest`.
- **Environment Agnostic:** Tests should pass on both CPU and GPU-enabled environments (using mocks where appropriate).

## Acceptance Criteria
- Full test suite passes with a single command.
- Code coverage report shows >80% for the core package.
- Regression tests for `NoneType` path errors and config resolution are included and passing.

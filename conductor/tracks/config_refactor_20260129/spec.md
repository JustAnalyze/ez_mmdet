# Track Specification: Refactor base.py Configuration Logic

## Overview
This track focuses on improving the readability and maintainability of the `EZMMDetector` class by refactoring the `train` and `_apply_common_overrides` methods. The current implementation is becoming "crowded," making it difficult to debug and extend. We will move complex MMDetection configuration logic into dedicated, data-driven handler classes derived from a general `BaseConfigHandler`.

## Objectives
- Reduce the cyclomatic complexity of `EZMMDetector`.
- Decouple the core training workflow from the low-level MMDetection configuration details.
- Improve debuggability by isolating configuration steps (dataloaders, visualizers, optimizers) into separate units.
- Establish a general handler pattern (`BaseConfigHandler`) that can be reused for future model expansions (e.g., EZMMPose).

## Functional Requirements
- **Extract Configuration Handlers:** Create dedicated classes (e.g., `DataloaderHandler`, `RuntimeHandler`) to manage specific sections of the MMDetection `Config` object.
- **Base Pattern:** Implement a general `BaseConfigHandler` protocol or abstract base class that defines how handlers consume `Config` and `UserConfig`.
- **Data-Driven Delegation:** The main detector class will initialize these handlers and delegate the modification of the `Config` object to them, passing in the validated `UserConfig`.
- **Maintain API Stability:** This refactor must not change the external API of `EZMMDetector`. Existing user code and CLI commands must continue to work without modification.

## Non-Functional Requirements
- **Readability:** The `train` method should serve as a high-level "clean" orchestrator.
- **Testability:** The new handler classes should be independently testable.

## Acceptance Criteria
- `EZMMDetector.train` and `_apply_common_overrides` are significantly shorter and easier to read.
- All existing unit and integration tests pass.
- No regression in the `user_config.toml` artifact generation.

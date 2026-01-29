# Implementation Plan: Build core inference API and CLI

## Phase 1: Inference API [checkpoint: 2adee04]

- [x] Task: Implement `predict()` method in `EZDetector`
    - [x] Write unit tests for `predict()` behavior (mocking `DetInferencer`)
    - [x] Implement `predict()` logic using MMDetection's high-level API
- [x] Task: Add result structure and visualization support
    - [x] Write tests for result parsing and output directory creation
    - [x] Implement detection result structuring and visualization logic
- [x] Task: Conductor - User Manual Verification 'Inference API' (Protocol in workflow.md)

## Phase 2: Command Line Interface [checkpoint: e33c1f4]

- [x] Task: Scaffold `ez-mmdet` CLI with Typer
    - [x] Write tests for CLI argument parsing and error handling
    - [x] Create `src/ez_openmmlab/cli.py` and define the main entry point
- [x] Task: Implement `train` command
    - [x] Write integration tests for the `train` command (mocking the training loop)
    - [x] Implement `train` command logic, bridging CLI args to `EZDetector.train()`
- [x] Task: Implement `predict` command
    - [x] Write integration tests for the `predict` command
    - [x] Implement `predict` command logic, bridging CLI args to `EZDetector.predict()`
- [x] Task: Conductor - User Manual Verification 'Command Line Interface' (Protocol in workflow.md)

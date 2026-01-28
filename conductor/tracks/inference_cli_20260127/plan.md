# Implementation Plan: Build core inference API and CLI

## Phase 1: Inference API [checkpoint: 2adee04]

- [x] Task: Implement `predict()` method in `EZDetector`
    - [x] Write unit tests for `predict()` behavior (mocking `DetInferencer`)
    - [x] Implement `predict()` logic using MMDetection's high-level API
- [x] Task: Add result structure and visualization support
    - [x] Write tests for result parsing and output directory creation
    - [x] Implement detection result structuring and visualization logic
- [x] Task: Conductor - User Manual Verification 'Inference API' (Protocol in workflow.md)

## Phase 2: Command Line Interface [checkpoint: PENDING]

- [x] Task: Scaffold `ez-mmdet` CLI with Typer
    - [x] Write tests for CLI argument parsing and error handling
    - [x] Create `src/ez_mmdetection/cli.py` and define the main entry point
- [ ] Task: Implement `train` command
    - [ ] Write integration tests for the `train` command (mocking the training loop)
    - [ ] Implement `train` command logic, bridging CLI args to `EZDetector.train()`
- [ ] Task: Implement `predict` command
    - [ ] Write integration tests for the `predict` command
    - [ ] Implement `predict` command logic, bridging CLI args to `EZDetector.predict()`
- [ ] Task: Conductor - User Manual Verification 'Command Line Interface' (Protocol in workflow.md)

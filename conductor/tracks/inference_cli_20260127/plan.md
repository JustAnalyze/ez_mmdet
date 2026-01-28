# Implementation Plan: Build core inference API and CLI

## Phase 1: Inference API [checkpoint: PENDING]

- [ ] Task: Implement `predict()` method in `EZDetector`
    - [ ] Write unit tests for `predict()` behavior (mocking `DetInferencer`)
    - [ ] Implement `predict()` logic using MMDetection's high-level API
- [ ] Task: Add result structure and visualization support
    - [ ] Write tests for result parsing and output directory creation
    - [ ] Implement detection result structuring and visualization logic
- [ ] Task: Conductor - User Manual Verification 'Inference API' (Protocol in workflow.md)

## Phase 2: Command Line Interface [checkpoint: PENDING]

- [ ] Task: Scaffold `ez-mmdet` CLI with Typer
    - [ ] Write tests for CLI argument parsing and error handling
    - [ ] Create `src/ez_mmdetection/cli.py` and define the main entry point
- [ ] Task: Implement `train` command
    - [ ] Write integration tests for the `train` command (mocking the training loop)
    - [ ] Implement `train` command logic, bridging CLI args to `EZDetector.train()`
- [ ] Task: Implement `predict` command
    - [ ] Write integration tests for the `predict` command
    - [ ] Implement `predict` command logic, bridging CLI args to `EZDetector.predict()`
- [ ] Task: Conductor - User Manual Verification 'Command Line Interface' (Protocol in workflow.md)

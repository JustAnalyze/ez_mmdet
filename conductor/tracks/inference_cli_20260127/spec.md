# Track Specification: Build core inference API and CLI

## Overview
This track focuses on enabling end-to-end usage of `ez_mmdet` by implementing a high-level inference API (`predict()` method) and a user-friendly Command Line Interface (`ez-mmdet`).

## Objectives
- Implement a robust `predict()` method in the `EZDetector` base class.
- Leverage MMDetection's `DetInferencer` for simplified model initialization and inference execution.
- Create a CLI using `Typer` to support core workflows (training and prediction).
- Ensure high code quality and reliability through Test-Driven Development (TDD).

## Requirements

### Inference API
- **Location:** `src/ez_openmmlab/core/base.py` (added to `EZDetector`).
- **Functionality:**
    - Accept an image (path or numpy array) and a checkpoint path.
    - Initialize the underlying MMDetection inferencer on the fly if not already initialized.
    - Return a structured detection result (boxes, labels, scores).
    - Support optional visualization (saving the result image to a specified directory).

### Command Line Interface (CLI)
- **Framework:** `Typer`.
- **Command: `train`**
    - Usage: `ez-mmdet train <dataset_config_path> [options]`
    - Options: `--epochs`, `--batch-size`, `--work-dir`, etc.
- **Command: `predict`**
    - Usage: `ez-mmdet predict <model_name> <checkpoint_path> <image_path> [options]`
    - Options: `--out-dir`, `--device`.

## Technical Constraints
- Must adhere to the Template Method Pattern established in the codebase.
- Inference must correctly handle the `classes` and `num_classes` inferred from the model/checkpoint.
- CLI must provide helpful, color-coded output using `rich`.
- All new code must achieve >80% test coverage.

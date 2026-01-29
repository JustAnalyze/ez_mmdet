# Technology Stack: ez_mmdet

## Core Languages & Runtimes
- **Python (>=3.9, <3.11):** The primary programming language. The version constraint is strictly enforced by the underlying MMDetection and MMCV requirements.

## Frameworks & Libraries
- **MMDetection (v3.3.0):** The base framework for all computer vision tasks. Integrated as a Git submodule in `libs/mmdetection`.
- **MMEngine (>=0.10.7):** The underlying training engine used by MMDetection.
- **Pydantic (>=2.0.0):** Used for strict data validation of all user-provided TOML configurations and API parameters.
- **Typer (>=0.9.0):** The framework for building the `ez-mmdet` CLI.
- **Loguru (>=0.7.3):** The primary logging library, providing structured and colored logs.
- **Rich (>=13.0.0):** Used for advanced terminal output formatting, including tables and progress bars.

## Data & Configuration
- **TOML:** The standard configuration format for datasets (`dataset.toml`) and training runs (`user_config.toml`).
- **tomli / tomli-w:** Libraries for robust TOML parsing and writing.

## Development & Build Tools
- **uv:** The primary tool for dependency management, environment isolation, and project synchronization.
- **setuptools:** Used as the build backend for the project.
- **pytest:** The standard testing framework for unit and integration tests.
- **pytest-cov:** Used for generating code coverage reports, with a project-wide target of >80% coverage.
- **ruff:** The unified Python linter and formatter.

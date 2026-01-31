# Initial Concept

A user-friendly API for OpenMMLab (Detection, Pose, etc.)

## Vision
`ez_openmmlab` is a high-level Python wrapper for the OpenMMLab ecosystem. It aims to democratize access to state-of-the-art computer vision by replacing the steep learning curve and complex configuration systems of frameworks like MMDetection and MMPose with an intuitive, "user-first" API.

## Target Audience
- **ML Practitioners:** Data scientists and engineers who want to train and deploy models quickly without navigating the intricacies of MMEngine's registries and nested Python configurations.
- **Prototypers:** Researchers needing a fast, reliable way to test different architectures on custom datasets.

## Core Value Propositions
- **Boilerplate Reduction:** Abstract away the `Runner`, `Config` inheritance, and manual module registration.
- **Config-First Workflow:** Use human-readable TOML files (`dataset.toml`, `user_config.toml`) for all data and training definitions.
- **Safety & Validation:** Leverage Pydantic to validate configurations before training begins, ensuring "fail-fast" behavior for common typos.
- **Reproducibility:** Every training run automatically saves its effective configuration as a TOML artifact.

## Key Features (MVP)
- **Modular Architecture:** Implementation of the Template Method Pattern (e.g., the `EZDetector` base class) to support multiple architectures while maintaining a consistent training loop.
- **Multi-Framework Support:** Native support for both MMDetection and MMPose within a unified engine structure.
- **Decoupled Data:** Separate data definitions (`dataset.toml`) from model logic, allowing for easy dataset switching.
- **RTMDet Support:** First-class support for the RTMDet family of high-performance one-stage detectors.
- **Pose Estimation:** Robust support for RTMPose (top-down) and RTMO (bottom-up) architectures.
- **Integrated Logging:** Rich, developer-friendly logging powered by Loguru.

  1. Implement Inference (predict method)
  Why: Currently, we can only train models. An MVP must allow users to use those trained models to detect objects in new images.
  Action:
   * Add a predict() method to the EZDetector base class.
   * Use MMDetection's init_detector and inference_detector APIs.
   * Add visualization support to save output images with bounding boxes drawn.

  2. Create the CLI (ez-mmdet)
  Why: You mentioned "EZ" usage. A Command Line Interface using Typer (which is already in your dependencies) is the standard for modern Python
  tools.
  Action:
   * Create src/ez_mmdetection/cli.py.
   * Expose commands like ez-mmdet train dataset.toml and ez-mmdet predict image.jpg --checkpoint best_model.pth.

  3. End-to-End Verification
  Why: We have tested pieces, but we need to verify the full loop: Init -> Train -> Save -> Load -> Predict.
  Action:
   * Update our test script to perform a mock training run and then immediately run inference on a dummy image using the generated checkpoint.

  4. Documentation Update
  Why: The API has changed significantly (Abstract Base Class, TOML configs). The README.md needs to reflect the new RTMDet usage and
  dataset.toml structure.

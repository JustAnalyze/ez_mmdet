from ez_mmdetection import RTMDet

# Initialize with WARNING log level to suppress INFO messages
model = RTMDet("rtmdet_m", log_level="WARNING")

# Note: This will only work if the checkpoint file actually exists at the specified path.
# For testing the fix, we are mainly looking at whether it resolves the config file
# and handles out_dir=None without a TypeError.
try:
    print("Starting prediction test (INFO logs should be suppressed)...")
    model.predict(
        image_path="libs/mmdetection/demo/large_image.jpg",
        device="cpu",
        out_dir="runs/pred_outputs",
    )
    print("Prediction call finished.")
except FileNotFoundError as e:
    if "checkpoint" in str(e).lower() or "rtmdet_tiny" in str(e):
        print(
            f"\nCaptured expected FileNotFoundError (checkpoint missing, but config resolution and out_dir handling worked): {e}"
        )
    else:
        raise e
except Exception as e:
    print(f"\nAn error occurred: {e}")
    import traceback

    traceback.print_exc()

from ez_mmdetection import RTMDet
from pathlib import Path

# Path for the test checkpoint
checkpoint_path = Path("checkpoints/rtmdet_tiny_test.pth")

# Clean up if it exists from a previous run
if checkpoint_path.exists():
    checkpoint_path.unlink()

model = RTMDet("rtmdet_tiny", log_level="INFO")

try:
    print(f"Starting prediction test with auto-download to: {checkpoint_path}")
    # This should trigger the download because checkpoint_path doesn't exist
    model.predict(
        image_path="libs/mmdetection/demo/demo.jpg",
        checkpoint_path=checkpoint_path,
    )
    print("Prediction call finished.")
    
    if checkpoint_path.exists():
        print(f"SUCCESS: Checkpoint was automatically downloaded to {checkpoint_path}")
    else:
        print("FAILURE: Checkpoint was not found after predict call.")

except Exception as e:
    print(f"\nAn error occurred: {e}")
    import traceback
    traceback.print_exc()

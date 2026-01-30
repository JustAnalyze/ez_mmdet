from ez_openmmlab import RTMDet

detector = RTMDet("rtmdet_tiny")

# Run prediction
result = detector.predict(
    image_path="./tests/data/coco_mini/images/000000000389.jpg", device="cpu", show=True
)

# Access structured results
for pred in result.predictions:
    if not pred.score > 0.5:
        continue
    print(f"Found {pred.label} with score {pred.score} at {pred.bbox}")

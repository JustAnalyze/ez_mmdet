from ez_openmmlab import RTMDet

detector = RTMDet("rtmdet-ins_s")

# Run prediction
result = detector.predict(
    image_path="./tests/data/coco_mini/images/000000000389.jpg",
    device="cpu",
    show=True,
    confidence=0.5,
    out_dir="./runs/rtmdet_preds",
)

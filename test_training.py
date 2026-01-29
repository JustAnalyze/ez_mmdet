from ez_mmdetection import RTMDet

model = RTMDet("rtmdet_tiny", log_level="ERROR")

model.train(
    "datasets/coco128_coco/dataset.toml",
    epochs=10,
    device="cpu",
    amp=False,
    num_workers=4,
    enable_tensorboard=True,
)

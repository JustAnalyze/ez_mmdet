from ez_mmdetection import RTMDet

model = RTMDet("rtmdet_tiny", log_level="INFO")

model.train(
    "datasets/coco128_coco/dataset.toml",
    epochs=5,
    device="cpu",
    amp=False,
    num_workers=4,
    enable_tensorboard=True,
    work_dir="./runs/test_train",
)

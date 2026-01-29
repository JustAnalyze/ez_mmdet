from ez_openmmlab import RTMDet

model = RTMDet("rtmdet_tiny")
model.train(
    "datasets/coco128_coco/dataset.toml",
    work_dir="./runs/sample_train",
    device="cpu",
    amp=False,
)

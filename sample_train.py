from ez_openmmlab import RTMDet


model = RTMDet(model_name="rtmdet_tiny")
model.train(
    "./tests/data/coco_mini/dataset.toml",
    work_dir="./runs/sample_train",
    device="cpu",
    amp=False,
    epochs=10,
)

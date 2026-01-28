from ez_mmdetection import RTMDet

model = RTMDet("rtmdet_tiny", log_level="WARNING")

model.train("datasets/coco128_coco/dataset.toml", epochs=10, device="cpu", amp=False)

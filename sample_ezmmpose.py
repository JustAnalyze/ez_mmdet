from ez_openmmlab import RTMPose

model = RTMPose("rtmpose_s")

model.predict(
    image_path="./tests/data/test_image.jpg",
    device="cpu",
    show=True,
    out_dir="./runs/rtmpose_preds",
    bbox_thr=0.4,
    kpt_thr=0.5,
)

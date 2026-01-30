from ez_openmmlab import RTMPose

model = RTMPose("rtmpose_s")

model.predict(
    image_path="./tests/data/coco_mini/images/000000000389.jpg",
    device="cpu",
    show=True,
    out_dir="./runs/rtmpose_preds",
    bbox_thr=0.5,
    kpt_thr=0.5
)

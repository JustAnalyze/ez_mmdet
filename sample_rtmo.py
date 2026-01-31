from ez_openmmlab import RTMO

# RTMO is a bottom-up model, it detects bboxes and keypoints in one go.
model = RTMO("rtmo_s")

model.predict(
    image_path="./tests/data/coco_mini/images/000000000564.jpg",
    device="cpu",
    show=True,
    out_dir="./runs/rtmo_preds",
    bbox_thr=0.7,
    kpt_thr=0.7,
)

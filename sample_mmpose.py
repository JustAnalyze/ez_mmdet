# from mmcv.image import imread

from mmcv.image import imread
from mmpose.apis import inference_topdown, init_model
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples

# Keep this
model_cfg = "./libs/mmpose/configs/body_2d_keypoint/rtmpose/coco/rtmpose-m_8xb256-420e_coco-256x192.py"

# Use the matching COCO checkpoint
ckpt = (
    "./checkpoints/rtmpose-m_simcc-coco_pt-aic-coco_420e-256x192-d8dd5ca4_20230127.pth"
)

device = "cpu"

# init model
model = init_model(model_cfg, ckpt, device=device)

img_path = "./tests/data/coco_mini/images/000000000389.jpg"

# inference on a single image
batch_results = inference_topdown(model, img_path)

# merge results as a single data sample
results = merge_data_samples(batch_results)

# build the visualizer
visualizer = VISUALIZERS.build(model.cfg.visualizer)

# set skeleton, colormap and joint connection rule
visualizer.set_dataset_meta(model.dataset_meta)

img = imread(img_path, channel_order="rgb")

# visualize the results
visualizer.add_datasample("result", img, data_sample=results, show=True)

# ANOTHER WAY TO VISUALIZE
from mmpose.apis import visualize

pred_instances = batch_results[0].pred_instances

keypoints = pred_instances.keypoints
keypoint_scores = pred_instances.keypoint_scores

metainfo = "config/_base_/datasets/coco.py"

visualize(img_path, keypoints, keypoint_scores, metainfo=metainfo, show=True)

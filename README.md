# MMDetection Project Setup: `ez_mmdet`

## 🛠 Installation Verification

Before starting any development or training, ensure your environment is correctly configured.

### 1. Environment Audit

Run the standard OpenMMLab environment collection script to check versions of PyTorch, MMCV, and MMDetection.

```bash
uv run python mmdetection/mmdet/utils/collect_env.py

```

### 2. Functional Sanity Check

Run the custom verification script to ensure the C++ extensions (`_ext`) are correctly loaded and operational on your hardware.

```bash
uv run python verify_install.py

```

_A successful output should show: `✅ Success! Output shape: torch.Size([1, 3, 7, 7])`._

### 3. Inference Demo (CPU/GPU)

Run a basic inference test on a sample image to verify the full model pipeline.

```bash
uv run python -c "
import torch
from mmdet.apis import DetInferencer

# Automatically detect the best available hardware
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Running inference on: {device}')

# Initialize the inferencer
inferencer = DetInferencer(model='rtmdet_tiny_8xb32-300e_coco', device=device)

# Perform inference
inferencer('mmdetection/demo/demo.jpg', out_dir='./output_test')
print(f'\nSuccess! Results saved to ./output_test/vis/demo.jpg')
"
```

# MMDetection Project Setup: `ez_mmdet`

---

## 🚀 Quick Start

### 1. Prerequisites

Ensure you have `uv` installed. If not, run:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh

```

### 2. Standard Installation (Automated)

This project is configured to handle the complex MMDetection build dependencies automatically via `uv sync`.

```bash
# Clone the repository
git clone https://github.com/JustAnalyze/ez_mmdet.git
cd ez_mmdet

# Install everything (including CPU-specific Torch and MMCV)
uv sync --extra cpu --preview

```

### 3. Manual Bootstrap (If `uv sync` fails)

If you encounter `ModuleNotFoundError` during the sync, the environment may need a manual "seed" to help MMDetection's legacy `setup.py` run:

```bash
# 1. Install build-essential tools first
uv pip install setuptools==69.5.1 --index-strategy unsafe-best-match

# 2. Install the CPU engine
uv pip install wheel torch==1.13.1+cpu --index https://download.pytorch.org/whl/cpu

# 3. Finalize the project sync
uv sync --extra cpu --no-build-isolation --preview

```

---

## 🛠 Installation Verification

Before starting any development or training, ensure your environment is correctly configured.

### 1. Environment Audit

Run the standard OpenMMLab environment collection script to check versions of PyTorch, MMCV, and MMDetection.

```bash
uv run python mmdetection/mmdet/utils/collect_env.py

```

### 2. Inference Demo (CPU/GPU)

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

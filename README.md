# 🚀 ez_mmdet: Object Detection Made Simple

`ez_mmdet` is a user-friendly Python wrapper for the powerful [MMDetection](https://github.com/open-mmlab/mmdetection) framework. It eliminates the complexity of nested Python configurations and MMEngine boilerplate, providing a streamlined, "Config-First" workflow for training and inference.

---

## ✨ Key Features

- **Intuitive API:** Train and predict with simple classes like `RTMDet`.
- **Config-First Workflow:** Decouple your data from your model using human-readable `dataset.toml` files.
- **Auto-Magic Checkpoints:** Missing a model? `ez_mmdet` automatically downloads official checkpoints to your `checkpoints/` folder with clean, simplified names.
- **Strict Validation:** Powered by Pydantic to catch configuration errors before you start a 10-hour training run.
- **Built-in CLI:** Run experiments directly from your terminal with the `ez-mmdet` command.

---

## 🛠️ Installation

```bash
# Clone the repository (including the MMDetection submodule)
git clone --recursive https://github.com/JustAnalyze/ez_mmdet.git
cd ez_mmdet

# Install for CPU
uv sync --extra cpu --preview

# OR: Install for GPU (CUDA 11.7)
uv sync --extra gpu --preview
```

### 💡 Troubleshooting `uv sync`

If `uv sync` fails (often due to MMDetection's complex build requirements), you can manually "bootstrap" the environment with the core build dependencies:

**For CPU:**

```bash
# 1. Install build-essential tools first
uv pip install "setuptools>=69.5.1" --index-strategy unsafe-best-match
uv pip install wheels

# 2. Install the Torch engine required for building
uv pip install torch==2.0.1+cpu --index https://download.pytorch.org/whl/cpu

# 3. Retry the project sync
uv sync --extra cpu --preview
```

**For GPU (CUDA 11.7):**

```bash
# 1. Install build-essential tools first
uv pip install "setuptools>=69.5.1" --index-strategy unsafe-best-match
uv pip install wheels

# 2. Install the Torch engine required for building
uv pip install torch==2.0.1+cu117 --index https://download.pytorch.org/whl/cu117

# 3. Retry the project sync
uv sync --extra gpu --preview
```

---

## 📖 Quick Start

### 1. Define your Data (`dataset.toml`)

Create a file to describe your dataset structure. No more editing framework internals.

```toml
data_root = "datasets/my_project"
classes = ["cat", "dog"]

[train]
ann_file = "annotations/train.json"
img_dir = "images/train"

[val]
ann_file = "annotations/val.json"
img_dir = "images/val"
```

### 2. Train a Model

#### Using Python

```python
from ez_mmdetection import RTMDet

# Initialize (choices: rtmdet_tiny, rtmdet_s, rtmdet-ins_tiny, etc.)
detector = RTMDet("rtmdet_tiny")

# Start training
detector.train(
    dataset_config_path="dataset.toml",
    epochs=50,
    batch_size=8,
    work_dir="./runs/my_experiment"
)
```

#### Using the CLI

```bash
ez-mmdet train rtmdet_tiny dataset.toml --epochs 50 --batch-size 8
```

### 3. Run Inference

`ez_mmdet` automatically manages your checkpoints. If you don't provide a path, it downloads the best official model for you.

```python
from ez_mmdetection import RTMDet

detector = RTMDet("rtmdet_tiny")

# Run prediction
result = detector.predict(image_path="sample.jpg")

# Access structured results
for pred in result.predictions:
    print(f"Found {pred.label} with score {pred.score} at {pred.bbox}")
```

---

## 🗺️ Roadmap & Future Plans

We are building `ez_mmdet` to be the easiest entry point into the OpenMMLab ecosystem. Here is what we're working on:

- [ ] **Deployment Support:** Native `export` method to convert your trained `.pth` models to **ONNX** and TensorRT formats for production.
- [ ] **Architecture Expansion:** Beyond RTMDet, we plan to bring the "EZ" treatment to **YOLOv8**, **Faster R-CNN**, and **DINO**.
- [ ] **MMPose Integration:** Supporting human pose estimation via a similar `EZPose` API.

---

## 🤝 Contributing

This project is in its early stages (MVP). We value your feedback! If you find a bug or have a feature request, please open an issue.

**Current Supported Models:**

- **Detection:** `rtmdet_tiny`, `rtmdet_s`, `rtmdet_m`, `rtmdet_l`, `rtmdet_x`
- **Instance Segmentation:** `rtmdet-ins_tiny`, `rtmdet-ins_s`, `rtmdet-ins_m`, `rtmdet-ins_l`, `rtmdet-ins_x`

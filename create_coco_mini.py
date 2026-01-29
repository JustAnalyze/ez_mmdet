import json
import shutil
from pathlib import Path

def create_mini_dataset(source_dir: Path, dest_dir: Path, num_images: int = 5):
    # 1. Create directories
    (dest_dir / "annotations").mkdir(parents=True, exist_ok=True)
    (dest_dir / "images").mkdir(parents=True, exist_ok=True)

    # 2. Load source annotations
    ann_path = source_dir / "annotations" / "train.json"
    with open(ann_path, 'r') as f:
        coco_data = json.load(f)

    # 3. Select a subset of images
    selected_images = coco_data['images'][:num_images]
    selected_image_ids = {img['id'] for img in selected_images}
    selected_filenames = {img['file_name'] for img in selected_images}

    # 4. Filter annotations
    selected_annotations = [
        ann for ann in coco_data['annotations'] 
        if ann['image_id'] in selected_image_ids
    ]

    # 5. Create mini COCO data
    mini_coco = {
        "images": selected_images,
        "annotations": selected_annotations,
        "categories": coco_data['categories']
    }

    # 6. Save mini annotations
    with open(dest_dir / "annotations" / "train.json", 'w') as f:
        json.dump(mini_coco, f, indent=4)

    # 7. Copy images
    for filename in selected_filenames:
        src_img = source_dir / "images" / filename
        dst_img = dest_dir / "images" / filename
        if src_img.exists():
            shutil.copy(src_img, dst_img)
            print(f"Copied {filename}")
        else:
            print(f"Warning: {filename} not found in source")

    # 8. Create dataset.toml
    toml_content = f"""
data_root = "{dest_dir.resolve().relative_to(Path.cwd().resolve())}"
classes = {[cat['name'] for cat in coco_data['categories']]}

[train]
ann_file = "annotations/train.json"
img_dir = "images"

[val]
ann_file = "annotations/train.json"
img_dir = "images"
"""
    (dest_dir / "dataset.toml").write_text(toml_content)
    print(f"\nMini dataset created at {dest_dir}")

if __name__ == "__main__":
    source = Path("datasets/coco128_coco")
    dest = Path("datasets/coco_mini")
    create_mini_dataset(source, dest)

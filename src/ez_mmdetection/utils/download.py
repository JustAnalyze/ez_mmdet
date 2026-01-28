from pathlib import Path
from typing import Dict, Optional, Union

import requests
from loguru import logger
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

from ez_mmdetection.schemas.model import ModelName

# Mapping of model names to their official MMDetection checkpoint URLs
MODEL_URLS: Dict[str, str] = {
    # Bounding Box detection
    ModelName.RTM_DET_TINY.value: "https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_tiny_8xb32-300e_coco/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth",
    ModelName.RTM_DET_S.value: "https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_s_8xb32-300e_coco/rtmdet_s_8xb32-300e_coco_20220905_161602-387a891e.pth",
    ModelName.RTM_DET_M.value: "https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_m_8xb32-300e_coco/rtmdet_m_8xb32-300e_coco_20220719_112220-229f527c.pth",
    ModelName.RTM_DET_L.value: "https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_l_8xb32-300e_coco/rtmdet_l_8xb32-300e_coco_20220719_112030-5a0be7c4.pth",
    ModelName.RTM_DET_X.value: "https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_x_8xb32-300e_coco/rtmdet_x_8xb32-300e_coco_20220715_230555-cc79b9ae.pth",
    # Instance Segmentation
    ModelName.RTM_DET_INS_TINY.value: "https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet-ins_tiny_8xb32-300e_coco/rtmdet-ins_tiny_8xb32-300e_coco_20221130_151727-ec670f7e.pth",
    ModelName.RTM_DET_INS_S.value: "https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet-ins_s_8xb32-300e_coco/rtmdet-ins_s_8xb32-300e_coco_20221121_212604-fdc5d7ec.pth",
    ModelName.RTM_DET_INS_M.value: "https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet-ins_m_8xb32-300e_coco/rtmdet-ins_m_8xb32-300e_coco_20221123_001039-6eba602e.pth",
    ModelName.RTM_DET_INS_L.value: "https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet-ins_l_8xb32-300e_coco/rtmdet-ins_l_8xb32-300e_coco_20221124_103237-78d1d652.pth",
    ModelName.RTM_DET_INS_X.value: "https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet-ins_x_8xb16-300e_coco/rtmdet-ins_x_8xb16-300e_coco_20221124_111313-33d4595b.pth",
}


def download_checkpoint(url: str, dest_path: Path) -> None:
    """Downloads a file from a URL to a destination path with a progress bar."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get("content-length", 0))

    filename = dest_path.name
    logger.info(f"Downloading checkpoint to {dest_path}...")

    progress = Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
    )

    with progress:
        task_id = progress.add_task(
            f"Downloading {filename}", total=total_size
        )
        with open(dest_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                progress.update(task_id, advance=len(chunk))

    logger.info(f"Successfully downloaded {filename}")


def ensure_model_checkpoint(
    model_name: str, checkpoint_path: Optional[Union[str, Path]] = None
) -> Path:
    """Checks if a checkpoint exists. If not, attempts to download it if it's a known model.
    Simplified names like 'rtmdet_tiny.pth' are used for auto-downloads.
    """
    project_root = Path.cwd()
    checkpoint_dir = project_root / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if checkpoint_path:
        path = Path(checkpoint_path)
        # If it's just a filename, put it in checkpoints dir
        if not path.parent or str(path.parent) == ".":
            path = checkpoint_dir / path
    else:
        # Default name for auto-download
        path = checkpoint_dir / f"{model_name}.pth"

    if path.exists():
        return path

    url = MODEL_URLS.get(model_name)
    if not url:
        if checkpoint_path:
            logger.error(
                f"Checkpoint not found at {path} and no download URL for {model_name}"
            )
            raise FileNotFoundError(f"Checkpoint not found at {path}")
        logger.warning(
            f"No default checkpoint URL found for model: {model_name}"
        )
        return path

    logger.info(
        f"Checkpoint not found. Attempting automatic download to {path}..."
    )
    download_checkpoint(url, path)
    return path

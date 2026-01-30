from pathlib import Path
from typing import Optional, Union

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

from ez_openmmlab.schemas.model import ModelName


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
        task_id = progress.add_task(f"Downloading {filename}", total=total_size)
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

    try:
        model = ModelName(model_name)
        url = model.weights_url
    except ValueError:
        if checkpoint_path:
            logger.error(
                f"Checkpoint not found at {path} and no download URL for {model_name}"
            )
            raise FileNotFoundError(f"Checkpoint not found at {path}")
        logger.warning(f"No default checkpoint URL found for model: {model_name}")
        return path

    logger.info(f"Checkpoint not found. Attempting automatic download to {path}...")
    download_checkpoint(url, path)
    return path


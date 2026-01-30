from pathlib import Path
from typing import List, Optional

import tomli
from pydantic import BaseModel, Field


class SplitConfig(BaseModel):
    """Configuration for a specific data split (train/val/test)."""

    ann_file: str  # Path relative to data_root (e.g., 'annotations/train.json')
    img_dir: str  # Path relative to data_root (e.g., 'train2017/')


class DatasetConfig(BaseModel):
    """Master configuration for the dataset."""

    data_root: Path
    train: SplitConfig
    val: SplitConfig
    test: Optional[SplitConfig] = None
    classes: Optional[List[str]] = Field(
        None, description="Explicit class names for safety"
    )
    metainfo: Optional[dict] = Field(
        None, description="Task-specific metadata (e.g., keypoints for pose)"
    )

    @classmethod
    def from_toml(cls, path: Path) -> "DatasetConfig":
        """Parses a TOML file into a strict DatasetConfig object."""
        if not path.exists():
            raise FileNotFoundError(f"Dataset config not found at {path}")

        with open(path, "rb") as f:
            data = tomli.load(f)

        return cls(**data)

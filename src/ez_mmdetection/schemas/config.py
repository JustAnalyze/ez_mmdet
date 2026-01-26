from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field, validator


class DataConfig(BaseModel):
    """Configuration for dataset paths and parameters."""

    data_root: Path = Field(..., description="Root directory of the dataset")
    train_ann: str = Field(
        "train/_annotations.coco.json",
        description="Path to train annotations relative to root",
    )
    val_ann: str = Field(
        "valid/_annotations.coco.json",
        description="Path to validation annotations relative to root",
    )
    train_img: str = Field(
        "train", description="Path to train images relative to root"
    )
    val_img: str = Field(
        "valid", description="Path to validation images relative to root"
    )
    batch_size: int = Field(8, gt=0, description="Batch size per GPU")
    num_workers: int = Field(2, ge=0, description="Dataloader workers")

    @validator("data_root")
    def validate_root(cls, v: Path) -> Path:
        if not v.exists():
            raise ValueError(f"Data root {v} does not exist.")
        return v


class ModelConfig(BaseModel):
    """Configuration for the model architecture."""

    base_model: str = Field(
        "rtmdet_tiny", description="Base architecture name"
    )
    num_classes: int = Field(..., gt=0, description="Number of object classes")
    load_from: Optional[Path] = Field(
        None, description="Path to pretrained weights"
    )


class TrainingArgs(BaseModel):
    """Aggregated arguments for a training session."""

    data: DataConfig
    model: ModelConfig
    epochs: int = Field(100, gt=0)
    learning_rate: float = Field(0.001, gt=0.0)
    work_dir: Path = Field(
        Path("./runs"), description="Directory to save outputs"
    )
    device: Literal["cuda", "cpu"] = "cuda"

from pathlib import Path
from typing import List, Optional

import tomli
import tomli_w
from pydantic import BaseModel, Field


# --- Pydantic Models for Validation ---
class ModelSection(BaseModel):
    name: str = "rtmdet_tiny"
    num_classes: int = Field(..., gt=0)
    load_from: Optional[str] = None


class DataSection(BaseModel):
    root: str
    train_ann: str = "annotations/instances_train2017.json"
    train_img: str = "train2017/"
    val_ann: str = "annotations/instances_val2017.json"
    val_img: str = "val2017/"
    classes: Optional[List[str]] = None


class TrainingSection(BaseModel):
    epochs: int = Field(100, gt=0)
    batch_size: int = Field(8, gt=0)
    learning_rate: float = Field(0.001, gt=0.0)
    device: str = "cuda"
    work_dir: str = "./runs/train"


class UserConfig(BaseModel):
    """The master schema for config.toml."""

    model: ModelSection
    data: DataSection
    training: TrainingSection


# --- Utilities ---
def load_user_config(path: Path) -> UserConfig:
    """Reads a TOML file and validates it against the schema."""
    with open(path, "rb") as f:
        data = tomli.load(f)
    return UserConfig(**data)


def save_user_config(config: UserConfig, path: Path) -> None:
    """Writes the configuration to a TOML file."""
    # Convert Pydantic model to dict, filtering None to keep it clean
    data = config.model_dump(exclude_none=True)

    # Write to file
    with open(path, "wb") as f:
        tomli_w.dump(data, f)

from pathlib import Path
from typing import Optional

import typer

from ez_openmmlab.models.mmdet import RTMDet
from ez_openmmlab.models.mmpose import RTMPose
from ez_openmmlab.schemas.model import ModelName

app = typer.Typer(help="ez_mmdet: A user-friendly CLI for MMDetection")


@app.command()
def train(
    model_name: ModelName = typer.Argument(..., help="Name of the model architecture"),
    dataset_config_path: Path = typer.Argument(
        ..., help="Path to the dataset.toml file"
    ),
    epochs: int = typer.Option(100, help="Number of training epochs"),
    batch_size: int = typer.Option(8, help="Batch size per GPU"),
    num_workers: int = typer.Option(2, help="Number of dataloader workers"),
    work_dir: str = typer.Option(
        "./runs/train", help="Directory to save logs and checkpoints"
    ),
    device: str = typer.Option("cuda", help="Training device"),
    learning_rate: float = typer.Option(0.001, help="Initial learning rate"),
    amp: bool = typer.Option(True, help="Enable Automatic Mixed Precision training"),
    tensorboard: bool = typer.Option(False, help="Enable TensorBoard logging"),
):
    """Starts model training using a dataset configuration."""
    detector = RTMDet(model_name=model_name)
    detector.train(
        dataset_config_path=dataset_config_path,
        epochs=epochs,
        batch_size=batch_size,
        num_workers=num_workers,
        work_dir=work_dir,
        device=device,
        learning_rate=learning_rate,
        amp=amp,
        enable_tensorboard=tensorboard,
    )


@app.command()
def predict(
    model_name: ModelName = typer.Argument(..., help="Name of the model architecture"),
    image_path: Path = typer.Argument(..., help="Path to the image for inference"),
    checkpoint_path: Optional[Path] = typer.Option(
        None, help="Optional path to a custom model checkpoint (.pth)"
    ),
    confidence: float = typer.Option(0.3, help="Confidence threshold for detections"),
    bbox_thr: float = typer.Option(
        0.3, help="Bounding box score threshold (pose estimation)"
    ),
    kpt_thr: float = typer.Option(
        0.3, help="Keypoint score threshold (pose estimation)"
    ),
    out_dir: Optional[str] = typer.Option(
        "runs/preds", help="Directory to save visualization results"
    ),
    device: str = typer.Option("cpu", help="Computing device"),
):
    """Performs object detection or pose estimation on an image."""
    if "rtmpose" in model_name.value:
        detector = RTMPose(model_name=model_name, checkpoint_path=checkpoint_path)
        detector.predict(
            image_path=image_path,
            bbox_thr=bbox_thr,
            kpt_thr=kpt_thr,
            out_dir=out_dir,
            device=device,
        )
    else:
        detector = RTMDet(model_name=model_name, checkpoint_path=checkpoint_path)
        detector.predict(
            image_path=image_path,
            confidence=confidence,
            out_dir=out_dir,
            device=device,
        )


if __name__ == "__main__":
    app()

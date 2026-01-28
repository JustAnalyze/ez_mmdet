import typer
from pathlib import Path
from typing import Optional

app = typer.Typer(help="ez_mmdet: A user-friendly CLI for MMDetection")

@app.command()
def train(
    dataset_config_path: Path = typer.Argument(..., help="Path to the dataset.toml file"),
    epochs: int = typer.Option(100, help="Number of training epochs"),
    batch_size: int = typer.Option(8, help="Batch size per GPU"),
    work_dir: str = typer.Option("./runs/train", help="Directory to save logs and checkpoints"),
    device: str = typer.Option("cuda", help="Training device"),
):
    """Starts model training using a dataset configuration."""
    typer.echo(f"Training with {dataset_config_path}")

@app.command()
def predict(
    model_name: str = typer.Argument(..., help="Name of the model architecture"),
    checkpoint_path: Path = typer.Argument(..., help="Path to the model checkpoint"),
    image_path: Path = typer.Argument(..., help="Path to the image for inference"),
    out_dir: Optional[str] = typer.Option(None, help="Directory to save visualization results"),
):
    """Performs object detection on an image."""
    typer.echo(f"Predicting with {model_name}")

if __name__ == "__main__":
    app()

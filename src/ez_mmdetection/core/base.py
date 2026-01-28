from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Union

from loguru import logger
from mmdet.apis import DetInferencer # New import
from mmdet.utils import register_all_modules
from mmengine.config import Config
from mmengine.runner import Runner

from ez_mmdetection.core.config_loader import get_config_file
from ez_mmdetection.schemas.dataset import DatasetConfig
from ez_mmdetection.utils.toml_config import (
    DataSection,
    ModelSection,
    TrainingSection,
    UserConfig,
    save_user_config,
)

# Force registration of MMDet modules
register_all_modules()


class EZMMDetector(ABC):
    """Abstract base class for training and inference using MMDetection.

    Implements the Template Method Pattern for the training workflow.
    """

    def __init__(self, model_name: str):
        """Initializes the detector with a base model.

        Args:
            model_name: The name of the architecture (e.g., 'rtmdet_tiny').
        """
        logger.info(
            f"Initializing {self.__class__.__name__} with base model: '{model_name}'"
        )
        self.model_name: str = model_name
        self._cfg: Optional[Config] = None
        self._inferencer: Optional[DetInferencer] = None

    def predict(
        self,
        image_path: Union[str, Path],
        checkpoint_path: Union[str, Path],
        device: str = "cpu",
        out_dir: Optional[str] = None,
        show: bool = False,
    ) -> dict:
        """Performs object detection on an image.

        Args:
            image_path: Path to the image file.
            checkpoint_path: Path to the model checkpoint (.pth).
            device: Computing device (default: 'cpu').
            out_dir: Directory to save visualization results.
            show: Whether to display the image.

        Returns:
            A dictionary containing the detection results.
        """
        if self._inferencer is None:
            logger.info(f"Initializing inferencer for model: {self.model_name}")
            self._inferencer = DetInferencer(
                model=self.model_name, weights=str(checkpoint_path), device=device
            )

        logger.info(f"Running inference on: {image_path}")
        results = self._inferencer(str(image_path), out_dir=out_dir, show=show)
        return results

    def train(
        self,
        dataset_config_path: Union[str, Path],
        epochs: int = 100,
        batch_size: int = 8,
        device: str = "cuda",
        work_dir: str = "./runs/train",
        learning_rate: float = 0.001,
        load_from: Optional[str] = None,
        log_level: str = "INFO",
    ) -> None:
        """The Template Method defining the training workflow.

        Args:
            dataset_config_path: Path to the dataset.toml file.
            epochs: Number of training epochs.
            batch_size: Batch size per GPU.
            device: Training device ('cuda' or 'cpu').
            work_dir: Directory to save logs and checkpoints.
            learning_rate: Base learning rate.
            load_from: Path to a checkpoint to resume from or load weights.
            log_level: Logging level (e.g., 'INFO', 'WARNING'). Default is 'INFO'.
        """
        logger.info(
            f"Loading dataset configuration from: {dataset_config_path}"
        )
        dataset_cfg = DatasetConfig.from_toml(Path(dataset_config_path))
        # Populate internal state from dataset config
        self.classes = dataset_cfg.classes
        self.num_classes: int = (
            len(dataset_cfg.classes) if dataset_cfg.classes else 80
        )
        # Construct the UserConfig artifact
        user_config = UserConfig(
            model=ModelSection(
                name=self.model_name,
                num_classes=self.num_classes,
                load_from=load_from,
            ),
            data=DataSection(
                root=str(dataset_cfg.data_root),
                train_ann=dataset_cfg.train.ann_file,
                train_img=dataset_cfg.train.img_dir,
                val_ann=dataset_cfg.val.ann_file,
                val_img=dataset_cfg.val.img_dir,
                classes=self.classes,
            ),
            training=TrainingSection(
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                device=device,
                work_dir=work_dir,
                log_level=log_level,
            ),
        )

        self._run_training_workflow(user_config)

    def _run_training_workflow(self, config: UserConfig) -> None:
        """Orchestrates the internal MMDetection setup and execution."""
        # 1. Reproducibility: Save the effective config
        work_dir = Path(config.training.work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        save_user_config(config, work_dir / "user_config.toml")
        logger.info(
            f"User configuration saved to: {work_dir / 'user_config.toml'}"
        )

        # 2. Load and Apply Overrides
        self._cfg = self._load_base_config(config.model.name)
        self._apply_common_overrides(config)

        # 3. Architecture Specifics (Template Method Gap)
        logger.info(
            f"Configuring architecture specifics for {self.__class__.__name__}..."
        )
        self._configure_model_specifics(config)

        # 4. Execute Runner
        logger.info("Starting MMEngine Runner...")
        runner = Runner.from_cfg(self._cfg)
        runner.train()

    def _load_base_config(self, model_name: str) -> Config:
        config_path = get_config_file(model_name)
        return Config.fromfile(config_path)

    def _apply_common_overrides(self, config: UserConfig) -> None:
        """Applies configuration changes common to all architectures."""
        if not self._cfg:
            raise RuntimeError("Base config not loaded.")

        self._cfg.work_dir = config.training.work_dir
        self._cfg.train_cfg.max_epochs = config.training.epochs
        self._cfg.load_from = config.model.load_from

        if hasattr(self._cfg, "optim_wrapper") and hasattr(
            self._cfg.optim_wrapper, "optimizer"
        ):
            self._cfg.optim_wrapper.optimizer.lr = (
                config.training.learning_rate
            )

        self._cfg.log_level = config.training.log_level

        # Data roots and paths
        self._cfg.data_root = config.data.root
        for key in ["train_dataloader", "val_dataloader", "test_dataloader"]:
            if hasattr(self._cfg, key):
                dl = getattr(self._cfg, key)
                dl.dataset.data_root = config.data.root
                dl.batch_size = config.training.batch_size

                # Annotation files and image prefixes
                if key == "train_dataloader":
                    dl.dataset.ann_file = config.data.train_ann
                    dl.dataset.data_prefix = {"img": config.data.train_img}
                else:
                    dl.dataset.ann_file = config.data.val_ann
                    dl.dataset.data_prefix = {"img": config.data.val_img}

                if config.data.classes:
                    dl.dataset.metainfo = {"classes": config.data.classes}

        # Also set metainfo at the top level of the config if possible
        if config.data.classes:
            self._cfg.metainfo = {"classes": config.data.classes}

    @abstractmethod
    def _configure_model_specifics(self, config: UserConfig) -> None:
        """Subclasses must implement this to handle architecture-specific overrides."""
        pass

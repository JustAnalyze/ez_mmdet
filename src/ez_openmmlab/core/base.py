from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Union

from loguru import logger
from mmengine.config import Config
from mmengine.runner import Runner

from ez_openmmlab.core.config_loader import get_config_file
from ez_openmmlab.core.handlers import DataloaderHandler, RuntimeHandler
from ez_openmmlab.schemas.dataset import DatasetConfig
from ez_openmmlab.schemas.inference import InferenceResult
from ez_openmmlab.schemas.model import ModelName
from ez_openmmlab.utils.download import ensure_model_checkpoint
from ez_openmmlab.utils.toml_config import (
    DataSection,
    ModelSection,
    TrainingSection,
    UserConfig,
    save_user_config,
)


class EZMMLab(ABC):
    """Abstract base class for all OpenMMLab libraries (Detection, Pose, etc.).

    Implements shared logic for configuration management and training workflows.
    """

    model_name: str
    log_level: str
    checkpoint_path: Path
    _cfg: Optional[Config]

    def __init__(
        self,
        model_name: ModelName,
        checkpoint_path: Optional[Union[str, Path]] = None,
        log_level: str = "INFO",
    ):
        """Initializes the library wrapper with a base model.

        Args:
            model_name: The name of the architecture (e.g., 'rtmdet_tiny').
            checkpoint_path: Path to a specific checkpoint (.pth or .pt).
            log_level: Global logging level. Default is 'INFO'.
        """
        logger.info(
            f"Initializing {self.__class__.__name__} with base model: '{model_name}'"
        )
        self.model_name: str = (
            model_name.value if isinstance(model_name, ModelName) else model_name
        )
        self.log_level: str = log_level
        self._cfg: Optional[Config] = None

        # Resolve or download checkpoint
        self.checkpoint_path = ensure_model_checkpoint(self.model_name, checkpoint_path)

        # Configure loguru level
        try:
            logger.remove()
            import sys

            logger.add(sys.stderr, level=log_level)
        except Exception as e:
            logger.warning(f"Failed to set log level: {e}")

    @abstractmethod
    def predict(self, *args, **kwargs):
        """Abstract method for performing inference."""
        pass

    def train(
        self,
        dataset_config_path: Union[str, Path],
        epochs: int = 100,
        batch_size: int = 8,
        device: str = "cuda",
        work_dir: str = "./runs/train",
        learning_rate: float = 0.001,
        amp: bool = True,
        num_workers: int = 4,
        enable_tensorboard: bool = False,
        log_level: Optional[str] = None,
    ) -> None:
        """Runs the end-to-end training pipeline.

        Args:
            dataset_config_path: Path to the dataset.toml definition.
            epochs: Number of training epochs.
            batch_size: Total batch size for training.
            device: Training hardware ('cuda', 'cpu', 'mps').
            work_dir: Directory for logs and checkpoints.
            learning_rate: Initial learning rate.
            amp: Enable Automatic Mixed Precision.
            num_workers: Number of data loading workers.
            enable_tensorboard: Enable TensorBoard visualization.
            log_level: Override for internal framework logging.
        """
        target_log_level = log_level or self.log_level

        logger.info(f"Loading dataset configuration from: {dataset_config_path}")
        dataset_cfg = DatasetConfig.from_toml(Path(dataset_config_path))

        self.classes = dataset_cfg.classes
        self.num_classes: int = len(dataset_cfg.classes) if dataset_cfg.classes else 80

        user_config = UserConfig(
            model=ModelSection(
                name=self.model_name,
                num_classes=self.num_classes,
                load_from=str(self.checkpoint_path),
            ),
            data=DataSection(
                root=str(dataset_cfg.data_root),
                train_ann=dataset_cfg.train.ann_file,
                train_img=dataset_cfg.train.img_dir,
                val_ann=dataset_cfg.val.ann_file,
                val_img=dataset_cfg.val.img_dir,
                test_ann=dataset_cfg.test.ann_file if dataset_cfg.test else None,
                test_img=dataset_cfg.test.img_dir if dataset_cfg.test else None,
                classes=self.classes,
            ),
            training=TrainingSection(
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                device=device,
                work_dir=work_dir,
                log_level=target_log_level,
                amp=amp,
                num_workers=num_workers,
                enable_tensorboard=enable_tensorboard,
            ),
        )

        self._run_training_workflow(user_config)

    def _run_training_workflow(self, config: UserConfig) -> None:
        """Orchestrates the internal OpenMMLab setup and execution."""
        work_dir = Path(config.training.work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        save_user_config(config, work_dir / "user_config.toml")
        logger.info(f"User configuration saved to: {work_dir / 'user_config.toml'}")

        self._cfg = self._load_base_config(config.model.name)
        self._apply_common_overrides(config)

        logger.info(
            f"Configuring architecture specifics for {self.__class__.__name__}..."
        )
        self._configure_model_specifics(config)

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

        DataloaderHandler().apply(self._cfg, config)
        RuntimeHandler().apply(self._cfg, config)

    @abstractmethod
    def _configure_model_specifics(self, config: UserConfig) -> None:
        """Architecture-specific overrides."""
        pass

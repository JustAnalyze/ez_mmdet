from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Union

from loguru import logger
from mmdet.apis import DetInferencer
from mmdet.utils import register_all_modules
from mmengine.config import Config
from mmengine.runner import Runner

from ez_mmdetection.core.config_loader import get_config_file
from ez_mmdetection.schemas.dataset import DatasetConfig
from ez_mmdetection.schemas.inference import InferenceResult
from ez_mmdetection.schemas.model import ModelName
from ez_mmdetection.utils.download import ensure_model_checkpoint
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

    def __init__(
        self,
        model_name: ModelName,
        checkpoint_path: Optional[Union[str, Path]] = None,
        log_level: str = "INFO",
    ):
        """Initializes the detector with a base model.

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
        self._inferencer: Optional[DetInferencer] = None

        # Resolve or download checkpoint
        self.checkpoint_path = ensure_model_checkpoint(self.model_name, checkpoint_path)

        # Configure loguru level
        try:
            logger.remove()
            import sys

            logger.add(sys.stderr, level=log_level)
        except Exception as e:
            logger.warning(f"Failed to set log level: {e}")

    def predict(
        self,
        image_path: Union[str, Path],
        checkpoint_path: Optional[Union[str, Path]] = None,
        device: str = "cuda",
        out_dir: Optional[str] = None,
        show: bool = False,
    ) -> InferenceResult:
        """Performs object detection on an image.

        Args:
            image_path: Path to the image file.
            checkpoint_path: Optional override for the model checkpoint (.pth).
            device: Computing device (default: 'cuda').
            out_dir: Directory to save visualization results.
            show: Whether to display the image.

        Returns:
            A structured InferenceResult object.
        """
        # Prioritize method-level checkpoint, then instance-level
        target_checkpoint = self.checkpoint_path
        if checkpoint_path:
            target_checkpoint = ensure_model_checkpoint(
                self.model_name, checkpoint_path
            )

        if self._inferencer is None:
            # Resolve model name to config file path
            config_path = get_config_file(self.model_name)
            logger.info(
                f"Initializing inferencer for model: {self.model_name} (using config: {config_path})"
            )
            self._inferencer = DetInferencer(
                model=str(config_path),
                weights=str(target_checkpoint),
                device=device,
            )

        logger.info(f"Running inference on: {image_path}")
        # Ensure out_dir is not None, as DetInferencer expects a string or PathLike
        results = self._inferencer(str(image_path), out_dir=out_dir or "", show=show)
        return InferenceResult.from_mmdet(results)

    def train(
        self,
        dataset_config_path: Union[str, Path],
        epochs: int = 100,
        batch_size: int = 8,
        device: str = "cuda",
        work_dir: str = "./runs/train",
        learning_rate: float = 0.001,
        load_from: Optional[str] = None,
        log_level: Optional[str] = None,
        amp: bool = True,
        num_workers: int = 2,
        enable_tensorboard: bool = True,
    ) -> None:
        """The Template Method defining the training workflow.

        Args:
            dataset_config_path: Path to the dataset.toml file.
            epochs: Number of training epochs.
            batch_size: Batch size per GPU.
            device: Training device ('cuda' or 'cpu').
            work_dir: Directory to save logs and checkpoints.
            learning_rate: Base learning rate.
            load_from: Optional checkpoint to resume from. Defaults to instance checkpoint.
            log_level: Logging level. Defaults to instance log_level.
            amp: Whether to enable Automatic Mixed Precision training. Defaults to True.
            num_workers: Number of dataloader workers. Defaults to 2.
            enable_tensorboard: Whether to enable TensorBoard logging. Defaults to True.
        """
        target_log_level = log_level or self.log_level
        # Use provided load_from or the one from initialization
        final_load_from = load_from or str(self.checkpoint_path)

        logger.info(f"Loading dataset configuration from: {dataset_config_path}")
        dataset_cfg = DatasetConfig.from_toml(Path(dataset_config_path))

        self.classes = dataset_cfg.classes
        self.num_classes: int = len(dataset_cfg.classes) if dataset_cfg.classes else 80

        user_config = UserConfig(
            model=ModelSection(
                name=self.model_name,
                num_classes=self.num_classes,
                load_from=final_load_from,
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
        """Orchestrates the internal MMDetection setup and execution."""
        # 1. Reproducibility: Save the effective config
        work_dir = Path(config.training.work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        save_user_config(config, work_dir / "user_config.toml")
        logger.info(f"User configuration saved to: {work_dir / 'user_config.toml'}")

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

        if hasattr(self._cfg, "optim_wrapper"):
            if config.training.amp:
                self._cfg.optim_wrapper.type = "AmpOptimWrapper"
                self._cfg.optim_wrapper.loss_scale = "dynamic"
            else:
                self._cfg.optim_wrapper.type = "OptimWrapper"
                if hasattr(self._cfg.optim_wrapper, "loss_scale"):
                    del self._cfg.optim_wrapper["loss_scale"]

            if hasattr(self._cfg.optim_wrapper, "optimizer"):
                self._cfg.optim_wrapper.optimizer.lr = config.training.learning_rate

        self._cfg.log_level = config.training.log_level

        # Data roots and paths
        data_root = Path(config.data.root)
        self._cfg.data_root = str(data_root)
        
        for key in ["train_dataloader", "val_dataloader", "test_dataloader"]:
            if hasattr(self._cfg, key):
                dl = getattr(self._cfg, key)
                # Setting data_root to empty string to prevent double-joining
                dl.dataset.data_root = ""
                dl.batch_size = config.training.batch_size
                dl.num_workers = config.training.num_workers

                # Use absolute paths for annotations and images
                if key == "train_dataloader":
                    dl.dataset.ann_file = str(data_root / config.data.train_ann)
                    dl.dataset.data_prefix = {"img": str(data_root / config.data.train_img)}
                else:
                    dl.dataset.ann_file = str(data_root / config.data.val_ann)
                    dl.dataset.data_prefix = {"img": str(data_root / config.data.val_img)}

                if config.data.classes:
                    dl.dataset.metainfo = {"classes": config.data.classes}

        # Also set metainfo at the top level of the config if possible
        if config.data.classes:
            self._cfg.metainfo = {"classes": config.data.classes}

        # Configure TensorBoard backend
        if config.training.enable_tensorboard:
            # Ensure visualizer exists and has vis_backends list
            if not hasattr(self._cfg, 'visualizer'):
                self._cfg.visualizer = dict(type='DetLocalVisualizer', vis_backends=[dict(type='LocalVisBackend')])
            
            if 'vis_backends' not in self._cfg.visualizer:
                self._cfg.visualizer['vis_backends'] = [dict(type='LocalVisBackend')]
            
            # Check if TensorboardVisBackend is already there
            has_tb = False
            for backend in self._cfg.visualizer['vis_backends']:
                if backend['type'] == 'TensorboardVisBackend':
                    has_tb = True
                    break
            
            if not has_tb:
                self._cfg.visualizer['vis_backends'].append(dict(type='TensorboardVisBackend'))

        # Override Evaluators
        # Evaluators often need absolute paths if data_root isn't explicitly used by them
        if hasattr(self._cfg, "val_evaluator"):
            self._cfg.val_evaluator.ann_file = str(data_root / config.data.val_ann)
        if hasattr(self._cfg, "test_evaluator"):
            test_ann = config.data.test_ann or config.data.val_ann
            self._cfg.test_evaluator.ann_file = str(data_root / test_ann)

    @abstractmethod
    def _configure_model_specifics(self, config: UserConfig) -> None:
        """Subclasses must implement this to handle architecture-specific overrides."""
        pass


# TODO: Implement EZMMPose
class EZMMPose: ...

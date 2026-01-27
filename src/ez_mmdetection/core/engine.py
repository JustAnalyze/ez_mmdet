from pathlib import Path
from typing import List, Optional, Union

from loguru import logger
from mmdet.utils import register_all_modules
from mmengine.config import Config
from mmengine.runner import Runner

from ez_mmdetection.core.config_loader import get_config_file
from ez_mmdetection.schemas.dataset import DatasetConfig  # New import
from ez_mmdetection.utils.toml_config import (
    DataSection,  # Re-added
    ModelSection,  # Re-added
    TrainingSection,
    UserConfig,
    save_user_config,
)

# Force registration of MMDet modules
register_all_modules()


class EZDetector:
    """Main interface for training and inference using MMDetection."""

    def __init__(self, model_name: str = "rtmdet_tiny"):
        """Initializes the detector with a base model.

        Args:
            model_name: The name of the architecture (e.g., 'rtmdet_tiny').
        """
        logger.info(f"Initializing EZDetector with base model: '{model_name}'")
        self.model_name = model_name
        # These will be populated from DatasetConfig
        self.num_classes: Optional[int] = None
        self.classes: Optional[List[str]] = None
        # Placeholder for the internal MMDet Config object
        self._cfg: Optional[Config] = None

    def train(
        self,
        dataset_config_path: Union[str, Path],
        epochs: int = 100,
        batch_size: int = 8,
        device: str = "cuda",
        work_dir: str = "./runs/train",
        learning_rate: float = 0.001,
        load_from: Optional[str] = None,
    ) -> None:
        """Launches the training process using a dataset configuration file.

        Args:
            dataset_config_path: Path to the `dataset.toml` file defining the dataset.
            epochs: Total training epochs (default: 100).
            batch_size: Batch size per device (default: 8).
            device: Computing device (default: 'cuda').
            work_dir: Directory for output files (default: './runs/train').
            learning_rate: Initial learning rate (default: 0.001).
            load_from: Optional path to a checkpoint to load from.
        """
        logger.info(f"Loading dataset configuration from: {dataset_config_path}")
        dataset_cfg = DatasetConfig.from_toml(Path(dataset_config_path))
        # Set internal state for num_classes and classes
        self.classes = dataset_cfg.classes
        self.num_classes = (
            len(dataset_cfg.classes) if dataset_cfg.classes else 80
        )  # Default to 80 for COCO if not specified

        # Ensure num_classes is set and valid
        if self.num_classes is None or self.num_classes <= 0:
            raise ValueError(
                "Dataset config must provide valid 'classes' or detector must be initialized with a 'num_classes'."
            )

        logger.info(
            f"Constructing UserConfig for training with dataset: '{dataset_config_path}'"
        )
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
                # test=dataset_cfg.test, # Include test split if available. Note: DataSection currently doesn't support 'test' explicitly based on previous file content, need to check utils/toml_config.py
            ),
            training=TrainingSection(
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                device=device,
                work_dir=work_dir,
            ),
        )

        self._train_from_config(user_config)

    def _train_from_config(self, config: UserConfig) -> None:
        """The core training logic, driven by a validated UserConfig."""
        # 1. Save the user's config to the working directory for reproducibility
        work_dir = Path(config.training.work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        saved_config_path = work_dir / "user_config.toml"
        save_user_config(config, saved_config_path)
        logger.info(f"User configuration saved to: {saved_config_path}")

        # 2. Load Base MMDetection Config
        self._cfg = self._load_base_config(config.model.name)
        logger.info(f"Loaded base config for model: {config.model.name}")

        # 3. Apply Overrides
        self._apply_config_overrides(config)

        # 4. Instantiate Runner
        logger.info("Instantiating MMEngine Runner...")
        runner = Runner.from_cfg(self._cfg)

        # 5. Execute
        logger.info("Calling runner.train()...")
        runner.train()
        logger.info("Training finished.")

    def _load_base_config(self, model_name: str) -> Config:
        """Loads the raw MMDetection config file."""
        config_path = get_config_file(model_name)
        logger.info(f"Loading MMDetection config from: {config_path}")
        cfg = Config.fromfile(config_path)
        return cfg

    def _apply_config_overrides(self, config: UserConfig) -> None:
        """Mutates the internal Config object based on the UserConfig."""
        logger.info("Applying configuration overrides...")
        if not self._cfg:
            raise RuntimeError("Config not loaded.")

        # Extract sections for clarity
        model_cfg = config.model
        data_cfg = config.data
        train_cfg = config.training

        # --- Apply Overrides ---
        self._cfg.work_dir = train_cfg.work_dir
        self._cfg.train_cfg.max_epochs = train_cfg.epochs
        self._cfg.load_from = model_cfg.load_from

        # Optimizer
        if hasattr(self._cfg, "optim_wrapper") and hasattr(
            self._cfg.optim_wrapper, "optimizer"
        ):
            self._cfg.optim_wrapper.optimizer.lr = train_cfg.learning_rate

        # Data roots and paths
        self._cfg.data_root = data_cfg.root
        for key in ["train_dataloader", "val_dataloader", "test_dataloader"]:
            if hasattr(self._cfg, key):
                getattr(self._cfg, key).dataset.data_root = data_cfg.root
                getattr(self._cfg, key).batch_size = train_cfg.batch_size

                # Set annotation and image paths
                if key == "train_dataloader":
                    getattr(self._cfg, key).dataset.ann_file = data_cfg.train_ann
                    getattr(self._cfg, key).dataset.data_prefix = {
                        "img": data_cfg.train_img
                    }
                else:  # val and test use the same settings
                    getattr(self._cfg, key).dataset.ann_file = data_cfg.val_ann
                    getattr(self._cfg, key).dataset.data_prefix = {
                        "img": data_cfg.val_img
                    }

        # Handle classes and num_classes consistently
        if data_cfg.classes:
            self._cfg.metainfo = {"classes": data_cfg.classes}
            for key in [
                "train_dataloader",
                "val_dataloader",
                "test_dataloader",
            ]:
                if hasattr(self._cfg, key):
                    getattr(self._cfg, key).dataset.metainfo = {
                        "classes": data_cfg.classes
                    }

        # This is the most brittle part - needs a robust strategy
        logger.info(
            f"  - Overriding 'model.bbox_head.num_classes': {model_cfg.num_classes}"
        )
        if hasattr(self._cfg.model, "bbox_head"):
            if isinstance(self._cfg.model.bbox_head, list):
                for head in self._cfg.model.bbox_head:
                    head.num_classes = model_cfg.num_classes
            else:
                self._cfg.model.bbox_head.num_classes = model_cfg.num_classes

        logger.info("Finished applying overrides.")

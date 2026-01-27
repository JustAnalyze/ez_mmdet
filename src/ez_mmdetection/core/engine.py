from pathlib import Path
from typing import Optional, Union, List

from loguru import logger
from mmdet.utils import register_all_modules
from mmengine.config import Config
from mmengine.runner import Runner

from ez_mmdetection.core.config_loader import get_config_file
from ez_mmdetection.utils.toml_config import (
    UserConfig,
    ModelSection,
    DataSection,
    TrainingSection,
    save_user_config,
    load_user_config,
)

# Force registration of MMDet modules
register_all_modules()


class EZDetector:
    """Main interface for training and inference using MMDetection."""

    def __init__(self, model_name: str = "rtmdet_tiny", num_classes: Optional[int] = None, classes: Optional[List[str]] = None):
        """Initializes the detector with a base model.

        Args:
            model_name: The name of the architecture (e.g., 'rtmdet_tiny').
            num_classes: Number of classes in the dataset. If `classes` is provided, this is inferred.
            classes: A list of class names for the dataset.
        """
        if classes:
            num_classes = len(classes)
        elif num_classes is None:
            raise ValueError("Either `num_classes` or `classes` must be provided.")

        logger.info(f"Initializing EZDetector with model: '{model_name}', num_classes: {num_classes}")
        self.model_name = model_name
        self.num_classes = num_classes
        self.classes = classes
        # Placeholder for the internal MMDet Config object
        self._cfg: Optional[Config] = None

    def train(
        self,
        config_file: Optional[Union[str, Path]] = None,
        data_root: Optional[Union[str, Path]] = None,
        epochs: int = 100,
        batch_size: int = 8,
        device: str = "cuda",
        work_dir: str = "./runs/train",
        train_ann: str = "annotations/instances_train2017.json",
        train_img: str = "train2017/",
        val_ann: str = "annotations/instances_val2017.json",
        val_img: str = "val2017/",
        learning_rate: float = 0.001,
        load_from: Optional[str] = None,
    ) -> None:
        """Launches the training process from a config file or parameters.

        This method supports two workflows:
        1.  **Config-file based:** Provide the `config_file` argument.
            >>> detector.train(config_file="path/to/your/config.toml")

        2.  **Parameter-based:** Provide `data_root` and other arguments. A
            `user_config.toml` will be generated in your `work_dir`.
            >>> detector.train(data_root="path/to/data", epochs=50)
        """
        if config_file:
            logger.info(f"Loading training configuration from: {config_file}")
            config = load_user_config(Path(config_file))
        elif data_root:
            logger.info("Constructing training configuration from parameters...")
            config = UserConfig(
                model=ModelSection(
                    name=self.model_name,
                    num_classes=self.num_classes,
                    load_from=load_from,
                ),
                data=DataSection(
                    root=str(data_root),
                    train_ann=train_ann,
                    train_img=train_img,
                    val_ann=val_ann,
                    val_img=val_img,
                    classes=self.classes,
                ),
                training=TrainingSection(
                    epochs=epochs,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    device=device,
                    work_dir=work_dir,
                ),
            )
        else:
            raise ValueError(
                "Either `config_file` or `data_root` must be provided to start training."
            )

        self._train_from_config(config)

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
        if hasattr(self._cfg, "optim_wrapper") and hasattr(self._cfg.optim_wrapper, "optimizer"):
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
                    getattr(self._cfg, key).dataset.data_prefix = {"img": data_cfg.train_img}
                else: # val and test use the same settings
                    getattr(self._cfg, key).dataset.ann_file = data_cfg.val_ann
                    getattr(self._cfg, key).dataset.data_prefix = {"img": data_cfg.val_img}

        # Handle classes and num_classes consistently
        if data_cfg.classes:
            self._cfg.metainfo = {"classes": data_cfg.classes}
            for key in ["train_dataloader", "val_dataloader", "test_dataloader"]:
                if hasattr(self._cfg, key):
                    getattr(self._cfg, key).dataset.metainfo = {"classes": data_cfg.classes}

        # This is the most brittle part - needs a robust strategy
        logger.info(f"  - Overriding 'model.bbox_head.num_classes': {model_cfg.num_classes}")
        if hasattr(self._cfg.model, "bbox_head"):
            if isinstance(self._cfg.model.bbox_head, list):
                for head in self._cfg.model.bbox_head:
                    head.num_classes = model_cfg.num_classes
            else:
                self._cfg.model.bbox_head.num_classes = model_cfg.num_classes

        logger.info("Finished applying overrides.")



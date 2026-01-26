from pathlib import Path
from typing import Optional, Union

from loguru import logger
from mmdet.utils import register_all_modules
from mmengine.config import Config
from mmengine.runner import Runner

from ez_mmdetection.schemas.config import DataConfig, ModelConfig, TrainingArgs

# Force registration of MMDet modules
register_all_modules()


class EZDetector:
    """Main interface for training and inference using MMDetection."""

    def __init__(self, model_name: str = "rtmdet_tiny", num_classes: int = 80):
        """Initializes the detector with a base model.

        Args:
            model_name: The name of the architecture (e.g., 'rtmdet_tiny').
            num_classes: Number of classes in the dataset.
        """
        logger.info(f"Initializing EZDetector with model: '{model_name}', num_classes: {num_classes}")
        self.model_name = model_name
        self.num_classes = num_classes
        # Placeholder for the internal MMDet Config object
        self._cfg: Optional[Config] = None

    def train(
        self,
        data_root: Union[str, Path],
        epochs: int = 100,
        batch_size: int = 8,
        device: str = "cuda",
    ) -> None:
        """Starts the training loop.

        Args:
            data_root: Path to the dataset directory.
            epochs: Total training epochs.
            batch_size: Batch size per device.
            device: Computing device ('cuda' or 'cpu').
        """
        logger.info("Starting training loop...")
        # 1. Validate Inputs via Pydantic
        # This ensures we fail fast if types/paths are wrong
        logger.info("Validating training arguments...")
        args = TrainingArgs(
            data=DataConfig(data_root=Path(data_root), batch_size=batch_size),
            model=ModelConfig(
                base_model=self.model_name, num_classes=self.num_classes
            ),
            epochs=epochs,
            device=device,
        )
        logger.info(f"Training arguments validated successfully: {args.dict()}")

        # 2. Load Base Config (You need a strategy to locate these files)
        # For this prototype, we assume a local lookup or standard MMDet path
        # TODO: Implement Config Loader logic
        self._cfg = self._load_base_config(args.model.base_model)
        logger.info(f"Loaded base config for model: {args.model.base_model}")

        # 3. Apply Overrides (The complex part)
        self._apply_config_overrides(args)

        # 4. Instantiate Runner
        logger.info("Instantiating MMEngine Runner...")
        runner = Runner.from_cfg(self._cfg)

        # 5. Execute
        logger.info("Calling runner.train()...")
        runner.train()
        logger.info("Training finished.")

    def _load_base_config(self, model_name: str) -> Config:
        """Loads the raw MMDetection config file."""
        from ez_mmdetection.core.config_loader import get_config_file # Local import to avoid circular dependency potentially

        config_path = get_config_file(model_name)
        logger.info(f"Loading MMDetection config from: {config_path}")
        cfg = Config.fromfile(config_path)
        return cfg

    def _apply_config_overrides(self, args: TrainingArgs) -> None:
        """Mutates the internal Config object based on validated args."""
        logger.info("Applying configuration overrides...")
        if not self._cfg:
            raise RuntimeError("Config not loaded.")

        # Override Logic
        logger.info(f"  - Overriding 'work_dir': {args.work_dir}")
        self._cfg.work_dir = str(args.work_dir)

        logger.info(f"  - Overriding 'optim_wrapper.optimizer.lr': {args.learning_rate}")
        self._cfg.optim_wrapper.optimizer.lr = args.learning_rate

        logger.info(f"  - Overriding 'train_cfg.max_epochs': {args.epochs}")
        self._cfg.train_cfg.max_epochs = args.epochs

        # Dataset overrides (Crucial for generic use)
        logger.info(f"  - Overriding 'data_root': {args.data.data_root}")
        self._cfg.data_root = str(args.data.data_root)

        # Head modification for num_classes
        # Note: This varies strictly by architecture (YOLO vs R-CNN)
        # We need a robust strategy here.
        logger.info(f"  - Overriding 'model.bbox_head.num_classes': {args.model.num_classes}")
        if hasattr(self._cfg.model, "bbox_head"):
            # Generic one-stage detector override
            if isinstance(self._cfg.model.bbox_head, list):
                for head in self._cfg.model.bbox_head:
                    head.num_classes = args.model.num_classes
            else:
                self._cfg.model.bbox_head.num_classes = args.model.num_classes
        logger.info("Finished applying overrides.")



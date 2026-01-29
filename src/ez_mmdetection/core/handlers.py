from abc import ABC, abstractmethod
from pathlib import Path
from mmengine.config import Config
from ez_mmdetection.utils.toml_config import UserConfig

class BaseConfigHandler(ABC):
    """
    Abstract base class for configuration handlers.
    Each handler is responsible for configuring a specific aspect of the MMDetection config.
    """

    @abstractmethod
    def apply(self, cfg: Config, user_config: UserConfig) -> None:
        """
        Applies configuration updates to the MMDetection Config object.

        Args:
            cfg: The mutable MMDetection Config object.
            user_config: The validated user configuration.
        """
        pass

class DataloaderHandler(BaseConfigHandler):
    """Configures dataset paths, batch sizes, and workers for train/val/test loaders."""

    def apply(self, cfg: Config, user_config: UserConfig) -> None:
        data_root = Path(user_config.data.root)
        cfg.data_root = str(data_root)

        for key in ["train_dataloader", "val_dataloader", "test_dataloader"]:
            if hasattr(cfg, key):
                dl = getattr(cfg, key)
                # Setting data_root to empty string to prevent double-joining
                dl.dataset.data_root = ""
                dl.batch_size = user_config.training.batch_size
                dl.num_workers = user_config.training.num_workers

                # Use absolute paths for annotations and images
                if key == "train_dataloader":
                    dl.dataset.ann_file = str(data_root / user_config.data.train_ann)
                    dl.dataset.data_prefix = {"img": str(data_root / user_config.data.train_img)}
                else:
                    # Default val/test to use validation set if test set not explicit (common pattern)
                    # For now, we mirror what was in base.py which used val for both.
                    # TODO: If DataSection supports test split, use it for test_dataloader
                    dl.dataset.ann_file = str(data_root / user_config.data.val_ann)
                    dl.dataset.data_prefix = {"img": str(data_root / user_config.data.val_img)}

                if user_config.data.classes:
                    dl.dataset.metainfo = {"classes": user_config.data.classes}

        # Also set metainfo at the top level of the config if possible
        if user_config.data.classes:
            cfg.metainfo = {"classes": user_config.data.classes}

class RuntimeHandler(BaseConfigHandler):
    """Configures general runtime settings including optimizer, AMP, and visualization."""

    def apply(self, cfg: Config, user_config: UserConfig) -> None:
        training = user_config.training
        model_cfg = user_config.model

        # --- General Runtime ---
        cfg.work_dir = training.work_dir
        cfg.train_cfg.max_epochs = training.epochs
        cfg.load_from = model_cfg.load_from
        cfg.log_level = training.log_level

        # --- Optimizer & AMP ---
        if hasattr(cfg, "optim_wrapper"):
            if training.amp:
                cfg.optim_wrapper.type = "AmpOptimWrapper"
                cfg.optim_wrapper.loss_scale = "dynamic"
            else:
                cfg.optim_wrapper.type = "OptimWrapper"
                if hasattr(cfg.optim_wrapper, "loss_scale"):
                    del cfg.optim_wrapper["loss_scale"]

            if hasattr(cfg.optim_wrapper, "optimizer"):
                cfg.optim_wrapper.optimizer.lr = training.learning_rate

        # --- Visualization (TensorBoard) ---
        if training.enable_tensorboard:
            # Ensure visualizer exists and has vis_backends list
            if not hasattr(cfg, 'visualizer'):
                cfg.visualizer = dict(
                    type='DetLocalVisualizer', 
                    vis_backends=[dict(type='LocalVisBackend')]
                )
            
            if 'vis_backends' not in cfg.visualizer:
                cfg.visualizer['vis_backends'] = [dict(type='LocalVisBackend')]
            
            # Add Tensorboard if not present
            vis_backends = cfg.visualizer['vis_backends']
            if not any(b['type'] == 'TensorboardVisBackend' for b in vis_backends):
                vis_backends.append(dict(type='TensorboardVisBackend'))

        # --- Evaluator Path Overrides ---
        data_root = Path(user_config.data.root)
        if hasattr(cfg, "val_evaluator"):
            cfg.val_evaluator.ann_file = str(data_root / user_config.data.val_ann)
        if hasattr(cfg, "test_evaluator"):
            # Fallback to val_ann if test_ann isn't explicitly defined in DataSection yet
            # (Note: DataSection in toml_config.py currently doesn't have test_ann)
            cfg.test_evaluator.ann_file = str(data_root / user_config.data.val_ann)


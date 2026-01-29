from mmengine.config import Config
from ez_mmdetection.core.handlers import DataloaderHandler, RuntimeHandler
from ez_mmdetection.utils.toml_config import UserConfig, DataSection, TrainingSection, ModelSection

def verify():
    print("--- Manual Verification: Configuration Handlers ---")
    
    # 1. Setup sample data
    user_config = UserConfig(
        model=ModelSection(name="rtmdet_tiny", num_classes=2),
        data=DataSection(
            root="data/coco",
            train_ann="annotations/train.json",
            train_img="train2017",
            val_ann="annotations/val.json",
            val_img="val2017",
            classes=["cat", "dog"]
        ),
        training=TrainingSection(
            epochs=10,
            batch_size=4,
            num_workers=2,
            learning_rate=0.01,
            amp=True,
            enable_tensorboard=True
        )
    )
    
    cfg = Config(dict(
        train_dataloader=dict(dataset=dict()),
        val_dataloader=dict(dataset=dict()),
        optim_wrapper=dict(optimizer=dict()),
        train_cfg=dict()
    ))
    
    # 2. Test DataloaderHandler
    print("Testing DataloaderHandler...")
    DataloaderHandler().apply(cfg, user_config)
    assert cfg.data_root == "data/coco"
    assert cfg.train_dataloader.num_workers == 2
    assert cfg.train_dataloader.dataset.ann_file == "data/coco/annotations/train.json"
    print("  - DataloaderHandler applied successfully.")
    
    # 3. Test RuntimeHandler
    print("Testing RuntimeHandler...")
    RuntimeHandler().apply(cfg, user_config)
    assert cfg.train_cfg.max_epochs == 10
    assert cfg.optim_wrapper.type == "AmpOptimWrapper"
    assert any(b['type'] == 'TensorboardVisBackend' for b in cfg.visualizer.vis_backends)
    print("  - RuntimeHandler applied successfully.")
    
    print("\nSUCCESS: Both handlers verified manually.")

if __name__ == "__main__":
    verify()

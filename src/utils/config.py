"""
Central configuration via YAML + env overrides.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class ModelConfig:
    name: str = "resnet34"
    pretrained: bool = True
    num_classes: int = 2
    dropout: float = 0.4


@dataclass
class TrainingConfig:
    batch_size: int = 8
    epochs: int = 30
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    early_stop_patience: int = 8
    mixup_alpha: float = 0.2
    temperature_scaling: bool = True


@dataclass
class DataConfig:
    raw_dir: str = r"C:\Users\32333\raw_data_hierarchical"
    val_split: float = 0.1
    test_split: float = 0.1
    image_size: int = 224
    crop_left_third: bool = True
    cold_tone_filter: bool = True


@dataclass
class EnsembleConfig:
    models: list[str] = field(default_factory=lambda: ["resnet34", "efficientnet_b0", "convnext_tiny"])
    strategy: str = "soft_voting"   # soft_voting, hard_voting, weighted
    tta_views: int = 4


@dataclass
class ActiveLearningConfig:
    enabled: bool = True
    uncertainty_threshold: float = 0.15   # margin between top-2 class probs
    queue_max_size: int = 500
    retrain_min_samples: int = 20
    retrain_trigger_count: int = 50
    model_registry_dir: str = "./model_registry"


@dataclass
class ABTestConfig:
    enabled: bool = True
    control_traffic: float = 0.8   # 80% to current model
    treatment_traffic: float = 0.2  # 20% to candidate
    promotion_threshold: float = 0.02   # candidate must beat control by 2% accuracy
    min_samples_before_promotion: int = 1000


@dataclass
class PLCConfig:
    host: str = "0.0.0.0"
    port: int = 8500
    timeout: float = 5.0


@dataclass
class CameraConfig:
    ip1: str = "192.168.10.13"
    ip2: str = "192.168.10.12"
    exposure1: float = 50.0
    exposure2: float = 10.0
    gain1: float = 10.0
    gain2: float = 10.0


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)
    active_learning: ActiveLearningConfig = field(default_factory=ActiveLearningConfig)
    ab_test: ABTestConfig = field(default_factory=ABTestConfig)
    plc: PLCConfig = field(default_factory=PLCConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    output_dir: str = "./output"
    device: str = "cuda:0"

    @classmethod
    def from_yaml(cls, path: str = None) -> "Config":
        cfg = cls()
        if path and os.path.exists(path):
            import yaml
            with open(path) as f:
                data = yaml.safe_load(f)
            # ... merge logic (simplified for now)
        if torch_available():
            import torch
            cfg.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        return cfg


def torch_available():
    try:
        import torch
        return True
    except ImportError:
        return False

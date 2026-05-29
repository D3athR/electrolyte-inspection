"""
Model architectures: ResNet34, EfficientNet-B0, ConvNeXt-Tiny.

All models share the same head structure (MultiScaleConvBlock + calibrated FC)
and support loading from both pretrained weights and custom checkpoints.
"""

import torch
import torch.nn as nn
from torchvision import models


class MultiScaleConvBlock(nn.Module):
    """Multi-scale feature fusion with residual connection."""
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.15):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        f = self.conv(x)
        s = self.shortcut(x)
        return self.dropout(self.relu(self.bn(f + s)))


class ClassificationHead(nn.Module):
    """Calibrated classification head with deep bottleneck."""
    def __init__(self, in_features: int, num_classes: int = 2, dropout: float = 0.4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.fc(x)


def _replace_head(model: nn.Module, in_features: int, num_classes: int, dropout: float,
                  num_msc_blocks: int = 2):
    """Replace layer4 and fc with custom MultiScaleConvBlock + ClassificationHead."""
    # Build new layer4
    blocks = [model.layer4]
    for _ in range(num_msc_blocks):
        blocks.append(MultiScaleConvBlock(512, 512, dropout * 0.4))
    model.layer4 = nn.Sequential(*blocks)

    # Replace FC
    model.fc = ClassificationHead(in_features, num_classes, dropout)
    return model


def _freeze_backbone(model: nn.Module):
    """Freeze all layers except layer4 and fc."""
    for p in model.parameters():
        p.requires_grad = False
    for p in model.layer4.parameters():
        p.requires_grad = True
    for p in model.fc.parameters():
        p.requires_grad = True


def build_resnet34(num_classes: int = 2, pretrained: bool = True,
                   dropout: float = 0.4, weights_path: str = None) -> nn.Module:
    model = models.resnet34(weights=None)
    if pretrained and weights_path:
        model.load_state_dict(torch.load(weights_path, map_location="cpu"), strict=False)
    model = _replace_head(model, 512, num_classes, dropout, num_msc_blocks=2)
    _freeze_backbone(model)
    return model


def build_efficientnet_b0(num_classes: int = 2, pretrained: bool = True,
                          dropout: float = 0.4) -> nn.Module:
    model = models.efficientnet_b0(weights="IMAGENET1K_V1" if pretrained else None)
    in_features = model.classifier[1].in_features
    model.classifier = ClassificationHead(in_features, num_classes, dropout)
    for p in model.parameters():
        p.requires_grad = False
    for p in model.classifier.parameters():
        p.requires_grad = True
    return model


def build_convnext_tiny(num_classes: int = 2, pretrained: bool = True,
                        dropout: float = 0.4) -> nn.Module:
    model = models.convnext_tiny(weights="IMAGENET1K_V1" if pretrained else None)
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, num_classes)
    for p in model.parameters():
        p.requires_grad = False
    for p in model.classifier.parameters():
        p.requires_grad = True
    return model


MODEL_BUILDERS = {
    "resnet34": build_resnet34,
    "efficientnet_b0": build_efficientnet_b0,
    "convnext_tiny": build_convnext_tiny,
}


def build_model(name: str, **kwargs) -> nn.Module:
    if name not in MODEL_BUILDERS:
        raise ValueError(f"Unknown model: {name}. Available: {list(MODEL_BUILDERS.keys())}")
    return MODEL_BUILDERS[name](**kwargs)

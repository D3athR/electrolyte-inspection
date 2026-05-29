"""
Image transforms: cold-tone filter, TTA, training augmentations.
"""

import numpy as np
import torch
from PIL import Image
from torchvision import transforms


class ColdToneFilter:
    """Apply blue*1.3, red*0.8 — matches production camera setup."""

    def __call__(self, img: Image.Image) -> Image.Image:
        arr = np.array(img, dtype=np.float32)
        arr[:, :, 0] = np.clip(arr[:, :, 0] * 1.3, 0, 255)  # Blue
        arr[:, :, 2] = np.clip(arr[:, :, 2] * 0.8, 0, 255)  # Red
        return Image.fromarray(arr.astype(np.uint8))


class CropLeftThird:
    """Crop left 1/3 of image — precipitation detection ROI."""

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        return img.crop((0, 0, w // 3, h))


def get_train_transforms(image_size: int = 224):
    """Training augmentations including cold-tone filter."""
    return transforms.Compose([
        CropLeftThird(),
        ColdToneFilter(),
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.3),
        transforms.RandomRotation(20),
        transforms.RandomAffine(0, translate=(0.1, 0.1), scale=(0.85, 1.15)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
        transforms.RandomGrayscale(0.15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.Lambda(lambda t: t + torch.randn_like(t) * 0.02),
    ])


def get_val_transforms(image_size: int = 224):
    """Validation transforms (no augmentation)."""
    return transforms.Compose([
        CropLeftThird(),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def get_tta_transforms(image_size: int = 224) -> list[transforms.Compose]:
    """Test-time augmentation: 4 views."""
    views = []
    for hflip, vflip in [(False, False), (True, False), (False, True), (True, True)]:
        ops = [CropLeftThird(), transforms.Resize((image_size, image_size))]
        if hflip:
            ops.append(transforms.RandomHorizontalFlip(1.0))
        if vflip:
            ops.append(transforms.RandomVerticalFlip(1.0))
        ops += [
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
        views.append(transforms.Compose(ops))
    return views

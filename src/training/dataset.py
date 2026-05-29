"""
Data loading with active learning queue support.
"""

import os
import shutil
from pathlib import Path

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder

from src.detection.transforms import get_train_transforms, get_val_transforms


class ElectrolyteDataset(Dataset):
    def __init__(self, root: str, transform=None):
        self.dataset = ImageFolder(root)
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        img = img.convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def split_dataset(raw_dir: str, split_dir: str, val_size=0.1, test_size=0.1, seed=42):
    """Split raw data into train/val/test."""
    if os.path.exists(split_dir):
        shutil.rmtree(split_dir)

    for split in ["train", "val", "test"]:
        for cls_name in os.listdir(raw_dir):
            cls_path = os.path.join(raw_dir, cls_name)
            if os.path.isdir(cls_path):
                os.makedirs(os.path.join(split_dir, split, cls_name))

    for cls_name in os.listdir(raw_dir):
        cls_path = os.path.join(raw_dir, cls_name)
        if not os.path.isdir(cls_path):
            continue
        imgs = [f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        if len(imgs) < 3:
            continue

        train_val, test = train_test_split(imgs, test_size=test_size, random_state=seed)
        train, val = train_test_split(train_val, test_size=val_size/(1-test_size), random_state=seed)

        for split, paths in zip(["train", "val", "test"], [train, val, test]):
            for f in paths:
                shutil.copy2(os.path.join(cls_path, f), os.path.join(split_dir, split, cls_name, f))


def create_dataloaders(split_dir: str, batch_size: int = 8, image_size: int = 224,
                       num_workers: int = 0) -> tuple:
    train_ds = ElectrolyteDataset(os.path.join(split_dir, "train"), get_train_transforms(image_size))
    val_ds = ElectrolyteDataset(os.path.join(split_dir, "val"), get_val_transforms(image_size))
    test_ds = ElectrolyteDataset(os.path.join(split_dir, "test"), get_val_transforms(image_size))

    train_loader = DataLoader(train_ds, batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, train_ds.dataset.class_to_idx

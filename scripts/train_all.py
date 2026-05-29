#!/usr/bin/env python3
"""
Train all ensemble models with the improved pipeline.

Usage:
    python scripts/train_all.py                     # Train all 3 models
    python scripts/train_all.py --model resnet34    # Train single model
    python scripts/train_all.py --raw_dir <path>    # Custom data dir
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.detection.models import build_model
from src.training.dataset import split_dataset, create_dataloaders
from src.training.trainer import Trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="all")
    parser.add_argument("--raw_dir", type=str, default=r"C:\Users\32333\raw_data_hierarchical")
    parser.add_argument("--output_dir", type=str, default="./model_registry")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Split data
    split_dir = os.path.join(args.output_dir, "split_data")
    split_dataset(args.raw_dir, split_dir)

    models_to_train = ["resnet34", "efficientnet_b0", "convnext_tiny"] if args.model == "all" else [args.model]

    for model_name in models_to_train:
        print(f"\n{'='*60}")
        print(f"Training: {model_name}")
        print(f"{'='*60}")

        model = build_model(model_name, num_classes=2, pretrained=True)
        model = model.to(device)

        train_loader, val_loader, test_loader, class_map = create_dataloaders(
            split_dir, args.batch_size)

        print(f"Classes: {class_map}")
        print(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, "
              f"Test: {len(test_loader.dataset)}")

        trainer = Trainer(model, device)
        config = trainer.run(
            train_loader, val_loader, args.epochs,
            model_save_path=os.path.join(args.output_dir, f"{model_name}.pth"),
            config_save_path=os.path.join(args.output_dir, f"{model_name}_config.json"),
        )

        # Test evaluation
        _, test_acc, test_probs, test_labels = trainer.validate(
            test_loader, config["temperature"])
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(test_labels, test_probs) if len(set(test_labels)) > 1 else 0.5
        print(f"Test: acc={test_acc:.3f}, AUC={auc:.3f}")

    print(f"\nAll models saved to {args.output_dir}")


if __name__ == "__main__":
    main()

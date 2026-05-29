#!/usr/bin/env python3
"""
Model evaluation: ROC curves, confusion matrix, threshold analysis.

Usage:
    python scripts/evaluate.py --model_path model_registry/resnet34.pth
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (roc_curve, auc, confusion_matrix,
                              precision_recall_curve, classification_report)

from src.detection.inference import Predictor
from src.training.dataset import split_dataset, create_dataloaders


def plot_roc_curve(probs, labels, save_path):
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, 'b-', lw=2, label=f'ROC (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'r--', lw=1, label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return roc_auc


def plot_threshold_analysis(probs, labels, save_path):
    thresholds = np.linspace(0, 1, 100)
    accuracies, f1_scores = [], []

    for t in thresholds:
        preds = (probs >= t).astype(int)
        acc = (preds == labels).mean()
        accuracies.append(acc)
        from sklearn.metrics import f1_score
        f1_scores.append(f1_score(labels, preds, zero_division=0))

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(thresholds, accuracies, 'b-', lw=2, label='Accuracy')
    ax1.plot(thresholds, f1_scores, 'g-', lw=2, label='F1 Score')
    ax1.set_xlabel('Threshold')
    ax1.set_ylabel('Score')
    ax1.legend(loc='center left')
    ax1.grid(alpha=0.3)

    # Mark current best threshold
    best_idx = np.argmax(f1_scores)
    best_t = thresholds[best_idx]
    ax1.axvline(x=best_t, color='r', linestyle='--', alpha=0.5,
                label=f'Best F1 threshold: {best_t:.2f}')
    ax1.legend(loc='lower left')

    plt.title('Threshold vs Accuracy/F1')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return best_t


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="resnet34")
    parser.add_argument("--config_path", type=str, default=None)
    parser.add_argument("--raw_dir", type=str, default=r"C:\Users\32333\raw_data_hierarchical")
    parser.add_argument("--output_dir", type=str, default="./evaluation")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Load model
    predictor = Predictor.from_checkpoint(
        args.model_path, args.model_name, args.config_path, device)

    # Load test data
    split_dir = os.path.join(args.output_dir, "split_data")
    split_dataset(args.raw_dir, split_dir)
    _, _, test_loader, class_map = create_dataloaders(split_dir, batch_size=1)
    print(f"Classes: {class_map}")
    print(f"Test samples: {len(test_loader.dataset)}")

    # Collect predictions
    all_probs, all_labels = [], []
    for x, y in test_loader:
        # Use raw predict to bypass TTA for evaluation speed
        from src.detection.transforms import get_val_transforms, ColdToneFilter, CropLeftThird
        from PIL import Image
        import torchvision.transforms.functional as TF

        logits = predictor.model(x.to(device))
        probs = torch.softmax(logits / predictor.temperature, dim=1)[0, 1].item()
        all_probs.append(probs)
        all_labels.append(y.item())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    # ROC
    roc_auc = plot_roc_curve(all_probs, all_labels,
                              os.path.join(args.output_dir, "roc_curve.png"))
    print(f"ROC AUC: {roc_auc:.3f}")

    # Threshold analysis
    best_t = plot_threshold_analysis(all_probs, all_labels,
                                      os.path.join(args.output_dir, "threshold_analysis.png"))
    print(f"Best F1 threshold: {best_t:.3f}")

    # Classification report
    preds = (all_probs >= best_t).astype(int)
    print("\nClassification Report:")
    print(classification_report(all_labels, preds,
                                target_names=list(class_map.keys())))

    # Confusion matrix
    cm = confusion_matrix(all_labels, preds)
    print(f"\nConfusion Matrix:")
    print(f"  TN={cm[0,0]:3d}  FP={cm[0,1]:3d}")
    print(f"  FN={cm[1,0]:3d}  TP={cm[1,1]:3d}")

    print(f"\nCharts saved to {args.output_dir}")


if __name__ == "__main__":
    main()

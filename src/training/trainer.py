"""
Training pipeline with MixUp, temperature scaling, threshold calibration.
"""

import json
import os
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score, precision_recall_curve
from torch.utils.data import DataLoader

from src.training.dataset import create_dataloaders


def mixup_data(x, y, alpha=0.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    idx = torch.randperm(x.size(0)).to(x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def calibrate_temperature(logits: torch.Tensor, labels: torch.Tensor,
                          lr=0.01, max_iter=100) -> float:
    """Learn optimal temperature to calibrate probabilities."""
    temp = nn.Parameter(torch.ones(1) * 1.5)
    criterion = nn.CrossEntropyLoss()
    opt = optim.LBFGS([temp], lr=lr, max_iter=max_iter)

    def closure():
        opt.zero_grad()
        loss = criterion(logits / temp, labels)
        loss.backward()
        return loss

    opt.step(closure)
    return temp.item()


def calibrate_threshold(probs, labels):
    """Find threshold maximizing F1 score."""
    precision, recall, thresholds = precision_recall_curve(labels, probs)
    with np.errstate(divide='ignore', invalid='ignore'):
        f1 = 2 * precision * recall / (precision + recall)
        f1[np.isnan(f1)] = 0
    idx = np.argmax(f1)
    return thresholds[idx] if idx < len(thresholds) else 0.5, f1[idx]


class Trainer:
    """Training loop with MixUp, early stopping, temperature scaling."""

    def __init__(self, model: nn.Module, device: str = "cpu",
                 lr: float = 1e-4, wd: float = 1e-4,
                 patience: int = 8, mixup_alpha: float = 0.2):
        self.model = model
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        trainable = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = optim.AdamW(trainable, lr=lr, weight_decay=wd)
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=5, T_mult=2)
        self.patience = patience
        self.mixup_alpha = mixup_alpha
        self.best_state = None
        self.best_val_acc = 0.0

    def train_epoch(self, loader: DataLoader, use_mixup: bool = True) -> tuple[float, float]:
        self.model.train()
        total_loss, correct = 0.0, 0
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()

            if use_mixup and np.random.random() < 0.5:
                x_m, ya, yb, lam = mixup_data(x, y, self.mixup_alpha)
                pred = self.model(x_m)
                loss = mixup_criterion(self.criterion, pred, ya, yb, lam)
            else:
                pred = self.model(x)
                loss = self.criterion(pred, y)

            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * x.size(0)
            correct += (pred.argmax(1) == y).sum().item()
        return total_loss / len(loader.dataset), correct / len(loader.dataset)

    @torch.no_grad()
    def validate(self, loader: DataLoader, temperature: float = 1.0) -> tuple:
        self.model.eval()
        total_loss, correct = 0.0, 0
        all_probs, all_labels = [], []
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            logits = self.model(x) / temperature
            total_loss += self.criterion(logits, y).item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            all_probs.extend(torch.softmax(logits, 1)[:, 1].cpu().tolist())
            all_labels.extend(y.cpu().tolist())
        return (total_loss / len(loader.dataset), correct / len(loader.dataset),
                np.array(all_probs), np.array(all_labels))

    def run(self, train_loader: DataLoader, val_loader: DataLoader,
            epochs: int, model_save_path: str, config_save_path: str) -> dict:
        patience_ctr = 0

        for epoch in range(epochs):
            t_loss, t_acc = self.train_epoch(train_loader)
            v_loss, v_acc, v_probs, v_labels = self.validate(val_loader)
            self.scheduler.step()

            auc = roc_auc_score(v_labels, v_probs) if len(set(v_labels)) > 1 else 0.5
            print(f"E{epoch+1:2d} | T loss={t_loss:.3f} acc={t_acc:.3f} | "
                  f"V loss={v_loss:.3f} acc={v_acc:.3f} AUC={auc:.3f}")

            if v_acc > self.best_val_acc:
                self.best_val_acc = v_acc
                self.best_state = self.model.state_dict().copy()
                patience_ctr = 0
                print(f"  -> best (acc={v_acc:.3f})")
            else:
                patience_ctr += 1
                if patience_ctr >= self.patience:
                    print(f"  Early stop at epoch {epoch+1}")
                    break

        # Restore best
        self.model.load_state_dict(self.best_state)

        # Temperature scaling
        _, _, val_probs, val_labels = self.validate(val_loader)
        logits_list = []
        for x, _ in val_loader:
            logits_list.append(self.model(x.to(self.device)).cpu())
        all_logits = torch.cat(logits_list)
        temperature = calibrate_temperature(all_logits, torch.tensor(val_labels))

        # Re-evaluate with temperature
        _, _, val_probs_cal, _ = self.validate(val_loader, temperature)
        threshold, f1 = calibrate_threshold(val_probs_cal, np.array(val_labels))

        # Save
        torch.save(self.best_state, model_save_path)
        config = {
            "model_path": model_save_path,
            "temperature": temperature,
            "threshold": float(threshold),
            "val_accuracy": float(self.best_val_acc),
            "val_f1": float(f1),
            "trained_at": datetime.now().isoformat(),
        }
        with open(config_save_path, "w") as f:
            json.dump(config, f, indent=2)

        print(f"\nSaved: {model_save_path}")
        print(f"Temperature: {temperature:.3f}, Threshold: {threshold:.3f}, F1: {f1:.3f}")
        return config

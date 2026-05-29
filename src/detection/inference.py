"""
Inference engine with TTA, temperature scaling, and ensemble support.
"""

import json
import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from src.detection.models import build_model
from src.detection.transforms import get_tta_transforms, ColdToneFilter, CropLeftThird


class Predictor:
    """Single model predictor with TTA and temperature scaling."""

    def __init__(self, model: nn.Module, temperature: float = 1.0,
                 threshold: float = 0.5, device: str = "cpu"):
        self.model = model.to(device)
        self.model.eval()
        self.temperature = temperature
        self.threshold = threshold
        self.device = device
        self.tta_views = get_tta_transforms()
        self.cold_tone = ColdToneFilter()
        self.crop = CropLeftThird()

    @classmethod
    def from_checkpoint(cls, model_path: str, model_name: str = "resnet34",
                        config_path: str = None, device: str = "cpu") -> "Predictor":
        model = build_model(model_name, num_classes=2, pretrained=False)
        model.load_state_dict(torch.load(model_path, map_location=device))
        temperature = 1.0
        threshold = 0.5
        if config_path and os.path.exists(config_path):
            with open(config_path) as f:
                cfg = json.load(f)
            temperature = cfg.get("temperature", 1.0)
            threshold = cfg.get("threshold", 0.5)
        return cls(model, temperature, threshold, device)

    @torch.no_grad()
    def predict(self, img: Image.Image) -> dict:
        """Predict with TTA — returns {label, probability, confidence, is_ok}."""
        img = self.crop(self.cold_tone(img))

        all_logits = []
        for tf in self.tta_views:
            tensor = tf(img).unsqueeze(0).to(self.device)
            logits = self.model(tensor)
            all_logits.append(logits.cpu())

        avg_logits = torch.stack(all_logits).mean(0)
        scaled = avg_logits / self.temperature
        probs = torch.softmax(scaled, dim=1)[0]
        ng_prob = probs[1].item()

        is_ng = ng_prob >= self.threshold
        return {
            "label": "NG" if is_ng else "OK",
            "probability": ng_prob,
            "confidence": ng_prob if is_ng else 1 - ng_prob,
            "is_ok": not is_ng,
            "logits": avg_logits.numpy().tolist()[0],
        }

    @torch.no_grad()
    def predict_raw(self, img: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
        """Return logits and probs without thresholding (for ensemble)."""
        img = self.crop(self.cold_tone(img))
        all_logits = []
        for tf in self.tta_views:
            tensor = tf(img).unsqueeze(0).to(self.device)
            all_logits.append(self.model(tensor).cpu())
        avg = torch.stack(all_logits).mean(0) / self.temperature
        probs = torch.softmax(avg, dim=1)
        return avg, probs

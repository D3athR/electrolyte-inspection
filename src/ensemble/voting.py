"""
Multi-model ensemble with soft/hard voting and weighted strategies.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from PIL import Image

from src.detection.inference import Predictor


@dataclass
class EnsembleResult:
    label: str
    probability: float
    confidence: float
    is_ok: bool
    model_votes: list[dict]   # Per-model results
    agreement: float          # 0-1, how much models agree


class EnsemblePredictor:
    """Combine multiple models via soft voting."""

    def __init__(self, predictors: list[Predictor], strategy: str = "soft_voting",
                 weights: list[float] = None):
        self.predictors = predictors
        self.strategy = strategy
        self.weights = weights or [1.0] * len(predictors)
        # Normalize
        total = sum(self.weights)
        self.weights = [w / total for w in self.weights]

    def predict(self, img: Image.Image) -> EnsembleResult:
        """Ensemble prediction with per-model breakdown."""
        model_votes = []
        weighted_probs = []

        for i, pred in enumerate(self.predictors):
            result = pred.predict(img)
            model_votes.append(result)
            weighted_probs.append(result["probability"] * self.weights[i])

        # Soft voting: weighted average of NG probabilities
        ng_prob = sum(weighted_probs)

        # Agreement metric: std of per-model NG probs (lower = more agreement)
        raw_probs = [r["probability"] for r in model_votes]
        agreement = 1.0 - min(float(np.std(raw_probs)) * 3, 1.0)

        # Hard voting for tie-breaking
        n_ng = sum(1 for r in model_votes if r["label"] == "NG")

        if self.strategy == "hard_voting":
            is_ng = n_ng > len(self.predictors) // 2
        else:
            is_ng = ng_prob >= 0.5

        return EnsembleResult(
            label="NG" if is_ng else "OK",
            probability=ng_prob,
            confidence=ng_prob if is_ng else 1 - ng_prob,
            is_ok=not is_ng,
            model_votes=model_votes,
            agreement=agreement,
        )

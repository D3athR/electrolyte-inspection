"""
A/B testing framework for model comparison and automatic promotion.

Control = current production model (80% traffic by default)
Treatment = candidate model (20% traffic)
Promotion when treatment beats control by threshold on min_samples.
"""

import json
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class ABMetrics:
    """Per-model metrics collector."""
    total: int = 0
    ok_count: int = 0
    ng_count: int = 0
    correct: int = 0    # If ground truth available
    sum_confidence: float = 0.0
    sum_latency: float = 0.0
    confusion: dict = field(default_factory=lambda: {"TP": 0, "TN": 0, "FP": 0, "FN": 0})

    def update(self, predicted_ok: bool, ground_truth_ok: Optional[bool] = None,
               confidence: float = 0, latency: float = 0):
        self.total += 1
        if predicted_ok:
            self.ok_count += 1
        else:
            self.ng_count += 1
        self.sum_confidence += confidence
        self.sum_latency += latency

        if ground_truth_ok is not None:
            if predicted_ok and ground_truth_ok:
                self.confusion["TN"] += 1
                self.correct += 1
            elif not predicted_ok and not ground_truth_ok:
                self.confusion["TP"] += 1
                self.correct += 1
            elif not predicted_ok and ground_truth_ok:
                self.confusion["FP"] += 1
            elif predicted_ok and not ground_truth_ok:
                self.confusion["FN"] += 1

    @property
    def accuracy(self) -> float:
        return self.correct / max(self.total, 1)

    @property
    def avg_confidence(self) -> float:
        return self.sum_confidence / max(self.total, 1)


class ABTestManager:
    """A/B testing with automatic promotion logic."""

    def __init__(self, control_name: str = "resnet34",
                 promotion_threshold: float = 0.02,
                 min_samples: int = 1000):
        self.control_name = control_name
        self.treatment_name: Optional[str] = None
        self.metrics: dict[str, ABMetrics] = defaultdict(ABMetrics)
        self.promotion_threshold = promotion_threshold
        self.min_samples = min_samples
        self.promotion_history: list[dict] = []
        self.started_at = datetime.now().isoformat()

    def set_treatment(self, name: str):
        self.treatment_name = name

    def record(self, model_name: str, predicted_ok: bool,
               ground_truth_ok: Optional[bool] = None,
               confidence: float = 0, latency: float = 0):
        self.metrics[model_name].update(predicted_ok, ground_truth_ok, confidence, latency)

    def should_promote(self) -> tuple[bool, str]:
        """Check if treatment should replace control."""
        if not self.treatment_name:
            return False, "No treatment model set"

        ctrl = self.metrics[self.control_name]
        trt = self.metrics[self.treatment_name]

        if trt.total < self.min_samples:
            return False, f"Treatment samples {trt.total}/{self.min_samples}"

        if trt.accuracy >= ctrl.accuracy + self.promotion_threshold:
            reason = (f"Treatment ({self.treatment_name}) accuracy {trt.accuracy:.3f} >= "
                      f"Control ({self.control_name}) {ctrl.accuracy:.3f} + {self.promotion_threshold}")
            return True, reason

        return False, f"Treatment not better ({trt.accuracy:.3f} vs {ctrl.accuracy:.3f})"

    def promote(self) -> str:
        """Promote treatment to control, archive old control."""
        self.promotion_history.append({
            "old_control": self.control_name,
            "new_control": self.treatment_name,
            "control_metrics": self._metrics_dict(self.metrics[self.control_name]),
            "treatment_metrics": self._metrics_dict(self.metrics[self.treatment_name]),
            "promoted_at": datetime.now().isoformat(),
        })
        self.control_name = self.treatment_name
        self.treatment_name = None
        # Reset metrics for new control
        self.metrics = defaultdict(ABMetrics)
        return self.control_name

    def report(self) -> dict:
        return {
            "control": {"name": self.control_name, **self._metrics_dict(self.metrics[self.control_name])},
            "treatment": {"name": self.treatment_name, **self._metrics_dict(self.metrics[self.treatment_name])}
            if self.treatment_name else None,
            "should_promote": self.should_promote()[0],
            "started_at": self.started_at,
        }

    def _metrics_dict(self, m: ABMetrics) -> dict:
        return {
            "total": m.total, "ok_count": m.ok_count, "ng_count": m.ng_count,
            "accuracy": m.accuracy, "avg_confidence": m.avg_confidence,
            "confusion": m.confusion,
        }

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump({
                "control_name": self.control_name,
                "treatment_name": self.treatment_name,
                "metrics": {k: self._metrics_dict(v) for k, v in self.metrics.items()},
                "promotion_history": self.promotion_history,
                "started_at": self.started_at,
            }, f, indent=2, default=str)

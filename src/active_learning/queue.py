"""
Active Learning Queue — collects low-confidence samples for human review.

When model uncertainty is high (top-2 class probs are close),
the sample is flagged and queued for labeling.
"""

import json
import os
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass
class QueuedSample:
    image_path: str
    timestamp: str
    model_name: str
    probability: float
    uncertainty: float       # 1 - abs(p_ok - p_ng), 0=confident, 1=unsure
    top2_probs: list[float]  # [p_ok, p_ng] or [p_ng, p_ok]
    label: Optional[str] = None      # Human-annotated label
    labeled_at: Optional[str] = None


class ActiveLearningQueue:
    """Manages queue of uncertain samples for human labeling."""

    def __init__(self, queue_dir: str = "./al_queue",
                 uncertainty_threshold: float = 0.15,
                 max_size: int = 500):
        self.queue_dir = Path(queue_dir)
        self.uncertainty_threshold = uncertainty_threshold
        self.max_size = max_size
        self._samples: list[QueuedSample] = []
        self._load()

    def _load(self):
        """Restore queue from disk."""
        index_path = self.queue_dir / "queue_index.json"
        if index_path.exists():
            with open(index_path) as f:
                data = json.load(f)
            self._samples = [QueuedSample(**s) for s in data]

    def _save(self):
        self.queue_dir.mkdir(parents=True, exist_ok=True)
        with open(self.queue_dir / "queue_index.json", "w") as f:
            json.dump([s.__dict__ for s in self._samples], f, indent=2, default=str)

    def should_queue(self, probs: list[float]) -> tuple[bool, float]:
        """Check if prediction is uncertain enough to queue."""
        # Margin sampling: difference between top 2 probabilities
        sorted_probs = sorted(probs, reverse=True)
        margin = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) >= 2 else 1.0
        uncertainty = 1.0 - margin
        return uncertainty > self.uncertainty_threshold, uncertainty

    def add(self, image_path: str, model_name: str, probs: list[float]):
        """Queue a sample for labeling."""
        _, uncertainty = self.should_queue(probs)

        # Copy image to queue directory
        img_name = os.path.basename(image_path)
        dest = self.queue_dir / "images" / f"{int(time.time())}_{img_name}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_path, str(dest))

        sample = QueuedSample(
            image_path=str(dest),
            timestamp=datetime.now().isoformat(),
            model_name=model_name,
            probability=probs[1] if len(probs) > 1 else probs[0],
            uncertainty=uncertainty,
            top2_probs=sorted(probs, reverse=True)[:2],
        )
        self._samples.append(sample)

        # Trim queue
        if len(self._samples) > self.max_size:
            # Keep most uncertain
            self._samples.sort(key=lambda s: s.uncertainty, reverse=True)
            removed = self._samples[self.max_size:]
            self._samples = self._samples[:self.max_size]
            for s in removed:
                if os.path.exists(s.image_path):
                    os.remove(s.image_path)

        self._save()

    def label(self, index: int, label: str):
        """Record human annotation."""
        if 0 <= index < len(self._samples):
            self._samples[index].label = label
            self._samples[index].labeled_at = datetime.now().isoformat()
            self._save()

    def get_labeled(self) -> list[QueuedSample]:
        """Get all labeled samples ready for retraining."""
        return [s for s in self._samples if s.label is not None]

    def get_pending(self) -> list[QueuedSample]:
        """Get unlabeled samples."""
        return [s for s in self._samples if s.label is None]

    @property
    def pending_count(self) -> int:
        return len(self.get_pending())

    @property
    def labeled_count(self) -> int:
        return len(self.get_labeled())

    def export_labeled(self, raw_data_dir: str):
        """Export labeled samples back to training data directory."""
        for s in self.get_labeled():
            cls_dir = os.path.join(raw_data_dir, s.label)
            os.makedirs(cls_dir, exist_ok=True)
            shutil.copy2(s.image_path, os.path.join(cls_dir, os.path.basename(s.image_path)))
        # Clear exported samples
        self._samples = [s for s in self._samples if s.label is None]
        self._save()

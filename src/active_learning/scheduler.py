"""
Auto-retraining scheduler — triggers retraining when enough labeled data accumulates.
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional


class RetrainScheduler:
    """Monitors active learning queue and triggers retraining."""

    def __init__(self, queue, model_registry_dir: str = "./model_registry",
                 trigger_count: int = 50, min_samples: int = 20):
        self.queue = queue
        self.registry_dir = Path(model_registry_dir)
        self.trigger_count = trigger_count
        self.min_samples = min_samples
        self.history: list[dict] = []
        self._load_history()

    def _load_history(self):
        hp = self.registry_dir / "retrain_history.json"
        if hp.exists():
            with open(hp) as f:
                self.history = json.load(f)

    def _save_history(self):
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        with open(self.registry_dir / "retrain_history.json", "w") as f:
            json.dump(self.history, f, indent=2, default=str)

    def should_retrain(self) -> tuple[bool, str]:
        """Check if enough labeled data has accumulated."""
        labeled = self.queue.labeled_count
        if labeled < self.min_samples:
            return False, f"Need {self.min_samples} labeled samples, have {labeled}"
        if labeled >= self.trigger_count:
            return True, f"Trigger: {labeled} >= {self.trigger_count}"
        return False, f"Waiting: {labeled}/{self.trigger_count}"

    def retrain(self, raw_data_dir: str, train_script: str = "scripts/train_all.py"):
        """Export labeled data, retrain all models, evaluate, register."""
        print(f"\n{'='*60}")
        print(f"Auto-retrain triggered at {datetime.now().isoformat()}")
        print(f"{'='*60}")

        # Export labeled samples to training data
        before_count = self.queue.labeled_count
        self.queue.export_labeled(raw_data_dir)
        print(f"Exported {before_count} labeled samples to {raw_data_dir}")

        # Run training script
        result = subprocess.run(
            [sys.executable, train_script, "--raw_dir", raw_data_dir],
            capture_output=True, text=True, timeout=1800,
        )
        print(result.stdout[-1000:])

        # Find best model
        best_model = self._find_best_model()
        record = {
            "timestamp": datetime.now().isoformat(),
            "samples_added": before_count,
            "best_model": best_model,
            "success": result.returncode == 0,
        }
        self.history.append(record)
        self._save_history()

        return record

    def _find_best_model(self) -> Optional[str]:
        """Find the best model from latest training artifacts."""
        # Simplified: check model registry for latest checkpoint
        checkpoints = sorted(self.registry_dir.glob("*_config.json"),
                             key=os.path.getmtime, reverse=True)
        if not checkpoints:
            return None
        with open(checkpoints[0]) as f:
            cfg = json.load(f)
        return cfg.get("model_path", str(checkpoints[0]))

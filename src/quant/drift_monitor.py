"""
Simple feature drift monitor.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Dict, List


class DriftMonitor:
    def __init__(self, feature_names: List[str], window: int = 600):
        self.feature_names = list(feature_names)
        self.window = int(max(60, window))
        self.recent: Deque[Dict[str, float]] = deque(maxlen=self.window)
        self.baseline_mean: Dict[str, float] = {f: 0.0 for f in self.feature_names}
        self.baseline_std: Dict[str, float] = {f: 1.0 for f in self.feature_names}
        self.initialized = False
        self.current_score: float = 0.0

    def observe(self, features: Dict[str, float]) -> float:
        row = {f: float(features.get(f, 0.0)) for f in self.feature_names}
        self.recent.append(row)
        if not self.initialized and len(self.recent) >= 120:
            self._set_baseline()
        if self.initialized:
            self.current_score = self._compute_score()
        return self.current_score

    def _set_baseline(self) -> None:
        n = len(self.recent)
        if n < 30:
            return
        for f in self.feature_names:
            vals = [r[f] for r in self.recent]
            mean = sum(vals) / n
            var = sum((v - mean) ** 2 for v in vals) / max(1, n - 1)
            self.baseline_mean[f] = mean
            self.baseline_std[f] = max(1e-6, var**0.5)
        self.initialized = True

    def _compute_score(self) -> float:
        n = len(self.recent)
        if n < 30:
            return 0.0
        shift_scores: List[float] = []
        for f in self.feature_names:
            vals = [r[f] for r in self.recent]
            mean_recent = sum(vals) / n
            z = abs(mean_recent - self.baseline_mean[f]) / self.baseline_std[f]
            shift_scores.append(min(3.0, z) / 3.0)
        if not shift_scores:
            return 0.0
        return max(0.0, min(1.0, sum(shift_scores) / len(shift_scores)))

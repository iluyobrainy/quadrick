"""
Probability calibration (Platt / isotonic-like monotonic mapping).
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple


def _clip_p(p: float) -> float:
    return max(1e-4, min(1.0 - 1e-4, float(p)))


class ProbabilityCalibrator:
    def __init__(self, method: str = "platt"):
        self.method = method if method in {"platt", "isotonic"} else "platt"
        self.A = 1.0
        self.B = 0.0
        self.iso_breaks: List[float] = []
        self.iso_values: List[float] = []
        self.samples: List[Tuple[float, int]] = []
        self.last_error: float = 0.12

    @staticmethod
    def _logit(p: float) -> float:
        p = _clip_p(p)
        return math.log(p / (1.0 - p))

    def add_sample(self, prob_raw: float, y: int, max_samples: int = 1500) -> None:
        self.samples.append((_clip_p(prob_raw), int(1 if y else 0)))
        if len(self.samples) > max_samples:
            self.samples = self.samples[-max_samples:]

    def fit(self) -> None:
        if len(self.samples) < 40:
            return
        if self.method == "platt":
            self._fit_platt()
        else:
            self._fit_isotonic()
        self.last_error = self._compute_ece(self.samples)

    def _fit_platt(self) -> None:
        xs = [self._logit(p) for p, _ in self.samples]
        ys = [y for _, y in self.samples]
        A = self.A
        B = self.B
        lr = 0.01
        for _ in range(180):
            gA = 0.0
            gB = 0.0
            n = max(1, len(xs))
            for x, y in zip(xs, ys):
                z = A * x + B
                pred = 1.0 / (1.0 + math.exp(-z))
                err = pred - y
                gA += err * x
                gB += err
            A -= lr * (gA / n)
            B -= lr * (gB / n)
        self.A = A
        self.B = B

    def _fit_isotonic(self) -> None:
        # Monotonic bin mapping via pooled adjacent violators on sorted bins.
        sorted_samples = sorted(self.samples, key=lambda x: x[0])
        bin_size = max(20, len(sorted_samples) // 25)
        bins: List[List[Tuple[float, int]]] = []
        for i in range(0, len(sorted_samples), bin_size):
            bins.append(sorted_samples[i : i + bin_size])
        means_p = [sum(p for p, _ in b) / len(b) for b in bins]
        means_y = [sum(y for _, y in b) / len(b) for b in bins]
        counts = [len(b) for b in bins]

        # PAV merge when monotonicity violated.
        i = 0
        while i < len(means_y) - 1:
            if means_y[i] <= means_y[i + 1]:
                i += 1
                continue
            total_c = counts[i] + counts[i + 1]
            if total_c <= 0:
                i += 1
                continue
            merged_y = ((means_y[i] * counts[i]) + (means_y[i + 1] * counts[i + 1])) / total_c
            merged_p = ((means_p[i] * counts[i]) + (means_p[i + 1] * counts[i + 1])) / total_c
            means_y[i] = merged_y
            means_p[i] = merged_p
            counts[i] = total_c
            del means_y[i + 1]
            del means_p[i + 1]
            del counts[i + 1]
            if i > 0:
                i -= 1

        self.iso_breaks = means_p
        self.iso_values = [max(0.001, min(0.999, y)) for y in means_y]

    @staticmethod
    def _compute_ece(samples: List[Tuple[float, int]], bins: int = 10) -> float:
        if not samples:
            return 0.5
        ece = 0.0
        n = len(samples)
        for b in range(bins):
            lo = b / bins
            hi = (b + 1) / bins
            bucket = [(p, y) for p, y in samples if lo <= p < hi or (b == bins - 1 and p == hi)]
            if not bucket:
                continue
            conf = sum(p for p, _ in bucket) / len(bucket)
            acc = sum(y for _, y in bucket) / len(bucket)
            ece += (len(bucket) / n) * abs(conf - acc)
        return float(max(0.0, min(1.0, ece)))

    def calibrate(self, prob_raw: float) -> float:
        p = _clip_p(prob_raw)
        if self.method == "platt":
            z = self.A * self._logit(p) + self.B
            out = 1.0 / (1.0 + math.exp(-z))
            return max(0.001, min(0.999, out))
        if not self.iso_breaks or not self.iso_values:
            return p
        if p <= self.iso_breaks[0]:
            return self.iso_values[0]
        for i in range(1, len(self.iso_breaks)):
            if p <= self.iso_breaks[i]:
                p0, p1 = self.iso_breaks[i - 1], self.iso_breaks[i]
                y0, y1 = self.iso_values[i - 1], self.iso_values[i]
                if p1 <= p0:
                    return y1
                alpha = (p - p0) / (p1 - p0)
                return max(0.001, min(0.999, y0 + alpha * (y1 - y0)))
        return self.iso_values[-1]

    def to_dict(self) -> Dict[str, object]:
        return {
            "method": self.method,
            "A": self.A,
            "B": self.B,
            "iso_breaks": self.iso_breaks,
            "iso_values": self.iso_values,
            "last_error": self.last_error,
            "samples": self.samples[-500:],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "ProbabilityCalibrator":
        cal = cls(method=str(data.get("method", "platt")))
        cal.A = float(data.get("A", 1.0))
        cal.B = float(data.get("B", 0.0))
        cal.iso_breaks = [float(x) for x in data.get("iso_breaks", [])]
        cal.iso_values = [float(x) for x in data.get("iso_values", [])]
        cal.last_error = float(data.get("last_error", 0.12))
        cal.samples = [(float(p), int(y)) for p, y in data.get("samples", [])]
        return cal

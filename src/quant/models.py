"""
Lightweight online models for quant forecasting.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


@dataclass
class OnlineLogisticModel:
    feature_names: List[str]
    lr: float = 0.02
    l2: float = 1e-4

    def __post_init__(self) -> None:
        self.weights: Dict[str, float] = {name: 0.0 for name in self.feature_names}
        self.bias: float = 0.0
        self.updates: int = 0

    def predict_proba(self, features: Dict[str, float]) -> float:
        score = self.bias
        for name in self.feature_names:
            score += self.weights.get(name, 0.0) * float(features.get(name, 0.0))
        return max(0.001, min(0.999, _sigmoid(score)))

    def partial_fit(self, features: Dict[str, float], y: float, sample_weight: float = 1.0) -> None:
        target = max(0.0, min(1.0, float(y)))
        p = self.predict_proba(features)
        error = (p - target) * float(sample_weight)
        for name in self.feature_names:
            x = float(features.get(name, 0.0))
            grad = (error * x) + (self.l2 * self.weights.get(name, 0.0))
            self.weights[name] = self.weights.get(name, 0.0) - (self.lr * grad)
        self.bias -= self.lr * error
        self.updates += 1

    def fit_batch(self, samples: Iterable[Dict[str, float]], labels: Iterable[float], epochs: int = 2) -> None:
        xs = list(samples)
        ys = list(labels)
        if not xs or not ys:
            return
        for _ in range(max(1, int(epochs))):
            for x, y in zip(xs, ys):
                self.partial_fit(x, y)

    def to_dict(self) -> Dict[str, object]:
        return {
            "feature_names": self.feature_names,
            "weights": self.weights,
            "bias": self.bias,
            "updates": self.updates,
            "lr": self.lr,
            "l2": self.l2,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "OnlineLogisticModel":
        model = cls(
            feature_names=[str(x) for x in data.get("feature_names", [])],
            lr=float(data.get("lr", 0.02)),
            l2=float(data.get("l2", 1e-4)),
        )
        model.weights = {str(k): float(v) for k, v in dict(data.get("weights", {})).items()}
        model.bias = float(data.get("bias", 0.0))
        model.updates = int(data.get("updates", 0))
        return model


@dataclass
class OnlineRegressor:
    feature_names: List[str]
    lr: float = 0.015
    l2: float = 1e-4

    def __post_init__(self) -> None:
        self.weights: Dict[str, float] = {name: 0.0 for name in self.feature_names}
        self.bias: float = 0.0
        self.updates: int = 0

    def predict(self, features: Dict[str, float]) -> float:
        out = self.bias
        for name in self.feature_names:
            out += self.weights.get(name, 0.0) * float(features.get(name, 0.0))
        return out

    def partial_fit(self, features: Dict[str, float], y: float, sample_weight: float = 1.0) -> None:
        pred = self.predict(features)
        error = (pred - float(y)) * float(sample_weight)
        for name in self.feature_names:
            x = float(features.get(name, 0.0))
            grad = (error * x) + (self.l2 * self.weights.get(name, 0.0))
            self.weights[name] = self.weights.get(name, 0.0) - (self.lr * grad)
        self.bias -= self.lr * error
        self.updates += 1

    def fit_batch(self, samples: Iterable[Dict[str, float]], labels: Iterable[float], epochs: int = 2) -> None:
        xs = list(samples)
        ys = list(labels)
        if not xs or not ys:
            return
        for _ in range(max(1, int(epochs))):
            for x, y in zip(xs, ys):
                self.partial_fit(x, y)

    def to_dict(self) -> Dict[str, object]:
        return {
            "feature_names": self.feature_names,
            "weights": self.weights,
            "bias": self.bias,
            "updates": self.updates,
            "lr": self.lr,
            "l2": self.l2,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "OnlineRegressor":
        model = cls(
            feature_names=[str(x) for x in data.get("feature_names", [])],
            lr=float(data.get("lr", 0.015)),
            l2=float(data.get("l2", 1e-4)),
        )
        model.weights = {str(k): float(v) for k, v in dict(data.get("weights", {})).items()}
        model.bias = float(data.get("bias", 0.0))
        model.updates = int(data.get("updates", 0))
        return model


class HorizonExpert:
    def __init__(self, feature_names: List[str]):
        self.direction = OnlineLogisticModel(feature_names=feature_names, lr=0.02)
        self.move_size = OnlineRegressor(feature_names=feature_names, lr=0.01)
        self.volatility = OnlineRegressor(feature_names=feature_names, lr=0.01)

    def predict(self, features: Dict[str, float]) -> Dict[str, float]:
        raw_prob = self.direction.predict_proba(features)
        expected_move = abs(self.move_size.predict(features))
        volatility_pct = abs(self.volatility.predict(features))
        return {
            "prob_up_raw": max(0.001, min(0.999, raw_prob)),
            "expected_move_pct": max(0.02, expected_move),
            "volatility_pct": max(0.03, volatility_pct),
        }

    def update(self, features: Dict[str, float], label: Dict[str, float]) -> None:
        direction_label = int(label.get("direction_label", 0))
        if direction_label == 1:
            y = 1.0
            w = 1.0
        elif direction_label == -1:
            y = 0.0
            w = 1.0
        else:
            # Neutral outcomes are less informative.
            y = 0.5
            w = 0.35
        self.direction.partial_fit(features, y=y, sample_weight=w)
        self.move_size.partial_fit(features, y=abs(float(label.get("move_pct", 0.0))))
        self.volatility.partial_fit(features, y=abs(float(label.get("realized_volatility", 0.0))))

    def fit_batch(self, rows: List[Dict[str, float]], labels: List[Dict[str, float]], epochs: int = 2) -> None:
        if not rows or not labels:
            return
        for _ in range(max(1, int(epochs))):
            for features, label in zip(rows, labels):
                self.update(features, label)

    @property
    def sample_count(self) -> int:
        return int(self.direction.updates)

    def to_dict(self) -> Dict[str, object]:
        return {
            "direction": self.direction.to_dict(),
            "move_size": self.move_size.to_dict(),
            "volatility": self.volatility.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object], feature_names: List[str]) -> "HorizonExpert":
        expert = cls(feature_names=feature_names)
        if isinstance(data.get("direction"), dict):
            expert.direction = OnlineLogisticModel.from_dict(dict(data["direction"]))
        if isinstance(data.get("move_size"), dict):
            expert.move_size = OnlineRegressor.from_dict(dict(data["move_size"]))
        if isinstance(data.get("volatility"), dict):
            expert.volatility = OnlineRegressor.from_dict(dict(data["volatility"]))
        return expert

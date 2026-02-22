"""
Mixture-of-experts forecast switcher by market regime.
"""

from __future__ import annotations

from typing import Any, Dict, List

from .models import HorizonExpert


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


class RegimeSwitcher:
    def __init__(self, feature_names: List[str], horizons: List[int]):
        self.feature_names = list(feature_names)
        self.horizons = list(horizons)
        self.regimes = ["trend", "range", "volatile"]
        self.experts: Dict[str, Dict[int, HorizonExpert]] = {
            r: {h: HorizonExpert(feature_names=self.feature_names) for h in self.horizons}
            for r in self.regimes
        }
        self.global_experts: Dict[int, HorizonExpert] = {
            h: HorizonExpert(feature_names=self.feature_names) for h in self.horizons
        }

    def _heuristic_prior(self, horizon: int, features: Dict[str, float]) -> Dict[str, float]:
        trend_score = (
            (features.get("trend_5m", 0.0) * 0.18)
            + (features.get("trend_15m", 0.0) * 0.34)
            + (features.get("trend_1h", 0.0) * 0.48)
        )
        momentum_score = (
            (features.get("rsi_5m_norm", 0.0) * 0.10)
            + (features.get("rsi_15m_norm", 0.0) * 0.16)
            + (features.get("rsi_1h_norm", 0.0) * 0.18)
            + (features.get("macd_5m", 0.0) * 0.12)
            + (features.get("macd_15m", 0.0) * 0.14)
        )
        orderflow_score = (
            (features.get("order_imbalance_norm", 0.0) * 0.12)
            + (features.get("order_depth_bias_norm", 0.0) * 0.11)
        )
        market_context_score = (
            (features.get("btc_24h_change_norm", 0.0) * 0.08)
            + (features.get("price_24h_change_norm", 0.0) * 0.10)
            - (features.get("funding_rate_norm", 0.0) * 0.05)
        )

        signal = trend_score + momentum_score + orderflow_score + market_context_score
        horizon_scale = {5: 0.95, 15: 1.05, 30: 1.15}.get(int(horizon), 1.0)
        signal *= horizon_scale

        friction_penalty = (
            (features.get("spread_bps_norm", 0.0) * 0.015)
            + (features.get("reject_streak_norm", 0.0) * 0.020)
        )
        prob_up_raw = _clamp(0.5 + (signal * 0.18) - friction_penalty, 0.08, 0.92)

        atr15 = abs(float(features.get("atr_pct_15m", 0.0)))
        atr1h = abs(float(features.get("atr_pct_1h", 0.0)))
        bb_width = abs(float(features.get("bb_width_15m", 0.0)))
        directional_pressure = min(1.0, abs(signal))
        expected_move = (
            (atr15 * 0.70)
            + (atr1h * 0.25)
            + (bb_width * 1.80)
            + (directional_pressure * 0.22)
        )
        expected_move *= {5: 0.95, 15: 1.10, 30: 1.30}.get(int(horizon), 1.0)
        expected_move = _clamp(expected_move, 0.08, 3.20)

        volatility = (
            (atr15 * 0.85)
            + (bb_width * 1.60)
            + (features.get("spread_bps_norm", 0.0) * 0.03)
        )
        volatility = _clamp(volatility, 0.04, 4.00)

        return {
            "prob_up_raw": prob_up_raw,
            "expected_move_pct": expected_move,
            "volatility_pct": volatility,
        }

    def _sample_strength(self, regime: str, horizon: int) -> float:
        regime_key = regime if regime in self.experts else "range"
        local_samples = float(self.experts[regime_key][horizon].sample_count)
        global_samples = float(self.global_experts[horizon].sample_count)
        weighted_samples = (local_samples * 0.65) + (global_samples * 0.35)
        return _clamp(weighted_samples / 180.0, 0.0, 1.0)

    def predict(self, regime: str, horizon: int, features: Dict[str, float]) -> Dict[str, float]:
        regime_key = regime if regime in self.experts else "range"
        local = self.experts[regime_key][horizon].predict(features)
        global_pred = self.global_experts[horizon].predict(features)
        heuristic = self._heuristic_prior(horizon=horizon, features=features)
        sample_strength = self._sample_strength(regime=regime_key, horizon=horizon)

        # Blend local expert with global fallback for robustness.
        model_blend = {
            "prob_up_raw": (0.70 * local["prob_up_raw"]) + (0.30 * global_pred["prob_up_raw"]),
            "expected_move_pct": (0.70 * local["expected_move_pct"]) + (0.30 * global_pred["expected_move_pct"]),
            "volatility_pct": (0.70 * local["volatility_pct"]) + (0.30 * global_pred["volatility_pct"]),
        }

        return {
            "prob_up_raw": (sample_strength * model_blend["prob_up_raw"])
            + ((1.0 - sample_strength) * heuristic["prob_up_raw"]),
            "expected_move_pct": (sample_strength * model_blend["expected_move_pct"])
            + ((1.0 - sample_strength) * heuristic["expected_move_pct"]),
            "volatility_pct": (sample_strength * model_blend["volatility_pct"])
            + ((1.0 - sample_strength) * heuristic["volatility_pct"]),
            "sample_strength": sample_strength,
        }

    def update(self, regime: str, horizon: int, features: Dict[str, float], label: Dict[str, float]) -> None:
        regime_key = regime if regime in self.experts else "range"
        self.experts[regime_key][horizon].update(features, label)
        self.global_experts[horizon].update(features, label)

    def fit_walk_forward(
        self,
        horizon: int,
        rows: List[Dict[str, float]],
        labels: List[Dict[str, float]],
        regimes: List[str],
        epochs: int = 2,
    ) -> None:
        if not rows or not labels or not regimes:
            return
        buckets: Dict[str, List[int]] = {r: [] for r in self.regimes}
        for idx, regime in enumerate(regimes):
            buckets[regime if regime in buckets else "range"].append(idx)

        for regime_key, indices in buckets.items():
            if not indices:
                continue
            r_rows = [rows[i] for i in indices]
            r_labels = [labels[i] for i in indices]
            self.experts[regime_key][horizon].fit_batch(r_rows, r_labels, epochs=epochs)
        self.global_experts[horizon].fit_batch(rows, labels, epochs=epochs)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature_names": self.feature_names,
            "horizons": self.horizons,
            "experts": {
                regime: {str(h): expert.to_dict() for h, expert in horizon_map.items()}
                for regime, horizon_map in self.experts.items()
            },
            "global_experts": {str(h): expert.to_dict() for h, expert in self.global_experts.items()},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RegimeSwitcher":
        feature_names = [str(x) for x in data.get("feature_names", [])]
        horizons = [int(x) for x in data.get("horizons", [5, 15, 30])]
        obj = cls(feature_names=feature_names, horizons=horizons)
        experts_data = data.get("experts", {})
        for regime, horizon_map in experts_data.items():
            if regime not in obj.experts:
                continue
            for h_str, expert_data in dict(horizon_map).items():
                horizon = int(h_str)
                if horizon not in obj.experts[regime]:
                    continue
                obj.experts[regime][horizon] = HorizonExpert.from_dict(expert_data, feature_names=obj.feature_names)
        global_data = data.get("global_experts", {})
        for h_str, expert_data in dict(global_data).items():
            horizon = int(h_str)
            if horizon not in obj.global_experts:
                continue
            obj.global_experts[horizon] = HorizonExpert.from_dict(expert_data, feature_names=obj.feature_names)
        return obj

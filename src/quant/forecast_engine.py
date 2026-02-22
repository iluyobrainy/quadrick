"""
Deterministic multi-horizon forecast helper.

This module does not place trades directly; it produces directional priors that
can be blended with existing EV scoring.
"""

from __future__ import annotations

from typing import Any, Dict


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


class ForecastEngine:
    """Feature-based directional forecast across 5m/15m/30m horizons."""

    def __init__(self) -> None:
        self._horizon_map = {
            "5m": "5m",
            "15m": "15m",
            "30m": "1h",
        }

    @staticmethod
    def _f(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _trend_bias(trend: Any) -> float:
        t = str(trend or "").lower()
        if t in {"uptrend", "bullish", "trending_up", "up"}:
            return 1.0
        if t in {"downtrend", "bearish", "trending_down", "down"}:
            return -1.0
        return 0.0

    def _horizon_forecast(self, tf_data: Dict[str, Any], current_price: float, horizon: str) -> Dict[str, Any]:
        rsi = self._f(tf_data.get("rsi"), 50.0)
        adx = self._f(tf_data.get("adx"), 20.0)
        macd_hist = self._f(tf_data.get("macd_histogram"), 0.0)
        volume_ratio = self._f(tf_data.get("volume_ratio"), 1.0)
        atr = self._f(tf_data.get("atr"), 0.0)
        trend = self._trend_bias(tf_data.get("trend"))

        rsi_component = _clamp((rsi - 50.0) / 25.0, -1.0, 1.0)
        macd_component = _clamp(macd_hist * 75.0, -1.0, 1.0)
        volume_component = _clamp((volume_ratio - 1.0) / 1.5, -0.5, 0.5)
        trend_component = trend

        directional_score = (
            (0.38 * trend_component)
            + (0.26 * rsi_component)
            + (0.26 * macd_component)
            + (0.10 * volume_component)
        )
        directional_score = _clamp(directional_score, -1.0, 1.0)

        trend_strength = _clamp(adx / 35.0, 0.0, 1.0)
        confidence = _clamp((abs(directional_score) * 0.70) + (trend_strength * 0.30), 0.0, 1.0)

        prob_up = _clamp(0.5 + (directional_score * 0.24), 0.05, 0.95)
        prob_down = _clamp(1.0 - prob_up, 0.05, 0.95)

        atr_pct = (atr / current_price * 100.0) if current_price > 0 and atr > 0 else 0.0
        horizon_mult = {"5m": 0.40, "15m": 0.75, "30m": 1.15}.get(horizon, 0.70)
        expected_move_pct = max(0.03, atr_pct * horizon_mult)

        direction = "up" if prob_up >= prob_down else "down"
        return {
            "direction": direction,
            "prob_up": round(prob_up, 4),
            "prob_down": round(prob_down, 4),
            "confidence": round(confidence, 4),
            "expected_move_pct": round(expected_move_pct, 4),
        }

    def predict(self, symbol_analysis: Dict[str, Any]) -> Dict[str, Any]:
        tf = (symbol_analysis or {}).get("timeframe_analysis", {}) or {}
        current_price = self._f((symbol_analysis or {}).get("current_price"), 0.0)

        horizons: Dict[str, Dict[str, Any]] = {}
        for horizon, tf_key in self._horizon_map.items():
            horizons[horizon] = self._horizon_forecast(
                tf_data=tf.get(tf_key, {}) or {},
                current_price=current_price,
                horizon=horizon,
            )

        prob_up_values = [h["prob_up"] for h in horizons.values()]
        conf_values = [h["confidence"] for h in horizons.values()]
        avg_prob_up = sum(prob_up_values) / len(prob_up_values) if prob_up_values else 0.5
        avg_conf = sum(conf_values) / len(conf_values) if conf_values else 0.0
        consensus = "neutral"
        if avg_prob_up >= 0.55:
            consensus = "up"
        elif avg_prob_up <= 0.45:
            consensus = "down"

        return {
            "horizons": horizons,
            "aggregate": {
                "prob_up": round(avg_prob_up, 4),
                "prob_down": round(1.0 - avg_prob_up, 4),
                "confidence": round(avg_conf, 4),
                "consensus": consensus,
            },
        }


"""
Feature extraction for quant models.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple


def _f(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _trend_to_num(value: Any) -> float:
    trend = str(value or "").lower()
    if trend in {"uptrend", "bullish", "trending_up", "up"}:
        return 1.0
    if trend in {"downtrend", "bearish", "trending_down", "down"}:
        return -1.0
    return 0.0


class FeatureStore:
    FEATURE_NAMES = [
        "trend_5m",
        "trend_15m",
        "trend_1h",
        "rsi_5m_norm",
        "rsi_15m_norm",
        "rsi_1h_norm",
        "macd_5m",
        "macd_15m",
        "adx_15m_norm",
        "adx_1h_norm",
        "atr_pct_15m",
        "atr_pct_1h",
        "bb_width_15m",
        "volume_ratio_5m_norm",
        "volume_ratio_15m_norm",
        "price_24h_change_norm",
        "funding_rate_norm",
        "spread_bps_norm",
        "order_imbalance_norm",
        "order_depth_bias_norm",
        "btc_24h_change_norm",
        "open_positions_norm",
        "reject_streak_norm",
    ]

    def extract(
        self,
        symbol: str,
        symbol_analysis: Dict[str, Any],
        market_data: Dict[str, Any],
        open_positions_count: int,
        reject_streak: int,
    ) -> Tuple[Dict[str, float], str]:
        tf = (symbol_analysis or {}).get("timeframe_analysis", {}) or {}
        tf5 = tf.get("5m", {}) or {}
        tf15 = tf.get("15m", {}) or {}
        tf1h = tf.get("1h", {}) or {}

        ticker = (market_data.get("tickers", {}) or {}).get(symbol)
        price = _f((symbol_analysis or {}).get("current_price"), 0.0)
        bid = _f(getattr(ticker, "bid_price", 0.0), 0.0)
        ask = _f(getattr(ticker, "ask_price", 0.0), 0.0)
        if price <= 0:
            price = _f(getattr(ticker, "last_price", 0.0), 0.0)
        if price <= 0:
            price = 1.0

        spread_bps = ((ask - bid) / price * 10000.0) if ask > 0 and bid > 0 and price > 0 else 0.0
        funding_rate = _f((market_data.get("funding_rates", {}) or {}).get(symbol), 0.0)
        btc_change = _f(market_data.get("btc_24h_change"), 0.0)
        price_24h_change = _f(getattr(ticker, "price_24h_change", 0.0), 0.0) * 100.0

        order_flow = (market_data.get("order_flow", {}) or {}).get(symbol, {}) or {}
        imbalance = _f(order_flow.get("bid_ask_imbalance"), 1.0)
        bid_depth = _f(order_flow.get("bid_depth_10_levels_usd"), 0.0)
        ask_depth = _f(order_flow.get("ask_depth_10_levels_usd"), 0.0)
        depth_bias = ((bid_depth - ask_depth) / max(1.0, bid_depth + ask_depth))

        atr_15m = _f(tf15.get("atr"), 0.0)
        atr_1h = _f(tf1h.get("atr"), 0.0)
        features = {
            "trend_5m": _trend_to_num(tf5.get("trend")),
            "trend_15m": _trend_to_num(tf15.get("trend")),
            "trend_1h": _trend_to_num(tf1h.get("trend")),
            "rsi_5m_norm": _clamp((_f(tf5.get("rsi"), 50.0) - 50.0) / 50.0, -1.0, 1.0),
            "rsi_15m_norm": _clamp((_f(tf15.get("rsi"), 50.0) - 50.0) / 50.0, -1.0, 1.0),
            "rsi_1h_norm": _clamp((_f(tf1h.get("rsi"), 50.0) - 50.0) / 50.0, -1.0, 1.0),
            "macd_5m": _clamp(_f(tf5.get("macd_histogram"), 0.0) * 100.0, -1.0, 1.0),
            "macd_15m": _clamp(_f(tf15.get("macd_histogram"), 0.0) * 100.0, -1.0, 1.0),
            "adx_15m_norm": _clamp(_f(tf15.get("adx"), 20.0) / 50.0, 0.0, 2.0),
            "adx_1h_norm": _clamp(_f(tf1h.get("adx"), 20.0) / 50.0, 0.0, 2.0),
            "atr_pct_15m": _clamp((atr_15m / price) * 100.0 if price > 0 else 0.0, 0.0, 8.0),
            "atr_pct_1h": _clamp((atr_1h / price) * 100.0 if price > 0 else 0.0, 0.0, 12.0),
            "bb_width_15m": _clamp(_f(tf15.get("bb_width"), 0.02), 0.0, 0.5),
            "volume_ratio_5m_norm": _clamp(_f(tf5.get("volume_ratio"), 1.0) / 5.0, 0.0, 2.0),
            "volume_ratio_15m_norm": _clamp(_f(tf15.get("volume_ratio"), 1.0) / 5.0, 0.0, 2.0),
            "price_24h_change_norm": _clamp(price_24h_change / 10.0, -2.5, 2.5),
            "funding_rate_norm": _clamp(funding_rate * 10000.0, -2.5, 2.5),
            "spread_bps_norm": _clamp(spread_bps / 10.0, 0.0, 5.0),
            "order_imbalance_norm": _clamp((imbalance - 1.0), -2.0, 2.0),
            "order_depth_bias_norm": _clamp(depth_bias, -1.0, 1.0),
            "btc_24h_change_norm": _clamp(btc_change / 10.0, -2.5, 2.5),
            "open_positions_norm": _clamp(open_positions_count / 5.0, 0.0, 1.5),
            "reject_streak_norm": _clamp(reject_streak / 5.0, 0.0, 2.0),
        }

        # Guarantee full vector keys.
        for name in self.FEATURE_NAMES:
            features.setdefault(name, 0.0)

        regime = self._detect_regime(features)
        return features, regime

    @staticmethod
    def _detect_regime(features: Dict[str, float]) -> str:
        adx = features.get("adx_1h_norm", 0.0) * 50.0
        trend_strength = abs(features.get("trend_1h", 0.0))
        bb_width = features.get("bb_width_15m", 0.02)
        atr_pct = features.get("atr_pct_15m", 0.0)

        if adx >= 25 and trend_strength >= 0.8:
            return "trend"
        if bb_width >= 0.05 or atr_pct >= 1.2:
            return "volatile"
        return "range"

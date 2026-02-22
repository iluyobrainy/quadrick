"""
Adaptive symbol+side+regime policy memory.

Tracks bucket performance and outputs soft controls:
- state: green / yellow / red
- risk and ranking multipliers
- quality and edge-floor adjustments
- probe cadence for red buckets
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _f(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _bucket_key(symbol: str, side: str, regime: str) -> str:
    sym = str(symbol or "").strip().upper()
    sd = str(side or "unknown").strip().title()
    rg = str(regime or "unknown").strip().lower()
    if sd not in {"Buy", "Sell"}:
        sd = "unknown"
    if not rg:
        rg = "unknown"
    return f"{sym}|{sd}|{rg}"


@dataclass
class BucketState:
    symbol: str
    side: str
    regime: str
    state: str = "green"
    weighted_closed_trades: float = 0.0
    closed_trades: float = 0.0
    raw_expectancy_pct: float = 0.0
    shrunk_expectancy_pct: float = 0.0
    raw_win_rate: float = 0.5
    shrunk_win_rate: float = 0.5
    avg_slippage_bps: float = 0.0
    tail_loss_3_rate: float = 0.0
    tail_loss_5_rate: float = 0.0
    tail_loss_7_rate: float = 0.0
    unstable: bool = False
    short_expectancy_pct: float = 0.0
    updated_at_utc: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "side": self.side,
            "regime": self.regime,
            "state": self.state,
            "weighted_closed_trades": self.weighted_closed_trades,
            "closed_trades": self.closed_trades,
            "raw_expectancy_pct": self.raw_expectancy_pct,
            "shrunk_expectancy_pct": self.shrunk_expectancy_pct,
            "raw_win_rate": self.raw_win_rate,
            "shrunk_win_rate": self.shrunk_win_rate,
            "avg_slippage_bps": self.avg_slippage_bps,
            "tail_loss_3_rate": self.tail_loss_3_rate,
            "tail_loss_5_rate": self.tail_loss_5_rate,
            "tail_loss_7_rate": self.tail_loss_7_rate,
            "unstable": bool(self.unstable),
            "short_expectancy_pct": self.short_expectancy_pct,
            "updated_at_utc": self.updated_at_utc,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BucketState":
        return cls(
            symbol=str(data.get("symbol") or ""),
            side=str(data.get("side") or "unknown"),
            regime=str(data.get("regime") or "unknown"),
            state=str(data.get("state") or "green"),
            weighted_closed_trades=_f(data.get("weighted_closed_trades"), 0.0),
            closed_trades=_f(data.get("closed_trades"), 0.0),
            raw_expectancy_pct=_f(data.get("raw_expectancy_pct"), 0.0),
            shrunk_expectancy_pct=_f(data.get("shrunk_expectancy_pct"), 0.0),
            raw_win_rate=_f(data.get("raw_win_rate"), 0.5),
            shrunk_win_rate=_f(data.get("shrunk_win_rate"), 0.5),
            avg_slippage_bps=_f(data.get("avg_slippage_bps"), 0.0),
            tail_loss_3_rate=_f(data.get("tail_loss_3_rate"), 0.0),
            tail_loss_5_rate=_f(data.get("tail_loss_5_rate"), 0.0),
            tail_loss_7_rate=_f(data.get("tail_loss_7_rate"), 0.0),
            unstable=bool(data.get("unstable", False)),
            short_expectancy_pct=_f(data.get("short_expectancy_pct"), 0.0),
            updated_at_utc=str(data.get("updated_at_utc") or ""),
        )


class SymbolSideRegimePolicy:
    def __init__(self, settings):
        cfg = settings.trading
        self.enabled = bool(getattr(cfg, "symbol_side_regime_policy_enabled", True))

        self.lookback_minutes = int(max(60, int(getattr(cfg, "symbol_policy_lookback_minutes", 1440))))
        self.short_lookback_minutes = int(max(30, int(getattr(cfg, "symbol_policy_short_lookback_minutes", 180))))
        self.half_life_minutes = int(max(60, int(getattr(cfg, "symbol_policy_decay_half_life_minutes", 720))))
        self.min_trades_yellow = int(max(1, int(getattr(cfg, "symbol_policy_min_trades_yellow", 6))))
        self.min_trades_red = int(max(1, int(getattr(cfg, "symbol_policy_min_trades_red", 10))))
        self.bayes_kappa = max(0.0, _f(getattr(cfg, "symbol_policy_bayes_kappa", 6.0), 6.0))
        self.prior_alpha = max(0.1, _f(getattr(cfg, "symbol_policy_prior_alpha", 5.0), 5.0))
        self.prior_beta = max(0.1, _f(getattr(cfg, "symbol_policy_prior_beta", 5.0), 5.0))
        self.yellow_expectancy = _f(getattr(cfg, "symbol_policy_yellow_expectancy_pct", 0.0), 0.0)
        self.red_expectancy = _f(getattr(cfg, "symbol_policy_red_expectancy_pct", -0.2), -0.2)

        self.green_risk_mult = _f(getattr(cfg, "symbol_policy_green_risk_mult", 1.0), 1.0)
        self.yellow_risk_mult = _f(getattr(cfg, "symbol_policy_yellow_risk_mult", 0.75), 0.75)
        self.red_risk_mult = _f(getattr(cfg, "symbol_policy_red_risk_mult", 0.35), 0.35)
        self.green_weight = _f(getattr(cfg, "symbol_policy_green_weight", 1.05), 1.05)
        self.yellow_weight = _f(getattr(cfg, "symbol_policy_yellow_weight", 0.85), 0.85)
        self.red_weight = _f(getattr(cfg, "symbol_policy_red_weight", 0.55), 0.55)
        self.yellow_quality_penalty = _f(getattr(cfg, "symbol_policy_yellow_quality_penalty", 5.0), 5.0)
        self.red_quality_penalty = _f(getattr(cfg, "symbol_policy_red_quality_penalty", 12.0), 12.0)
        self.yellow_edge_floor_delta = _f(
            getattr(cfg, "symbol_policy_yellow_edge_floor_delta_pct", 0.03), 0.03
        )
        self.red_edge_floor_delta = _f(
            getattr(cfg, "symbol_policy_red_edge_floor_delta_pct", 0.08), 0.08
        )
        self.yellow_slippage_mult = _f(getattr(cfg, "symbol_policy_yellow_slippage_mult", 0.9), 0.9)
        self.red_slippage_mult = _f(getattr(cfg, "symbol_policy_red_slippage_mult", 0.75), 0.75)
        self.red_probe_interval_minutes = int(
            max(1, int(getattr(cfg, "symbol_policy_red_probe_interval_minutes", 30)))
        )
        self.red_probe_risk_mult = _f(getattr(cfg, "symbol_policy_red_probe_risk_mult", 0.5), 0.5)
        self.drift_flip_threshold = _f(getattr(cfg, "symbol_policy_drift_flip_threshold_pct", 0.35), 0.35)

        self.buckets: Dict[str, BucketState] = {}
        self.last_refresh_utc: str = ""
        self.last_red_probe_at: Dict[str, str] = {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "last_refresh_utc": self.last_refresh_utc,
            "buckets": {k: v.to_dict() for k, v in self.buckets.items()},
            "last_red_probe_at": dict(self.last_red_probe_at),
        }

    def load_state(self, data: Dict[str, Any]) -> None:
        if not isinstance(data, dict):
            return
        self.last_refresh_utc = str(data.get("last_refresh_utc") or "")
        buckets_raw = data.get("buckets") or {}
        if isinstance(buckets_raw, dict):
            loaded: Dict[str, BucketState] = {}
            for key, payload in buckets_raw.items():
                if not isinstance(payload, dict):
                    continue
                loaded[str(key)] = BucketState.from_dict(payload)
            self.buckets = loaded
        red_probe = data.get("last_red_probe_at") or {}
        if isinstance(red_probe, dict):
            self.last_red_probe_at = {str(k): str(v) for k, v in red_probe.items()}

    def refresh(self, data_lake, now_utc: Optional[datetime] = None) -> None:
        if not self.enabled:
            return
        now_utc = now_utc or _utc_now()
        long_perf = data_lake.get_bucket_performance(
            minutes=self.lookback_minutes,
            half_life_minutes=self.half_life_minutes,
        )
        short_perf = data_lake.get_bucket_performance(
            minutes=self.short_lookback_minutes,
            half_life_minutes=max(60, int(self.short_lookback_minutes)),
        )
        new_buckets: Dict[str, BucketState] = {}
        for key, row in long_perf.items():
            weighted_n = max(0.0, _f(row.get("weighted_closed_trades"), 0.0))
            closed_n = max(0.0, _f(row.get("closed_trades"), 0.0))
            raw_exp = _f(row.get("weighted_expectancy_pct"), 0.0)
            raw_wr = _clamp(_f(row.get("weighted_win_rate"), 0.5), 0.0, 1.0)
            shrink_factor = weighted_n / (weighted_n + self.bayes_kappa) if (weighted_n + self.bayes_kappa) > 0 else 0.0
            shrunk_exp = raw_exp * shrink_factor
            shrunk_wr = (raw_wr * weighted_n + self.prior_alpha) / (weighted_n + self.prior_alpha + self.prior_beta)

            short_row = short_perf.get(key) or {}
            short_exp = _f(short_row.get("weighted_expectancy_pct"), raw_exp)
            unstable = False
            if weighted_n >= self.min_trades_yellow:
                if (short_exp * shrunk_exp) < 0 and abs(short_exp - shrunk_exp) >= self.drift_flip_threshold:
                    unstable = True

            tail3 = _f(row.get("tail_loss_3_rate"), 0.0)
            tail5 = _f(row.get("tail_loss_5_rate"), 0.0)
            tail7 = _f(row.get("tail_loss_7_rate"), 0.0)
            avg_slip = _f(row.get("weighted_avg_slippage_bps"), 0.0)

            if weighted_n < self.min_trades_yellow:
                state = "green"
            else:
                red_cond = (
                    weighted_n >= self.min_trades_red
                    and (
                        shrunk_exp <= self.red_expectancy
                        or shrunk_wr < 0.44
                        or tail5 >= 0.20
                        or tail7 >= 0.08
                    )
                )
                yellow_cond = (
                    shrunk_exp <= self.yellow_expectancy
                    or shrunk_wr < 0.50
                    or tail3 >= 0.35
                    or avg_slip >= 80.0
                )
                if red_cond:
                    state = "red"
                elif yellow_cond:
                    state = "yellow"
                else:
                    state = "green"
            if unstable and state == "green":
                state = "yellow"

            new_buckets[key] = BucketState(
                symbol=str(row.get("symbol") or ""),
                side=str(row.get("side") or "unknown"),
                regime=str(row.get("regime") or "unknown"),
                state=state,
                weighted_closed_trades=weighted_n,
                closed_trades=closed_n,
                raw_expectancy_pct=raw_exp,
                shrunk_expectancy_pct=shrunk_exp,
                raw_win_rate=raw_wr,
                shrunk_win_rate=shrunk_wr,
                avg_slippage_bps=avg_slip,
                tail_loss_3_rate=tail3,
                tail_loss_5_rate=tail5,
                tail_loss_7_rate=tail7,
                unstable=unstable,
                short_expectancy_pct=short_exp,
                updated_at_utc=now_utc.isoformat(),
            )

        self.buckets = new_buckets
        self.last_refresh_utc = now_utc.isoformat()

    def mark_red_probe(self, symbol: str, side: str, regime: str, now_utc: Optional[datetime] = None) -> None:
        key = _bucket_key(symbol, side, regime)
        now_utc = now_utc or _utc_now()
        self.last_red_probe_at[key] = now_utc.isoformat()

    def _can_red_probe(self, key: str, now_utc: Optional[datetime] = None) -> bool:
        now_utc = now_utc or _utc_now()
        raw = self.last_red_probe_at.get(key)
        if not raw:
            return True
        try:
            last_dt = datetime.fromisoformat(str(raw))
            if last_dt.tzinfo is None:
                last_dt = last_dt.replace(tzinfo=timezone.utc)
        except Exception:
            return True
        minutes = (now_utc - last_dt).total_seconds() / 60.0
        return minutes >= float(self.red_probe_interval_minutes)

    def evaluate(
        self,
        symbol: str,
        side: str,
        regime: str,
        now_utc: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        now_utc = now_utc or _utc_now()
        key = _bucket_key(symbol, side, regime)
        bucket = self.buckets.get(key)
        if bucket is None:
            for fallback_key in (
                _bucket_key(symbol, "unknown", regime),
                _bucket_key(symbol, side, "unknown"),
                _bucket_key(symbol, "unknown", "unknown"),
            ):
                bucket = self.buckets.get(fallback_key)
                if bucket is not None:
                    key = fallback_key
                    break
        if not bucket:
            return {
                "key": key,
                "state": "green",
                "risk_multiplier": self.green_risk_mult,
                "symbol_weight_multiplier": self.green_weight,
                "quality_penalty": 0.0,
                "edge_floor_delta_pct": 0.0,
                "slippage_limit_multiplier": 1.0,
                "force_probe_only": False,
                "allow_counter_trend_bypass": True,
                "unstable": False,
                "shrunk_expectancy_pct": 0.0,
                "shrunk_win_rate": 0.5,
                "weighted_closed_trades": 0.0,
                "tail_loss_3_rate": 0.0,
                "tail_loss_5_rate": 0.0,
                "tail_loss_7_rate": 0.0,
                "red_probe_allowed_now": True,
            }

        if bucket.state == "red":
            risk_mult = self.red_risk_mult
            weight_mult = self.red_weight
            q_penalty = self.red_quality_penalty
            edge_delta = self.red_edge_floor_delta
            slip_mult = self.red_slippage_mult
            force_probe_only = True
        elif bucket.state == "yellow":
            risk_mult = self.yellow_risk_mult
            weight_mult = self.yellow_weight
            q_penalty = self.yellow_quality_penalty
            edge_delta = self.yellow_edge_floor_delta
            slip_mult = self.yellow_slippage_mult
            force_probe_only = False
        else:
            risk_mult = self.green_risk_mult
            weight_mult = self.green_weight
            q_penalty = 0.0
            edge_delta = 0.0
            slip_mult = 1.0
            force_probe_only = False

        # Counter-trend bypass is disallowed on weak/negative buckets.
        allow_counter_bypass = bool(
            bucket.state == "green"
            and bucket.shrunk_expectancy_pct >= 0.0
            and bucket.shrunk_win_rate >= 0.50
            and not bucket.unstable
        )
        red_probe_allowed = self._can_red_probe(key, now_utc=now_utc)

        return {
            "key": key,
            "state": bucket.state,
            "risk_multiplier": _clamp(risk_mult, 0.01, 2.0),
            "symbol_weight_multiplier": _clamp(weight_mult, 0.01, 3.0),
            "quality_penalty": _clamp(q_penalty, 0.0, 100.0),
            "edge_floor_delta_pct": _clamp(edge_delta, 0.0, 20.0),
            "slippage_limit_multiplier": _clamp(slip_mult, 0.05, 3.0),
            "force_probe_only": bool(force_probe_only),
            "allow_counter_trend_bypass": allow_counter_bypass,
            "unstable": bool(bucket.unstable),
            "shrunk_expectancy_pct": float(bucket.shrunk_expectancy_pct),
            "shrunk_win_rate": float(bucket.shrunk_win_rate),
            "weighted_closed_trades": float(bucket.weighted_closed_trades),
            "tail_loss_3_rate": float(bucket.tail_loss_3_rate),
            "tail_loss_5_rate": float(bucket.tail_loss_5_rate),
            "tail_loss_7_rate": float(bucket.tail_loss_7_rate),
            "avg_slippage_bps": float(bucket.avg_slippage_bps),
            "red_probe_allowed_now": bool(red_probe_allowed),
        }

"""
Soft adaptive risk governor for continuous trading.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _f(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


@dataclass
class GovernorState:
    risk_multiplier: float = 1.0
    mode: str = "neutral"  # recovering | neutral | degrading
    last_update_utc: Optional[str] = None
    degrade_reason: str = ""
    health_snapshot: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "risk_multiplier": float(self.risk_multiplier),
            "mode": str(self.mode),
            "last_update_utc": self.last_update_utc,
            "degrade_reason": str(self.degrade_reason or ""),
            "health_snapshot": dict(self.health_snapshot or {}),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GovernorState":
        if not isinstance(data, dict):
            return cls()
        return cls(
            risk_multiplier=_f(data.get("risk_multiplier"), 1.0),
            mode=str(data.get("mode") or "neutral"),
            last_update_utc=data.get("last_update_utc"),
            degrade_reason=str(data.get("degrade_reason") or ""),
            health_snapshot={
                str(k): _f(v)
                for k, v in (data.get("health_snapshot") or {}).items()
            },
        )


class SoftRiskGovernor:
    def __init__(self, settings):
        cfg = settings.trading
        self.enabled = bool(getattr(cfg, "soft_governor_enabled", True))
        self.min_multiplier = _f(getattr(cfg, "soft_governor_min_multiplier", 0.2), 0.2)
        self.max_multiplier = _f(getattr(cfg, "soft_governor_max_multiplier", 1.2), 1.2)
        self.decay = _f(getattr(cfg, "soft_governor_decay", 0.85), 0.85)
        self.recovery = _f(getattr(cfg, "soft_governor_recovery", 1.05), 1.05)
        self.update_minutes = int(max(1, int(getattr(cfg, "soft_governor_update_minutes", 10))))
        self.min_objective_closed_trades = int(
            max(0, int(getattr(cfg, "soft_governor_min_objective_closed_trades", 6)))
        )
        self.min_execution_events = int(
            max(0, int(getattr(cfg, "soft_governor_min_execution_events", 6)))
        )
        self.degrade_drawdown_cap_pct = _f(getattr(cfg, "soft_drawdown_degrade_cap_pct", 6.0), 6.0)
        self.catastrophic_drawdown_pct = _f(
            getattr(cfg, "soft_catastrophic_drawdown_pct", 12.0),
            12.0,
        )
        self.state = GovernorState(
            risk_multiplier=_clamp(1.0, self.min_multiplier, self.max_multiplier),
            mode="neutral",
        )

    def load_state(self, state_data: Dict[str, Any]) -> None:
        state = GovernorState.from_dict(state_data or {})
        state.risk_multiplier = _clamp(state.risk_multiplier, self.min_multiplier, self.max_multiplier)
        self.state = state

    def to_dict(self) -> Dict[str, Any]:
        return self.state.to_dict()

    def should_block_catastrophic(self, drawdown_pct: float) -> bool:
        if not self.enabled:
            return False
        return _f(drawdown_pct, 0.0) >= self.catastrophic_drawdown_pct

    def update(
        self,
        now_utc: Optional[datetime],
        objective_metrics: Dict[str, Any],
        execution_metrics: Dict[str, Any],
        drawdown_pct: float,
        no_accept_cycles: int,
    ) -> GovernorState:
        now_utc = now_utc or _utc_now()
        current_multiplier = _clamp(self.state.risk_multiplier, self.min_multiplier, self.max_multiplier)

        if not self.enabled:
            self.state.risk_multiplier = 1.0
            self.state.mode = "neutral"
            self.state.last_update_utc = now_utc.isoformat()
            self.state.degrade_reason = "disabled"
            self.state.health_snapshot = {}
            return self.state

        if self.state.last_update_utc:
            try:
                prev = datetime.fromisoformat(str(self.state.last_update_utc))
                if prev.tzinfo is None:
                    prev = prev.replace(tzinfo=timezone.utc)
                if now_utc < (prev + timedelta(minutes=self.update_minutes)):
                    return self.state
            except Exception:
                pass

        eph = _f(objective_metrics.get("expectancy_per_hour_pct"), 0.0)
        tph = _f(objective_metrics.get("trades_per_hour"), 0.0)
        objective_closed = _f(objective_metrics.get("closed_trades"), 0.0)
        execution_events = _f(execution_metrics.get("events"), 0.0)
        reject_rate = _f(execution_metrics.get("reject_rate"), 0.0)
        avg_slippage_bps = _f(execution_metrics.get("avg_slippage_bps"), 0.0)
        dd = max(0.0, _f(drawdown_pct, 0.0))
        objective_sample_ready = objective_closed >= float(self.min_objective_closed_trades)
        execution_sample_ready = execution_events >= float(self.min_execution_events)

        bad_reasons = []
        bad_score = 0
        good_score = 0

        if dd >= self.degrade_drawdown_cap_pct:
            bad_score += 3
            bad_reasons.append("drawdown_cap")
        elif dd >= (self.degrade_drawdown_cap_pct * 0.75):
            bad_score += 2
            bad_reasons.append("drawdown_rising")
        elif dd <= (self.degrade_drawdown_cap_pct * 0.30):
            good_score += 1

        if objective_sample_ready:
            if eph < -0.12:
                bad_score += 2
                bad_reasons.append("negative_eph")
            elif eph > 0.04:
                good_score += 1

        if execution_sample_ready:
            if reject_rate > 0.40:
                bad_score += 2
                bad_reasons.append("high_reject_rate")
            elif reject_rate < 0.12:
                good_score += 1

            if avg_slippage_bps > 80:
                bad_score += 2
                bad_reasons.append("high_slippage")
            elif avg_slippage_bps < 25:
                good_score += 1

        if no_accept_cycles >= 14:
            bad_score += 1
            bad_reasons.append("proposal_starvation")
        elif no_accept_cycles <= 2 and tph >= 0.8:
            good_score += 1

        if dd >= self.catastrophic_drawdown_pct:
            current_multiplier = self.min_multiplier
            self.state.mode = "degrading"
            self.state.degrade_reason = "catastrophic_drawdown"
        elif bad_score > good_score:
            decay_factor = self.decay ** max(1, bad_score - good_score)
            current_multiplier *= decay_factor
            self.state.mode = "degrading"
            self.state.degrade_reason = ",".join(bad_reasons) if bad_reasons else "degrade"
        elif good_score > bad_score:
            recover_factor = self.recovery ** max(1, good_score - bad_score)
            current_multiplier *= recover_factor
            self.state.mode = "recovering"
            self.state.degrade_reason = ""
        else:
            self.state.mode = "neutral"
            self.state.degrade_reason = ""

        self.state.risk_multiplier = _clamp(current_multiplier, self.min_multiplier, self.max_multiplier)
        self.state.last_update_utc = now_utc.isoformat()
        self.state.health_snapshot = {
            "expectancy_per_hour_pct": eph,
            "trades_per_hour": tph,
            "closed_trades": objective_closed,
            "execution_events": execution_events,
            "reject_rate": reject_rate,
            "avg_slippage_bps": avg_slippage_bps,
            "drawdown_pct": dd,
            "no_accept_cycles": float(max(0, no_accept_cycles)),
            "objective_sample_ready": 1.0 if objective_sample_ready else 0.0,
            "execution_sample_ready": 1.0 if execution_sample_ready else 0.0,
            "bad_score": float(bad_score),
            "good_score": float(good_score),
        }
        return self.state

"""
Typed structures for quant decision pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class HorizonPrediction:
    horizon_min: int
    prob_up_raw: float
    prob_up_calibrated: float
    expected_move_pct: float
    volatility_pct: float
    uncertainty: float
    expert: str


@dataclass
class EVProposal:
    symbol: str
    side: str  # Buy or Sell
    entry_price: float
    stop_loss: float
    take_profit: float
    risk_pct: float
    leverage: int
    expected_edge_pct: float
    win_probability: float
    rr_ratio: float
    confidence: float
    regime: str
    expected_hold_minutes: int = 30
    expectancy_per_hour_pct: float = 0.0
    quality_score: float = 0.0
    quality_score_raw: float = 0.0
    quality_score_adjusted: float = 0.0
    entry_tier: str = "full"  # full | probe
    symbol_weight: float = 1.0
    policy_state: str = "green"
    policy_key: str = ""
    estimated_slippage_bps: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QuantCycleMetrics:
    cycle_latency_ms: float = 0.0
    candidates_scored: int = 0
    proposals_generated: int = 0
    uncertainty_passed: int = 0
    portfolio_passed: int = 0
    affordability_passed: int = 0
    selected: int = 0
    proposals_accepted: int = 0
    drift_score: float = 0.0
    recent_execution_events: float = 0.0
    recent_reject_rate: float = 0.0
    recent_fill_slippage_bps: float = 0.0
    objective_trades_per_hour: float = 0.0
    objective_expectancy_per_trade_pct: float = 0.0
    objective_expectancy_per_hour_pct: float = 0.0
    objective_closed_trades: int = 0
    objective_edge_floor_pct: float = 0.0
    objective_score: float = 0.0
    risk_multiplier: float = 1.0
    governor_mode: str = "neutral"
    governor_reason: str = ""
    governor_snapshot: Optional[Dict[str, Any]] = None
    symbol_funnel: Optional[Dict[str, Dict[str, Any]]] = None
    reject_reason_counts: Optional[Dict[str, int]] = None
    monitor_alerts: Optional[list[dict]] = None
    retrained: bool = False
    retrain_summary: Optional[Dict[str, Any]] = None

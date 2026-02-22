"""
End-to-end quant engine:
- data lake ingestion
- feature extraction
- multi-horizon expert forecasts
- calibration and uncertainty gating
- EV proposal construction
- portfolio-aware selection
- walk-forward retraining and drift handling
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from .calibration import ProbabilityCalibrator
from .data_lake import QuantDataLake
from .drift_monitor import DriftMonitor
from .ev_constructor import EVTradeConstructor
from .feature_store import FeatureStore
from .labeling import BarrierLabeler
from .portfolio_optimizer import PortfolioOptimizer
from .regime_switcher import RegimeSwitcher
from .risk_governor import SoftRiskGovernor
from .symbol_side_regime_policy import SymbolSideRegimePolicy
from .retraining import WalkForwardTrainer
from .types import EVProposal, HorizonPrediction, QuantCycleMetrics
from .uncertainty import UncertaintyGate


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


def _f(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


class QuantEngine:
    def __init__(self, settings):
        self.settings = settings
        self.horizons = [5, 15, 30]
        self.data_lake = QuantDataLake(self.settings.trading.quant_data_lake_path)
        self.feature_store = FeatureStore()
        self.regime_switcher = RegimeSwitcher(
            feature_names=self.feature_store.FEATURE_NAMES,
            horizons=self.horizons,
        )
        self.calibrators: Dict[int, ProbabilityCalibrator] = {
            5: ProbabilityCalibrator(method="platt"),
            15: ProbabilityCalibrator(method="platt"),
            30: ProbabilityCalibrator(method="platt"),
        }
        self.uncertainty_gate = UncertaintyGate(
            min_confidence=self.settings.trading.quant_min_calibrated_confidence,
            max_uncertainty=self.settings.trading.quant_uncertainty_max,
        )
        self.ev_constructor = EVTradeConstructor(
            min_edge_pct=self.settings.trading.quant_min_edge_per_trade_pct,
            min_expected_move_pct=self.settings.trading.quant_min_expected_move_pct,
            min_tp_pct=self.settings.trading.quant_min_tp_pct,
            max_tp_pct=self.settings.trading.quant_max_tp_pct,
            min_sl_pct=self.settings.trading.quant_min_sl_pct,
            max_sl_pct=self.settings.trading.quant_max_sl_pct,
            round_trip_cost_pct=self.settings.trading.estimated_round_trip_cost_pct,
            realized_cost_slippage_weight=self.settings.trading.realized_cost_slippage_weight,
            realized_cost_slippage_clip_bps=self.settings.trading.realized_cost_slippage_clip_bps,
        )
        self.portfolio_optimizer = PortfolioOptimizer(
            correlation_cap=self.settings.trading.quant_correlation_cap,
            max_risk_budget_pct=self.settings.trading.quant_max_portfolio_risk_budget_pct,
        )
        self.drift_monitor = DriftMonitor(feature_names=self.feature_store.FEATURE_NAMES, window=600)
        self.labeler = BarrierLabeler(fee_buffer_pct=self.settings.trading.estimated_round_trip_cost_pct)
        self.trainer = WalkForwardTrainer(
            data_lake=self.data_lake,
            regime_switcher=self.regime_switcher,
            calibrators=self.calibrators,
            retrain_interval_minutes=self.settings.trading.quant_retrain_interval_minutes,
            lookback_rows=self.settings.trading.quant_training_lookback_rows,
        )
        self.symbol_policy = SymbolSideRegimePolicy(self.settings)
        self.risk_governor = SoftRiskGovernor(self.settings)
        self._no_proposal_cycles = 0
        self._last_alert_at: Dict[str, datetime] = {}
        self._session_started_at = _utc_now()
        self._session_scope_metrics = bool(getattr(self.settings.trading, "quant_session_scope_metrics", True))
        raw_major_symbols = str(
            getattr(self.settings.trading, "major_symbols_csv", "BTCUSDT,ETHUSDT") or ""
        )
        self._major_symbols = {
            token.strip().upper()
            for token in raw_major_symbols.split(",")
            if token.strip()
        } or {"BTCUSDT", "ETHUSDT"}
        self._load_state()
        logger.info("Quant engine initialized (primary quant decision path)")

    def _load_state(self) -> None:
        state = self.data_lake.load_model_state("quant_engine_state")
        if not state:
            return
        try:
            if isinstance(state.get("regime_switcher"), dict):
                self.regime_switcher = RegimeSwitcher.from_dict(state["regime_switcher"])
            if isinstance(state.get("calibrators"), dict):
                for h_str, payload in state["calibrators"].items():
                    horizon = int(h_str)
                    if horizon in self.calibrators and isinstance(payload, dict):
                        self.calibrators[horizon] = ProbabilityCalibrator.from_dict(payload)
            reset_governor_on_start = bool(
                getattr(self.settings.trading, "focus_recovery_reset_governor_on_start", True)
            ) and bool(getattr(self.settings.trading, "focus_recovery_mode_enabled", True))
            if (not reset_governor_on_start) and isinstance(state.get("soft_governor_state"), dict):
                self.risk_governor.load_state(state.get("soft_governor_state") or {})
            if isinstance(state.get("symbol_side_regime_policy_state"), dict):
                self.symbol_policy.load_state(state.get("symbol_side_regime_policy_state") or {})
            last_retrain = state.get("last_retrain_at")
            if last_retrain:
                self.trainer.last_retrain_at = datetime.fromisoformat(str(last_retrain))
        except Exception as exc:
            logger.warning(f"Failed to load quant engine state: {exc}")

    def _save_state(self) -> None:
        payload = {
            "regime_switcher": self.regime_switcher.to_dict(),
            "calibrators": {str(h): c.to_dict() for h, c in self.calibrators.items()},
            "soft_governor_state": self.risk_governor.to_dict(),
            "symbol_side_regime_policy_state": self.symbol_policy.to_dict(),
            "last_retrain_at": _iso(self.trainer.last_retrain_at) if self.trainer.last_retrain_at else None,
        }
        self.data_lake.save_model_state("quant_engine_state", payload)

    def record_execution_event(
        self,
        event_type: str,
        symbol: Optional[str],
        reason_code: Optional[str] = None,
        pnl_pct: Optional[float] = None,
        slippage_bps: Optional[float] = None,
        latency_ms: Optional[float] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.data_lake.save_execution_event(
            {
                "ts": _iso(_utc_now()),
                "symbol": symbol,
                "event_type": event_type,
                "reason_code": reason_code,
                "pnl_pct": pnl_pct,
                "slippage_bps": slippage_bps,
                "latency_ms": latency_ms,
                "details": details or {},
            }
        )

    def _can_emit_alert(self, code: str, now_utc: datetime) -> bool:
        min_gap = int(max(1, self.settings.trading.quant_monitor_min_alert_interval_minutes))
        last = self._last_alert_at.get(code)
        if last is None:
            return True
        return now_utc >= (last + timedelta(minutes=min_gap))

    def _emit_alert(
        self,
        now_utc: datetime,
        severity: str,
        code: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        payload = {
            "timestamp_utc": _iso(now_utc),
            "severity": severity,
            "code": code,
            "message": message,
            "details": details or {},
        }
        self.data_lake.save_alert(
            severity=severity,
            code=code,
            message=message,
            details=payload["details"],
        )
        self._last_alert_at[code] = now_utc
        return payload

    def _derive_dynamic_edge_floor(self, objective_metrics: Dict[str, float]) -> float:
        base_floor = float(self.settings.trading.quant_min_edge_per_trade_pct)
        target_tph = float(self.settings.trading.quant_target_trades_per_hour)
        min_closed = int(self.settings.trading.quant_min_closed_trades_for_objective)
        floor_hour = float(self.settings.trading.quant_expectancy_hour_floor_pct)

        tph = float(objective_metrics.get("trades_per_hour", 0.0))
        eph = float(objective_metrics.get("expectancy_per_hour_pct", 0.0))
        closed = int(objective_metrics.get("closed_trades", 0.0))

        floor = base_floor
        no_prop_alert_cycles = int(self.settings.trading.quant_monitor_no_proposal_cycles_alert)

        # Warm-up period: prioritize participation over strict edge thresholds.
        warmup_gate = max(4, int(min_closed * 0.30))
        if closed < warmup_gate:
            floor -= 0.03
        elif closed < min_closed:
            floor -= 0.015
        if closed <= 1:
            floor -= 0.04

        # If execution frequency is lagging, lower entry edge floor to avoid starvation.
        if tph < (target_tph * 0.85):
            floor -= 0.04
        if tph < (target_tph * 0.60):
            floor -= 0.03

        # Prolonged no-proposal streak should aggressively relax edge gating.
        if self._no_proposal_cycles >= max(2, no_prop_alert_cycles // 2):
            floor -= 0.03
        if self._no_proposal_cycles >= no_prop_alert_cycles:
            floor -= 0.03

        # Tighten only when sample quality is sufficient and market participation is healthy.
        if (
            closed >= min_closed
            and tph >= (target_tph * 0.90)
            and eph < floor_hour
            and self._no_proposal_cycles < max(2, no_prop_alert_cycles // 2)
        ):
            deficit = max(0.0, floor_hour - eph)
            floor += min(0.03, deficit * 0.20)

        return max(-0.12, min(1.50, floor))

    def _symbol_weight(self, symbol: str) -> float:
        m30 = self.data_lake.get_symbol_expectancy_metrics(symbol=symbol, minutes=30)
        m2h = self.data_lake.get_symbol_expectancy_metrics(symbol=symbol, minutes=120)
        m1d = self.data_lake.get_symbol_expectancy_metrics(symbol=symbol, minutes=1440)

        e30 = _f(m30.get("expectancy_per_trade_pct"), 0.0)
        e2h = _f(m2h.get("expectancy_per_trade_pct"), 0.0)
        e1d = _f(m1d.get("expectancy_per_trade_pct"), 0.0)
        n2h = _f(m2h.get("closed_trades"), 0.0)
        n1d = _f(m1d.get("closed_trades"), 0.0)

        if (n2h + n1d) < 4:
            return 1.0

        weighted = (0.45 * e30) + (0.35 * e2h) + (0.20 * e1d)
        if weighted >= 0.50:
            return 1.2
        if weighted >= 0.20:
            return 1.1
        if weighted <= -0.50:
            return 0.5
        if weighted <= -0.20:
            return 0.7
        if weighted < 0:
            return 0.85
        return 1.0

    def _apply_symbol_diversity_ranking(
        self,
        ranked: List[EVProposal],
        now_utc: datetime,
        since_ts: Optional[Any] = None,
    ) -> List[EVProposal]:
        if not ranked:
            return ranked
        if not bool(getattr(self.settings.trading, "symbol_diversity_enabled", True)):
            return ranked

        lookback_minutes = int(max(30, int(self.settings.trading.symbol_diversity_lookback_minutes)))
        close_counts = self.data_lake.get_recent_close_symbol_counts(
            minutes=lookback_minutes,
            since_ts=since_ts,
        )
        total_closes = float(sum(close_counts.values()))
        max_share = max(0.01, min(1.0, float(self.settings.trading.symbol_diversity_max_share_pct) / 100.0))
        repeat_penalty_scale = max(0.0, float(self.settings.trading.symbol_diversity_repeat_penalty_scale))
        underused_bonus = max(0.0, float(self.settings.trading.symbol_diversity_underused_bonus))
        underused_min_closed = int(max(0, int(self.settings.trading.symbol_diversity_underused_min_closed_trades)))

        major_boost_enabled = bool(getattr(self.settings.trading, "major_symbol_boost_enabled", True))
        major_target = max(0.0, min(1.0, float(self.settings.trading.major_symbol_target_share_pct) / 100.0))
        major_bonus = max(0.0, float(self.settings.trading.major_symbol_bonus))
        major_min_quality = int(max(1, int(self.settings.trading.major_symbol_min_quality)))
        major_closed = float(sum(close_counts.get(symbol, 0) for symbol in self._major_symbols))
        major_share = (major_closed / total_closes) if total_closes > 0 else 0.0
        major_share_deficit = max(0.0, major_target - major_share)

        rescored: List[Tuple[float, EVProposal]] = []
        for proposal in ranked:
            symbol = str(proposal.symbol or "").upper()
            base_score = _f(
                (proposal.metadata or {}).get("portfolio_objective_score"),
                proposal.expectancy_per_hour_pct,
            )

            symbol_closed = float(close_counts.get(symbol, 0))
            symbol_share = (symbol_closed / total_closes) if total_closes > 0 else 0.0
            over_share = max(0.0, symbol_share - max_share)
            repeat_penalty = over_share * repeat_penalty_scale * 2.0

            coverage_bonus = 0.0
            if symbol_closed <= float(underused_min_closed):
                scarcity = 1.0
                if total_closes > 0 and max_share > 0:
                    scarcity = max(0.0, min(1.0, 1.0 - (symbol_share / max_share)))
                coverage_bonus = underused_bonus * max(0.25, scarcity)

            major_symbol_bonus = 0.0
            quality_adj = _f(getattr(proposal, "quality_score_adjusted", proposal.quality_score), proposal.quality_score)
            if (
                major_boost_enabled
                and symbol in self._major_symbols
                and quality_adj >= float(major_min_quality)
            ):
                base_major_boost = major_bonus * max(0.0, min(1.0, major_share_deficit / max(0.01, major_target or 1.0)))
                if total_closes > 0 and symbol_closed == 0:
                    base_major_boost += (major_bonus * 0.35)
                major_symbol_bonus = base_major_boost

            final_score = base_score + coverage_bonus + major_symbol_bonus - repeat_penalty
            proposal.metadata["symbol_diversity"] = {
                "enabled": True,
                "lookback_minutes": lookback_minutes,
                "symbol_closed_trades": symbol_closed,
                "total_closed_trades": total_closes,
                "symbol_share": symbol_share,
                "max_share": max_share,
                "repeat_penalty": repeat_penalty,
                "coverage_bonus": coverage_bonus,
                "major_symbol_bonus": major_symbol_bonus,
                "major_share": major_share,
                "major_target_share": major_target,
            }
            proposal.metadata["portfolio_objective_score_diversified"] = final_score
            rescored.append((final_score, proposal))

        rescored.sort(key=lambda item: item[0], reverse=True)
        return [proposal for _, proposal in rescored]

    def _is_chop_regime(self, features: Dict[str, float], regime: str) -> bool:
        if not bool(getattr(self.settings.trading, "anti_chop_enabled", True)):
            return False
        if str(regime).lower() != "range":
            return False
        adx_15m = _f(features.get("adx_15m_norm"), 0.0) * 50.0
        bb_width = _f(features.get("bb_width_15m"), 0.0)
        atr_pct_15m = _f(features.get("atr_pct_15m"), 0.0)
        trend_1h = abs(_f(features.get("trend_1h"), 0.0))
        return bool(
            adx_15m <= float(self.settings.trading.anti_chop_max_adx_15m)
            and bb_width <= float(self.settings.trading.anti_chop_max_bb_width_15m)
            and atr_pct_15m <= float(self.settings.trading.anti_chop_max_atr_pct_15m)
            and trend_1h <= 0.5
        )

    def _expectancy_floor_guard_state(self, minutes: int = 240) -> Dict[str, Any]:
        if not bool(getattr(self.settings.trading, "quant_expectancy_floor_guard_enabled", True)):
            return {"active": False}
        session_since = self._session_started_at if self._session_scope_metrics else None
        dist = self.data_lake.get_recent_close_distribution(
            minutes=max(30, int(minutes)),
            since_ts=session_since,
        )
        closed = int(_f(dist.get("closed_trades"), 0.0))
        floor = float(self.settings.trading.quant_expectancy_floor_guard_floor_pct)
        min_closed = int(self.settings.trading.quant_expectancy_floor_guard_min_closed_trades)
        tail_limit = float(self.settings.trading.quant_expectancy_floor_guard_tail_loss_5_rate)
        expectancy = _f(dist.get("expectancy_per_trade_pct"), 0.0)
        tail5 = _f(dist.get("tail_loss_5_rate"), 0.0)
        active = bool(closed >= min_closed and (expectancy < floor or tail5 > tail_limit))
        return {
            "active": active,
            "closed_trades": closed,
            "expectancy_per_trade_pct": expectancy,
            "tail_loss_5_rate": tail5,
            "floor_pct": floor,
            "tail_loss_5_limit": tail_limit,
            "probe_only": bool(getattr(self.settings.trading, "quant_expectancy_floor_guard_probe_only", True)),
        }

    def _label_matured_rows(self, now_utc: datetime) -> int:
        labeled = 0
        for horizon in self.horizons:
            rows = self.data_lake.get_mature_feature_rows(horizon_min=horizon, now_utc=now_utc, limit=500)
            for row in rows:
                start_ts = str(row["ts"])
                end_ts = _iso(datetime.fromisoformat(start_ts) + timedelta(minutes=horizon))
                path = self.data_lake.get_price_path(str(row["symbol"]), start_ts, end_ts)
                if len(path) < 2:
                    continue
                try:
                    features = json.loads(str(row["features_json"]))
                except Exception:
                    features = {}
                label = self.labeler.label_path(
                    entry_price=float(row["price"]),
                    price_path=path,
                    volatility_pct=abs(_f(features.get("atr_pct_15m"), 0.2)),
                )
                self.data_lake.insert_label(int(row["id"]), horizon, label)
                self.data_lake.mark_labeled(int(row["id"]), horizon)

                regime = str(row["regime"] or "range")
                feature_vector = {k: _f(v) for k, v in dict(features).items()}
                self.regime_switcher.update(regime=regime, horizon=horizon, features=feature_vector, label=label)

                pred_row = self.data_lake.get_prediction_for_feature(int(row["id"]), horizon)
                if pred_row and int(label.get("direction_label", 0)) != 0:
                    y = 1 if int(label.get("direction_label", 0)) > 0 else 0
                    self.calibrators[horizon].add_sample(float(pred_row["prob_up_raw"]), y)
                labeled += 1

        for calibrator in self.calibrators.values():
            calibrator.fit()
        if labeled > 0:
            self._save_state()
        return labeled

    def ingest_and_score_cycle(
        self,
        market_data: Dict[str, Any],
        analysis: Dict[str, Any],
        open_positions: List[object],
        symbol_reject_streak: Dict[str, int],
        current_total_risk_pct: float,
        account_balance: float = 0.0,
        account_drawdown_pct: float = 0.0,
    ) -> Tuple[List[EVProposal], QuantCycleMetrics]:
        start = time.perf_counter()
        now_utc = _utc_now()
        session_since = self._session_started_at if self._session_scope_metrics else None

        snapshots: List[Dict[str, Any]] = []
        proposals: List[EVProposal] = []
        metrics = QuantCycleMetrics()
        reject_reason_counts: Dict[str, int] = {}
        symbol_funnel: Dict[str, Dict[str, int]] = {}

        def _funnel_bump(symbol: str, stage: str, inc: int = 1) -> None:
            sym = str(symbol or "").strip().upper()
            if not sym:
                return
            row = symbol_funnel.setdefault(
                sym,
                {
                    "seen": 0,
                    "proposed": 0,
                    "quality_passed": 0,
                    "uncertainty_passed": 0,
                    "portfolio_passed": 0,
                    "selected": 0,
                },
            )
            row[stage] = int(row.get(stage, 0)) + int(max(0, inc))
        objective_metrics = self.data_lake.get_recent_expectancy_metrics(
            minutes=120,
            since_ts=session_since,
        )
        dynamic_edge_floor = self._derive_dynamic_edge_floor(objective_metrics)
        pre_exec_metrics = self.data_lake.get_recent_execution_metrics(
            minutes=90,
            since_ts=session_since,
        )
        governor_state = self.risk_governor.update(
            now_utc=now_utc,
            objective_metrics=objective_metrics,
            execution_metrics=pre_exec_metrics,
            drawdown_pct=account_drawdown_pct,
            no_accept_cycles=self._no_proposal_cycles,
        )
        metrics.objective_trades_per_hour = float(objective_metrics.get("trades_per_hour", 0.0))
        metrics.objective_expectancy_per_trade_pct = float(
            objective_metrics.get("expectancy_per_trade_pct", 0.0)
        )
        metrics.objective_expectancy_per_hour_pct = float(
            objective_metrics.get("expectancy_per_hour_pct", 0.0)
        )
        metrics.objective_closed_trades = int(objective_metrics.get("closed_trades", 0.0))
        metrics.objective_edge_floor_pct = float(dynamic_edge_floor)
        metrics.objective_score = float(metrics.objective_expectancy_per_hour_pct)
        metrics.risk_multiplier = float(governor_state.risk_multiplier)
        metrics.governor_mode = str(governor_state.mode)
        metrics.governor_reason = str(governor_state.degrade_reason or "")
        metrics.governor_snapshot = dict(governor_state.health_snapshot or {})
        expectancy_guard = self._expectancy_floor_guard_state(minutes=240)
        if expectancy_guard.get("active"):
            metrics.governor_snapshot = dict(metrics.governor_snapshot or {})
            metrics.governor_snapshot["expectancy_floor_guard"] = {
                "active": True,
                "expectancy_per_trade_pct": _f(expectancy_guard.get("expectancy_per_trade_pct"), 0.0),
                "floor_pct": _f(expectancy_guard.get("floor_pct"), 0.0),
                "tail_loss_5_rate": _f(expectancy_guard.get("tail_loss_5_rate"), 0.0),
                "tail_loss_5_limit": _f(expectancy_guard.get("tail_loss_5_limit"), 0.0),
                "closed_trades": int(expectancy_guard.get("closed_trades") or 0),
            }
        self.symbol_policy.refresh(self.data_lake, now_utc=now_utc)
        legacy_relaxed_mode = bool(self.settings.trading.relaxed_trade_gating) and not bool(
            self.settings.trading.win_rate_mode_enabled
        )

        open_positions_count = len([p for p in open_positions or [] if abs(_f(getattr(p, "size", 0))) > 0])

        for symbol, symbol_analysis in (analysis or {}).items():
            ticker = (market_data.get("tickers", {}) or {}).get(symbol)
            if ticker is None:
                continue
            metrics.candidates_scored += 1
            _funnel_bump(symbol, "seen")
            price = _f(getattr(ticker, "last_price", 0.0), 0.0)
            if price <= 0:
                continue
            bid = _f(getattr(ticker, "bid_price", 0.0), 0.0)
            ask = _f(getattr(ticker, "ask_price", 0.0), 0.0)
            spread_bps = ((ask - bid) / price * 10000.0) if ask > 0 and bid > 0 and price > 0 else 0.0
            order_flow = (market_data.get("order_flow", {}) or {}).get(symbol, {}) or {}
            reject_streak = int(symbol_reject_streak.get(symbol, 0))

            features, regime = self.feature_store.extract(
                symbol=symbol,
                symbol_analysis=symbol_analysis,
                market_data=market_data,
                open_positions_count=open_positions_count,
                reject_streak=reject_streak,
            )
            is_chop_regime = False if legacy_relaxed_mode else self._is_chop_regime(features=features, regime=regime)
            atr_pct_15m = abs(_f(features.get("atr_pct_15m"), 0.0))
            atr_bps_15m = max(0.0, atr_pct_15m * 100.0)
            spread_to_atr_ratio = float(spread_bps) / max(1e-9, atr_bps_15m)
            if is_chop_regime and not legacy_relaxed_mode:
                anti_chop_spread_limit = float(self.settings.trading.anti_chop_max_spread_to_atr_ratio)
                if spread_to_atr_ratio > anti_chop_spread_limit:
                    reject_reason_counts["anti_chop_spread_reject"] = (
                        reject_reason_counts.get("anti_chop_spread_reject", 0) + 1
                    )
                    continue

            realized_cost_profile = self.data_lake.get_recent_fill_cost_profile(
                symbol=symbol,
                minutes=int(self.settings.trading.realized_cost_lookback_minutes),
                regime=regime,
                since_ts=session_since,
            )
            if _f(realized_cost_profile.get("samples"), 0.0) < float(self.settings.trading.realized_cost_min_samples):
                realized_cost_profile = self.data_lake.get_recent_fill_cost_profile(
                    symbol=symbol,
                    minutes=int(self.settings.trading.realized_cost_lookback_minutes),
                    since_ts=session_since,
                )
            drift_score = self.drift_monitor.observe(features)
            metrics.drift_score = max(metrics.drift_score, drift_score)
            feature_id = self.data_lake.insert_feature_row(
                ts=_iso(now_utc),
                symbol=symbol,
                price=price,
                regime=regime,
                features=features,
            )

            snapshots.append(
                {
                    "ts": _iso(now_utc),
                    "symbol": symbol,
                    "price": price,
                    "bid_price": bid,
                    "ask_price": ask,
                    "volume_24h": _f(getattr(ticker, "volume_24h", 0.0), 0.0),
                    "funding_rate": _f((market_data.get("funding_rates", {}) or {}).get(symbol), 0.0),
                    "open_interest": _f(getattr(ticker, "open_interest", 0.0), 0.0),
                    "basis": _f(getattr(ticker, "basis", 0.0), 0.0),
                    "spread_bps": spread_bps,
                    "bid_depth_10": _f(order_flow.get("bid_depth_10_levels_usd"), 0.0),
                    "ask_depth_10": _f(order_flow.get("ask_depth_10_levels_usd"), 0.0),
                    "bid_ask_imbalance": _f(order_flow.get("bid_ask_imbalance"), 1.0),
                    "reject_streak": reject_streak,
                }
            )

            horizon_predictions: Dict[int, HorizonPrediction] = {}
            horizon_sample_strength: Dict[int, float] = {}
            for horizon in self.horizons:
                raw = self.regime_switcher.predict(regime=regime, horizon=horizon, features=features)
                prob_raw = _f(raw.get("prob_up_raw"), 0.5)
                prob_cal = self.calibrators[horizon].calibrate(prob_raw)
                sample_strength = max(0.0, min(1.0, _f(raw.get("sample_strength"), 0.0)))
                horizon_sample_strength[horizon] = sample_strength
                uncertainty = self.uncertainty_gate.score(
                    prob_calibrated=prob_cal,
                    calibration_error=self.calibrators[horizon].last_error,
                    drift_score=drift_score,
                    volatility_pct=_f(raw.get("volatility_pct"), 0.0),
                )
                pred = HorizonPrediction(
                    horizon_min=horizon,
                    prob_up_raw=prob_raw,
                    prob_up_calibrated=prob_cal,
                    expected_move_pct=_f(raw.get("expected_move_pct"), 0.0),
                    volatility_pct=_f(raw.get("volatility_pct"), 0.0),
                    uncertainty=uncertainty,
                    expert=f"{regime}_h{horizon}",
                )
                horizon_predictions[horizon] = pred
                self.data_lake.save_prediction(
                    {
                        "ts": _iso(now_utc),
                        "feature_id": feature_id,
                        "symbol": symbol,
                        "horizon_min": horizon,
                        "regime": regime,
                        "prob_up_raw": pred.prob_up_raw,
                        "prob_up_calibrated": pred.prob_up_calibrated,
                        "expected_move_pct": pred.expected_move_pct,
                        "volatility_pct": pred.volatility_pct,
                        "uncertainty": pred.uncertainty,
                        "expert": pred.expert,
                        "sample_strength": sample_strength,
                    }
                )

            proposal = self.ev_constructor.construct(
                symbol=symbol,
                current_price=price,
                regime=regime,
                horizon_predictions=horizon_predictions,
                barrier_profiles={
                    h: self.data_lake.get_barrier_profile(
                        symbol=symbol,
                        regime=regime,
                        horizon_min=h,
                        lookback=900,
                    )
                    for h in self.horizons
                },
                spread_bps=spread_bps,
                reject_streak=reject_streak,
                realized_cost_profile=realized_cost_profile,
            )
            if proposal is None:
                constructor_reason = str(
                    getattr(self.ev_constructor, "last_reject_reason", "") or "constructor_none"
                )
                reject_reason_counts[constructor_reason] = reject_reason_counts.get(constructor_reason, 0) + 1
                continue
            _funnel_bump(symbol, "proposed")

            sample_strength_avg = (
                sum(horizon_sample_strength.values()) / max(1, len(horizon_sample_strength))
            )
            uncertainty_now = max(pred.uncertainty for pred in horizon_predictions.values())
            confidence_now = max(proposal.win_probability, 1.0 - proposal.win_probability)

            proposal.quality_score_raw = float(proposal.quality_score or 0.0)
            proposal.quality_score_adjusted = float(proposal.quality_score_raw)
            proposal.policy_key = f"{str(symbol).upper()}|{proposal.side}|{str(regime).lower()}"

            policy_eval = self.symbol_policy.evaluate(
                symbol=symbol,
                side=proposal.side,
                regime=regime,
                now_utc=now_utc,
            )
            proposal.policy_state = str(policy_eval.get("state") or "green")
            proposal.policy_key = str(policy_eval.get("key") or proposal.policy_key)

            base_symbol_weight = self._symbol_weight(symbol)
            policy_weight_mult = _f(policy_eval.get("symbol_weight_multiplier"), 1.0)
            proposal.symbol_weight = max(0.10, float(base_symbol_weight) * max(0.10, policy_weight_mult))

            uncertainty_bias = float(self.settings.trading.quant_uncertainty_soft_penalty_bias)
            uncertainty_scale = float(self.settings.trading.quant_uncertainty_soft_penalty_scale)
            uncertainty_scaled = 0.0
            if uncertainty_now > uncertainty_bias:
                denom = max(0.01, 1.0 - uncertainty_bias)
                uncertainty_scaled = min(1.0, (uncertainty_now - uncertainty_bias) / denom)
            uncertainty_penalty = uncertainty_scale * uncertainty_scaled
            policy_quality_penalty = _f(policy_eval.get("quality_penalty"), 0.0)

            proposal.quality_score_adjusted = max(
                0.0,
                float(proposal.quality_score_raw) - uncertainty_penalty - policy_quality_penalty,
            )
            proposal.quality_score = proposal.quality_score_adjusted
            proposal.metadata["anti_chop_detected"] = bool(is_chop_regime)
            proposal.metadata["realized_cost_profile"] = realized_cost_profile

            if is_chop_regime:
                anti_chop_skip_quality_max = int(self.settings.trading.anti_chop_skip_quality_max)
                if proposal.quality_score_adjusted < anti_chop_skip_quality_max:
                    reject_reason_counts["anti_chop_quality_reject"] = (
                        reject_reason_counts.get("anti_chop_quality_reject", 0) + 1
                    )
                    continue

            force_probe_from_chop = False if legacy_relaxed_mode else bool(
                is_chop_regime and bool(self.settings.trading.anti_chop_probe_only)
            )
            force_probe_from_expectancy_guard = False if legacy_relaxed_mode else bool(
                expectancy_guard.get("active") and expectancy_guard.get("probe_only")
            )

            dynamic_edge_floor_for_proposal = float(dynamic_edge_floor) + _f(
                policy_eval.get("edge_floor_delta_pct"),
                0.0,
            )
            if proposal.expected_edge_pct < dynamic_edge_floor_for_proposal:
                reject_reason_counts["edge_below_floor"] = reject_reason_counts.get("edge_below_floor", 0) + 1
                continue

            _funnel_bump(symbol, "quality_passed")

            metrics.proposals_generated += 1

            warmup_relax = max(0.0, 1.0 - sample_strength_avg)
            starvation_threshold = int(self.settings.trading.probe_after_no_accept_cycles)
            starvation_active = self._no_proposal_cycles >= starvation_threshold
            starvation_pressure = max(0, self._no_proposal_cycles - starvation_threshold)
            dynamic_min_confidence = max(
                0.50,
                float(self.settings.trading.quant_min_calibrated_confidence) - (warmup_relax * 0.04),
            )
            dynamic_max_uncertainty = min(
                0.82,
                float(self.settings.trading.quant_uncertainty_max) + (warmup_relax * 0.10),
            )
            if starvation_active:
                dynamic_min_confidence = max(
                    0.47,
                    dynamic_min_confidence - min(0.07, (0.015 * max(1, starvation_pressure + 1))),
                )
                dynamic_max_uncertainty = min(
                    0.90,
                    dynamic_max_uncertainty + min(0.10, (0.02 * max(1, starvation_pressure + 1))),
                )
            if legacy_relaxed_mode:
                probe_quality_min = max(35, int(self.settings.trading.quality_score_probe_min))
                full_quality_min = max(
                    probe_quality_min + 8,
                    int(self.settings.trading.quality_score_full_min),
                )
            else:
                probe_quality_min = int(self.settings.trading.quality_score_probe_min)
                full_quality_min = int(self.settings.trading.quality_score_full_min)
                if self._no_proposal_cycles >= max(2, starvation_threshold // 2):
                    probe_quality_min -= 5
                    full_quality_min -= 3
                if starvation_active:
                    probe_quality_min -= 8
                    full_quality_min -= 5
                if self._no_proposal_cycles >= (starvation_threshold + 4):
                    probe_quality_min -= 6
                    full_quality_min -= 3
                if is_chop_regime:
                    probe_quality_min = max(
                        probe_quality_min,
                        int(self.settings.trading.anti_chop_probe_quality_min),
                    )
                    full_quality_min = max(full_quality_min, probe_quality_min + 12)
                if expectancy_guard.get("active"):
                    probe_quality_min = max(probe_quality_min, 50)
                    full_quality_min = max(full_quality_min, 68)
                probe_quality_min = max(40, min(95, probe_quality_min))
                full_quality_min = max(probe_quality_min + 8, min(98, full_quality_min))
            probe_scale = float(self.settings.trading.probe_risk_scale)
            quality_override = False
            if proposal.quality_score_adjusted >= full_quality_min:
                proposal.entry_tier = "full"
            elif proposal.quality_score_adjusted >= probe_quality_min:
                proposal.entry_tier = "probe"
                proposal.risk_pct = max(0.3, float(proposal.risk_pct) * probe_scale)
                proposal.expected_hold_minutes = max(5, int(proposal.expected_hold_minutes * 0.75))
            else:
                starvation_probe_min = max(35, probe_quality_min - 10)
                if starvation_active and proposal.quality_score_adjusted >= starvation_probe_min:
                    proposal.entry_tier = "probe"
                    proposal.risk_pct = max(0.25, float(proposal.risk_pct) * min(probe_scale, 0.30))
                    proposal.expected_hold_minutes = max(5, int(proposal.expected_hold_minutes * 0.65))
                    proposal.metadata["gate_override"] = "probe_starvation_unlock"
                    proposal.metadata["starvation_probe_min"] = starvation_probe_min
                    reject_reason_counts["probe_starvation_unlock"] = (
                        reject_reason_counts.get("probe_starvation_unlock", 0) + 1
                    )
                    quality_override = True
                else:
                    reject_reason_counts["probe_rejected_quality"] = reject_reason_counts.get("probe_rejected_quality", 0) + 1
                    continue

            if force_probe_from_chop:
                anti_chop_probe_quality_min = int(self.settings.trading.anti_chop_probe_quality_min)
                if proposal.quality_score_adjusted < anti_chop_probe_quality_min:
                    reject_reason_counts["anti_chop_probe_reject"] = (
                        reject_reason_counts.get("anti_chop_probe_reject", 0) + 1
                    )
                    continue
                if proposal.entry_tier != "probe":
                    proposal.entry_tier = "probe"
                    proposal.risk_pct = max(0.25, float(proposal.risk_pct) * min(probe_scale, 0.30))
                    proposal.expected_hold_minutes = max(5, int(proposal.expected_hold_minutes * 0.70))
                proposal.metadata["gate_override"] = proposal.metadata.get("gate_override") or "anti_chop_probe_only"

            if force_probe_from_expectancy_guard:
                if proposal.entry_tier != "probe":
                    proposal.entry_tier = "probe"
                proposal.risk_pct = max(0.20, float(proposal.risk_pct) * 0.60)
                proposal.expected_hold_minutes = max(5, int(proposal.expected_hold_minutes * 0.70))
                proposal.metadata["gate_override"] = (
                    proposal.metadata.get("gate_override") or "expectancy_floor_guard_probe_only"
                )

            proposal.metadata["atr_pct_15m"] = atr_pct_15m
            proposal.metadata["atr_bps_15m"] = atr_bps_15m
            proposal.metadata["spread_to_atr_ratio"] = spread_to_atr_ratio

            stable_ratio = 1.0
            if not legacy_relaxed_mode:
                # Full-tier only: require spread to remain small relative to local ATR.
                max_spread_to_atr_ratio = max(
                    0.01,
                    float(self.settings.trading.full_entry_max_spread_to_atr_ratio),
                )
                if proposal.entry_tier == "full":
                    if atr_bps_15m <= 0 or spread_to_atr_ratio > max_spread_to_atr_ratio:
                        proposal.entry_tier = "probe"
                        proposal.risk_pct = max(0.25, float(proposal.risk_pct) * min(probe_scale, 0.30))
                        proposal.expected_hold_minutes = max(5, int(proposal.expected_hold_minutes * 0.80))
                        proposal.metadata["gate_override"] = proposal.metadata.get("gate_override") or "full_downgraded_spread_to_atr"
                        proposal.metadata["spread_to_atr_limit"] = max_spread_to_atr_ratio
                        reject_reason_counts["full_downgraded_spread_to_atr"] = (
                            reject_reason_counts.get("full_downgraded_spread_to_atr", 0) + 1
                        )

                # Full-tier only: require regime stability over the last N bars.
                stability_bars = max(1, int(self.settings.trading.full_entry_regime_stability_bars))
                stability_min_ratio = max(
                    0.10,
                    min(1.0, float(self.settings.trading.full_entry_regime_stability_min_ratio)),
                )
                recent_regimes = self.data_lake.get_recent_symbol_regimes(symbol=symbol, limit=stability_bars)
                stable_ratio = 0.0
                if recent_regimes:
                    stable_hits = sum(1 for value in recent_regimes if value == str(regime).lower())
                    stable_ratio = stable_hits / max(1, len(recent_regimes))
                regime_stable = len(recent_regimes) >= stability_bars and stable_ratio >= stability_min_ratio
                proposal.metadata["regime_stability_bars"] = len(recent_regimes)
                proposal.metadata["regime_stability_ratio"] = stable_ratio
                proposal.metadata["regime_stability_required_bars"] = stability_bars
                proposal.metadata["regime_stability_required_ratio"] = stability_min_ratio

                if proposal.entry_tier == "full" and not regime_stable:
                    proposal.entry_tier = "probe"
                    proposal.risk_pct = max(0.25, float(proposal.risk_pct) * min(probe_scale, 0.30))
                    proposal.expected_hold_minutes = max(5, int(proposal.expected_hold_minutes * 0.85))
                    proposal.metadata["gate_override"] = proposal.metadata.get("gate_override") or "full_downgraded_regime_unstable"
                    reject_reason_counts["full_downgraded_regime_unstable"] = (
                        reject_reason_counts.get("full_downgraded_regime_unstable", 0) + 1
                    )

            force_probe_only = False
            if not legacy_relaxed_mode:
                force_probe_only = bool(policy_eval.get("force_probe_only", False))
                red_probe_allowed = bool(policy_eval.get("red_probe_allowed_now", True))
                if force_probe_only:
                    if not red_probe_allowed:
                        reject_reason_counts["policy_red_probe_cooldown"] = (
                            reject_reason_counts.get("policy_red_probe_cooldown", 0) + 1
                        )
                        continue
                    if proposal.entry_tier != "probe":
                        proposal.entry_tier = "probe"
                        proposal.metadata["policy_probe_enforced"] = True
                    proposal.risk_pct = max(
                        0.25,
                        float(proposal.risk_pct)
                        * min(probe_scale, float(self.settings.trading.symbol_policy_red_probe_risk_mult)),
                    )
                    proposal.expected_hold_minutes = max(5, int(proposal.expected_hold_minutes * 0.75))
                    proposal.metadata["gate_override"] = proposal.metadata.get("gate_override") or "policy_red_probe_only"

            allowed_slippage_bps = (
                float(self.settings.trading.probe_entry_max_est_slippage_bps)
                if proposal.entry_tier == "probe"
                else float(self.settings.trading.full_entry_max_est_slippage_bps)
            )
            allowed_slippage_bps = max(
                5.0,
                allowed_slippage_bps * max(0.05, _f(policy_eval.get("slippage_limit_multiplier"), 1.0)),
            )
            if float(proposal.estimated_slippage_bps) > allowed_slippage_bps:
                if proposal.entry_tier == "probe":
                    reject_reason_counts["probe_rejected_cost"] = reject_reason_counts.get("probe_rejected_cost", 0) + 1
                else:
                    reject_reason_counts["full_rejected_cost"] = reject_reason_counts.get("full_rejected_cost", 0) + 1
                continue

            gate = self.uncertainty_gate.allow(
                prob_calibrated=confidence_now,
                uncertainty=uncertainty_now,
                min_confidence_override=dynamic_min_confidence,
                max_uncertainty_override=dynamic_max_uncertainty,
            )
            proposal.metadata["gate"] = gate
            proposal.metadata["feature_id"] = feature_id
            proposal.metadata["sample_strength_avg"] = sample_strength_avg
            proposal.metadata["dynamic_min_confidence"] = dynamic_min_confidence
            proposal.metadata["dynamic_max_uncertainty"] = dynamic_max_uncertainty
            proposal.metadata["dynamic_probe_quality_min"] = probe_quality_min
            proposal.metadata["dynamic_full_quality_min"] = full_quality_min
            proposal.metadata["quality_override"] = bool(quality_override)
            proposal.metadata["objective_edge_floor_pct"] = dynamic_edge_floor
            proposal.metadata["dynamic_edge_floor_for_proposal_pct"] = dynamic_edge_floor_for_proposal
            proposal.metadata["expectancy_floor_guard_active"] = bool(expectancy_guard.get("active"))
            proposal.metadata["expectancy_floor_guard"] = dict(expectancy_guard)
            proposal.metadata["risk_multiplier"] = float(governor_state.risk_multiplier)
            proposal.metadata["policy"] = policy_eval
            proposal.metadata["quality_score_raw"] = proposal.quality_score_raw
            proposal.metadata["quality_score_adjusted"] = proposal.quality_score_adjusted
            proposal.metadata["uncertainty_penalty"] = uncertainty_penalty
            proposal.metadata["policy_quality_penalty"] = policy_quality_penalty
            proposal.metadata["uncertainty_scaled"] = uncertainty_scaled
            proposal.metadata["edge_net_after_cost_pct"] = float(proposal.expected_edge_pct)
            proposal.metadata["candidate_diag"] = {
                "symbol": proposal.symbol,
                "side": proposal.side,
                "regime": proposal.regime,
                "q_raw": float(proposal.quality_score_raw),
                "q_adj": float(proposal.quality_score_adjusted),
                "edge_net_pct": float(proposal.expected_edge_pct),
                "cost_est_bps": float(proposal.estimated_slippage_bps),
                "spread_to_atr_ratio": float(spread_to_atr_ratio),
                "atr_bps_15m": float(atr_bps_15m),
                "regime_stability_ratio": float(stable_ratio),
                "cap_clipped": False,
                "final_size": None,
                "entry_tier": proposal.entry_tier,
                "anti_chop_detected": bool(is_chop_regime),
                "expectancy_floor_guard_active": bool(expectancy_guard.get("active")),
            }
            proposal.metadata["horizon_predictions"] = {
                h: {
                    "prob_up_raw": p.prob_up_raw,
                    "prob_up_calibrated": p.prob_up_calibrated,
                    "expected_move_pct": p.expected_move_pct,
                    "volatility_pct": p.volatility_pct,
                    "uncertainty": p.uncertainty,
                    "expert": p.expert,
                    "sample_strength": horizon_sample_strength.get(h, 0.0),
                }
                for h, p in horizon_predictions.items()
            }
            gate_override = False
            if not gate.get("allowed"):
                hard_uncertainty_max = float(self.settings.trading.quant_uncertainty_hard_max)
                if uncertainty_now > hard_uncertainty_max:
                    reject_reason_counts["uncertainty_hard_reject"] = (
                        reject_reason_counts.get("uncertainty_hard_reject", 0) + 1
                    )
                    continue
                confidence_gap = dynamic_min_confidence - confidence_now
                if confidence_gap > 0.08:
                    reject_reason_counts["confidence_hard_reject"] = (
                        reject_reason_counts.get("confidence_hard_reject", 0) + 1
                    )
                    continue
                soft_penalty = max(0.0, confidence_gap * 20.0) + max(
                    0.0,
                    (uncertainty_now - dynamic_max_uncertainty) * 30.0,
                )
                proposal.quality_score_adjusted = max(
                    0.0,
                    float(proposal.quality_score_adjusted) - soft_penalty,
                )
                proposal.quality_score = proposal.quality_score_adjusted
                proposal.risk_pct = max(0.25, float(proposal.risk_pct) * 0.75)
                proposal.metadata["gate_override"] = "uncertainty_soft_penalty"
                proposal.metadata["uncertainty_soft_penalty"] = soft_penalty
                gate_override = True
                reject_reason_counts["uncertainty_soft_pass"] = (
                    reject_reason_counts.get("uncertainty_soft_pass", 0) + 1
                )
                if not gate_override:
                    reject_reason_counts["uncertainty_gate"] = reject_reason_counts.get("uncertainty_gate", 0) + 1
                    continue
            metrics.uncertainty_passed += 1
            _funnel_bump(symbol, "uncertainty_passed")

            if proposal.entry_tier == "probe":
                probe_fills_last_hour = self.data_lake.count_execution_events(
                    event_type="fill",
                    reason_code="probe_entry_filled",
                    minutes=60,
                    since_ts=session_since,
                )
                if probe_fills_last_hour >= int(self.settings.trading.probe_max_per_hour):
                    reject_reason_counts["probe_hourly_cap"] = reject_reason_counts.get("probe_hourly_cap", 0) + 1
                    continue
                if force_probe_only:
                    self.symbol_policy.mark_red_probe(
                        symbol=proposal.symbol,
                        side=proposal.side,
                        regime=proposal.regime,
                        now_utc=now_utc,
                    )
            proposals.append(proposal)

        self.data_lake.record_market_snapshots(snapshots)
        labeled_count = self._label_matured_rows(now_utc)
        retrain_report = self.trainer.maybe_retrain(drift_score=metrics.drift_score)
        if retrain_report:
            metrics.retrained = True
            metrics.retrain_summary = retrain_report
            self._save_state()

        if proposals:
            min_risk = float(self.settings.trading.min_risk_pct)
            max_risk = float(self.settings.trading.max_risk_pct)
            for proposal in proposals:
                policy_risk_mult = _f(
                    (proposal.metadata or {}).get("policy", {}).get("risk_multiplier"),
                    1.0,
                )
                proposal.risk_pct = max(
                    min_risk,
                    min(
                        max_risk,
                        float(proposal.risk_pct)
                        * float(governor_state.risk_multiplier)
                        * max(0.05, policy_risk_mult),
                    ),
                )

        ranked = self.portfolio_optimizer.select_best(
            proposals=proposals,
            open_positions=open_positions,
            data_lake=self.data_lake,
            current_total_risk_pct=current_total_risk_pct,
            account_balance=account_balance,
            symbol_max_margin_pct=float(self.settings.trading.symbol_max_margin_pct),
            portfolio_max_margin_pct=float(self.settings.trading.portfolio_max_margin_pct),
            allow_probe_override=True,
        )
        ranked = self._apply_symbol_diversity_ranking(
            ranked=ranked,
            now_utc=now_utc,
            since_ts=session_since,
        )
        for proposal in ranked:
            _funnel_bump(proposal.symbol, "portfolio_passed")
        if ranked:
            _funnel_bump(ranked[0].symbol, "selected")
        metrics.portfolio_passed = len(ranked)
        metrics.proposals_accepted = len(ranked)
        metrics.selected = len(ranked)
        if metrics.proposals_accepted > 0:
            self._no_proposal_cycles = 0
        else:
            self._no_proposal_cycles += 1

        exec_metrics = self.data_lake.get_recent_execution_metrics(minutes=90, since_ts=session_since)
        metrics.recent_execution_events = float(exec_metrics.get("events", 0.0))
        metrics.recent_reject_rate = float(exec_metrics.get("reject_rate", 0.0))
        metrics.recent_fill_slippage_bps = float(exec_metrics.get("avg_slippage_bps", 0.0))
        metrics.affordability_passed = len(ranked)
        recent_reject_reasons = self.data_lake.get_recent_reject_reason_counts(
            minutes=120,
            since_ts=session_since,
        )
        merged_reject_counts: Dict[str, int] = dict(reject_reason_counts)
        for key, value in recent_reject_reasons.items():
            merged_reject_counts[f"recent_{key}"] = int(value)
        metrics.reject_reason_counts = merged_reject_counts
        metrics.symbol_funnel = symbol_funnel
        metrics.cycle_latency_ms = (time.perf_counter() - start) * 1000.0
        policy_state_counts: Dict[str, int] = {}
        for p in proposals:
            state = str(getattr(p, "policy_state", "green") or "green")
            policy_state_counts[state] = int(policy_state_counts.get(state, 0)) + 1
        bucket_summary = {
            "tracked_buckets": len(getattr(self.symbol_policy, "buckets", {}) or {}),
            "states": policy_state_counts,
        }
        alerts: List[Dict[str, Any]] = []
        reject_alert = float(self.settings.trading.quant_monitor_reject_alert)
        reject_min_events = int(self.settings.trading.quant_reject_kill_switch_min_events)
        latency_alert_ms = float(self.settings.trading.quant_monitor_latency_alert_ms)
        drift_alert = float(self.settings.trading.quant_monitor_drift_alert)
        no_prop_alert_cycles = int(self.settings.trading.quant_monitor_no_proposal_cycles_alert)
        floor_hour = float(self.settings.trading.quant_expectancy_hour_floor_pct)
        min_closed = int(self.settings.trading.quant_min_closed_trades_for_objective)

        if (
            metrics.recent_execution_events >= reject_min_events
            and metrics.recent_reject_rate > reject_alert
            and self._can_emit_alert("high_reject_rate", now_utc)
        ):
            alerts.append(
                self._emit_alert(
                    now_utc=now_utc,
                    severity="critical",
                    code="high_reject_rate",
                    message=(
                        f"Reject rate {metrics.recent_reject_rate:.2%} exceeded "
                        f"threshold {reject_alert:.2%}"
                    ),
                    details={
                        "events": metrics.recent_execution_events,
                        "reject_rate": metrics.recent_reject_rate,
                        "threshold": reject_alert,
                        "min_events": reject_min_events,
                    },
                )
            )

        if metrics.cycle_latency_ms > latency_alert_ms and self._can_emit_alert("high_cycle_latency", now_utc):
            alerts.append(
                self._emit_alert(
                    now_utc=now_utc,
                    severity="warning",
                    code="high_cycle_latency",
                    message=(
                        f"Quant cycle latency {metrics.cycle_latency_ms:.1f}ms exceeded "
                        f"threshold {latency_alert_ms:.1f}ms"
                    ),
                    details={
                        "latency_ms": metrics.cycle_latency_ms,
                        "threshold_ms": latency_alert_ms,
                    },
                )
            )

        if metrics.drift_score > drift_alert and self._can_emit_alert("high_feature_drift", now_utc):
            alerts.append(
                self._emit_alert(
                    now_utc=now_utc,
                    severity="warning",
                    code="high_feature_drift",
                    message=(
                        f"Feature drift score {metrics.drift_score:.3f} exceeded "
                        f"threshold {drift_alert:.3f}"
                    ),
                    details={
                        "drift_score": metrics.drift_score,
                        "threshold": drift_alert,
                    },
                )
            )

        if self._no_proposal_cycles >= no_prop_alert_cycles and self._can_emit_alert("proposal_starvation", now_utc):
            alerts.append(
                self._emit_alert(
                    now_utc=now_utc,
                    severity="warning",
                    code="proposal_starvation",
                    message=(
                        f"No accepted proposals for {self._no_proposal_cycles} consecutive cycles"
                    ),
                    details={
                        "no_proposal_cycles": self._no_proposal_cycles,
                        "threshold_cycles": no_prop_alert_cycles,
                    },
                )
            )

        if (
            metrics.objective_closed_trades >= min_closed
            and metrics.objective_expectancy_per_hour_pct < floor_hour
            and self._can_emit_alert("negative_expectancy_hour", now_utc)
        ):
            alerts.append(
                self._emit_alert(
                    now_utc=now_utc,
                    severity="warning",
                    code="negative_expectancy_hour",
                    message=(
                        f"Expectancy/hour {metrics.objective_expectancy_per_hour_pct:+.3f}% below floor "
                        f"{floor_hour:+.3f}%"
                    ),
                    details={
                        "expectancy_per_hour_pct": metrics.objective_expectancy_per_hour_pct,
                        "floor_pct": floor_hour,
                        "closed_trades": metrics.objective_closed_trades,
                    },
                )
            )
        if expectancy_guard.get("active") and self._can_emit_alert("expectancy_floor_guard", now_utc):
            alerts.append(
                self._emit_alert(
                    now_utc=now_utc,
                    severity="warning",
                    code="expectancy_floor_guard",
                    message=(
                        "Expectancy-floor guard active: forcing probe-only de-risking "
                        f"(exp/trade={_f(expectancy_guard.get('expectancy_per_trade_pct'), 0.0):+.3f}%, "
                        f"tail5={_f(expectancy_guard.get('tail_loss_5_rate'), 0.0):.2%})"
                    ),
                    details=dict(expectancy_guard),
                )
            )
        if (
            metrics.governor_mode == "degrading"
            and metrics.governor_reason
            and self._can_emit_alert("soft_governor_degrading", now_utc)
        ):
            alerts.append(
                self._emit_alert(
                    now_utc=now_utc,
                    severity="warning",
                    code="soft_governor_degrading",
                    message=(
                        f"Soft governor degrading risk to x{metrics.risk_multiplier:.2f} "
                        f"({metrics.governor_reason})"
                    ),
                    details={
                        "risk_multiplier": metrics.risk_multiplier,
                        "mode": metrics.governor_mode,
                        "reason": metrics.governor_reason,
                        "snapshot": metrics.governor_snapshot or {},
                    },
                )
            )
        metrics.monitor_alerts = alerts

        self.data_lake.save_monitor_metrics(
            {
                "timestamp_utc": _iso(now_utc),
                "candidates_scored": metrics.candidates_scored,
                "proposals_generated": metrics.proposals_generated,
                "uncertainty_passed": metrics.uncertainty_passed,
                "portfolio_passed": metrics.portfolio_passed,
                "affordability_passed": metrics.affordability_passed,
                "selected": metrics.selected,
                "proposals_accepted": metrics.proposals_accepted,
                "objective_trades_per_hour": metrics.objective_trades_per_hour,
                "objective_expectancy_per_trade_pct": metrics.objective_expectancy_per_trade_pct,
                "objective_expectancy_per_hour_pct": metrics.objective_expectancy_per_hour_pct,
                "objective_closed_trades": metrics.objective_closed_trades,
                "objective_edge_floor_pct": metrics.objective_edge_floor_pct,
                "risk_multiplier": metrics.risk_multiplier,
                "governor_mode": metrics.governor_mode,
                "governor_reason": metrics.governor_reason,
                "governor_snapshot": metrics.governor_snapshot or {},
                "expectancy_floor_guard": dict(expectancy_guard),
                "symbol_policy_summary": bucket_summary,
                "symbol_funnel": metrics.symbol_funnel or {},
                "reject_reason_counts": metrics.reject_reason_counts or {},
                "drift_score": metrics.drift_score,
                "recent_reject_rate": metrics.recent_reject_rate,
                "recent_fill_slippage_bps": metrics.recent_fill_slippage_bps,
                "cycle_latency_ms": metrics.cycle_latency_ms,
                "labeled_count": labeled_count,
                "retrained": metrics.retrained,
                "retrain_summary": metrics.retrain_summary,
                "alerts": alerts,
            }
        )
        self._save_state()
        return ranked, metrics

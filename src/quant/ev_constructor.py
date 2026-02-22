"""
Construct EV-maximizing trade proposals from probabilistic forecasts.
"""

from __future__ import annotations

from typing import Dict, Optional

from .types import EVProposal, HorizonPrediction


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


class EVTradeConstructor:
    def __init__(
        self,
        min_edge_pct: float,
        min_expected_move_pct: float,
        min_tp_pct: float,
        max_tp_pct: float,
        min_sl_pct: float,
        max_sl_pct: float,
        round_trip_cost_pct: float,
        realized_cost_slippage_weight: float = 0.70,
        realized_cost_slippage_clip_bps: float = 180.0,
    ):
        self.min_edge_pct = float(min_edge_pct)
        self.min_expected_move_pct = float(min_expected_move_pct)
        self.min_tp_pct = float(min_tp_pct)
        self.max_tp_pct = float(max_tp_pct)
        self.min_sl_pct = float(min_sl_pct)
        self.max_sl_pct = float(max_sl_pct)
        self.round_trip_cost_pct = float(round_trip_cost_pct)
        self.realized_cost_slippage_weight = float(realized_cost_slippage_weight)
        self.realized_cost_slippage_clip_bps = float(realized_cost_slippage_clip_bps)
        self.last_reject_reason: str = ""
        self.last_reject_details: Dict[str, float] = {}

    def _reject(self, reason: str, **details: float) -> None:
        self.last_reject_reason = str(reason or "constructor_none")
        self.last_reject_details = {str(k): float(v) for k, v in details.items()}

    @staticmethod
    def _quality_score(
        confidence: float,
        expected_edge_pct: float,
        uncertainty: float,
        spread_bps: float,
        reject_streak: int,
        regime: str,
        realized_slippage_bps: float = 0.0,
    ) -> float:
        score = 0.0
        # Keep quality scoring permissive enough for live exploration while still
        # prioritizing confidence, edge, and low uncertainty.
        score += max(0.0, min(1.0, (confidence - 0.48) / 0.30)) * 40.0
        score += max(0.0, min(1.0, (expected_edge_pct + 0.08) / 0.55)) * 38.0
        score += max(0.0, min(1.0, 1.0 - uncertainty)) * 16.0
        spread_penalty = min(12.0, max(0.0, spread_bps / 12.0))
        realized_cost_penalty = min(10.0, max(0.0, realized_slippage_bps / 20.0))
        reject_penalty = min(7.0, max(0.0, reject_streak * 1.35))
        score -= (spread_penalty + reject_penalty + realized_cost_penalty)
        if regime == "volatile":
            score -= 2.0
        elif regime == "trend":
            score += 4.0
        elif regime == "range":
            score += 1.5
        return _clamp(score, 0.0, 100.0)

    def _aggregate(self, preds: Dict[int, HorizonPrediction]) -> Dict[str, float]:
        if not preds:
            return {
                "prob_up": 0.5,
                "expected_move_pct": 0.0,
                "volatility_pct": 0.0,
                "uncertainty": 1.0,
            }
        weights = {5: 0.5, 15: 0.3, 30: 0.2}
        total_w = 0.0
        out = {"prob_up": 0.0, "expected_move_pct": 0.0, "volatility_pct": 0.0, "uncertainty": 0.0}
        for h, pred in preds.items():
            w = float(weights.get(h, 0.0))
            if w <= 0:
                continue
            total_w += w
            out["prob_up"] += pred.prob_up_calibrated * w
            out["expected_move_pct"] += pred.expected_move_pct * w
            out["volatility_pct"] += pred.volatility_pct * w
            out["uncertainty"] += pred.uncertainty * w
        if total_w <= 0:
            return {
                "prob_up": 0.5,
                "expected_move_pct": 0.0,
                "volatility_pct": 0.0,
                "uncertainty": 1.0,
            }
        for k in out:
            out[k] /= total_w
        return out

    def construct(
        self,
        symbol: str,
        current_price: float,
        regime: str,
        horizon_predictions: Dict[int, HorizonPrediction],
        spread_bps: float,
        reject_streak: int,
        barrier_profiles: Optional[Dict[int, Dict[str, float]]] = None,
        realized_cost_profile: Optional[Dict[str, float]] = None,
    ) -> Optional[EVProposal]:
        self._reject("constructor_none")
        if current_price <= 0:
            self._reject("invalid_price")
            return None

        agg = self._aggregate(horizon_predictions)
        p_up = _clamp(float(agg["prob_up"]), 0.0, 1.0)
        expected_move = max(self.min_expected_move_pct, float(agg["expected_move_pct"]))
        volatility = max(0.04, float(agg["volatility_pct"]))
        uncertainty = _clamp(float(agg["uncertainty"]), 0.0, 1.0)

        side_threshold = 0.505 + min(0.015, (max(0.0, spread_bps) / 1800.0) + (reject_streak * 0.003))
        if p_up >= side_threshold:
            side = "Buy"
            p_side = p_up
        elif p_up <= (1.0 - side_threshold):
            side = "Sell"
            p_side = 1.0 - p_up
        else:
            self._reject(
                "side_not_decisive",
                prob_up=p_up,
                side_threshold=side_threshold,
            )
            return None

        # Execution friction penalty.
        p_side = _clamp(p_side - ((spread_bps / 10000.0) * 3.5) - (reject_streak * 0.02), 0.05, 0.95)

        barrier_profiles = barrier_profiles or {}
        barrier_samples = 0.0
        barrier_favorable = 0.5
        barrier_volatility = max(0.05, volatility)
        horizon_weights = {5: 0.50, 15: 0.30, 30: 0.20}
        weighted_favorable = 0.0
        weighted_volatility = 0.0
        total_weight = 0.0
        for horizon, profile in barrier_profiles.items():
            samples = float(profile.get("samples", 0.0) or 0.0)
            if samples <= 0:
                continue
            w = horizon_weights.get(int(horizon), 0.0) * min(1.0, (samples / 180.0))
            if w <= 0:
                continue
            upper = _clamp(float(profile.get("upper_hit_rate", 0.5) or 0.5), 0.0, 1.0)
            lower = _clamp(float(profile.get("lower_hit_rate", 0.5) or 0.5), 0.0, 1.0)
            favorable = upper if side == "Buy" else lower
            weighted_favorable += favorable * w
            weighted_volatility += max(0.04, float(profile.get("avg_realized_volatility", volatility) or volatility)) * w
            total_weight += w
            barrier_samples += samples
        if total_weight > 0:
            barrier_favorable = weighted_favorable / total_weight
            barrier_volatility = weighted_volatility / total_weight

        barrier_confidence = _clamp(barrier_samples / 420.0, 0.0, 1.0)
        # Blend model probability with observed TP-before-SL barrier outcomes.
        p_side = _clamp(
            (1.0 - (0.38 * barrier_confidence)) * p_side
            + ((0.38 * barrier_confidence) * barrier_favorable),
            0.05,
            0.95,
        )
        volatility = max(volatility, barrier_volatility * 0.75)

        best_horizon = max(
            horizon_predictions.values(),
            key=lambda pred: abs(pred.prob_up_calibrated - 0.5),
        ).horizon_min if horizon_predictions else 15
        expected_hold_minutes = int(max(5, min(180, int(best_horizon * (1.6 if regime == "range" else 1.2)))))

        realized_cost_profile = realized_cost_profile or {}
        realized_slippage_samples = max(0.0, float(realized_cost_profile.get("samples", 0.0) or 0.0))
        realized_slippage_avg_bps = max(
            0.0,
            float(
                realized_cost_profile.get("p75_slippage_bps")
                or realized_cost_profile.get("avg_slippage_bps")
                or 0.0
            ),
        )
        realized_slippage_clip_bps = max(1.0, float(self.realized_cost_slippage_clip_bps))
        realized_slippage_weight = max(0.0, float(self.realized_cost_slippage_weight))
        if realized_slippage_samples > 0:
            realized_cost_pct = (min(realized_slippage_avg_bps, realized_slippage_clip_bps) / 100.0) * realized_slippage_weight
        else:
            realized_cost_pct = 0.0

        best = None
        for tp_mult in (0.8, 1.0, 1.2, 1.4, 1.6):
            for sl_mult in (0.65, 0.85, 1.0, 1.15):
                tp_pct = _clamp(expected_move * tp_mult, self.min_tp_pct, self.max_tp_pct)
                # Dynamic stop with volatility + uncertainty control.
                sl_base = (volatility * 0.6 * sl_mult) * (1.0 + (uncertainty * 0.35))
                sl_pct = _clamp(sl_base, self.min_sl_pct, self.max_sl_pct)
                rr = tp_pct / sl_pct if sl_pct > 0 else 0.0
                if rr < 1.05:
                    continue
                dynamic_cost_pct = (
                    self.round_trip_cost_pct
                    + max(0.0, (float(spread_bps) / 100.0) * 0.85)
                    + max(0.0, float(reject_streak) * 0.008)
                    + max(0.0, float(uncertainty) * 0.02)
                    + (realized_cost_pct * 0.40)
                )
                edge = (p_side * tp_pct) - ((1.0 - p_side) * sl_pct) - dynamic_cost_pct
                expectancy_per_hour = edge / max(expected_hold_minutes / 60.0, 0.08)
                if best is None or expectancy_per_hour > best["expectancy_per_hour"]:
                    best = {
                        "edge": edge,
                        "dynamic_cost_pct": dynamic_cost_pct,
                        "tp_pct": tp_pct,
                        "sl_pct": sl_pct,
                        "rr": rr,
                        "expectancy_per_hour": expectancy_per_hour,
                    }

        if not best:
            self._reject(
                "rr_or_tp_sl_invalid",
                expected_move=expected_move,
                volatility=volatility,
            )
            return None
        if best["edge"] < self.min_edge_pct:
            self._reject(
                "edge_below_min",
                edge=float(best["edge"]),
                min_edge=self.min_edge_pct,
            )
            return None

        tp_pct = float(best["tp_pct"])
        sl_pct = float(best["sl_pct"])
        rr_ratio = float(best["rr"])

        if side == "Buy":
            stop_loss = current_price * (1.0 - sl_pct / 100.0)
            take_profit = current_price * (1.0 + tp_pct / 100.0)
        else:
            stop_loss = current_price * (1.0 + sl_pct / 100.0)
            take_profit = current_price * (1.0 - tp_pct / 100.0)

        confidence = _clamp(max(p_side, 1.0 - p_side), 0.0, 1.0)
        spread_est_bps = max(0.0, float(spread_bps) * (0.9 + (0.2 * max(0, reject_streak))))
        if realized_slippage_samples > 0:
            estimated_slippage_bps = max(
                spread_est_bps,
                (0.55 * spread_est_bps) + (0.45 * min(realized_slippage_avg_bps, realized_slippage_clip_bps)),
            )
        else:
            estimated_slippage_bps = spread_est_bps
        quality_score = self._quality_score(
            confidence=confidence,
            expected_edge_pct=float(best["edge"]),
            uncertainty=uncertainty,
            spread_bps=spread_bps,
            reject_streak=reject_streak,
            regime=regime,
            realized_slippage_bps=realized_slippage_avg_bps,
        )

        # Dynamic risk/leverage proposal.
        risk_pct = 1.0 + ((confidence - 0.5) * 3.0)
        if regime == "volatile":
            risk_pct *= 0.75
        risk_pct = _clamp(risk_pct, 0.8, 3.2)

        leverage = int(round(_clamp(3.0 + (confidence * 4.0), 2.0, 8.0)))
        if regime == "volatile":
            leverage = max(2, leverage - 1)

        return EVProposal(
            symbol=symbol,
            side=side,
            entry_price=float(current_price),
            stop_loss=float(stop_loss),
            take_profit=float(take_profit),
            risk_pct=float(risk_pct),
            leverage=int(leverage),
            expected_edge_pct=float(best["edge"]),
            win_probability=float(p_side),
            rr_ratio=rr_ratio,
            confidence=confidence,
            regime=regime,
            expected_hold_minutes=expected_hold_minutes,
            expectancy_per_hour_pct=float(best["expectancy_per_hour"]),
            quality_score=quality_score,
            quality_score_raw=quality_score,
            quality_score_adjusted=quality_score,
            entry_tier="full",
            symbol_weight=1.0,
            policy_state="green",
            policy_key=f"{str(symbol).upper()}|{side}|{str(regime).lower()}",
            estimated_slippage_bps=estimated_slippage_bps,
            metadata={
                "uncertainty": uncertainty,
                "spread_bps": float(spread_bps),
                "reject_streak": int(reject_streak),
                "expected_move_pct": expected_move,
                "volatility_pct": volatility,
                "prob_up": p_up,
                "barrier_favorable_rate": barrier_favorable,
                "barrier_samples": barrier_samples,
                "barrier_confidence": barrier_confidence,
                "dynamic_cost_pct": float(best.get("dynamic_cost_pct", self.round_trip_cost_pct)),
                "realized_cost_profile": {
                    "samples": float(realized_slippage_samples),
                    "slippage_bps": float(realized_slippage_avg_bps),
                    "cost_pct_component": float(realized_cost_pct),
                },
                "expectancy_per_hour_pct": float(best["expectancy_per_hour"]),
                "expected_hold_minutes": expected_hold_minutes,
                "quality_score": quality_score,
                "quality_score_raw": quality_score,
                "quality_score_adjusted": quality_score,
                "estimated_slippage_bps": estimated_slippage_bps,
            },
        )

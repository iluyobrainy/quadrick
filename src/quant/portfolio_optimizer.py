"""
Portfolio-aware proposal selection and risk allocation.
"""

from __future__ import annotations

from typing import List

from .data_lake import QuantDataLake
from .types import EVProposal


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


class PortfolioOptimizer:
    def __init__(self, correlation_cap: float = 0.75, max_risk_budget_pct: float = 8.0):
        self.correlation_cap = float(correlation_cap)
        self.max_risk_budget_pct = float(max_risk_budget_pct)

    @staticmethod
    def _extract_open_symbols(open_positions: List[object]) -> List[str]:
        symbols: List[str] = []
        for p in open_positions or []:
            symbol = getattr(p, "symbol", None)
            size = getattr(p, "size", 0)
            try:
                if symbol and abs(float(size)) > 0:
                    symbols.append(str(symbol))
            except (TypeError, ValueError):
                continue
        return symbols

    @staticmethod
    def _extract_symbol_margin_pct(open_positions: List[object], account_balance: float) -> dict[str, float]:
        symbol_margin: dict[str, float] = {}
        if account_balance <= 0:
            return symbol_margin
        for p in open_positions or []:
            try:
                symbol = str(getattr(p, "symbol", "") or "")
                size = abs(float(getattr(p, "size", 0) or 0))
                mark = float(getattr(p, "mark_price", 0) or 0)
                leverage = max(1.0, float(getattr(p, "leverage", 1) or 1))
                if not symbol or size <= 0 or mark <= 0:
                    continue
                margin = (size * mark) / leverage
                symbol_margin[symbol] = symbol_margin.get(symbol, 0.0) + ((margin / account_balance) * 100.0)
            except (TypeError, ValueError):
                continue
        return symbol_margin

    def select_best(
        self,
        proposals: List[EVProposal],
        open_positions: List[object],
        data_lake: QuantDataLake,
        current_total_risk_pct: float,
        account_balance: float = 0.0,
        symbol_max_margin_pct: float = 15.0,
        portfolio_max_margin_pct: float = 35.0,
        allow_probe_override: bool = True,
    ) -> List[EVProposal]:
        if not proposals:
            return []
        open_symbols = self._extract_open_symbols(open_positions)
        symbol_margin_pct = self._extract_symbol_margin_pct(open_positions, account_balance)
        ranked: List[tuple[float, EVProposal]] = []

        for proposal in proposals:
            if proposal.entry_tier == "probe" and not allow_probe_override:
                continue
            correlation_penalty = 0.0
            blocked = False
            for open_symbol in open_symbols:
                corr = abs(data_lake.get_symbol_correlation(proposal.symbol, open_symbol))
                if corr >= self.correlation_cap:
                    blocked = True
                    break
                correlation_penalty = max(correlation_penalty, corr * 0.45)
            if blocked:
                continue

            concentration_penalty = 0.0
            if proposal.symbol in open_symbols:
                concentration_penalty = 0.35

            portfolio_cap = max(1.0, float(portfolio_max_margin_pct))
            budget_left = max(0.0, portfolio_cap - float(current_total_risk_pct))
            if budget_left <= 0:
                continue

            # Treat proposal risk percentage as an approximate incremental margin load proxy.
            # This keeps selection aligned with runtime portfolio caps.
            proposal_margin_pct = max(0.5, float(proposal.risk_pct))
            current_symbol_margin = float(symbol_margin_pct.get(proposal.symbol, 0.0))
            if (current_symbol_margin + proposal_margin_pct) > float(symbol_max_margin_pct):
                continue
            if proposal_margin_pct > budget_left:
                continue

            risk_scale = _clamp(budget_left / max(0.01, proposal_margin_pct), 0.35, 1.0)
            adj_edge = proposal.expected_edge_pct - correlation_penalty - concentration_penalty
            expectancy_hour = float(
                proposal.metadata.get("expectancy_per_hour_pct", proposal.expectancy_per_hour_pct)
                or 0.0
            )
            quality_score = float(proposal.quality_score or 0.0)
            quality_bonus = max(0.0, (quality_score - 55.0) / 100.0)
            policy_meta = (proposal.metadata or {}).get("policy", {}) or {}
            policy_state = str(policy_meta.get("state") or "green")
            policy_unstable = bool(policy_meta.get("unstable", False))
            tail_penalty = (
                float(policy_meta.get("tail_loss_3_rate", 0.0)) * 0.20
                + float(policy_meta.get("tail_loss_5_rate", 0.0)) * 0.35
                + float(policy_meta.get("tail_loss_7_rate", 0.0)) * 0.45
            )
            if policy_state == "red" and proposal.entry_tier != "probe":
                continue
            exploration_override = (
                adj_edge > -0.18
                and expectancy_hour >= -0.05
                and float(proposal.win_probability) >= 0.50
                and float(proposal.confidence) >= 0.48
            )
            if proposal.entry_tier == "probe":
                exploration_override = True
            if adj_edge <= 0 and not exploration_override:
                continue

            proposal.risk_pct = _clamp(proposal.risk_pct * risk_scale, 0.5, max(0.5, budget_left))
            effective_edge = adj_edge
            if adj_edge <= 0 and exploration_override:
                effective_edge = max(0.005, (expectancy_hour * 0.30) + (adj_edge * 0.25))
            objective_score = (
                (0.50 * expectancy_hour)
                + (0.35 * effective_edge)
                + (0.15 * quality_bonus)
            ) * float(proposal.symbol_weight or 1.0)
            objective_score -= tail_penalty
            if policy_unstable:
                objective_score -= 0.08
            proposal.expectancy_per_hour_pct = expectancy_hour
            proposal.metadata["portfolio_correlation_penalty"] = correlation_penalty
            proposal.metadata["portfolio_concentration_penalty"] = concentration_penalty
            proposal.metadata["portfolio_budget_left_pct"] = budget_left
            proposal.metadata["portfolio_adjusted_edge_pct"] = adj_edge
            proposal.metadata["portfolio_effective_edge_pct"] = effective_edge
            proposal.metadata["portfolio_exploration_override"] = exploration_override
            proposal.metadata["portfolio_objective_score"] = objective_score
            proposal.metadata["portfolio_symbol_margin_pct"] = current_symbol_margin
            proposal.metadata["portfolio_proposal_margin_pct"] = proposal_margin_pct
            proposal.metadata["portfolio_symbol_cap_pct"] = float(symbol_max_margin_pct)
            proposal.metadata["portfolio_cap_pct"] = portfolio_cap
            ranked.append((objective_score, proposal))

        ranked.sort(key=lambda x: x[0], reverse=True)
        return [p for _, p in ranked]

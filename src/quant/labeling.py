"""
Label generation for multi-horizon quant training.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple


class BarrierLabeler:
    def __init__(self, fee_buffer_pct: float = 0.08):
        self.fee_buffer_pct = max(0.0, float(fee_buffer_pct))

    @staticmethod
    def _realized_volatility(prices: List[float]) -> float:
        if len(prices) < 3:
            return 0.0
        log_returns: List[float] = []
        for i in range(1, len(prices)):
            prev = prices[i - 1]
            cur = prices[i]
            if prev <= 0 or cur <= 0:
                continue
            log_returns.append(math.log(cur / prev))
        if len(log_returns) < 2:
            return 0.0
        mean_lr = sum(log_returns) / len(log_returns)
        var = sum((x - mean_lr) ** 2 for x in log_returns) / (len(log_returns) - 1)
        # Express as percentage move scale for the horizon window.
        return max(0.0, math.sqrt(var) * math.sqrt(len(log_returns)) * 100.0)

    def label_path(
        self,
        entry_price: float,
        price_path: List[Tuple[str, float]],
        volatility_pct: float,
    ) -> Dict[str, float]:
        prices = [float(p) for _, p in price_path if float(p) > 0]
        if entry_price <= 0 or len(prices) < 2:
            return {
                "direction_label": 0,
                "move_pct": 0.0,
                "tp_hit_first": 0,
                "sl_hit_first": 0,
                "realized_volatility": 0.0,
            }

        final_price = prices[-1]
        move_pct = ((final_price - entry_price) / entry_price) * 100.0
        vol_barrier_pct = max(0.08, min(3.5, max(volatility_pct * 0.85, 0.12)))
        upper = entry_price * (1.0 + vol_barrier_pct / 100.0)
        lower = entry_price * (1.0 - vol_barrier_pct / 100.0)

        tp_hit_first = 0
        sl_hit_first = 0
        for p in prices[1:]:
            if p >= upper:
                tp_hit_first = 1
                break
            if p <= lower:
                sl_hit_first = 1
                break

        fee = self.fee_buffer_pct
        if move_pct > fee:
            direction_label = 1
        elif move_pct < -fee:
            direction_label = -1
        else:
            direction_label = 0

        return {
            "direction_label": direction_label,
            "move_pct": move_pct,
            "tp_hit_first": tp_hit_first,
            "sl_hit_first": sl_hit_first,
            "realized_volatility": self._realized_volatility(prices),
        }

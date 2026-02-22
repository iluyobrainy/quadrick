"""
Uncertainty gating for trade admission.
"""

from __future__ import annotations

import math
from typing import Dict


class UncertaintyGate:
    def __init__(self, min_confidence: float = 0.56, max_uncertainty: float = 0.58):
        self.min_confidence = float(min_confidence)
        self.max_uncertainty = float(max_uncertainty)

    @staticmethod
    def _entropy(prob: float) -> float:
        p = max(1e-6, min(1.0 - 1e-6, float(prob)))
        h = -(p * math.log(p, 2) + (1.0 - p) * math.log(1.0 - p, 2))
        return max(0.0, min(1.0, h))

    def score(
        self,
        prob_calibrated: float,
        calibration_error: float,
        drift_score: float,
        volatility_pct: float,
    ) -> float:
        entropy = self._entropy(prob_calibrated)
        calib = max(0.0, min(1.0, float(calibration_error)))
        drift = max(0.0, min(1.0, float(drift_score)))
        vol = max(0.0, min(1.0, float(volatility_pct) / 5.0))
        uncertainty = (0.45 * entropy) + (0.25 * calib) + (0.20 * drift) + (0.10 * vol)
        return max(0.0, min(1.0, uncertainty))

    def allow(
        self,
        prob_calibrated: float,
        uncertainty: float,
        min_confidence_override: float | None = None,
        max_uncertainty_override: float | None = None,
    ) -> Dict[str, object]:
        p = max(0.0, min(1.0, float(prob_calibrated)))
        conf = max(p, 1.0 - p)
        min_confidence = (
            self.min_confidence
            if min_confidence_override is None
            else float(min_confidence_override)
        )
        max_uncertainty = (
            self.max_uncertainty
            if max_uncertainty_override is None
            else float(max_uncertainty_override)
        )
        if conf < min_confidence:
            return {
                "allowed": False,
                "reason": f"confidence {conf:.3f} < min {min_confidence:.3f}",
                "confidence": conf,
                "min_confidence": min_confidence,
            }
        if uncertainty > max_uncertainty:
            return {
                "allowed": False,
                "reason": f"uncertainty {uncertainty:.3f} > max {max_uncertainty:.3f}",
                "confidence": conf,
                "max_uncertainty": max_uncertainty,
            }
        return {
            "allowed": True,
            "reason": "ok",
            "confidence": conf,
            "min_confidence": min_confidence,
            "max_uncertainty": max_uncertainty,
        }

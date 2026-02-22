"""
Walk-forward retraining and calibration refresh.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

from .calibration import ProbabilityCalibrator
from .data_lake import QuantDataLake
from .regime_switcher import RegimeSwitcher


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class WalkForwardTrainer:
    def __init__(
        self,
        data_lake: QuantDataLake,
        regime_switcher: RegimeSwitcher,
        calibrators: Dict[int, ProbabilityCalibrator],
        retrain_interval_minutes: int,
        lookback_rows: int,
    ):
        self.data_lake = data_lake
        self.regime_switcher = regime_switcher
        self.calibrators = calibrators
        self.retrain_interval_minutes = int(max(15, retrain_interval_minutes))
        self.lookback_rows = int(max(200, lookback_rows))
        self.last_retrain_at: Optional[datetime] = None

    def _parse_features(self, raw: str) -> Dict[str, float]:
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return {str(k): float(v) for k, v in parsed.items()}
        except Exception:
            pass
        return {}

    def maybe_retrain(self, drift_score: float, force: bool = False) -> Optional[Dict[str, object]]:
        now = _utc_now()
        due = (
            self.last_retrain_at is None
            or now >= (self.last_retrain_at + timedelta(minutes=self.retrain_interval_minutes))
        )
        if not force and not due and drift_score < 0.6:
            return None

        report: Dict[str, object] = {
            "started_at_utc": now.isoformat(),
            "drift_score": float(drift_score),
            "horizons": {},
        }

        for horizon in [5, 15, 30]:
            rows = self.data_lake.get_training_rows(horizon_min=horizon, limit=self.lookback_rows)
            if len(rows) < 150:
                report["horizons"][horizon] = {"status": "insufficient_samples", "samples": len(rows)}
                continue

            # Oldest -> newest for walk-forward split.
            rows = list(reversed(rows))
            split = int(len(rows) * 0.7)
            train_rows = rows[:split]
            val_rows = rows[split:]
            if len(train_rows) < 80 or len(val_rows) < 20:
                report["horizons"][horizon] = {"status": "insufficient_split", "samples": len(rows)}
                continue

            train_features: List[Dict[str, float]] = []
            train_labels: List[Dict[str, float]] = []
            train_regimes: List[str] = []
            for r in train_rows:
                train_features.append(self._parse_features(str(r["features_json"])))
                train_labels.append(
                    {
                        "direction_label": int(r["direction_label"]),
                        "move_pct": float(r["move_pct"]),
                        "tp_hit_first": int(r["tp_hit_first"]),
                        "sl_hit_first": int(r["sl_hit_first"]),
                        "realized_volatility": float(r["realized_volatility"]),
                    }
                )
                train_regimes.append(str(r["regime"]))

            self.regime_switcher.fit_walk_forward(
                horizon=horizon,
                rows=train_features,
                labels=train_labels,
                regimes=train_regimes,
                epochs=2,
            )

            # Validation + calibrator refresh.
            correct = 0
            total = 0
            self.calibrators[horizon].samples.clear()
            for r in val_rows:
                features = self._parse_features(str(r["features_json"]))
                regime = str(r["regime"])
                pred = self.regime_switcher.predict(regime=regime, horizon=horizon, features=features)
                direction = int(r["direction_label"])
                if direction == 0:
                    continue
                y = 1 if direction > 0 else 0
                raw_p = float(pred["prob_up_raw"])
                self.calibrators[horizon].add_sample(raw_p, y)
                pred_y = 1 if raw_p >= 0.5 else 0
                if pred_y == y:
                    correct += 1
                total += 1
            self.calibrators[horizon].fit()

            report["horizons"][horizon] = {
                "status": "ok",
                "samples": len(rows),
                "train_samples": len(train_rows),
                "val_samples": len(val_rows),
                "val_direction_accuracy": (correct / total) if total > 0 else None,
                "calibration_error": float(self.calibrators[horizon].last_error),
            }

        report["finished_at_utc"] = _utc_now().isoformat()
        self.last_retrain_at = now
        return report

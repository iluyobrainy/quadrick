"""
SQLite-backed market/feature/label data lake for quant runtime.
"""

from __future__ import annotations

import json
import math
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


def _parse_ts(value: Any) -> datetime:
    if isinstance(value, datetime):
        dt = value
    else:
        raw = str(value or "")
        raw = raw.replace("Z", "+00:00")
        dt = datetime.fromisoformat(raw)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _merge_cutoff(cutoff: str, since_ts: Optional[Any]) -> str:
    if since_ts is None:
        return cutoff
    try:
        cutoff_dt = _parse_ts(cutoff)
        since_dt = _parse_ts(since_ts)
        return _iso(max(cutoff_dt, since_dt))
    except Exception:
        return cutoff


class QuantDataLake:
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA synchronous=NORMAL;")
        self._init_schema()

    def _init_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS market_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                symbol TEXT NOT NULL,
                price REAL NOT NULL,
                bid_price REAL,
                ask_price REAL,
                volume_24h REAL,
                funding_rate REAL,
                open_interest REAL,
                basis REAL,
                spread_bps REAL,
                bid_depth_10 REAL,
                ask_depth_10 REAL,
                bid_ask_imbalance REAL,
                reject_streak INTEGER DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_market_snapshots_symbol_ts ON market_snapshots(symbol, ts);

            CREATE TABLE IF NOT EXISTS feature_rows (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                symbol TEXT NOT NULL,
                price REAL NOT NULL,
                regime TEXT NOT NULL,
                features_json TEXT NOT NULL,
                label_status_5m INTEGER NOT NULL DEFAULT 0,
                label_status_15m INTEGER NOT NULL DEFAULT 0,
                label_status_30m INTEGER NOT NULL DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_feature_rows_symbol_ts ON feature_rows(symbol, ts);

            CREATE TABLE IF NOT EXISTS labels (
                feature_id INTEGER NOT NULL,
                horizon_min INTEGER NOT NULL,
                direction_label INTEGER NOT NULL,
                move_pct REAL NOT NULL,
                tp_hit_first INTEGER NOT NULL,
                sl_hit_first INTEGER NOT NULL,
                realized_volatility REAL NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (feature_id, horizon_min)
            );
            CREATE INDEX IF NOT EXISTS idx_labels_horizon ON labels(horizon_min);

            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                feature_id INTEGER NOT NULL,
                symbol TEXT NOT NULL,
                horizon_min INTEGER NOT NULL,
                regime TEXT NOT NULL,
                prob_up_raw REAL NOT NULL,
                prob_up_calibrated REAL NOT NULL,
                expected_move_pct REAL NOT NULL,
                volatility_pct REAL NOT NULL,
                uncertainty REAL NOT NULL,
                expert TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_predictions_feature_horizon ON predictions(feature_id, horizon_min);

            CREATE TABLE IF NOT EXISTS execution_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                symbol TEXT,
                event_type TEXT NOT NULL,
                reason_code TEXT,
                pnl_pct REAL,
                slippage_bps REAL,
                latency_ms REAL,
                details_json TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_execution_events_ts ON execution_events(ts);
            CREATE INDEX IF NOT EXISTS idx_execution_events_type ON execution_events(event_type);

            CREATE TABLE IF NOT EXISTS quant_monitor (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                metrics_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_quant_monitor_ts ON quant_monitor(ts);

            CREATE TABLE IF NOT EXISTS quant_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                severity TEXT NOT NULL,
                code TEXT NOT NULL,
                message TEXT NOT NULL,
                details_json TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_quant_alerts_ts ON quant_alerts(ts);
            CREATE INDEX IF NOT EXISTS idx_quant_alerts_code ON quant_alerts(code);

            CREATE TABLE IF NOT EXISTS model_state (
                key TEXT PRIMARY KEY,
                value_json TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            """
        )
        self._ensure_column("predictions", "sample_strength", "REAL")
        self.conn.commit()

    def _ensure_column(self, table: str, column: str, definition: str) -> None:
        cur = self.conn.execute(f"PRAGMA table_info({table})")
        existing = {str(r["name"]) for r in cur.fetchall()}
        if column in existing:
            return
        self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

    @staticmethod
    def _label_column(horizon_min: int) -> str:
        mapping = {5: "label_status_5m", 15: "label_status_15m", 30: "label_status_30m"}
        if horizon_min not in mapping:
            raise ValueError(f"Unsupported horizon: {horizon_min}")
        return mapping[horizon_min]

    def record_market_snapshots(self, rows: Iterable[Dict[str, Any]]) -> None:
        payload = []
        for row in rows:
            payload.append(
                (
                    str(row.get("ts")),
                    str(row.get("symbol") or ""),
                    float(row.get("price") or 0.0),
                    float(row.get("bid_price") or 0.0),
                    float(row.get("ask_price") or 0.0),
                    float(row.get("volume_24h") or 0.0),
                    float(row.get("funding_rate") or 0.0),
                    float(row.get("open_interest") or 0.0),
                    float(row.get("basis") or 0.0),
                    float(row.get("spread_bps") or 0.0),
                    float(row.get("bid_depth_10") or 0.0),
                    float(row.get("ask_depth_10") or 0.0),
                    float(row.get("bid_ask_imbalance") or 1.0),
                    int(row.get("reject_streak") or 0),
                )
            )
        if not payload:
            return
        self.conn.executemany(
            """
            INSERT INTO market_snapshots(
                ts, symbol, price, bid_price, ask_price, volume_24h, funding_rate,
                open_interest, basis, spread_bps, bid_depth_10, ask_depth_10,
                bid_ask_imbalance, reject_streak
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            payload,
        )
        self.conn.commit()

    def insert_feature_row(
        self,
        ts: str,
        symbol: str,
        price: float,
        regime: str,
        features: Dict[str, float],
    ) -> int:
        cur = self.conn.execute(
            """
            INSERT INTO feature_rows(ts, symbol, price, regime, features_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            (str(ts), str(symbol), float(price), str(regime), json.dumps(features)),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def get_mature_feature_rows(self, horizon_min: int, now_utc: datetime, limit: int = 300) -> List[sqlite3.Row]:
        label_col = self._label_column(horizon_min)
        cutoff = _iso(now_utc - timedelta(minutes=horizon_min))
        cur = self.conn.execute(
            f"""
            SELECT id, ts, symbol, price, regime, features_json
            FROM feature_rows
            WHERE {label_col} = 0 AND ts <= ?
            ORDER BY ts ASC
            LIMIT ?
            """,
            (cutoff, int(limit)),
        )
        return cur.fetchall()

    def mark_labeled(self, feature_id: int, horizon_min: int) -> None:
        label_col = self._label_column(horizon_min)
        self.conn.execute(
            f"UPDATE feature_rows SET {label_col} = 1 WHERE id = ?",
            (int(feature_id),),
        )
        self.conn.commit()

    def insert_label(self, feature_id: int, horizon_min: int, label: Dict[str, Any]) -> None:
        self.conn.execute(
            """
            INSERT OR REPLACE INTO labels(
                feature_id, horizon_min, direction_label, move_pct,
                tp_hit_first, sl_hit_first, realized_volatility, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                int(feature_id),
                int(horizon_min),
                int(label.get("direction_label") or 0),
                float(label.get("move_pct") or 0.0),
                int(1 if label.get("tp_hit_first") else 0),
                int(1 if label.get("sl_hit_first") else 0),
                float(label.get("realized_volatility") or 0.0),
                _iso(_utc_now()),
            ),
        )
        self.conn.commit()

    def get_price_path(self, symbol: str, start_ts: str, end_ts: str) -> List[Tuple[str, float]]:
        cur = self.conn.execute(
            """
            SELECT ts, price
            FROM market_snapshots
            WHERE symbol = ? AND ts >= ? AND ts <= ?
            ORDER BY ts ASC
            """,
            (str(symbol), str(start_ts), str(end_ts)),
        )
        return [(str(r["ts"]), float(r["price"])) for r in cur.fetchall()]

    def save_prediction(self, payload: Dict[str, Any]) -> None:
        self.conn.execute(
            """
            INSERT INTO predictions(
                ts, feature_id, symbol, horizon_min, regime, prob_up_raw,
                prob_up_calibrated, expected_move_pct, volatility_pct, uncertainty, expert, sample_strength
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(payload.get("ts")),
                int(payload.get("feature_id")),
                str(payload.get("symbol")),
                int(payload.get("horizon_min")),
                str(payload.get("regime") or "unknown"),
                float(payload.get("prob_up_raw") or 0.5),
                float(payload.get("prob_up_calibrated") or 0.5),
                float(payload.get("expected_move_pct") or 0.0),
                float(payload.get("volatility_pct") or 0.0),
                float(payload.get("uncertainty") or 1.0),
                str(payload.get("expert") or "unknown"),
                float(payload.get("sample_strength") or 0.0),
            ),
        )
        self.conn.commit()

    def get_prediction_for_feature(self, feature_id: int, horizon_min: int) -> Optional[sqlite3.Row]:
        cur = self.conn.execute(
            """
            SELECT *
            FROM predictions
            WHERE feature_id = ? AND horizon_min = ?
            ORDER BY id DESC
            LIMIT 1
            """,
            (int(feature_id), int(horizon_min)),
        )
        return cur.fetchone()

    def get_training_rows(self, horizon_min: int, limit: int = 4000) -> List[sqlite3.Row]:
        cur = self.conn.execute(
            """
            SELECT
                f.id,
                f.ts,
                f.symbol,
                f.regime,
                f.features_json,
                l.direction_label,
                l.move_pct,
                l.tp_hit_first,
                l.sl_hit_first,
                l.realized_volatility
            FROM labels l
            JOIN feature_rows f ON f.id = l.feature_id
            WHERE l.horizon_min = ?
            ORDER BY f.ts DESC
            LIMIT ?
            """,
            (int(horizon_min), int(limit)),
        )
        return cur.fetchall()

    def save_execution_event(self, event: Dict[str, Any]) -> None:
        self.conn.execute(
            """
            INSERT INTO execution_events(
                ts, symbol, event_type, reason_code, pnl_pct, slippage_bps, latency_ms, details_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(event.get("ts") or _iso(_utc_now())),
                event.get("symbol"),
                str(event.get("event_type") or "unknown"),
                event.get("reason_code"),
                float(event.get("pnl_pct") or 0.0) if event.get("pnl_pct") is not None else None,
                float(event.get("slippage_bps") or 0.0) if event.get("slippage_bps") is not None else None,
                float(event.get("latency_ms") or 0.0) if event.get("latency_ms") is not None else None,
                json.dumps(event.get("details") or {}),
            ),
        )
        self.conn.commit()

    def get_recent_execution_metrics(self, minutes: int = 90, since_ts: Optional[Any] = None) -> Dict[str, float]:
        cutoff = _iso(_utc_now() - timedelta(minutes=max(1, int(minutes))))
        cutoff = _merge_cutoff(cutoff, since_ts)
        cur = self.conn.execute(
            """
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN event_type = 'reject' THEN 1 ELSE 0 END) as rejects,
                AVG(CASE WHEN slippage_bps IS NOT NULL THEN slippage_bps END) as avg_slippage_bps
            FROM execution_events
            WHERE ts >= ?
            """,
            (cutoff,),
        )
        row = cur.fetchone()
        total = float((row["total"] or 0) if row else 0)
        rejects = float((row["rejects"] or 0) if row else 0)
        avg_slippage = float((row["avg_slippage_bps"] or 0) if row else 0)
        reject_rate = (rejects / total) if total > 0 else 0.0
        return {
            "events": total,
            "rejects": rejects,
            "reject_rate": reject_rate,
            "avg_slippage_bps": avg_slippage,
        }

    def get_recent_expectancy_metrics(self, minutes: int = 360, since_ts: Optional[Any] = None) -> Dict[str, float]:
        cutoff = _iso(_utc_now() - timedelta(minutes=max(15, int(minutes))))
        cutoff = _merge_cutoff(cutoff, since_ts)
        cur = self.conn.execute(
            """
            SELECT ts, pnl_pct
            FROM execution_events
            WHERE ts >= ? AND event_type = 'close' AND pnl_pct IS NOT NULL
            ORDER BY ts ASC
            """,
            (cutoff,),
        )
        rows = cur.fetchall()
        pnls = [float(r["pnl_pct"]) for r in rows]
        closed_trades = len(pnls)
        window_hours = max(1.0 / 60.0, minutes / 60.0)
        trades_per_hour = float(closed_trades) / window_hours
        expectancy_per_trade = (sum(pnls) / closed_trades) if closed_trades > 0 else 0.0
        expectancy_per_hour = trades_per_hour * expectancy_per_trade
        wins = len([p for p in pnls if p > 0])
        win_rate = (wins / closed_trades) if closed_trades > 0 else 0.0
        return {
            "closed_trades": float(closed_trades),
            "trades_per_hour": float(trades_per_hour),
            "expectancy_per_trade_pct": float(expectancy_per_trade),
            "expectancy_per_hour_pct": float(expectancy_per_hour),
            "win_rate": float(win_rate),
        }

    def get_recent_close_distribution(self, minutes: int = 360, since_ts: Optional[Any] = None) -> Dict[str, float]:
        cutoff = _iso(_utc_now() - timedelta(minutes=max(15, int(minutes))))
        cutoff = _merge_cutoff(cutoff, since_ts)
        cur = self.conn.execute(
            """
            SELECT pnl_pct
            FROM execution_events
            WHERE ts >= ? AND event_type = 'close' AND pnl_pct IS NOT NULL
            ORDER BY ts ASC
            """,
            (cutoff,),
        )
        pnls = [float(r["pnl_pct"]) for r in cur.fetchall()]
        n = len(pnls)
        if n <= 0:
            return {
                "closed_trades": 0.0,
                "expectancy_per_trade_pct": 0.0,
                "avg_win_pct": 0.0,
                "avg_loss_pct": 0.0,
                "tail_loss_3_rate": 0.0,
                "tail_loss_5_rate": 0.0,
                "tail_loss_7_rate": 0.0,
            }

        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        return {
            "closed_trades": float(n),
            "expectancy_per_trade_pct": float(sum(pnls) / n),
            "avg_win_pct": float(sum(wins) / len(wins)) if wins else 0.0,
            "avg_loss_pct": float(sum(losses) / len(losses)) if losses else 0.0,
            "tail_loss_3_rate": float(len([p for p in pnls if p <= -3.0]) / n),
            "tail_loss_5_rate": float(len([p for p in pnls if p <= -5.0]) / n),
            "tail_loss_7_rate": float(len([p for p in pnls if p <= -7.0]) / n),
        }

    def get_recent_fill_cost_profile(
        self,
        symbol: str,
        minutes: int = 360,
        regime: Optional[str] = None,
        entry_tier: Optional[str] = None,
        limit: int = 300,
        since_ts: Optional[Any] = None,
    ) -> Dict[str, float]:
        cutoff = _iso(_utc_now() - timedelta(minutes=max(15, int(minutes))))
        cutoff = _merge_cutoff(cutoff, since_ts)
        cur = self.conn.execute(
            """
            SELECT slippage_bps, details_json
            FROM execution_events
            WHERE ts >= ?
              AND event_type = 'fill'
              AND symbol = ?
              AND slippage_bps IS NOT NULL
            ORDER BY id DESC
            LIMIT ?
            """,
            (cutoff, str(symbol), int(max(10, limit))),
        )
        slips: List[float] = []
        regime_filter = str(regime or "").strip().lower()
        tier_filter = str(entry_tier or "").strip().lower()
        for row in cur.fetchall():
            try:
                details = json.loads(str(row["details_json"] or "{}"))
            except (TypeError, ValueError, json.JSONDecodeError):
                details = {}
            row_regime = str(
                details.get("regime")
                or details.get("market_regime")
                or "unknown"
            ).strip().lower()
            row_tier = str(details.get("entry_tier") or "full").strip().lower()
            if regime_filter and row_regime != regime_filter:
                continue
            if tier_filter and row_tier != tier_filter:
                continue
            try:
                slips.append(float(row["slippage_bps"] or 0.0))
            except (TypeError, ValueError):
                continue

        if not slips:
            return {
                "samples": 0.0,
                "avg_slippage_bps": 0.0,
                "median_slippage_bps": 0.0,
                "p75_slippage_bps": 0.0,
                "p90_slippage_bps": 0.0,
            }

        slips_sorted = sorted(slips)
        n = len(slips_sorted)

        def _pct(p: float) -> float:
            idx = int(max(0, min(n - 1, round((n - 1) * p))))
            return float(slips_sorted[idx])

        mid = n // 2
        if n % 2 == 1:
            median = slips_sorted[mid]
        else:
            median = (slips_sorted[mid - 1] + slips_sorted[mid]) / 2.0

        return {
            "samples": float(n),
            "avg_slippage_bps": float(sum(slips_sorted) / n),
            "median_slippage_bps": float(median),
            "p75_slippage_bps": _pct(0.75),
            "p90_slippage_bps": _pct(0.90),
        }

    def get_symbol_expectancy_metrics(self, symbol: str, minutes: int = 120) -> Dict[str, float]:
        cutoff = _iso(_utc_now() - timedelta(minutes=max(15, int(minutes))))
        cur = self.conn.execute(
            """
            SELECT pnl_pct
            FROM execution_events
            WHERE ts >= ? AND event_type = 'close' AND symbol = ? AND pnl_pct IS NOT NULL
            ORDER BY ts ASC
            """,
            (cutoff, str(symbol)),
        )
        pnls = [float(r["pnl_pct"]) for r in cur.fetchall()]
        closed = len(pnls)
        win_rate = (len([p for p in pnls if p > 0]) / closed) if closed > 0 else 0.0
        expectancy = (sum(pnls) / closed) if closed > 0 else 0.0
        return {
            "closed_trades": float(closed),
            "expectancy_per_trade_pct": float(expectancy),
            "win_rate": float(win_rate),
        }

    def get_recent_symbol_regimes(self, symbol: str, limit: int = 12) -> List[str]:
        cur = self.conn.execute(
            """
            SELECT regime
            FROM feature_rows
            WHERE symbol = ?
            ORDER BY ts DESC
            LIMIT ?
            """,
            (str(symbol), int(max(1, limit))),
        )
        out: List[str] = []
        for row in cur.fetchall():
            regime = str(row["regime"] or "").strip().lower()
            if regime:
                out.append(regime)
        return out

    def count_execution_events(
        self,
        event_type: Optional[str] = None,
        reason_code: Optional[str] = None,
        minutes: int = 60,
        since_ts: Optional[Any] = None,
    ) -> int:
        cutoff = _iso(_utc_now() - timedelta(minutes=max(1, int(minutes))))
        cutoff = _merge_cutoff(cutoff, since_ts)
        sql = """
            SELECT COUNT(*) as n
            FROM execution_events
            WHERE ts >= ?
        """
        args: List[Any] = [cutoff]
        if event_type:
            sql += " AND event_type = ?"
            args.append(str(event_type))
        if reason_code:
            sql += " AND reason_code = ?"
            args.append(str(reason_code))
        cur = self.conn.execute(sql, tuple(args))
        row = cur.fetchone()
        return int((row["n"] or 0) if row else 0)

    def get_recent_reject_reason_counts(self, minutes: int = 120, since_ts: Optional[Any] = None) -> Dict[str, int]:
        cutoff = _iso(_utc_now() - timedelta(minutes=max(1, int(minutes))))
        cutoff = _merge_cutoff(cutoff, since_ts)
        cur = self.conn.execute(
            """
            SELECT COALESCE(reason_code, 'unknown') as reason_code, COUNT(*) as n
            FROM execution_events
            WHERE ts >= ? AND event_type = 'reject'
            GROUP BY COALESCE(reason_code, 'unknown')
            """,
            (cutoff,),
        )
        return {str(r["reason_code"]): int(r["n"] or 0) for r in cur.fetchall()}

    def get_bucket_performance(
        self,
        minutes: int = 1440,
        half_life_minutes: int = 720,
    ) -> Dict[str, Dict[str, float]]:
        cutoff_dt = _utc_now() - timedelta(minutes=max(15, int(minutes)))
        cutoff = _iso(cutoff_dt)
        cur = self.conn.execute(
            """
            SELECT ts, symbol, pnl_pct, slippage_bps, details_json
            FROM execution_events
            WHERE ts >= ? AND event_type = 'close' AND pnl_pct IS NOT NULL
            ORDER BY ts ASC
            """,
            (cutoff,),
        )
        buckets: Dict[str, Dict[str, float]] = {}
        hl = max(1.0, float(half_life_minutes))
        now_dt = _utc_now()

        for row in cur.fetchall():
            symbol = str(row["symbol"] or "").strip().upper()
            if not symbol:
                continue
            pnl = float(row["pnl_pct"] or 0.0)
            slip = float(row["slippage_bps"] or 0.0)
            try:
                details = json.loads(str(row["details_json"] or "{}"))
            except (TypeError, ValueError, json.JSONDecodeError):
                details = {}
            side = str(
                details.get("side")
                or details.get("entry_side")
                or details.get("position_side")
                or "unknown"
            ).strip().title()
            if side not in {"Buy", "Sell"}:
                side = "unknown"
            regime = str(
                details.get("regime")
                or details.get("market_regime")
                or details.get("bucket_regime")
                or "unknown"
            ).strip().lower()
            if not regime:
                regime = "unknown"
            bucket_key = f"{symbol}|{side}|{regime}"

            try:
                ts_dt = _parse_ts(row["ts"])
                age_minutes = max(0.0, (now_dt - ts_dt).total_seconds() / 60.0)
            except Exception:
                age_minutes = 0.0
            decay_weight = math.pow(0.5, age_minutes / hl)
            decay_weight = max(0.02, min(1.0, decay_weight))

            stats = buckets.setdefault(
                bucket_key,
                {
                    "symbol": symbol,
                    "side": side,
                    "regime": regime,
                    "closed_trades": 0.0,
                    "weighted_closed_trades": 0.0,
                    "weighted_wins": 0.0,
                    "weighted_losses": 0.0,
                    "weighted_pnl_sum": 0.0,
                    "weighted_slippage_sum": 0.0,
                    "tail_loss_3": 0.0,
                    "tail_loss_5": 0.0,
                    "tail_loss_7": 0.0,
                },
            )

            stats["closed_trades"] += 1.0
            stats["weighted_closed_trades"] += decay_weight
            stats["weighted_pnl_sum"] += pnl * decay_weight
            stats["weighted_slippage_sum"] += slip * decay_weight
            if pnl > 0:
                stats["weighted_wins"] += decay_weight
            elif pnl < 0:
                stats["weighted_losses"] += decay_weight
            if pnl <= -3.0:
                stats["tail_loss_3"] += decay_weight
            if pnl <= -5.0:
                stats["tail_loss_5"] += decay_weight
            if pnl <= -7.0:
                stats["tail_loss_7"] += decay_weight

        output: Dict[str, Dict[str, float]] = {}
        for key, stats in buckets.items():
            weighted_closed = max(0.0, float(stats.get("weighted_closed_trades", 0.0)))
            if weighted_closed <= 0:
                continue
            weighted_wins = float(stats.get("weighted_wins", 0.0))
            weighted_losses = float(stats.get("weighted_losses", 0.0))
            weighted_pnl_sum = float(stats.get("weighted_pnl_sum", 0.0))
            weighted_slippage_sum = float(stats.get("weighted_slippage_sum", 0.0))
            output[key] = {
                "symbol": str(stats.get("symbol", "")),
                "side": str(stats.get("side", "unknown")),
                "regime": str(stats.get("regime", "unknown")),
                "closed_trades": float(stats.get("closed_trades", 0.0)),
                "weighted_closed_trades": weighted_closed,
                "weighted_win_rate": weighted_wins / weighted_closed,
                "weighted_loss_rate": weighted_losses / weighted_closed,
                "weighted_expectancy_pct": weighted_pnl_sum / weighted_closed,
                "weighted_avg_slippage_bps": weighted_slippage_sum / weighted_closed,
                "tail_loss_3_rate": float(stats.get("tail_loss_3", 0.0)) / weighted_closed,
                "tail_loss_5_rate": float(stats.get("tail_loss_5", 0.0)) / weighted_closed,
                "tail_loss_7_rate": float(stats.get("tail_loss_7", 0.0)) / weighted_closed,
            }
        return output

    def minutes_since_last_event(
        self,
        event_type: Optional[str] = None,
        reason_code: Optional[str] = None,
        symbol: Optional[str] = None,
    ) -> Optional[float]:
        sql = """
            SELECT ts
            FROM execution_events
            WHERE 1=1
        """
        args: List[Any] = []
        if event_type:
            sql += " AND event_type = ?"
            args.append(str(event_type))
        if reason_code:
            sql += " AND reason_code = ?"
            args.append(str(reason_code))
        if symbol:
            sql += " AND symbol = ?"
            args.append(str(symbol))
        sql += " ORDER BY id DESC LIMIT 1"
        cur = self.conn.execute(sql, tuple(args))
        row = cur.fetchone()
        if not row:
            return None
        try:
            ts = _parse_ts(row["ts"])
            return max(0.0, (_utc_now() - ts).total_seconds() / 60.0)
        except Exception:
            return None

    def get_barrier_profile(
        self,
        symbol: str,
        regime: str,
        horizon_min: int,
        lookback: int = 1200,
    ) -> Dict[str, float]:
        cur = self.conn.execute(
            """
            SELECT
                l.tp_hit_first,
                l.sl_hit_first,
                l.move_pct,
                l.realized_volatility
            FROM labels l
            JOIN feature_rows f ON f.id = l.feature_id
            WHERE l.horizon_min = ?
              AND f.symbol = ?
              AND f.regime = ?
            ORDER BY f.ts DESC
            LIMIT ?
            """,
            (int(horizon_min), str(symbol), str(regime), int(max(30, lookback))),
        )
        rows = cur.fetchall()
        if len(rows) < 25:
            cur = self.conn.execute(
                """
                SELECT
                    l.tp_hit_first,
                    l.sl_hit_first,
                    l.move_pct,
                    l.realized_volatility
                FROM labels l
                JOIN feature_rows f ON f.id = l.feature_id
                WHERE l.horizon_min = ?
                  AND f.symbol = ?
                ORDER BY f.ts DESC
                LIMIT ?
                """,
                (int(horizon_min), str(symbol), int(max(30, lookback))),
            )
            rows = cur.fetchall()
        sample_count = len(rows)
        if sample_count == 0:
            return {
                "samples": 0.0,
                "upper_hit_rate": 0.5,
                "lower_hit_rate": 0.5,
                "avg_move_pct": 0.0,
                "avg_realized_volatility": 0.0,
            }
        upper = sum(int(r["tp_hit_first"] or 0) for r in rows) / sample_count
        lower = sum(int(r["sl_hit_first"] or 0) for r in rows) / sample_count
        avg_move = sum(float(r["move_pct"] or 0.0) for r in rows) / sample_count
        avg_vol = sum(float(r["realized_volatility"] or 0.0) for r in rows) / sample_count
        return {
            "samples": float(sample_count),
            "upper_hit_rate": float(upper),
            "lower_hit_rate": float(lower),
            "avg_move_pct": float(avg_move),
            "avg_realized_volatility": float(avg_vol),
        }

    def save_monitor_metrics(self, metrics: Dict[str, Any]) -> None:
        self.conn.execute(
            "INSERT INTO quant_monitor(ts, metrics_json) VALUES (?, ?)",
            (_iso(_utc_now()), json.dumps(metrics)),
        )
        self.conn.commit()

    def save_alert(
        self,
        severity: str,
        code: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO quant_alerts(ts, severity, code, message, details_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                _iso(_utc_now()),
                str(severity),
                str(code),
                str(message),
                json.dumps(details or {}),
            ),
        )
        self.conn.commit()

    def get_recent_alerts(self, limit: int = 100) -> List[Dict[str, Any]]:
        cur = self.conn.execute(
            """
            SELECT ts, severity, code, message, details_json
            FROM quant_alerts
            ORDER BY id DESC
            LIMIT ?
            """,
            (int(max(1, limit)),),
        )
        out: List[Dict[str, Any]] = []
        for row in cur.fetchall():
            try:
                details = json.loads(str(row["details_json"] or "{}"))
            except (TypeError, ValueError, json.JSONDecodeError):
                details = {}
            out.append(
                {
                    "timestamp_utc": str(row["ts"]),
                    "severity": str(row["severity"]),
                    "code": str(row["code"]),
                    "message": str(row["message"]),
                    "details": details,
                }
            )
        return out

    def get_recent_symbol_returns(self, symbol: str, lookback: int = 240) -> List[float]:
        cur = self.conn.execute(
            """
            SELECT price
            FROM market_snapshots
            WHERE symbol = ?
            ORDER BY ts DESC
            LIMIT ?
            """,
            (str(symbol), int(max(10, lookback))),
        )
        prices = [float(r["price"]) for r in cur.fetchall()]
        prices.reverse()
        returns: List[float] = []
        for i in range(1, len(prices)):
            prev = prices[i - 1]
            if prev <= 0:
                continue
            returns.append((prices[i] - prev) / prev)
        return returns

    def get_symbol_correlation(self, symbol_a: str, symbol_b: str, lookback: int = 240) -> float:
        returns_a = self.get_recent_symbol_returns(symbol_a, lookback=lookback)
        returns_b = self.get_recent_symbol_returns(symbol_b, lookback=lookback)
        n = min(len(returns_a), len(returns_b))
        if n < 15:
            return 0.0
        x = returns_a[-n:]
        y = returns_b[-n:]
        mean_x = sum(x) / n
        mean_y = sum(y) / n
        cov = sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y))
        var_x = sum((a - mean_x) ** 2 for a in x)
        var_y = sum((b - mean_y) ** 2 for b in y)
        if var_x <= 0 or var_y <= 0:
            return 0.0
        return float(max(-1.0, min(1.0, cov / ((var_x**0.5) * (var_y**0.5)))))

    def save_model_state(self, key: str, value: Dict[str, Any]) -> None:
        self.conn.execute(
            """
            INSERT INTO model_state(key, value_json, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET
                value_json = excluded.value_json,
                updated_at = excluded.updated_at
            """,
            (str(key), json.dumps(value), _iso(_utc_now())),
        )
        self.conn.commit()

    def load_model_state(self, key: str) -> Optional[Dict[str, Any]]:
        cur = self.conn.execute(
            "SELECT value_json FROM model_state WHERE key = ?",
            (str(key),),
        )
        row = cur.fetchone()
        if not row:
            return None
        try:
            return json.loads(str(row["value_json"]))
        except (TypeError, ValueError, json.JSONDecodeError):
            return None

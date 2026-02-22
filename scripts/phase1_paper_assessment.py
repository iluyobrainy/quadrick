#!/usr/bin/env python
"""
Phase 1 paper-trading assessment runner.

Runs the Quadrick bot on Bybit testnet for a fixed duration, then emits:
1) a machine-readable JSON report
2) a markdown summary report

The report includes normalized $100 equity outcomes from realized paper-trade returns.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import shutil
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv
from loguru import logger


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from main import QuadrickTradingBot  # noqa: E402
from src.llm.deepseek_client import DecisionType, TradingDecision  # noqa: E402
from src.quant.data_lake import QuantDataLake  # noqa: E402


@dataclass
class Snapshot:
    timestamp_utc: str
    account_balance: float
    available_balance: float
    open_positions: int
    closed_trades: int
    tracked_trades: int


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso_now() -> str:
    return utc_now().isoformat()


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def ensure_safe_env(start_equity: float, profile: str = "soft_balanced", respect_env: bool = False) -> None:
    # Hard safety guard: always force testnet and disable live trading for assessment.
    # This script is not allowed to place real-money orders.
    import os

    if respect_env:
        return

    base_profile: Dict[str, str] = {
        "BYBIT_TESTNET": "true",
        "ALLOW_LIVE_TRADING": "false",
        "MIN_RISK_PCT": "1.0",
        "MAX_RISK_PCT": "4.0",
        "MAX_DAILY_DRAWDOWN_PCT": "12.0",
        "MAX_LEVERAGE": "5",
        "MAX_CONCURRENT_POSITIONS": "2",
        "DECISION_INTERVAL_SECONDS": "60",
        "MIN_RR_RATIO": "1.5",
        "SMALL_ACCOUNT_BALANCE_THRESHOLD": "150",
        "SMALL_ACCOUNT_MAX_RISK_PCT": "3.0",
        "SMALL_ACCOUNT_MAX_LEVERAGE": "5",
        "SMALL_ACCOUNT_MIN_RR_RATIO": "1.8",
        "MIN_CONFIDENCE_SCORE": "0.60",
        "MAX_ENTRY_DRIFT_PCT": "1.20",
        "MIN_STOP_DISTANCE_PCT": "0.35",
        "MAX_STOP_DISTANCE_PCT": "8.0",
        "ESTIMATED_ROUND_TRIP_COST_PCT": "0.14",
        "MIN_EXPECTED_EDGE_PCT": "0.00",
        "AUTONOMOUS_MODE_ENABLED": "true",
        "RELAXED_TRADE_GATING": "true",
        "SOFT_GOVERNOR_ENABLED": "true",
        "SOFT_GOVERNOR_MIN_MULTIPLIER": "0.20",
        "SOFT_GOVERNOR_MAX_MULTIPLIER": "1.20",
        "SOFT_GOVERNOR_DECAY": "0.85",
        "SOFT_GOVERNOR_RECOVERY": "1.05",
        "SOFT_GOVERNOR_UPDATE_MINUTES": "10",
        "SOFT_GOVERNOR_MIN_OBJECTIVE_CLOSED_TRADES": "6",
        "SOFT_GOVERNOR_MIN_EXECUTION_EVENTS": "6",
        "SOFT_DRAWDOWN_DEGRADE_CAP_PCT": "6.0",
        "SOFT_CATASTROPHIC_DRAWDOWN_PCT": "12.0",
        "FOCUS_RECOVERY_MODE_ENABLED": "true",
        "FOCUS_RECOVERY_RESET_GOVERNOR_ON_START": "true",
        "FOCUS_RECOVERY_DISABLE_COUNTERTREND_SOFT_BYPASS": "true",
        "QUANT_SESSION_SCOPE_METRICS": "true",
        "QUALITY_SCORE_FULL_MIN": "62",
        "QUALITY_SCORE_PROBE_MIN": "45",
        "PROBE_RISK_SCALE": "0.35",
        "PROBE_AFTER_NO_ACCEPT_CYCLES": "8",
        "PROBE_MAX_PER_HOUR": "6",
        "SYMBOL_MAX_MARGIN_PCT": "25.0",
        "PORTFOLIO_MAX_MARGIN_PCT": "45.0",
        "FULL_ENTRY_MAX_EST_SLIPPAGE_BPS": "40",
        "PROBE_ENTRY_MAX_EST_SLIPPAGE_BPS": "60",
        "POST_FILL_MICRO_EXIT_BPS": "120",
        "ENFORCE_SINGLE_POSITION_PER_SYMBOL": "true",
        "ALLOW_SCALE_IN": "false",
        "MAX_CONSECUTIVE_SYMBOL_ENTRIES": "5",
        "SYMBOL_REPEAT_WINDOW": "8",
        "SYMBOL_REPEAT_PENALTY_PCT": "0.07",
        "SYMBOL_REPEAT_OVERRIDE_GAP_PCT": "0.22",
        "COUNTER_TREND_STRICT_MODE": "false",
        "COUNTER_TREND_DISABLE_SOFT_OVERRIDES": "true",
        "FLAT_SYMBOL_COOLDOWN_MINUTES": "0",
        "AFFORDABILITY_MARGIN_EPSILON_USD": "0.01",
        "MARKET_ORDER_SLIPPAGE_TOLERANCE_PCT": "0.45",
        "MARKET_ORDER_REJECT_COOLDOWN_MINUTES": "0",
        "MAX_REPRICE_ATTEMPTS": "2",
        "ENABLE_FORECAST_ENGINE": "true",
        "FORECAST_WEIGHT_PCT": "0.12",
        "FORECAST_MIN_CONFIDENCE": "0.54",
        "QUANT_PRIMARY_MODE": "true",
        "LLM_AUDIT_ENABLED": "false",
        "QUANT_MIN_EDGE_PER_TRADE_PCT": "0.00",
        "QUANT_MIN_CALIBRATED_CONFIDENCE": "0.52",
        "QUANT_UNCERTAINTY_MAX": "0.70",
        "QUANT_TRAINING_LOOKBACK_ROWS": "4000",
        "QUANT_RETRAIN_INTERVAL_MINUTES": "120",
        "QUANT_DRIFT_RETRAIN_THRESHOLD": "0.65",
        "QUANT_CORRELATION_CAP": "0.75",
        "QUANT_MAX_PORTFOLIO_RISK_BUDGET_PCT": "35.0",
        "QUANT_MIN_EXPECTED_MOVE_PCT": "0.10",
        "QUANT_MIN_TP_PCT": "0.20",
        "QUANT_MAX_TP_PCT": "2.20",
        "QUANT_MIN_SL_PCT": "0.15",
        "QUANT_MAX_SL_PCT": "1.20",
        "QUANT_LATENCY_KILL_SWITCH_MS": "60000",
        "QUANT_REJECT_RATE_KILL_SWITCH": "1.00",
        "QUANT_REJECT_KILL_SWITCH_MIN_EVENTS": "10000",
        "QUANT_ENFORCE_LLM_EXECUTION_LOCK": "true",
        "QUANT_TARGET_TRADES_PER_HOUR": "1.3",
        "QUANT_EXPECTANCY_HOUR_FLOOR_PCT": "0.04",
        "QUANT_MIN_CLOSED_TRADES_FOR_OBJECTIVE": "2000",
        "QUANT_MONITOR_REJECT_ALERT": "0.95",
        "QUANT_MONITOR_LATENCY_ALERT_MS": "1600",
        "QUANT_MONITOR_DRIFT_ALERT": "0.72",
        "QUANT_MONITOR_NO_PROPOSAL_CYCLES_ALERT": "6",
        "QUANT_MONITOR_MIN_ALERT_INTERVAL_MINUTES": "10",
        "MAX_CONSECUTIVE_WAITS": "6",
        "TELEGRAM_ENABLED": "false",
        "NOTIFICATIONS_ENABLED": "false",
    }
    profiles: Dict[str, Dict[str, str]] = {
        "soft_balanced": {},
    }
    selected_profile = str(profile or "soft_balanced").strip().lower()
    if selected_profile not in profiles:
        selected_profile = "soft_balanced"

    merged: Dict[str, str] = dict(base_profile)
    merged.update(profiles[selected_profile])
    if start_equity > 0:
        merged["INITIAL_BALANCE"] = str(start_equity)

    for key, value in merged.items():
        os.environ[key] = str(value)


def max_drawdown_pct(equity_points: List[float]) -> float:
    if not equity_points:
        return 0.0
    peak = equity_points[0]
    worst = 0.0
    for value in equity_points:
        if value > peak:
            peak = value
        if peak > 0:
            dd = (peak - value) / peak * 100.0
            if dd > worst:
                worst = dd
    return worst


def virtual_equity_from_returns(
    pnl_pct_returns: List[float], start_equity: float
) -> Tuple[List[Dict[str, float]], float]:
    equity = float(start_equity)
    curve: List[Dict[str, float]] = [{"trade_index": 0, "equity": round(equity, 6)}]

    for idx, pnl_pct in enumerate(pnl_pct_returns, start=1):
        multiplier = 1.0 + (pnl_pct / 100.0)
        multiplier = max(multiplier, 0.0)
        equity *= multiplier
        curve.append({"trade_index": idx, "equity": round(equity, 6)})

    total_return_pct = ((equity / start_equity) - 1.0) * 100.0 if start_equity > 0 else 0.0
    return curve, total_return_pct


def monte_carlo_projection(
    returns_pct: List[float],
    start_equity: float,
    projected_trades: int,
    paths: int = 2000,
    seed: int = 42,
) -> Dict[str, float]:
    if not returns_pct or projected_trades <= 0:
        return {
            "p10": start_equity,
            "p50": start_equity,
            "p90": start_equity,
        }
    # Prevent misleading explosive projections from tiny samples.
    if len(returns_pct) < 10:
        return {
            "p10": start_equity,
            "p50": start_equity,
            "p90": start_equity,
        }

    rng = random.Random(seed)
    finals: List[float] = []

    for _ in range(paths):
        equity = start_equity
        for _ in range(projected_trades):
            r = rng.choice(returns_pct)
            equity *= max(0.0, 1.0 + r / 100.0)
        finals.append(equity)

    finals.sort()
    p10 = finals[int(0.10 * (len(finals) - 1))]
    p50 = finals[int(0.50 * (len(finals) - 1))]
    p90 = finals[int(0.90 * (len(finals) - 1))]
    return {"p10": p10, "p50": p50, "p90": p90}


def engineering_grade(run_stats: Dict[str, Any]) -> Tuple[float, List[str]]:
    grade = 100.0
    notes: List[str] = []

    if not run_stats.get("initialized", False):
        return 0.0, ["Initialization failed."]

    fatal = bool(run_stats.get("fatal_error"))
    if fatal:
        grade -= 35
        notes.append("Fatal runtime error occurred during session.")

    runtime_errors = int(run_stats.get("runtime_error_count", 0))
    if runtime_errors > 0:
        penalty = min(20.0, runtime_errors * 4.0)
        grade -= penalty
        notes.append(f"Runtime errors observed: {runtime_errors} (penalty {penalty:.1f}).")

    snapshots = int(run_stats.get("snapshots_collected", 0))
    if snapshots < 5:
        grade -= 10
        notes.append("Short/noisy observation window.")

    closed = int(run_stats.get("closed_trades", 0))
    if closed == 0:
        grade -= 15
        notes.append("No closed trades recorded, limited execution evidence.")

    if run_stats.get("forced_task_cancelled", False):
        grade -= 8
        notes.append("Main loop required force-cancel on shutdown.")

    if run_stats.get("positions_after_cleanup", 0) > 0:
        grade -= 10
        notes.append("Open positions remained after cleanup.")

    grade = round(clamp(grade, 0.0, 100.0), 1)
    if not notes:
        notes.append("No major runtime reliability issues observed.")
    return grade, notes


def quant_grade(metrics: Dict[str, Any]) -> Tuple[float, List[str]]:
    closed = int(metrics.get("closed_trades", 0))
    win_rate = float(metrics.get("win_rate_pct", 0.0))
    expectancy = float(metrics.get("expectancy_pct_per_trade", 0.0))
    profit_factor = float(metrics.get("profit_factor", 0.0))
    max_dd = float(metrics.get("max_drawdown_pct_on_100", 0.0))

    notes: List[str] = []

    sample_score = min(20.0, closed * 1.25)
    edge_score = clamp((expectancy + 0.4) * 25.0, 0.0, 30.0)
    win_rate_score = clamp((win_rate - 45.0) * 0.8, 0.0, 20.0)
    profit_factor_score = clamp((profit_factor - 1.0) * 10.0, 0.0, 10.0)
    drawdown_score = clamp(20.0 - max_dd * 1.5, 0.0, 20.0)

    raw_grade = sample_score + edge_score + win_rate_score + profit_factor_score + drawdown_score

    # Small sample protection: do not over-grade with too few observations.
    if closed < 10:
        raw_grade = min(raw_grade, 60.0)
        notes.append("Grade capped at 60 due to small sample (<10 closed trades).")
    elif closed < 20:
        raw_grade = min(raw_grade, 75.0)
        notes.append("Grade capped at 75 due to modest sample (<20 closed trades).")

    if expectancy <= 0:
        notes.append("Non-positive expectancy observed.")
    if profit_factor < 1.0:
        notes.append("Profit factor below 1.0 indicates weak edge.")
    if max_dd > 15:
        notes.append("High drawdown profile on normalized $100 curve.")
    if closed == 0:
        notes.append("No realized trades, quant confidence is very low.")

    grade = round(clamp(raw_grade, 0.0, 100.0), 1)
    if not notes:
        notes.append("Sample shows positive edge with acceptable drawdown.")
    return grade, notes


def collect_bucket_attribution(bot: QuadrickTradingBot) -> Dict[str, Any]:
    try:
        db_path = str(getattr(bot.settings.trading, "quant_data_lake_path", "data/quant_lake/quant.db"))
        data_lake = QuantDataLake(db_path)
        long_perf = data_lake.get_bucket_performance(minutes=1440, half_life_minutes=720)
        short_perf = data_lake.get_bucket_performance(minutes=180, half_life_minutes=180)
    except Exception as exc:
        return {"error": f"bucket_attribution_unavailable: {exc}"}

    rows: List[Dict[str, Any]] = []
    for key, row in long_perf.items():
        short_row = short_perf.get(key, {})
        weighted_n = float(row.get("weighted_closed_trades", 0.0) or 0.0)
        expectancy = float(row.get("weighted_expectancy_pct", 0.0) or 0.0)
        edge_mass = weighted_n * expectancy
        rows.append(
            {
                "bucket": key,
                "symbol": str(row.get("symbol") or ""),
                "side": str(row.get("side") or "unknown"),
                "regime": str(row.get("regime") or "unknown"),
                "weighted_closed_trades": round(weighted_n, 4),
                "weighted_win_rate": round(float(row.get("weighted_win_rate", 0.0) or 0.0), 6),
                "weighted_expectancy_pct": round(expectancy, 6),
                "weighted_avg_slippage_bps": round(float(row.get("weighted_avg_slippage_bps", 0.0) or 0.0), 4),
                "tail_loss_3_rate": round(float(row.get("tail_loss_3_rate", 0.0) or 0.0), 6),
                "tail_loss_5_rate": round(float(row.get("tail_loss_5_rate", 0.0) or 0.0), 6),
                "tail_loss_7_rate": round(float(row.get("tail_loss_7_rate", 0.0) or 0.0), 6),
                "short_expectancy_pct": round(float(short_row.get("weighted_expectancy_pct", 0.0) or 0.0), 6),
                "edge_mass": round(edge_mass, 6),
            }
        )

    top_positive = sorted(rows, key=lambda r: r["edge_mass"], reverse=True)[:8]
    top_negative = sorted(rows, key=lambda r: r["edge_mass"])[:8]
    worst_tail = sorted(
        rows,
        key=lambda r: (r["tail_loss_7_rate"], r["tail_loss_5_rate"], r["tail_loss_3_rate"]),
        reverse=True,
    )[:8]
    return {
        "tracked_buckets": len(rows),
        "top_positive_buckets": top_positive,
        "top_negative_buckets": top_negative,
        "tail_risk_hotspots": worst_tail,
    }


async def close_all_positions(bot: QuadrickTradingBot, notes: List[str]) -> int:
    try:
        bot.positions = bot.bybit.get_positions()
    except Exception as exc:
        notes.append(f"Could not refresh positions before cleanup: {exc}")
        return 0

    open_positions = [p for p in bot.positions if float(getattr(p, "size", 0.0)) > 0]
    if not open_positions:
        return 0

    closed_count = 0
    for pos in open_positions:
        symbol = getattr(pos, "symbol", None)
        if not symbol:
            continue
        close_decision = TradingDecision(
            decision_id=f"PHASE1_CLOSE_{symbol}_{int(datetime.now().timestamp())}",
            timestamp_utc=utc_now(),
            decision_type=DecisionType.CLOSE_POSITION,
            symbol=symbol,
            strategy_tag="phase1_cleanup",
            reasoning={"action": "phase1_forced_cleanup"},
        )
        try:
            await bot._close_position(close_decision)  # Reuse internal bookkeeping path.
            closed_count += 1
            await asyncio.sleep(0.2)
        except Exception as exc:
            notes.append(f"Failed to close {symbol} during cleanup: {exc}")

    return closed_count


async def run_phase_assessment(
    duration_minutes: int,
    start_equity: float,
    poll_seconds: int,
    close_positions_on_end: bool,
    profile: str = "soft_balanced",
    respect_env: bool = False,
) -> Dict[str, Any]:
    ensure_safe_env(start_equity=start_equity, profile=profile, respect_env=respect_env)
    if respect_env:
        import os

        if str(os.getenv("BYBIT_TESTNET", "true")).strip().lower() != "true":
            raise RuntimeError("--respect-env requires BYBIT_TESTNET=true for safety")
        if str(os.getenv("ALLOW_LIVE_TRADING", "false")).strip().lower() == "true":
            raise RuntimeError("--respect-env requires ALLOW_LIVE_TRADING=false for safety")

    run_started_at = utc_now()
    snapshots: List[Snapshot] = []
    runtime_notes: List[str] = []
    runtime_errors: List[str] = []

    initialized = False
    forced_task_cancelled = False
    fatal_error = None

    bot = QuadrickTradingBot()
    run_task: asyncio.Task | None = None

    try:
        await bot.initialize()
        initialized = True

        # Ensure prior runtime state (bot_state.json) does not contaminate this assessment.
        try:
            if hasattr(bot, "emergency_controls") and bot.emergency_controls:
                bot.emergency_controls.resume_trading()
                bot.emergency_controls.consecutive_losses = 0
                bot.emergency_controls.cooldown_active = False
                bot.emergency_controls.cooldown_until = None
        except Exception as exc:
            runtime_notes.append(f"Failed to reset emergency controls cleanly: {exc}")

        loop = asyncio.get_running_loop()
        deadline = loop.time() + (duration_minutes * 60)
        run_task = asyncio.create_task(bot.run(), name="quadrick_main_loop")

        while loop.time() < deadline:
            await asyncio.sleep(max(5, poll_seconds))

            try:
                open_positions = [
                    p for p in getattr(bot, "positions", []) if float(getattr(p, "size", 0.0)) > 0
                ]
                snapshots.append(
                    Snapshot(
                        timestamp_utc=iso_now(),
                        account_balance=float(getattr(bot, "account_balance", 0.0)),
                        available_balance=float(getattr(bot, "available_balance", 0.0)),
                        open_positions=len(open_positions),
                        closed_trades=len(getattr(bot.performance, "trades", [])),
                        tracked_trades=len(getattr(bot, "trade_history", [])),
                    )
                )
            except Exception as exc:
                runtime_errors.append(f"Snapshot collection error: {exc}")

    except Exception as exc:
        fatal_error = f"{type(exc).__name__}: {exc}"
        runtime_errors.append(fatal_error)

    finally:
        bot.running = False

        if run_task:
            try:
                await asyncio.wait_for(run_task, timeout=max(120, bot.decision_interval * 8))
            except asyncio.TimeoutError:
                forced_task_cancelled = True
                run_task.cancel()
                try:
                    await run_task
                except asyncio.CancelledError:
                    runtime_notes.append("Main loop task cancelled on timeout.")
                except Exception as exc:
                    runtime_errors.append(f"Cancelled task raised error: {exc}")
            except Exception as exc:
                runtime_errors.append(f"Main loop exited with error: {exc}")

        cleaned_positions = 0
        if initialized and close_positions_on_end:
            cleaned_positions = await close_all_positions(bot, runtime_notes)
            if cleaned_positions:
                runtime_notes.append(f"Forced cleanup closed {cleaned_positions} open position(s).")

        try:
            if initialized:
                await bot._update_account_state()
        except Exception as exc:
            runtime_errors.append(f"Final account refresh failed: {exc}")

        try:
            await bot.shutdown()
        except Exception as exc:
            runtime_errors.append(f"Shutdown error: {exc}")

    run_finished_at = utc_now()
    elapsed_seconds = (run_finished_at - run_started_at).total_seconds()

    closed_trades = [
        t for t in getattr(bot.performance, "trades", []) if isinstance(t.get("pnl"), (float, int))
    ]
    pnl_returns = [float(t.get("pnl", 0.0)) for t in closed_trades]
    wins = [p for p in pnl_returns if p > 0]
    losses = [p for p in pnl_returns if p < 0]

    equity_curve_100, total_return_100 = virtual_equity_from_returns(pnl_returns, start_equity)
    equity_only = [p["equity"] for p in equity_curve_100]

    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (math.inf if gross_profit > 0 else 0.0)
    expectancy = mean(pnl_returns) if pnl_returns else 0.0
    win_rate_pct = (len(wins) / len(pnl_returns) * 100.0) if pnl_returns else 0.0

    elapsed_hours = elapsed_seconds / 3600.0 if elapsed_seconds > 0 else 0.0
    trades_per_hour = (len(pnl_returns) / elapsed_hours) if elapsed_hours > 0 else 0.0
    projected_30d_trades = int(trades_per_hour * 24.0 * 30.0)
    projected_7d_trades = int(trades_per_hour * 24.0 * 7.0)

    projection_7d = monte_carlo_projection(pnl_returns, start_equity, projected_7d_trades)
    projection_30d = monte_carlo_projection(pnl_returns, start_equity, projected_30d_trades)
    actual_start_equity = (
        float(snapshots[0].account_balance)
        if snapshots
        else float(getattr(bot, "starting_balance", 0.0) or 0.0)
    )
    actual_end_equity = float(getattr(bot, "account_balance", 0.0) or 0.0)
    actual_net_pnl = actual_end_equity - actual_start_equity
    actual_return_pct = ((actual_net_pnl / actual_start_equity) * 100.0) if actual_start_equity > 0 else 0.0

    metrics = {
        "closed_trades": len(pnl_returns),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate_pct": round(win_rate_pct, 3),
        "expectancy_pct_per_trade": round(expectancy, 6),
        "avg_win_pct": round(mean(wins), 6) if wins else 0.0,
        "avg_loss_pct": round(mean(losses), 6) if losses else 0.0,
        "gross_profit_pct": round(gross_profit, 6),
        "gross_loss_pct": round(-gross_loss, 6),
        "profit_factor": float("inf") if math.isinf(profit_factor) else round(profit_factor, 6),
        "max_drawdown_pct_on_100": round(max_drawdown_pct(equity_only), 6),
        "end_equity_100": round(equity_only[-1], 6) if equity_only else start_equity,
        "total_return_pct_on_100": round(total_return_100, 6),
        "net_pnl_on_100": round((equity_only[-1] - start_equity), 6) if equity_only else 0.0,
        "trades_per_hour": round(trades_per_hour, 6),
        "projected_7d_trade_count": projected_7d_trades,
        "projected_30d_trade_count": projected_30d_trades,
        "projection_7d_100": {k: round(v, 6) for k, v in projection_7d.items()},
        "projection_30d_100": {k: round(v, 6) for k, v in projection_30d.items()},
        "actual_start_equity": round(actual_start_equity, 6),
        "actual_end_equity": round(actual_end_equity, 6),
        "actual_net_pnl": round(actual_net_pnl, 6),
        "actual_return_pct": round(actual_return_pct, 6),
    }

    run_stats = {
        "initialized": initialized,
        "fatal_error": fatal_error,
        "runtime_error_count": len(runtime_errors),
        "runtime_errors": runtime_errors,
        "runtime_notes": runtime_notes,
        "forced_task_cancelled": forced_task_cancelled,
        "snapshots_collected": len(snapshots),
        "positions_after_cleanup": sum(1 for p in getattr(bot, "positions", []) if float(getattr(p, "size", 0.0)) > 0),
        "closed_trades": len(pnl_returns),
    }

    eng_grade, eng_notes = engineering_grade(run_stats)
    q_grade, q_notes = quant_grade(metrics)
    bucket_attribution = collect_bucket_attribution(bot)

    result: Dict[str, Any] = {
        "phase": "phase1_paper_assessment",
        "run_window": {
            "started_at_utc": run_started_at.isoformat(),
            "finished_at_utc": run_finished_at.isoformat(),
            "duration_minutes_target": duration_minutes,
            "duration_minutes_actual": round(elapsed_seconds / 60.0, 3),
            "poll_seconds": poll_seconds,
        },
        "mode": {
            "bybit_testnet": True,
            "allow_live_trading": False,
            "start_equity_for_projection": start_equity,
            "profile": profile,
            "respect_env": bool(respect_env),
        },
        "run_stats": run_stats,
        "metrics": metrics,
        "grades": {
            "engineering_grade": eng_grade,
            "engineering_notes": eng_notes,
            "quant_grade": q_grade,
            "quant_notes": q_notes,
        },
        "bucket_attribution": bucket_attribution,
        "trades": closed_trades,
        "equity_curve_100": equity_curve_100,
        "snapshots": [asdict(s) for s in snapshots],
    }
    return result


def report_markdown(data: Dict[str, Any]) -> str:
    grades = data["grades"]
    metrics = data["metrics"]
    run_stats = data["run_stats"]
    run_window = data["run_window"]
    runtime_note_lines = [f"- {n}" for n in run_stats["runtime_notes"]] or ["- None"]
    runtime_error_lines = [f"- {n}" for n in run_stats["runtime_errors"]] or ["- None"]
    bucket_data = data.get("bucket_attribution", {}) if isinstance(data.get("bucket_attribution"), dict) else {}
    top_positive = bucket_data.get("top_positive_buckets", []) if isinstance(bucket_data.get("top_positive_buckets"), list) else []
    top_negative = bucket_data.get("top_negative_buckets", []) if isinstance(bucket_data.get("top_negative_buckets"), list) else []
    tail_hotspots = bucket_data.get("tail_risk_hotspots", []) if isinstance(bucket_data.get("tail_risk_hotspots"), list) else []
    bucket_lines = [f"- Tracked buckets: {bucket_data.get('tracked_buckets', 0)}"]
    for row in top_positive[:3]:
        bucket_lines.append(
            f"- Top+: {row.get('symbol')} {row.get('side')} {row.get('regime')} "
            f"(exp={row.get('weighted_expectancy_pct')}%, n={row.get('weighted_closed_trades')})"
        )
    for row in top_negative[:3]:
        bucket_lines.append(
            f"- Top-: {row.get('symbol')} {row.get('side')} {row.get('regime')} "
            f"(exp={row.get('weighted_expectancy_pct')}%, n={row.get('weighted_closed_trades')})"
        )
    for row in tail_hotspots[:3]:
        bucket_lines.append(
            f"- Tail: {row.get('symbol')} {row.get('side')} {row.get('regime')} "
            f"(<=-5%={row.get('tail_loss_5_rate')}, <=-7%={row.get('tail_loss_7_rate')})"
        )
    if not bucket_lines:
        bucket_lines = ["- None"]

    return "\n".join(
        [
            "# Quadrick Phase 1 Paper Assessment",
            "",
            "## Run Window",
            f"- Start (UTC): {run_window['started_at_utc']}",
            f"- End (UTC): {run_window['finished_at_utc']}",
            f"- Target Duration (min): {run_window['duration_minutes_target']}",
            f"- Actual Duration (min): {run_window['duration_minutes_actual']}",
            f"- Snapshot Poll (sec): {run_window['poll_seconds']}",
            "",
            "## Runtime Reliability",
            f"- Initialized: {run_stats['initialized']}",
            f"- Fatal Error: {run_stats['fatal_error'] or 'None'}",
            f"- Runtime Error Count: {run_stats['runtime_error_count']}",
            f"- Forced Task Cancelled: {run_stats['forced_task_cancelled']}",
            f"- Snapshots Collected: {run_stats['snapshots_collected']}",
            "",
            "## Trading Metrics",
            f"- Closed Trades: {metrics['closed_trades']}",
            f"- Win Rate: {metrics['win_rate_pct']}%",
            f"- Expectancy: {metrics['expectancy_pct_per_trade']}% per trade",
            f"- Profit Factor: {metrics['profit_factor']}",
            f"- Max Drawdown on $100 curve: {metrics['max_drawdown_pct_on_100']}%",
            "",
            "## $100 Projection (Observed Sample)",
            f"- End Equity: ${metrics['end_equity_100']}",
            f"- Net PnL: ${metrics['net_pnl_on_100']}",
            f"- Total Return: {metrics['total_return_pct_on_100']}%",
            f"- Trades/Hour: {metrics['trades_per_hour']}",
            "",
            "## Actual Exchange-Equity Snapshot",
            f"- Start Equity: ${metrics['actual_start_equity']}",
            f"- End Equity: ${metrics['actual_end_equity']}",
            f"- Net PnL: ${metrics['actual_net_pnl']}",
            f"- Return: {metrics['actual_return_pct']}%",
            "",
            "## Monte Carlo Projection (Same Trade Distribution)",
            f"- 7D trade count estimate: {metrics['projected_7d_trade_count']}",
            f"- 7D P10/P50/P90 on $100: ${metrics['projection_7d_100']['p10']} / ${metrics['projection_7d_100']['p50']} / ${metrics['projection_7d_100']['p90']}",
            f"- 30D trade count estimate: {metrics['projected_30d_trade_count']}",
            f"- 30D P10/P50/P90 on $100: ${metrics['projection_30d_100']['p10']} / ${metrics['projection_30d_100']['p50']} / ${metrics['projection_30d_100']['p90']}",
            "",
            "## Grades",
            f"- Engineering Grade: {grades['engineering_grade']}/100",
            f"- Quant Grade: {grades['quant_grade']}/100",
            "",
            "## Engineering Notes",
            *[f"- {n}" for n in grades["engineering_notes"]],
            "",
            "## Quant Notes",
            *[f"- {n}" for n in grades["quant_notes"]],
            "",
            "## Bucket Attribution (Symbol+Side+Regime)",
            *bucket_lines,
            "",
            "## Runtime Notes",
            *runtime_note_lines,
            "",
            "## Runtime Errors",
            *runtime_error_lines,
            "",
            "## Caveats",
            "- This is testnet paper execution and does not include full live slippage/latency/psychology effects.",
            "- Small sample windows can strongly overfit recent market regime.",
            "- Projection assumes future trade distribution resembles observed sample.",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Quadrick Phase 1 paper-trading assessment")
    parser.add_argument("--duration-minutes", type=int, default=90, help="How long to run bot loop")
    parser.add_argument("--start-equity", type=float, default=100.0, help="Starting capital for normalized projection")
    parser.add_argument("--poll-seconds", type=int, default=15, help="Snapshot polling interval")
    parser.add_argument(
        "--no-close-on-end",
        action="store_true",
        help="Do not force close open testnet positions at end of run",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "reports" / "phase1",
        help="Directory to write reports",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default="soft_balanced",
        help="Execution profile name for env overrides",
    )
    parser.add_argument(
        "--respect-env",
        action="store_true",
        help="Use existing env values without profile overrides",
    )
    return parser.parse_args()


async def async_main(args: argparse.Namespace) -> int:
    logger.info(
        "Starting Phase 1 paper assessment: duration={} min, start_equity=${}, profile={}, respect_env={}",
        args.duration_minutes,
        args.start_equity,
        args.profile,
        bool(args.respect_env),
    )

    result = await run_phase_assessment(
        duration_minutes=args.duration_minutes,
        start_equity=args.start_equity,
        poll_seconds=args.poll_seconds,
        close_positions_on_end=not args.no_close_on_end,
        profile=args.profile,
        respect_env=bool(args.respect_env),
    )

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    stamp = utc_now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"phase1_{stamp}.json"
    md_path = output_dir / f"phase1_{stamp}.md"
    latest_json = output_dir / "latest.json"
    latest_md = output_dir / "latest.md"

    json_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    md_path.write_text(report_markdown(result), encoding="utf-8")
    shutil.copyfile(json_path, latest_json)
    shutil.copyfile(md_path, latest_md)

    logger.info("Phase 1 JSON report: {}", json_path)
    logger.info("Phase 1 markdown report: {}", md_path)
    logger.info("Latest JSON report: {}", latest_json)
    logger.info("Latest markdown report: {}", latest_md)

    return 0


def main() -> int:
    args = parse_args()
    try:
        return asyncio.run(async_main(args))
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())

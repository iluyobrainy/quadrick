"""
Walk-forward backtest harness for scalping strategies.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any
import argparse
import pandas as pd
import pandas_ta as ta


@dataclass
class BacktestTrade:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    side: str
    entry: float
    exit: float
    pnl_pct: float
    outcome: str


def _compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["rsi"] = ta.rsi(df["close"], length=14)
    macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
    df["macd_hist"] = macd["MACDh_12_26_9"]
    df["ema_9"] = ta.ema(df["close"], length=9)
    df["ema_21"] = ta.ema(df["close"], length=21)
    df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=14)
    df["volume_ratio"] = df["volume"] / df["volume"].rolling(20).mean()
    return df


def _trend(ema_fast: float, ema_slow: float) -> str:
    if ema_fast > ema_slow:
        return "uptrend"
    if ema_fast < ema_slow:
        return "downtrend"
    return "neutral"


def _generate_signal(row: pd.Series) -> Dict[str, Any]:
    volume_ratio = row["volume_ratio"]
    rsi = row["rsi"]
    macd_hist = row["macd_hist"]
    trend = _trend(row["ema_9"], row["ema_21"])

    if pd.isna(volume_ratio) or pd.isna(rsi) or pd.isna(macd_hist) or pd.isna(row["atr"]):
        return {"side": None}

    if volume_ratio >= 2.0 and macd_hist > 0 and trend == "uptrend" and 45 <= rsi <= 75:
        return {"side": "long"}
    if volume_ratio >= 2.0 and macd_hist < 0 and trend == "downtrend" and 25 <= rsi <= 55:
        return {"side": "short"}
    return {"side": None}


def _simulate_trades(df: pd.DataFrame, max_hold_bars: int = 9) -> List[BacktestTrade]:
    trades: List[BacktestTrade] = []
    in_trade = False
    entry_price = 0.0
    entry_time = None
    side = None
    stop_loss = 0.0
    take_profit = 0.0
    hold_bars = 0

    for idx in range(len(df)):
        row = df.iloc[idx]
        if not in_trade:
            signal = _generate_signal(row)
            if signal["side"] and not pd.isna(row["atr"]):
                side = signal["side"]
                entry_price = row["close"]
                entry_time = row["timestamp"]
                atr = row["atr"]
                if side == "long":
                    stop_loss = entry_price - (0.7 * atr)
                    take_profit = entry_price + (1.0 * atr)
                else:
                    stop_loss = entry_price + (0.7 * atr)
                    take_profit = entry_price - (1.0 * atr)
                in_trade = True
                hold_bars = 0
            continue

        hold_bars += 1
        high = row["high"]
        low = row["low"]
        exit_price = None
        outcome = None

        if side == "long":
            if low <= stop_loss:
                exit_price = stop_loss
                outcome = "loss"
            elif high >= take_profit:
                exit_price = take_profit
                outcome = "win"
        else:
            if high >= stop_loss:
                exit_price = stop_loss
                outcome = "loss"
            elif low <= take_profit:
                exit_price = take_profit
                outcome = "win"

        if exit_price is None and hold_bars >= max_hold_bars:
            exit_price = row["close"]
            outcome = "timeout"

        if exit_price is not None:
            pnl_pct = ((exit_price - entry_price) / entry_price) * 100 if side == "long" else ((entry_price - exit_price) / entry_price) * 100
            trades.append(
                BacktestTrade(
                    entry_time=entry_time,
                    exit_time=row["timestamp"],
                    side=side,
                    entry=entry_price,
                    exit=exit_price,
                    pnl_pct=pnl_pct,
                    outcome=outcome,
                )
            )
            in_trade = False
            side = None

    return trades


def run_walk_forward(csv_path: str, train_size: int, test_size: int) -> Dict[str, Any]:
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = _compute_indicators(df)
    df = df.dropna().reset_index(drop=True)

    results = []
    start = 0
    while start + train_size + test_size <= len(df):
        test_slice = df.iloc[start + train_size : start + train_size + test_size]
        trades = _simulate_trades(test_slice)
        pnl = sum(t.pnl_pct for t in trades)
        win_rate = (sum(1 for t in trades if t.outcome == "win") / len(trades)) * 100 if trades else 0
        results.append({"pnl_pct": pnl, "win_rate": win_rate, "trades": len(trades)})
        start += test_size

    aggregate = {
        "windows": len(results),
        "avg_pnl_pct": sum(r["pnl_pct"] for r in results) / len(results) if results else 0,
        "avg_win_rate": sum(r["win_rate"] for r in results) / len(results) if results else 0,
        "total_trades": sum(r["trades"] for r in results),
    }
    return {"results": results, "aggregate": aggregate}


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward backtest for scalping strategy")
    parser.add_argument("--csv", required=True, help="Path to CSV with OHLCV data")
    parser.add_argument("--train", type=int, default=1500, help="Training window size (rows)")
    parser.add_argument("--test", type=int, default=500, help="Testing window size (rows)")
    args = parser.parse_args()

    report = run_walk_forward(args.csv, args.train, args.test)
    print("Walk-forward summary:")
    print(report["aggregate"])


if __name__ == "__main__":
    main()

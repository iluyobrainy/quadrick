"""
Backtest Analyzer Module

Analyzes historical trades from Supabase to generate performance statistics
by symbol, direction, and strategy.
"""
import asyncio
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from loguru import logger


@dataclass
class SymbolPerformance:
    """Performance statistics for a single symbol"""
    symbol: str
    total_trades: int
    wins: int
    losses: int
    win_rate: float
    total_pnl_pct: float
    avg_pnl_pct: float
    avg_hold_time_minutes: float
    best_trade_pct: float
    worst_trade_pct: float
    long_win_rate: float
    short_win_rate: float


@dataclass
class DirectionStats:
    """Win rate statistics by direction"""
    long_trades: int
    short_trades: int
    long_wins: int
    short_wins: int
    long_win_rate: float
    short_win_rate: float
    long_avg_pnl: float
    short_avg_pnl: float


@dataclass
class ConfidenceCalibration:
    """Calibration between predicted confidence and actual win rate"""
    confidence_bucket: str  # e.g., "70-80%"
    predicted_win_rate: float  # Average confidence in bucket
    actual_win_rate: float     # Actual win rate in bucket
    trade_count: int
    is_well_calibrated: bool   # True if within 10% of prediction


class BacktestAnalyzer:
    """
    Analyze historical trades from Supabase to generate insights.
    
    Uses existing Supabase tables:
    - trades: Raw trade executions
    - trade_memories: Trades with vector embeddings
    """
    
    def __init__(self, supabase_client):
        """
        Initialize backtest analyzer.
        
        Args:
            supabase_client: Initialized SupabaseClient instance
        """
        self.db = supabase_client
    
    async def get_symbol_performance(self, symbol: str, days: int = 30) -> Optional[SymbolPerformance]:
        """
        Get performance statistics for a specific symbol.
        
        Args:
            symbol: Trading symbol (e.g., "BTCUSDT")
            days: Number of days to analyze
            
        Returns:
            SymbolPerformance or None if no data
        """
        if not self.db or not self.db.enabled:
            return None
        
        try:
            start_date = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
            
            response = await asyncio.to_thread(
                lambda: self.db.client.table("trades")
                    .select("*")
                    .eq("symbol", symbol)
                    .gte("timestamp", start_date)
                    .execute()
            )
            
            if not response.data:
                return None
            
            trades = response.data
            total = len(trades)
            wins = sum(1 for t in trades if t.get("pnl_pct", 0) > 0)
            losses = total - wins
            
            pnl_values = [t.get("pnl_pct", 0) for t in trades]
            long_trades = [t for t in trades if t.get("side") == "Buy"]
            short_trades = [t for t in trades if t.get("side") == "Sell"]
            
            long_wins = sum(1 for t in long_trades if t.get("pnl_pct", 0) > 0)
            short_wins = sum(1 for t in short_trades if t.get("pnl_pct", 0) > 0)
            
            return SymbolPerformance(
                symbol=symbol,
                total_trades=total,
                wins=wins,
                losses=losses,
                win_rate=(wins / total * 100) if total > 0 else 0,
                total_pnl_pct=sum(pnl_values),
                avg_pnl_pct=(sum(pnl_values) / total) if total > 0 else 0,
                avg_hold_time_minutes=self._calc_avg_hold_time(trades),
                best_trade_pct=max(pnl_values) if pnl_values else 0,
                worst_trade_pct=min(pnl_values) if pnl_values else 0,
                long_win_rate=(long_wins / len(long_trades) * 100) if long_trades else 0,
                short_win_rate=(short_wins / len(short_trades) * 100) if short_trades else 0,
            )
            
        except Exception as e:
            logger.error(f"Failed to get symbol performance: {e}")
            return None
    
    async def get_direction_stats(self, days: int = 30) -> Optional[DirectionStats]:
        """
        Get overall win rate statistics by direction (long vs short).
        
        Args:
            days: Number of days to analyze
            
        Returns:
            DirectionStats or None if no data
        """
        if not self.db or not self.db.enabled:
            return None
        
        try:
            start_date = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
            
            response = await asyncio.to_thread(
                lambda: self.db.client.table("trades")
                    .select("*")
                    .gte("timestamp", start_date)
                    .execute()
            )
            
            if not response.data:
                return None
            
            trades = response.data
            long_trades = [t for t in trades if t.get("side") == "Buy"]
            short_trades = [t for t in trades if t.get("side") == "Sell"]
            
            long_wins = sum(1 for t in long_trades if t.get("pnl_pct", 0) > 0)
            short_wins = sum(1 for t in short_trades if t.get("pnl_pct", 0) > 0)
            
            long_pnl = [t.get("pnl_pct", 0) for t in long_trades]
            short_pnl = [t.get("pnl_pct", 0) for t in short_trades]
            
            return DirectionStats(
                long_trades=len(long_trades),
                short_trades=len(short_trades),
                long_wins=long_wins,
                short_wins=short_wins,
                long_win_rate=(long_wins / len(long_trades) * 100) if long_trades else 0,
                short_win_rate=(short_wins / len(short_trades) * 100) if short_trades else 0,
                long_avg_pnl=(sum(long_pnl) / len(long_pnl)) if long_pnl else 0,
                short_avg_pnl=(sum(short_pnl) / len(short_pnl)) if short_pnl else 0,
            )
            
        except Exception as e:
            logger.error(f"Failed to get direction stats: {e}")
            return None
    
    async def get_confidence_calibration(self, days: int = 30) -> List[ConfidenceCalibration]:
        """
        Analyze if predicted confidence matches actual win rates.
        
        Groups trades by confidence bucket and compares predicted vs actual.
        
        Args:
            days: Number of days to analyze
            
        Returns:
            List of ConfidenceCalibration for each bucket
        """
        if not self.db or not self.db.enabled:
            return []
        
        try:
            start_date = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
            
            response = await asyncio.to_thread(
                lambda: self.db.client.table("decisions")
                    .select("*")
                    .gte("timestamp", start_date)
                    .execute()
            )
            
            if not response.data:
                return []
            
            # Group by confidence buckets
            buckets = {
                "60-70%": [],
                "70-80%": [],
                "80-90%": [],
                "90-100%": [],
            }
            
            for decision in response.data:
                conf = decision.get("confidence_score", 0) * 100
                win = decision.get("outcome", {}).get("win", False)
                
                if 60 <= conf < 70:
                    buckets["60-70%"].append({"conf": conf, "win": win})
                elif 70 <= conf < 80:
                    buckets["70-80%"].append({"conf": conf, "win": win})
                elif 80 <= conf < 90:
                    buckets["80-90%"].append({"conf": conf, "win": win})
                elif conf >= 90:
                    buckets["90-100%"].append({"conf": conf, "win": win})
            
            results = []
            for bucket_name, trades in buckets.items():
                if trades:
                    avg_conf = sum(t["conf"] for t in trades) / len(trades)
                    actual_wr = sum(1 for t in trades if t["win"]) / len(trades) * 100
                    well_calibrated = abs(avg_conf - actual_wr) <= 10
                    
                    results.append(ConfidenceCalibration(
                        confidence_bucket=bucket_name,
                        predicted_win_rate=avg_conf,
                        actual_win_rate=actual_wr,
                        trade_count=len(trades),
                        is_well_calibrated=well_calibrated,
                    ))
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to get confidence calibration: {e}")
            return []
    
    async def get_all_symbol_stats(self, symbols: List[str], days: int = 30) -> Dict[str, SymbolPerformance]:
        """
        Get performance for all symbols.
        
        Args:
            symbols: List of trading symbols
            days: Number of days to analyze
            
        Returns:
            Dict mapping symbol to performance
        """
        results = {}
        for symbol in symbols:
            perf = await self.get_symbol_performance(symbol, days)
            if perf:
                results[symbol] = perf
        return results
    
    async def generate_report(self, symbols: List[str], days: int = 30) -> str:
        """
        Generate a text report of performance.
        
        Args:
            symbols: List of trading symbols
            days: Number of days to analyze
            
        Returns:
            Formatted report string
        """
        lines = [f"=== BACKTEST REPORT ({days} days) ===\n"]
        
        # Direction stats
        dir_stats = await self.get_direction_stats(days)
        if dir_stats:
            lines.append("📊 DIRECTION STATS:")
            lines.append(f"  Long: {dir_stats.long_trades} trades, {dir_stats.long_win_rate:.1f}% WR, {dir_stats.long_avg_pnl:.2f}% avg")
            lines.append(f"  Short: {dir_stats.short_trades} trades, {dir_stats.short_win_rate:.1f}% WR, {dir_stats.short_avg_pnl:.2f}% avg\n")
        
        # Symbol stats
        lines.append("📈 SYMBOL PERFORMANCE:")
        for symbol in symbols:
            perf = await self.get_symbol_performance(symbol, days)
            if perf and perf.total_trades > 0:
                lines.append(f"  {symbol}: {perf.total_trades} trades, {perf.win_rate:.1f}% WR, {perf.total_pnl_pct:.2f}% total")
        
        return "\n".join(lines)
    
    def _calc_avg_hold_time(self, trades: List[Dict]) -> float:
        """Calculate average hold time in minutes from trades"""
        hold_times = []
        for t in trades:
            entry_time = t.get("entry_time")
            exit_time = t.get("exit_time")
            if entry_time and exit_time:
                try:
                    entry = datetime.fromisoformat(entry_time.replace("Z", "+00:00"))
                    exit = datetime.fromisoformat(exit_time.replace("Z", "+00:00"))
                    hold_times.append((exit - entry).total_seconds() / 60)
                except:
                    pass
        return (sum(hold_times) / len(hold_times)) if hold_times else 0

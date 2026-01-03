"""
Performance Tracking Module - Analyzes trading performance for DeepSeek feedback
"""
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from collections import defaultdict
from loguru import logger


class PerformanceTracker:
    """Track and analyze trading performance"""
    
    def __init__(self):
        """Initialize performance tracker"""
        self.trades: List[Dict[str, Any]] = []
        self.daily_stats: Dict[str, Dict[str, Any]] = {}
        logger.info("Performance tracker initialized")
    
    def add_trade(self, trade_data: Dict[str, Any]):
        """Add a completed trade"""
        self.trades.append({
            **trade_data,
            "timestamp": trade_data.get("timestamp", datetime.utcnow()),
        })
        logger.debug(f"Trade added to performance tracker: {trade_data.get('symbol')}")
    
    def get_recent_performance(self, limit: int = 20) -> Dict[str, Any]:
        """
        Get performance summary of recent trades
        
        Args:
            limit: Number of recent trades to analyze
        
        Returns:
            Performance summary for DeepSeek feedback
        """
        if not self.trades:
            return {
                "total_trades": 0,
                "message": "No trades yet - this is your first decision!",
            }
        
        # Get recent trades
        recent = self.trades[-limit:]
        
        # Calculate basic stats
        total = len(recent)
        wins = sum(1 for t in recent if t.get("pnl", 0) > 0)
        losses = total - wins
        win_rate = (wins / total * 100) if total > 0 else 0
        
        # Calculate P&L
        total_pnl = sum(t.get("pnl", 0) for t in recent)
        avg_win = sum(t.get("pnl", 0) for t in recent if t.get("pnl", 0) > 0) / wins if wins > 0 else 0
        avg_loss = sum(t.get("pnl", 0) for t in recent if t.get("pnl", 0) < 0) / losses if losses > 0 else 0
        
        best_trade = max(recent, key=lambda x: x.get("pnl", 0)) if recent else None
        worst_trade = min(recent, key=lambda x: x.get("pnl", 0)) if recent else None
        
        # Analyze by strategy
        by_strategy = defaultdict(lambda: {"wins": 0, "losses": 0, "total_pnl": 0, "trades": []})
        
        for trade in recent:
            strategy = trade.get("strategy_tag", "unknown")
            pnl = trade.get("pnl", 0)
            
            by_strategy[strategy]["trades"].append(trade)
            by_strategy[strategy]["total_pnl"] += pnl
            
            if pnl > 0:
                by_strategy[strategy]["wins"] += 1
            else:
                by_strategy[strategy]["losses"] += 1
        
        # Calculate win rate by strategy
        strategy_performance = {}
        for strategy, stats in by_strategy.items():
            total_strategy_trades = stats["wins"] + stats["losses"]
            strategy_performance[strategy] = {
                "wins": stats["wins"],
                "losses": stats["losses"],
                "total_trades": total_strategy_trades,
                "win_rate": (stats["wins"] / total_strategy_trades * 100) if total_strategy_trades > 0 else 0,
                "total_pnl": stats["total_pnl"],
                "avg_pnl": stats["total_pnl"] / total_strategy_trades if total_strategy_trades > 0 else 0,
            }
        
        # Find best and worst strategies
        best_strategy = max(strategy_performance.items(), key=lambda x: x[1]["total_pnl"]) if strategy_performance else None
        worst_strategy = min(strategy_performance.items(), key=lambda x: x[1]["total_pnl"]) if strategy_performance else None
        
        # Recent streak
        streak_type = None
        streak_count = 0
        
        for trade in reversed(recent):
            pnl = trade.get("pnl", 0)
            if streak_type is None:
                streak_type = "win" if pnl > 0 else "loss"
                streak_count = 1
            elif (streak_type == "win" and pnl > 0) or (streak_type == "loss" and pnl < 0):
                streak_count += 1
            else:
                break
        
        # Generate insights and lessons
        insights = self._generate_insights(recent, strategy_performance, win_rate, streak_type, streak_count)
        
        return {
            "total_trades": total,
            "wins": wins,
            "losses": losses,
            "win_rate": round(win_rate, 1),
            "total_pnl": round(total_pnl, 2),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "profit_factor": round(abs(avg_win * wins / (avg_loss * losses)), 2) if (avg_loss * losses) != 0 else 0,
            
            "best_trade": {
                "symbol": best_trade.get("symbol"),
                "strategy": best_trade.get("strategy_tag"),
                "pnl": round(best_trade.get("pnl", 0), 2),
            } if best_trade else None,
            
            "worst_trade": {
                "symbol": worst_trade.get("symbol"),
                "strategy": worst_trade.get("strategy_tag"),
                "pnl": round(worst_trade.get("pnl", 0), 2),
            } if worst_trade else None,
            
            "by_strategy": strategy_performance,
            
            "best_performing_strategy": {
                "name": best_strategy[0],
                "win_rate": round(best_strategy[1]["win_rate"], 1),
                "total_pnl": round(best_strategy[1]["total_pnl"], 2),
                "trades": best_strategy[1]["total_trades"],
            } if best_strategy else None,
            
            "worst_performing_strategy": {
                "name": worst_strategy[0],
                "win_rate": round(worst_strategy[1]["win_rate"], 1),
                "total_pnl": round(worst_strategy[1]["total_pnl"], 2),
                "trades": worst_strategy[1]["total_trades"],
            } if worst_strategy else None,
            
            "current_streak": {
                "type": streak_type,
                "count": streak_count,
            },
            
            "insights_and_lessons": insights,
        }
    
    def _generate_insights(
        self,
        trades: List[Dict[str, Any]],
        strategy_perf: Dict[str, Dict[str, Any]],
        overall_win_rate: float,
        streak_type: Optional[str],
        streak_count: int,
    ) -> str:
        """Generate actionable insights from performance data"""
        insights = []
        
        # Overall performance
        if overall_win_rate >= 70:
            insights.append("🔥 EXCELLENT: 70%+ win rate - your strategy is working! Keep doing what you're doing.")
        elif overall_win_rate >= 60:
            insights.append("✅ GOOD: 60-70% win rate - solid performance, maintain discipline.")
        elif overall_win_rate >= 50:
            insights.append("⚠️ BORDERLINE: 50-60% win rate - need higher R:R or better setups.")
        else:
            insights.append("🚨 CONCERNING: <50% win rate - strategy needs adjustment. Review what's failing.")
        
        # Strategy-specific insights
        if strategy_perf:
            best = max(strategy_perf.items(), key=lambda x: x[1]["total_pnl"])
            worst = min(strategy_perf.items(), key=lambda x: x[1]["total_pnl"])
            
            if best[1]["win_rate"] >= 70:
                insights.append(f"🎯 WORKING WELL: '{best[0]}' strategy has {best[1]['win_rate']:.0f}% win rate and +${best[1]['total_pnl']:.2f} profit. Focus on this!")
            
            if worst[1]["win_rate"] < 40 and worst[1]["total_trades"] >= 3:
                insights.append(f"❌ AVOID: '{worst[0]}' strategy only {worst[1]['win_rate']:.0f}% win rate, losing ${abs(worst[1]['total_pnl']):.2f}. Stop using this approach.")
        
        # Streak analysis
        if streak_type == "win" and streak_count >= 3:
            insights.append(f"🔥 HOT STREAK: {streak_count} wins in a row! Stay disciplined, don't get overconfident.")
        elif streak_type == "loss" and streak_count >= 3:
            insights.append(f"🚨 LOSING STREAK: {streak_count} losses in a row. Take a break, reassess market conditions. Market might have changed.")
        
        # Recent performance
        if len(trades) >= 5:
            last_5_pnl = sum(t.get("pnl", 0) for t in trades[-5:])
            if last_5_pnl > 0:
                insights.append(f"📈 RECENT MOMENTUM: Last 5 trades = +${last_5_pnl:.2f}. You're in rhythm.")
            else:
                insights.append(f"📉 RECENT STRUGGLES: Last 5 trades = -${abs(last_5_pnl):.2f}. Re-evaluate approach.")
        
        return " | ".join(insights) if insights else "Continue analyzing and adapting your strategy."
    
    def get_strategy_recommendations(self) -> str:
        """Get AI-generated recommendations based on performance"""
        if len(self.trades) < 5:
            return "Not enough trades yet to generate recommendations. Focus on quality setups."
        
        recent = self.trades[-20:]
        
        # Analyze what's working
        by_strategy = defaultdict(list)
        for trade in recent:
            strategy = trade.get("strategy_tag", "unknown")
            by_strategy[strategy].append(trade.get("pnl", 0))
        
        recommendations = []
        
        for strategy, pnls in by_strategy.items():
            wins = sum(1 for p in pnls if p > 0)
            total = len(pnls)
            win_rate = (wins / total * 100) if total > 0 else 0
            total_pnl = sum(pnls)
            
            if win_rate >= 70 and total >= 3:
                recommendations.append(f"✅ KEEP USING: '{strategy}' ({win_rate:.0f}% win rate, +${total_pnl:.2f})")
            elif win_rate < 40 and total >= 3:
                recommendations.append(f"❌ STOP USING: '{strategy}' ({win_rate:.0f}% win rate, -${abs(total_pnl):.2f})")
        
        return " | ".join(recommendations) if recommendations else "All strategies showing mixed results. Continue refining."
    
    def get_regime_performance(self, regime_filter: Optional[str] = None) -> Dict[str, Any]:
        """
        Get performance breakdown by market regime.
        
        Args:
            regime_filter: Optional filter for specific regime ("bull_trend", "bear_trend", "range_chop", "volatile")
        
        Returns:
            Performance metrics grouped by market regime
        """
        if not self.trades:
            return {"message": "No trades yet"}
        
        # Group trades by regime
        by_regime = defaultdict(lambda: {"wins": 0, "losses": 0, "total_pnl": 0.0, "strategies": defaultdict(list)})
        
        for trade in self.trades:
            regime = trade.get("regime", "unknown")
            
            # Apply filter if specified
            if regime_filter and regime != regime_filter:
                continue
            
            pnl = trade.get("pnl", 0)
            strategy = trade.get("strategy_tag", "unknown")
            
            by_regime[regime]["total_pnl"] += pnl
            by_regime[regime]["strategies"][strategy].append(pnl)
            
            if pnl > 0:
                by_regime[regime]["wins"] += 1
            else:
                by_regime[regime]["losses"] += 1
        
        # Calculate metrics for each regime
        regime_metrics = {}
        for regime, data in by_regime.items():
            total_trades = data["wins"] + data["losses"]
            win_rate = (data["wins"] / total_trades * 100) if total_trades > 0 else 0
            
            # Calculate best strategy for this regime
            best_strategy = None
            best_strategy_pnl = float('-inf')
            strategy_breakdown = {}
            
            for strategy, pnls in data["strategies"].items():
                strat_total = sum(pnls)
                strat_wins = sum(1 for p in pnls if p > 0)
                strat_total_trades = len(pnls)
                strat_win_rate = (strat_wins / strat_total_trades * 100) if strat_total_trades > 0 else 0
                
                strategy_breakdown[strategy] = {
                    "trades": strat_total_trades,
                    "win_rate": round(strat_win_rate, 1),
                    "total_pnl": round(strat_total, 2),
                }
                
                if strat_total > best_strategy_pnl:
                    best_strategy_pnl = strat_total
                    best_strategy = strategy
            
            regime_metrics[regime] = {
                "total_trades": total_trades,
                "wins": data["wins"],
                "losses": data["losses"],
                "win_rate": round(win_rate, 1),
                "total_pnl": round(data["total_pnl"], 2),
                "best_strategy": best_strategy,
                "strategy_breakdown": strategy_breakdown,
            }
        
        # Generate regime-specific recommendations
        recommendations = []
        for regime, metrics in regime_metrics.items():
            if metrics["win_rate"] >= 70 and metrics["total_trades"] >= 5:
                recommendations.append(
                    f"✅ Strong in '{regime}': {metrics['win_rate']:.0f}% win rate. "
                    f"Best strategy: {metrics['best_strategy']}"
                )
            elif metrics["win_rate"] < 40 and metrics["total_trades"] >= 5:
                recommendations.append(
                    f"⚠️ Weak in '{regime}': only {metrics['win_rate']:.0f}% win rate. "
                    f"Consider skipping trades or changing strategy"
                )
        
        return {
            "by_regime": regime_metrics,
            "recommendations": recommendations,
            "best_performing_regime": max(
                regime_metrics.items(),
                key=lambda x: x[1]["total_pnl"]
            )[0] if regime_metrics else None,
        }


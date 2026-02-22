"""
Strategy Optimizer - Simple reinforcement learning for trading strategies
"""
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from loguru import logger
import json


class StrategyOptimizer:
    """Simple strategy optimization using performance feedback"""

    def __init__(self):
        """Initialize strategy optimizer"""
        self.strategy_performance = {}
        self.market_regime_stats = {}
        self.parameter_suggestions = {}
        logger.info("Strategy optimizer initialized")

    def get_stats(self) -> Dict[str, Any]:
        """Return internal stats for persistence"""
        return {
            "strategy_performance": self.strategy_performance,
            "market_regime_stats": self.market_regime_stats,
            "parameter_suggestions": self.parameter_suggestions
        }

    def load_stats(self, stats: Dict[str, Any]):
        """Load stats from persistence"""
        if not stats:
            return
        self.strategy_performance = stats.get("strategy_performance", {})
        self.market_regime_stats = stats.get("market_regime_stats", {})
        self.parameter_suggestions = stats.get("parameter_suggestions", {})
        logger.info("Strategy optimizer stats loaded from persistence")

    def analyze_strategy_performance(
        self,
        strategy_name: str,
        pnl: float,
        win: Optional[bool],
        market_regime: str = "neutral",
        timeframe: str = "1h",
        leverage: float = 10.0,
        risk_pct: float = 20.0
    ) -> Dict[str, Any]:
        """
        Analyze strategy performance and provide insights

        Args:
            strategy_name: Name of the strategy used
            pnl: Profit/loss percentage
            win: Whether the trade was profitable; None means neutral/flat outcome
            market_regime: Current market regime
            timeframe: Timeframe used
            leverage: Leverage used
            risk_pct: Risk percentage used

        Returns:
            Performance analysis and suggestions
        """
        if strategy_name not in self.strategy_performance:
            self.strategy_performance[strategy_name] = {
                "total_trades": 0,
                "wins": 0,
                "losses": 0,
                "decisive_trades": 0,
                "total_pnl": 0,
                "avg_pnl": 0,
                "win_rate": 0,
                "best_pnl": float('-inf'),
                "worst_pnl": float('inf'),
                "regime_performance": {},
                "parameter_performance": {}
            }

        strategy = self.strategy_performance[strategy_name]
        if "decisive_trades" not in strategy:
            strategy["decisive_trades"] = strategy.get("wins", 0) + strategy.get("losses", 0)
        strategy["total_trades"] += 1
        strategy["total_pnl"] += pnl

        if win is True:
            strategy["wins"] += 1
            strategy["decisive_trades"] += 1
        elif win is False:
            strategy["losses"] += 1
            strategy["decisive_trades"] += 1

        strategy["avg_pnl"] = strategy["total_pnl"] / strategy["total_trades"]
        decisive_total = max(1, int(strategy.get("decisive_trades", 0)))
        strategy["win_rate"] = strategy["wins"] / decisive_total
        strategy["best_pnl"] = max(strategy["best_pnl"], pnl)
        strategy["worst_pnl"] = min(strategy["worst_pnl"], pnl)

        # Track regime performance
        if market_regime not in strategy["regime_performance"]:
            strategy["regime_performance"][market_regime] = {"trades": 0, "wins": 0, "decisive_trades": 0, "pnl": 0}

        regime_stats = strategy["regime_performance"][market_regime]
        if "decisive_trades" not in regime_stats:
            regime_stats["decisive_trades"] = regime_stats.get("wins", 0)
        regime_stats["trades"] += 1
        regime_stats["pnl"] += pnl
        if win is True:
            regime_stats["wins"] += 1
            regime_stats["decisive_trades"] += 1
        elif win is False:
            regime_stats["decisive_trades"] += 1

        # Track parameter performance (simplified)
        param_key = f"L{leverage}_R{risk_pct}_T{timeframe}"
        if param_key not in strategy["parameter_performance"]:
            strategy["parameter_performance"][param_key] = {"trades": 0, "wins": 0, "decisive_trades": 0, "pnl": 0}

        param_stats = strategy["parameter_performance"][param_key]
        if "decisive_trades" not in param_stats:
            param_stats["decisive_trades"] = param_stats.get("wins", 0)
        param_stats["trades"] += 1
        param_stats["pnl"] += pnl
        if win is True:
            param_stats["wins"] += 1
            param_stats["decisive_trades"] += 1
        elif win is False:
            param_stats["decisive_trades"] += 1

        return self._generate_insights(strategy_name, strategy)

    def _generate_insights(self, strategy_name: str, strategy_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate insights and suggestions based on performance"""

        insights = {
            "strategy_name": strategy_name,
            "overall_performance": {
                "total_trades": strategy_data["total_trades"],
                "win_rate": round(strategy_data["win_rate"] * 100, 1),
                "avg_pnl": round(strategy_data["avg_pnl"], 2),
                "total_pnl": round(strategy_data["total_pnl"], 2)
            },
            "recommendations": [],
            "warnings": [],
            "optimal_conditions": {}
        }

        # Performance assessment
        win_rate = strategy_data["win_rate"]
        avg_pnl = strategy_data["avg_pnl"]

        if win_rate < 0.4:
            insights["warnings"].append("Low win rate - consider strategy modification")
        elif win_rate > 0.65:
            insights["recommendations"].append("High win rate - consider increasing position size")

        if avg_pnl < -0.5:
            insights["warnings"].append("Negative average PnL - strategy not profitable")
        elif avg_pnl > 1.0:
            insights["recommendations"].append("Strong performance - continue using this strategy")

        # Regime analysis
        best_regime = None
        best_regime_win_rate = 0

        for regime, stats in strategy_data["regime_performance"].items():
            if stats["trades"] >= 3:  # Need minimum sample
                decisive = max(1, int(stats.get("decisive_trades", stats.get("trades", 0))))
                regime_win_rate = stats["wins"] / decisive
                if regime_win_rate > best_regime_win_rate:
                    best_regime = regime
                    best_regime_win_rate = regime_win_rate

        if best_regime:
            insights["optimal_conditions"]["best_market_regime"] = best_regime
            insights["recommendations"].append(f"Best performance in {best_regime} market regime")

        # Parameter optimization
        best_params = None
        best_param_pnl = float('-inf')

        for param_key, stats in strategy_data["parameter_performance"].items():
            if stats["trades"] >= 2:
                avg_param_pnl = stats["pnl"] / stats["trades"]
                if avg_param_pnl > best_param_pnl:
                    best_params = param_key
                    best_param_pnl = avg_param_pnl

        if best_params:
            insights["optimal_conditions"]["best_parameters"] = best_params
            insights["recommendations"].append(f"Best parameters: {best_params}")

        return insights

    def get_strategy_recommendations(self, available_strategies: List[str]) -> Dict[str, Any]:
        """
        Get recommendations for which strategies to use based on performance

        Args:
            available_strategies: List of available strategy names

        Returns:
            Strategy recommendations
        """
        recommendations = {
            "recommended_strategies": [],
            "avoid_strategies": [],
            "performance_summary": {}
        }

        for strategy in available_strategies:
            if strategy in self.strategy_performance:
                data = self.strategy_performance[strategy]
                win_rate = data["win_rate"]
                avg_pnl = data["avg_pnl"]

                recommendations["performance_summary"][strategy] = {
                    "win_rate": round(win_rate * 100, 1),
                    "avg_pnl": round(avg_pnl, 2),
                    "total_trades": data["total_trades"]
                }

                # Recommend strategies with good performance
                if win_rate > 0.55 and avg_pnl > 0.2 and data["total_trades"] >= 5:
                    recommendations["recommended_strategies"].append(strategy)
                elif win_rate < 0.35 or (avg_pnl < -0.5 and data["total_trades"] >= 3):
                    recommendations["avoid_strategies"].append(strategy)

        return recommendations

    def get_market_regime_adaptation(self, current_regime: str) -> Dict[str, Any]:
        """
        Get adaptation suggestions based on market regime

        Args:
            current_regime: Current market regime

        Returns:
            Regime-specific suggestions
        """
        adaptations = {
            "bullish": {
                "leverage_adjustment": 1.2,  # Increase leverage in bull markets
                "risk_adjustment": 0.9,     # Slightly reduce risk
                "preferred_strategies": ["momentum_continuation", "breakout"],
                "avoid_strategies": ["mean_reversion", "oversold_reversal"]
            },
            "bearish": {
                "leverage_adjustment": 0.8,  # Reduce leverage in bear markets
                "risk_adjustment": 1.1,     # Increase caution
                "preferred_strategies": ["short_squeeze", "bearish_breakdown"],
                "avoid_strategies": ["momentum_continuation"]
            },
            "sideways": {
                "leverage_adjustment": 0.9,  # Conservative in ranging markets
                "risk_adjustment": 1.0,     # Standard risk
                "preferred_strategies": ["range_trading", "mean_reversion"],
                "avoid_strategies": ["breakout", "momentum_continuation"]
            },
            "volatile": {
                "leverage_adjustment": 0.7,  # Much more conservative
                "risk_adjustment": 0.8,     # Reduce risk in volatility
                "preferred_strategies": ["volatility_breakout"],
                "avoid_strategies": ["scalping", "tight_ranges"]
            }
        }

        return adaptations.get(current_regime, {
            "leverage_adjustment": 1.0,
            "risk_adjustment": 1.0,
            "preferred_strategies": [],
            "avoid_strategies": []
        })

    def save_learning_data(self, filename: str = "strategy_learning.json"):
        """Save learning data for persistence"""
        try:
            data = {
                "strategy_performance": self.strategy_performance,
                "market_regime_stats": self.market_regime_stats,
                "last_updated": datetime.utcnow().isoformat()
            }

            with open(filename, 'w') as f:
                json.dump(data, f, indent=2, default=str)

            logger.info(f"Learning data saved to {filename}")
        except Exception as e:
            logger.warning(f"Failed to save learning data: {e}")

    def load_learning_data(self, filename: str = "strategy_learning.json"):
        """Load learning data from file"""
        try:
            with open(filename, 'r') as f:
                data = json.load(f)

            self.strategy_performance = data.get("strategy_performance", {})
            self.market_regime_stats = data.get("market_regime_stats", {})

            logger.info(f"Learning data loaded from {filename}")
        except FileNotFoundError:
            logger.info("No existing learning data found - starting fresh")
        except Exception as e:
            logger.warning(f"Failed to load learning data: {e}")

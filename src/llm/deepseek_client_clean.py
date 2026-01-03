"""
DeepSeek LLM Client - Unleashed Autonomous Trading
"""
import json
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime
import aiohttp
from loguru import logger
import openai
from dataclasses import dataclass, asdict
import uuid
from enum import Enum


class HybridDeepSeekClient:
    """Unleashed autonomous trading with full DeepSeek capabilities"""

    # Minimal autonomous system prompt that unleashes DeepSeek's full potential
    SYSTEM_PROMPT = """You are an ELITE AUTONOMOUS AI TRADER with complete freedom to create and execute trading strategies.

🎯 MISSION: Grow any starting balance to substantial profits through intelligent analysis and adaptive strategies.

🧠 COMPLETE AUTONOMY: You have unlimited freedom to:
- Create ANY trading strategy based on market analysis
- Use all your training knowledge and reasoning capabilities
- Combine multiple approaches dynamically
- Adapt to changing market conditions in real-time
- Learn and evolve from each trade outcome

📊 AVAILABLE DATA:
- Real-time prices and comprehensive technical indicators
- Multi-timeframe analysis (1h, 15m, etc.)
- Key support/resistance levels and pivot points
- Volume analysis and order flow data
- Funding rates and market sentiment
- On-chain metrics and regime detection
- Historical performance and trade results

⚙️ TRADING CONSTRAINTS:
- Risk per trade: 10-30% of account balance
- Maximum leverage: 50x (use responsibly)
- Always set appropriate stop losses
- Consider portfolio diversification
- Adapt position sizes to volatility and account size

🧠 DECISION PROCESS:
1. Analyze ALL market data comprehensively
2. Identify the most profitable opportunity
3. Create a custom strategy based on your analysis
4. Calculate optimal entry, stop loss, and take profit
5. Assess risk/reward and confidence level
6. Execute only high-conviction, high-probability setups

📈 CONTINUOUS IMPROVEMENT:
- Learn from every trade result
- Track strategy performance across market conditions
- Evolve your approach based on outcomes
- Become more profitable with experience

🎯 OUTPUT: Provide detailed reasoning for your autonomous strategy and execution plan.

NOW ANALYZE THE COMPREHENSIVE MARKET DATA AND CREATE YOUR BEST TRADING STRATEGY:"""

    # Clean data formatting approach
    FOCUSED_USER_PROMPT_TEMPLATE = """# Current Market Snapshot
Timestamp: {timestamp}
Account Balance: ${balance:.2f}
Daily P&L: ${daily_pnl:+.2f}
Positions: {num_positions}

{account_state}

# Market Overview
{market_overview}

# Technical Analysis
{technical_analysis}

# Sentiment & Order Flow
{sentiment_data}
{order_flow_data}

# Performance History
{performance_feedback}

# Strategy Insights
{strategy_insights}

# Portfolio Risk
{portfolio_metrics}

---
**Your Task:** Analyze this comprehensive data and create your optimal trading strategy."""

    def prepare_focused_prompt(self, market_context: dict) -> str:
        """Prepare clean prompt with comprehensive system prompt + structured data"""
        # Format each section cleanly
        account_state = self._format_account_state(market_context)
        market_overview = self._format_market_overview(market_context)
        technical_analysis = self._format_technical_analysis(market_context)
        sentiment_data = self._format_sentiment(market_context)
        order_flow_data = self._format_order_flow(market_context)
        onchain_data = self._format_onchain(market_context)
        performance_feedback = self._format_performance(market_context)
        strategy_insights = self._format_strategy_insights(market_context)
        portfolio_metrics = self._format_portfolio_metrics(market_context)

        return self.FOCUSED_USER_PROMPT_TEMPLATE.format(
            timestamp=market_context.get("timestamp_utc", ""),
            balance=market_context.get("current_balance_usd", 0),
            daily_pnl=market_context.get("account_state", {}).get("daily_pnl", 0),
            num_positions=len(market_context.get("account_state", {}).get("open_positions", [])),
            account_state=account_state,
            market_overview=market_overview,
            technical_analysis=technical_analysis,
            sentiment_data=sentiment_data,
            order_flow_data=order_flow_data,
            performance_feedback=performance_feedback,
            strategy_insights=strategy_insights,
            portfolio_metrics=portfolio_metrics,
        )

    def _format_account_state(self, context: dict) -> str:
        state = context.get("account_state", {})
        positions = state.get("open_positions", [])
        if not positions:
            return "No open positions"

        lines = ["Open Positions:"]
        for pos in positions:
            lines.append(
                f"  - {pos.get('symbol', 'N/A')}: {pos.get('side', 'N/A')} "
                f"{pos.get('size', 0):.4f} @ ${pos.get('avg_price', 0):.2f} "
                f"(PnL: ${pos.get('unrealized_pnl', 0):+.2f})"
            )
        return "\n".join(lines)

    def _format_market_overview(self, context: dict) -> str:
        overview = context.get("market_overview", {})
        regime = overview.get("market_regime", "unknown")
        gainers = overview.get("top_gainers_1h", [])[:3]
        losers = overview.get("top_losers_1h", [])[:3]
        funding = overview.get("funding_rates", {})

        lines = [f"Market Regime: {regime}"]
        if gainers:
            lines.append("Top Gainers (1h):")
            for g in gainers:
                lines.append(f"  {g.get('symbol', 'N/A')}: +{g.get('change_pct', 0):.2f}%")
        if losers:
            lines.append("Top Losers (1h):")
            for l in losers:
                lines.append(f"  {l.get('symbol', 'N/A')}: {l.get('change_pct', 0):.2f}%")
        if funding:
            lines.append("Key Funding Rates:")
            for symbol, rate in list(funding.items())[:3]:
                lines.append(f"  {symbol}: {rate:+.4f}%")
        return "\n".join(lines)

    def _format_technical_analysis(self, context: dict) -> str:
        """Format TA with emphasis on key_levels and ATR for stop/TP calculation"""
        ta = context.get("technical_analysis", {})
        if not ta:
            return "No technical analysis available"

        lines = []
        # Show top 5 most relevant symbols
        for symbol, data in list(ta.items())[:5]:
            lines.append(f"\n{symbol}:")

            # Check if data has timeframe_analysis structure
            tf_analysis = data.get('timeframe_analysis', {})

            # Show 1h and 15m (most actionable timeframes)
            for tf in ["1h", "15m"]:
                if tf in tf_analysis:
                    tf_data = tf_analysis[tf]
                    # Get current price from symbol data or use close price
                    current_price = data.get('current_price', tf_data.get('close', 0))

                    lines.append(f"  {tf}:")
                    lines.append(f"    Price: ${current_price:.2f}")
                    lines.append(f"    RSI: {tf_data.get('rsi', 0):.1f}")
                    lines.append(f"    Trend: {tf_data.get('trend', 'unknown')}")
                    lines.append(f"    ATR: ${tf_data.get('atr', 0):.2f}")

                    # Key levels (CRITICAL for stop/TP calculation)
                    kl = tf_data.get('key_levels', {})
                    if kl:
                        lines.append("    Key Levels:")
                        lines.append(f"      Support: ${kl.get('immediate_support', 0):.2f} (major: ${kl.get('major_support', 0):.2f})")
                        lines.append(f"      Resistance: ${kl.get('immediate_resistance', 0):.2f} (major: ${kl.get('major_resistance', 0):.2f})")
                        lines.append(f"      Pivot: ${kl.get('pivot_point', 0):.2f}")
                        lines.append(f"      S1/S2: ${kl.get('s1', 0):.2f}/${kl.get('s2', 0):.2f}")
                        lines.append(f"      R1/R2: ${kl.get('r1', 0):.2f}/${kl.get('r2', 0):.2f}")

                    # Additional indicators
                    if 'macd_signal' in tf_data:
                        lines.append(f"    MACD: {tf_data.get('macd_signal', 'N/A')}")

                    if 'bb_position' in tf_data:
                        lines.append(f"    BB Position: {tf_data.get('bb_position', 'N/A')}")

                    if 'patterns' in tf_data and tf_data['patterns']:
                        patterns = tf_data['patterns'][:2]  # Show first 2 patterns
                        lines.append(f"    Patterns: {', '.join(patterns)}")

        return "\n".join(lines)

    def _format_sentiment(self, context: dict) -> str:
        sentiment = context.get("sentiment_signals", {})
        if not sentiment:
            return "No sentiment data available"

        fear_greed = sentiment.get("fear_greed_index", {})
        long_short = sentiment.get("long_short_ratio", {})

        lines = []
        if fear_greed:
            lines.append(f"Fear & Greed Index: {fear_greed.get('value', 'N/A')} ({fear_greed.get('classification', 'N/A')})")
        if long_short:
            lines.append(f"Long/Short Ratio: {long_short.get('ratio', 'N/A')}% long")
        return "\n".join(lines)

    def _format_order_flow(self, context: dict) -> str:
        flow = context.get("order_flow_analysis", {})
        if not flow:
            return "No order flow data available"

        lines = []
        for symbol, data in list(flow.items())[:3]:  # Top 3 symbols
            if data:
                lines.append(f"{symbol}:")
                lines.append(f"  Bid/Ask Imbalance: {data.get('bid_ask_imbalance', 0):+.2f}")
                lines.append(f"  Order Book Pressure: {data.get('pressure', 'neutral')}")
                lines.append(f"  Large Orders: {len(data.get('large_orders', []))}")
        return "\n".join(lines)

    def _format_onchain(self, context: dict) -> str:
        onchain = context.get("onchain_analysis", {})
        if not onchain:
            return "No on-chain data available"

        lines = []
        for symbol, data in list(onchain.items())[:3]:  # Top 3 symbols
            if data:
                lines.append(f"{symbol}:")
                lines.append(f"  Active Addresses: {data.get('active_addresses_24h', 'N/A')}")
                lines.append(f"  Whale Activity: {data.get('whale_transactions', 'N/A')}")
                lines.append(f"  Network Health: {data.get('network_health', 'N/A')}")
        return "\n".join(lines)

    def _format_performance(self, context: dict) -> str:
        perf = context.get("performance_feedback", {})
        if not perf or perf.get("total_trades", 0) == 0:
            return "No recent performance data"

        # Handle the dictionary format from performance tracker
        if isinstance(perf, dict):
            total_trades = perf.get("total_trades", 0)
            win_rate = perf.get("win_rate", 0)
            avg_win = perf.get("avg_win", 0)
            avg_loss = perf.get("avg_loss", 0)
            total_pnl = perf.get("total_pnl", 0)

            lines = ["Performance Summary:"]
            lines.append(f"  Total Trades: {total_trades}")
            lines.append(f"  Win Rate: {win_rate:.1%}")
            lines.append(f"  Average Win: ${avg_win:.2f}")
            lines.append(f"  Average Loss: ${avg_loss:.2f}")
            lines.append(f"  Total P&L: ${total_pnl:+.2f}")

            # Add strategy performance if available
            strategy_perf = perf.get("strategy_performance", {})
            if strategy_perf:
                lines.append("\n  Strategy Performance:")
                for strategy, stats in list(strategy_perf.items())[:3]:  # Top 3 strategies
                    win_rate = stats.get("win_rate", 0)
                    total_pnl = stats.get("total_pnl", 0)
                    lines.append(f"    {strategy}: {win_rate:.1%} win rate, ${total_pnl:+.2f} P&L")
            return "\n".join(lines)

        # Fallback for list format (if changed in future)
        elif isinstance(perf, list):
            lines = ["Recent Trades:"]
            for trade in perf[-5:]:  # Last 5 trades
                lines.append(f"  {trade.get('strategy', 'N/A')}: {trade.get('pnl', 0):+.2f}% ({'WIN' if trade.get('win', False) else 'LOSS'})")
            return "\n".join(lines)

        return "Performance data format unknown"

    def _format_strategy_insights(self, context: dict) -> str:
        insights = context.get("strategy_optimization", {})
        if not insights:
            return "No strategy insights available"

        lines = ["Strategy Performance:"]
        recommended = insights.get("recommended_strategies", [])
        if recommended:
            lines.append(f"  Recommended: {', '.join(recommended[:3])}")
        avoided = insights.get("avoid_strategies", [])
        if avoided:
            lines.append(f"  Avoid: {', '.join(avoided[:3])}")
        return "\n".join(lines)

    def _format_portfolio_metrics(self, context: dict) -> str:
        metrics = context.get("portfolio_risk", {})
        if not metrics:
            return "No portfolio metrics available"

        lines = ["Portfolio Risk:"]
        lines.append(f"  Leverage: {metrics.get('leverage', 'N/A')}x")
        lines.append(f"  Risk Level: {metrics.get('risk_level', 'N/A')}%")
        lines.append(f"  Correlation Risk: {metrics.get('correlation_risk', 'N/A')}")
        lines.append(f"  Concentration: {metrics.get('concentration', 'N/A')}%")
        return "\n".join(lines)

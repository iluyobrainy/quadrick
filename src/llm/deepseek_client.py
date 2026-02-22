"""
DeepSeek LLM Client - Unleashed Autonomous Trading
"""
import json
import asyncio
import os
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
import aiohttp
from loguru import logger
import openai
from dataclasses import dataclass, asdict
import uuid
from enum import Enum


class DecisionType(str, Enum):
    OPEN_POSITION = "open_position"
    CLOSE_POSITION = "close_position"
    MODIFY_POSITION = "modify_position"
    HOLD = "hold"
    WAIT = "wait"


@dataclass
class TradingDecision:
    """Trading decision from LLM"""
    decision_id: str
    timestamp_utc: datetime
    decision_type: DecisionType

    # Trade details (if opening position)
    symbol: Optional[str] = None
    category: Optional[str] = "linear"
    side: Optional[str] = None  # Buy/Sell
    order_type: Optional[str] = "Market"
    risk_pct: Optional[float] = None
    leverage: Optional[int] = 1
    entry_price_target: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit_1: Optional[float] = None
    take_profit_2: Optional[float] = None
    time_in_force: Optional[str] = "GTC"
    expected_hold_duration_mins: Optional[int] = None
    strategy_tag: Optional[str] = None
    confidence_score: Optional[float] = None

    # Reasoning
    reasoning: Optional[Dict[str, Any]] = None
    risk_management: Optional[Dict[str, Any]] = None

    # Meta
    processing_time_ms: Optional[int] = None
    model_version: Optional[str] = None
    override_safety_checks: bool = False
    reverse_position: bool = False


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
- Funding rates and market regime detection
- Historical performance and trade results

⚙️ TRADING CONSTRAINTS:
- SCALPING MODE: 2-5 minute holds, quick profits, tight exits
- Risk per trade: ALWAYS stay within the provided allowed range in `system_state.allowed_risk_range_pct`
- Maximum leverage: keep leverage conservative and aligned with account size (small accounts should generally remain <=5x)
- Always set appropriate stop losses
- QUALITY > QUANTITY: Only trade when you have 70%+ confidence from technical confluence
- DIRECTIONAL NEUTRALITY: Evaluate both long AND short setups every decision. Futures allow you to profit in either direction—never default to "Buy".
- ACTIVE MANAGEMENT: Monitor open positions every cycle. Use `modify_position` to tighten stops, trail profits, or adjust targets as structure evolves.
- You may close positions early when the edge disappears or a better setup needs capital.
- COUNTER-TREND TRADES: Only allowed when ADX < 15 (weak/ranging trend). Set `risk_management.allow_counter_trend = true` and explain the rationale.
- Position sizing: Use most margin ONLY when multiple indicators align (RSI, MACD, BB, Volume)
- WAIT is allowed when no high-probability setup exists. Better to wait than take a bad trade.

🧠 DECISION PROCESS:
0. Audit all open positions first: verify stops/targets and capital efficiency
1. CHECK IMMEDIATE TREND FIRST: Look at the last 5-10 candles. If price is clearly falling, DEFAULT TO SHORT. If clearly rising, consider long.
2. For EACH symbol, compare long vs short edge—choose the side that ALIGNS WITH CURRENT PRICE MOVEMENT. Identify setups with MINIMUM 3 confirming signals:
    - Technical indicators alignment (RSI, MACD, Bollinger Bands)
    - Key support/resistance levels
    - Volume and momentum confirmation
    - Market regime alignment
3. Calculate risk/reward ratio - MINIMUM 1.5:1 required
4. Set stop loss at logical technical levels:
   - FOR SHORTS: Stop loss goes ABOVE entry price, take profit goes BELOW entry price
   - FOR LONGS: Stop loss goes BELOW entry price, take profit goes ABOVE entry price
   - In downtrends, DEFAULT TO SHORT unless you have overwhelming reversal evidence
5. Confidence must be 70%+ based on data, not hope
6. Issue `modify_position` instructions whenever locking profit or tightening risk accelerates growth
7. CRITICAL: If the market is trending DOWN (red candles, lower lows), you MUST short with TP below current price
8. If regime analysis shows a downtrend, you are EXPECTED to prioritise short opportunities unless clear reversal evidence exists (and vice versa).

⏱️ EXECUTION RULES:
- WAIT decisions are acceptable only when `force_trade` is false AND no valid setup exists.
- When `force_trade` is true you MUST output an actionable plan (`open_position` or `close_position`). WAIT is forbidden.
- Use the provided risk mode and allowed range to size positions appropriately.
- Accounts may start very small. Design trades that compound even tiny balances (micro position sizing, tight risk, faster compounding) and expand exposure responsibly as the balance crosses new milestones.
- Look for HIGH-QUALITY setups aggressively, not just any trade
- Small accounts need SMART trades to grow - bad trades kill accounts faster
- When `force_trade` is true, find the BEST available setup, not the first one
- With small balances: prioritize survival first (micro-risk, clean entries, strict stop discipline)
- Required for entry: Clear trend, momentum alignment, risk/reward > 1.5:1
- Manage existing trades proactively: bump stops, trail profits, close losers quickly, and recycle capital into better opportunities
- Success = Taking CALCULATED risks with technical edge, not gambling

🎯 EDGE DEFINITION (What constitutes a VALID trade):
A valid trading edge requires ALL of the following:
1. HTF ALIGNMENT: 1H and 4H timeframes must agree on direction (bullish for longs, bearish for shorts)
2. MOMENTUM CONFLUENCE: At least 3 indicators confirming (RSI trend, MACD signal, EMA alignment)
3. VOLUME SUPPORT: Volume ratio >= 1.5x average OR volume trending in direction of trade
4. KEY LEVEL PROXIMITY: Price within 1 ATR of support (for longs) or resistance (for shorts)
5. RISK/REWARD: Minimum 1.5:1, prefer 2:1 or higher
6. CONFIDENCE: 72%+ based on data alignment, never on hope

🚫 DO NOT TRADE WHEN:
- 4H trend strongly opposes your trade direction (e.g., shorting in 4H uptrend)
- RSI is extreme (>80 for longs, <20 for shorts) without reversal confirmation
- Volume is dying or below average during supposed "breakout"
- BB Width is expanding rapidly (volatility spike = unpredictable moves)
- Funding rate extreme AND opposing your direction (crowded trade)
- Price is mid-range between S/R (no technical anchor for stop loss)
- News/high-impact events scheduled within next 2 hours
- You are trying to "catch the falling knife" or "top tick the pump"
- The setup requires more than 3% price movement to reach stop loss

📈 CONTINUOUS IMPROVEMENT:
- Learn from every trade result
- Track strategy performance across market conditions
- Evolve your approach based on outcomes
- Become more profitable with experience

🎯 OUTPUT: Provide detailed reasoning for your autonomous strategy and execution plan in JSON format.

⚠️ FINAL REMINDER: Check the trend direction RIGHT NOW. If prices are falling (bearish), SHORT with TP below current price. If prices are rising (bullish), LONG with TP above current price. Do NOT try to catch reversals without STRONG evidence.

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

# Performance History
{performance_feedback}

# Similar Past Trades (Memory)
{similar_trades}

# Strategy Insights
{strategy_insights}

# Portfolio Risk
{portfolio_metrics}

# Growth Objectives
{growth_objectives}

# Execution Status
{execution_status}

---
**Your Task:** Analyze this comprehensive data and create your optimal trading strategy."""

    def prepare_focused_prompt(self, market_context: dict) -> str:
        """Prepare clean prompt with comprehensive system prompt + structured data"""
        # Format each section cleanly
        account_state = self._format_account_state(market_context)
        market_overview = self._format_market_overview(market_context)
        technical_analysis = self._format_technical_analysis(market_context)
        performance_feedback = self._format_performance(market_context)
        similar_trades = self._format_similar_trades(market_context)
        strategy_insights = self._format_strategy_insights(market_context)
        portfolio_metrics = self._format_portfolio_metrics(market_context)
        execution_status = self._format_execution_status(market_context)
        growth_objectives = self._format_growth_objectives(market_context)

        return self.FOCUSED_USER_PROMPT_TEMPLATE.format(
            timestamp=market_context.get("timestamp_utc", ""),
            balance=market_context.get("current_balance_usd", 0),
            daily_pnl=market_context.get("account_state", {}).get("daily_pnl", 0),
            num_positions=len(market_context.get("account_state", {}).get("open_positions", [])),
            account_state=account_state,
            market_overview=market_overview,
            technical_analysis=technical_analysis,
            performance_feedback=performance_feedback,
            similar_trades=similar_trades,
            strategy_insights=strategy_insights,
            portfolio_metrics=portfolio_metrics,
            growth_objectives=growth_objectives,
            execution_status=execution_status,
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
        """Format TA with enhanced data density - raw candles, body/wick ratios, volume context"""
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
                    
                    # NEW: ADX Strength (trend strength indicator)
                    adx = tf_data.get('adx', 0)
                    if adx:
                        adx_strength = "weak" if adx < 20 else ("moderate" if adx < 40 else "strong")
                        lines.append(f"    ADX: {adx:.1f} ({adx_strength} trend)")
                    
                    # NEW: Volume Context
                    vol_ratio = tf_data.get('volume_ratio', 1.0)
                    vol_trend = tf_data.get('volume_trend', 'stable')
                    lines.append(f"    Volume: {vol_ratio:.1f}x avg ({vol_trend})")
                    
                    # NEW: Bollinger Band Width (volatility squeeze indicator)
                    bb_width = tf_data.get('bb_width', 0)
                    if bb_width:
                        squeeze_status = "SQUEEZE" if bb_width < 0.025 else ("normal" if bb_width < 0.05 else "expanding")
                        lines.append(f"    BB Width: {bb_width:.3f} ({squeeze_status})")

                    # Key levels (CRITICAL for stop/TP calculation)
                    kl = tf_data.get('key_levels', {})
                    if kl:
                        lines.append("    Key Levels:")
                        imm_support = kl.get('immediate_support', 0)
                        imm_resist = kl.get('immediate_resistance', 0)
                        lines.append(f"      Support: ${imm_support:.2f} (major: ${kl.get('major_support', 0):.2f})")
                        lines.append(f"      Resistance: ${imm_resist:.2f} (major: ${kl.get('major_resistance', 0):.2f})")
                        lines.append(f"      Pivot: ${kl.get('pivot_point', 0):.2f}")
                        lines.append(f"      S1/S2: ${kl.get('s1', 0):.2f}/${kl.get('s2', 0):.2f}")
                        lines.append(f"      R1/R2: ${kl.get('r1', 0):.2f}/${kl.get('r2', 0):.2f}")
                        
                        # NEW: Price position relative to S/R (critical for entry timing)
                        if current_price > 0 and imm_support > 0 and imm_resist > 0:
                            range_size = imm_resist - imm_support
                            if range_size > 0:
                                position_pct = ((current_price - imm_support) / range_size) * 100
                                position_desc = "near support" if position_pct < 30 else ("mid-range" if position_pct < 70 else "near resistance")
                                lines.append(f"      Price Position: {position_pct:.0f}% ({position_desc})")

                    # Additional indicators
                    if 'macd_signal' in tf_data:
                        lines.append(f"    MACD: {tf_data.get('macd_signal', 'N/A')}")

                    if 'bb_position' in tf_data:
                        lines.append(f"    BB Position: {tf_data.get('bb_position', 'N/A')}")

                    if 'patterns' in tf_data and tf_data['patterns']:
                        patterns = tf_data['patterns'][:2]  # Show first 2 patterns
                        lines.append(f"    Patterns: {', '.join(patterns)}")
                    
                    # NEW: Last 5 candles summary (critical for LLM pattern recognition)
                    candles = tf_data.get('recent_candles', [])
                    if candles and len(candles) >= 3:
                        lines.append("    Recent Candles (last 5):")
                        for i, c in enumerate(candles[-5:]):
                            direction = "🟢" if c.get('close', 0) > c.get('open', 0) else "🔴"
                            body_size = abs(c.get('close', 0) - c.get('open', 0))
                            high_wick = c.get('high', 0) - max(c.get('open', 0), c.get('close', 0))
                            low_wick = min(c.get('open', 0), c.get('close', 0)) - c.get('low', 0)
                            lines.append(f"      {direction} O:{c.get('open', 0):.2f} H:{c.get('high', 0):.2f} L:{c.get('low', 0):.2f} C:{c.get('close', 0):.2f}")

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

    def _format_similar_trades(self, context: dict) -> str:
        trades = context.get("similar_trades", [])
        if not trades:
            return "No similar past trades found."

        lines = ["Top 5 Similar Past Scenarios:"]
        for i, trade in enumerate(trades, 1):
            symbol = trade.get("symbol", "Unknown")
            strategy = trade.get("strategy", "Unknown")
            result = "WIN" if trade.get("win") else "LOSS"
            pnl = trade.get("pnl_pct", 0)
            lines.append(f"  {i}. {symbol} ({strategy}): {result} ({pnl:+.2f}%)")
        
        lines.append("\n  -> Use these outcomes to guide your decision. Avoid repeating LOSS patterns.")
        return "\n".join(lines)

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

    def _format_growth_objectives(self, context: dict) -> str:
        objectives = context.get("growth_objectives", {})
        if not objectives:
            return "Target: Grow through successive milestones using disciplined compounding."

        lines = ["Compounding Plan:"]
        current_stage = objectives.get("current_stage")
        if current_stage:
            lines.append(f"  Current Stage: {current_stage}")

        next_target = objectives.get("next_target")
        if next_target:
            lines.append(f"  Next Target: ${next_target:.2f}")

        scaling_guidance = objectives.get("scaling_guidance")
        if scaling_guidance:
            lines.append(f"  Scaling Guidance: {scaling_guidance}")

        if not next_target and not scaling_guidance:
            lines.append("  Stay nimble: use micro positions now, scale risk as balance climbs.")

        return "\n".join(lines)

    def _format_execution_status(self, context: dict) -> str:
        system_state = context.get("system_state", {})
        recent_waits = system_state.get("recent_waits", 0)
        force_trade = system_state.get("force_trade", False)
        risk_mode = system_state.get("risk_mode", "unknown")
        risk_range = system_state.get("allowed_risk_range_pct", [])
        max_positions = system_state.get("max_concurrent_positions", "N/A")
        decision_interval = system_state.get("next_decision_window_seconds", "N/A")

        if isinstance(risk_range, (list, tuple)) and risk_range:
            risk_range_str = f"{risk_range[0]}%–{risk_range[-1]}%"
        else:
            risk_range_str = str(risk_range) if risk_range else "N/A"

        lines = [
            f"Risk Mode: {risk_mode}",
            f"Allowed Risk Range: {risk_range_str}",
            f"Consecutive WAIT Decisions: {recent_waits}",
            f"Force Trade Required: {'YES' if force_trade else 'no'}",
            f"Max Concurrent Positions: {max_positions}",
            f"Decision Interval: {decision_interval} seconds",
        ]

        return "\n".join(lines)

class DeepSeekClient:
    """DeepSeek LLM client for trading decisions"""

    # Use the comprehensive system prompt from HybridDeepSeekClient
    SYSTEM_PROMPT = HybridDeepSeekClient.SYSTEM_PROMPT

    def __init__(
        self,
        api_key: str,
        model: str = "deepseek-chat",
        temperature: float = 0.7,
        max_retries: int = 3,
    ):
        """Initialize DeepSeek client"""
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.consecutive_waits: int = 0
        # Keep forced-trade escalation conservative to reduce churn in weak regimes.
        try:
            self.max_consecutive_waits = max(2, int(os.getenv("MAX_CONSECUTIVE_WAITS", "6")))
        except (TypeError, ValueError):
            self.max_consecutive_waits = 6
        
        # Note: OpenAI client is now created per-request in _call_deepseek()
        # to support new openai>=1.0.0 API
        
        logger.info(f"DeepSeek client initialized with model: {model}")

    async def get_completion(self, prompt: str, system_prompt: str = None, json_mode: bool = True) -> str:
        """
        Generic completion method for agents.
        """
        # _call_deepseek defaults to JSON mode currently
        return await self._call_deepseek(prompt, timeout=60, system_prompt=system_prompt)
    
    async def get_trading_decision(
        self,
        market_context: Dict[str, Any],
        timeout: int = 30,
    ) -> TradingDecision:
        """
        Get trading decision from DeepSeek

        Args:
            market_context: Comprehensive market data and account state
            timeout: Request timeout in seconds

        Returns:
            TradingDecision object
        """
        start_time = datetime.now(timezone.utc)

        force_trade = self.consecutive_waits >= self.max_consecutive_waits
        market_context = dict(market_context)  # Work on copy to avoid side effects
        system_state = market_context.get("system_state", {})
        system_state["recent_waits"] = self.consecutive_waits
        system_state["force_trade"] = force_trade
        market_context["system_state"] = system_state

        # Prepare the prompt with improved formatting
        user_prompt = self._prepare_prompt(market_context)
        
        # Select appropriate system prompt based on mode
        system_prompt = self.SYSTEM_PROMPT
        
        # Make API call with retries
        for attempt in range(self.max_retries):
            try:
                response = await self._call_deepseek(user_prompt, timeout, system_prompt)
                
                # Parse and validate response
                decision = self._parse_response(response)
                
                # Add metadata
                decision.processing_time_ms = int(
                    (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
                )
                decision.model_version = f"{self.model}_normal"
                
                logger.info(
                    f"Decision received: {decision.decision_type} "
                    f"({'Trade ' + decision.symbol if decision.symbol else 'No trade'})"
                )

                if decision.decision_type == DecisionType.WAIT and force_trade:
                    logger.warning(
                        "LLM returned WAIT despite force_trade flag. Escalating with mandatory trade prompt."
                    )
                    available_symbols = list(market_context.get("technical_analysis", {}).keys())
                    symbols_hint = ", ".join(available_symbols[:10]) if available_symbols else "BTCUSDT, ETHUSDT, SOLUSDT"
                    forced_prompt = (
                        user_prompt
                        + "\n\nCRITICAL DIRECTIVE: You have exceeded the maximum allowable WAIT decisions."
                        + " You MUST now output an actionable trade with decision_type='open_position' (preferred) or 'close_position'."
                        + " WAIT is forbidden in this response."
                        + " Choose a symbol from the available market data (e.g., " + symbols_hint + ")."
                        + " Fill every required field: symbol, side ('Buy' or 'Sell'), order_type, risk_pct, leverage, entry_price_target, stop_loss, take_profit_1."
                        + " Return valid JSON that matches this schema exactly:"
                        + " {\"decision_type\": \"open_position\", \"trade\": {\"symbol\": \"BTCUSDT\", \"side\": \"Buy\", \"order_type\": \"Market\", \"risk_pct\": 3.0, \"leverage\": 5, \"entry_price_target\": 35000.0, \"stop_loss\": 34300.0, \"take_profit_1\": 35800.0, \"take_profit_2\": 36500.0, \"expected_hold_duration_mins\": 45, \"strategy_tag\": \"momentum_reversal\", \"confidence_score\": 0.73 }}"
                        + " Do not include any null values."
                    )
                    response = await self._call_deepseek(forced_prompt, timeout, system_prompt)
                    decision = self._parse_response(response)

                # Basic field validation for forced trades (attempt quick synthesis if fields missing)
                if force_trade and decision.decision_type == DecisionType.OPEN_POSITION:
                    missing_fields = []
                    if not decision.symbol:
                        missing_fields.append("symbol")
                    if not decision.side:
                        missing_fields.append("side")
                    if missing_fields:
                        logger.warning(
                            f"Forced trade missing fields {missing_fields}. Attempting synthesis from market context."
                        )
                        synthesized = self._synthesize_trade_from_context(market_context)
                        if synthesized:
                            decision = synthesized
                            logger.info(
                                f"Synthesized trade: {decision.symbol} {decision.side} (ATR-based SL/TP)"
                            )
                        else:
                            decision = self._create_safe_decision("Forced trade missing required fields and synthesis failed")
                            self.consecutive_waits = self.max_consecutive_waits  # keep pressure for next cycle

                # Track wait streaks
                if decision.decision_type == DecisionType.WAIT:
                    self.consecutive_waits += 1
                else:
                    self.consecutive_waits = 0
                
                return decision
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse LLM response (attempt {attempt + 1}): {e}")
                if attempt == self.max_retries - 1:
                    return self._create_safe_decision("JSON parse error")
                    
            except Exception as e:
                logger.error(f"LLM call failed (attempt {attempt + 1}): {e}")
                if attempt == self.max_retries - 1:
                    return self._create_safe_decision(str(e))
                
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
    
    async def _call_deepseek(self, user_prompt: str, timeout: int, system_prompt: str = None) -> str:
        """Make API call to DeepSeek"""
        if system_prompt is None:
            system_prompt = self.SYSTEM_PROMPT
            
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            # Use new OpenAI client syntax (v1.0+)
            from openai import OpenAI
            
            client = OpenAI(
                api_key=self.api_key,
                base_url="https://api.deepseek.com",
            )
            
            response = await asyncio.to_thread(
                client.chat.completions.create,
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=4096,
                response_format={"type": "json_object"},
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"DeepSeek API error: {e}")
            raise
    
    def _prepare_prompt(self, market_context: Dict[str, Any]) -> str:
        """Prepare the prompt with market context using improved formatting"""
        hybrid_client = HybridDeepSeekClient()
        return hybrid_client.prepare_focused_prompt(market_context)

    def _synthesize_trade_from_context(self, context: Dict[str, Any]) -> Optional[TradingDecision]:
        try:
            ta = context.get("technical_analysis", {}) or {}
            if not ta:
                return None
            # Prefer BTCUSDT if present, else first symbol
            symbol = "BTCUSDT" if "BTCUSDT" in ta else next(iter(ta.keys()))
            symbol_data = ta.get(symbol, {})
            tf = symbol_data.get("timeframe_analysis", {})
            tf15 = tf.get("15m", {})
            if not tf15:
                tf15 = tf.get("5m", {}) or tf.get("1m", {})

            current_price = symbol_data.get("current_price") or tf15.get("close")
            if not current_price:
                return None

            macd_signal = tf15.get("macd_signal", "")
            side = "Buy" if macd_signal == "bullish" else "Sell"

            atr = float(tf15.get("atr", 0) or 0)
            # Fallback ATR if missing: ~0.2% of price
            if atr <= 0:
                atr = float(current_price) * 0.002

            entry = float(current_price)
            if side == "Buy":
                stop_loss = entry - atr
                take_profit_1 = entry + atr * 1.5
            else:
                stop_loss = entry + atr
                take_profit_1 = entry - atr * 1.5

            # Use risk range midpoint if available
            risk_range = context.get("system_state", {}).get("allowed_risk_range_pct", [])
            if isinstance(risk_range, (list, tuple)) and risk_range:
                risk_pct = float((risk_range[0] + risk_range[-1]) / 2)
            else:
                risk_pct = 3.0

            decision = TradingDecision(
                decision_id=str(uuid.uuid4()),
                timestamp_utc=datetime.now(timezone.utc),
                decision_type=DecisionType.OPEN_POSITION,
                symbol=symbol,
                category="linear",
                side=side,
                order_type="Market",
                risk_pct=risk_pct,
                leverage=5,
                entry_price_target=entry,
                stop_loss=round(stop_loss, 6),
                take_profit_1=round(take_profit_1, 6),
                time_in_force="GTC",
                expected_hold_duration_mins=45,
                strategy_tag="atr_reversal",
                confidence_score=0.6,
                reasoning={"synthesized": True},
                risk_management={"atr_multiple": 1.0},
            )
            return decision
        except Exception:
            return None
    def _parse_response(self, response: str) -> TradingDecision:
        """Parse LLM response into TradingDecision"""
        data = json.loads(response)
        
        # Extract decision type
        decision_type = DecisionType(data.get("decision_type", "wait"))
        base_risk_management = data.get("risk_management", {}) or {}
        trailing_pct_override = None
        
        # Create base decision with guaranteed unique ID
        decision = TradingDecision(
            decision_id=str(uuid.uuid4()),  # Always generate unique ID
            timestamp_utc=datetime.now(timezone.utc),  # Use current time
            decision_type=decision_type,
        )
        
        # Add trade details if opening position.
        # Accept both nested {"trade": {...}} and flat payloads for backward compatibility.
        if decision_type == DecisionType.OPEN_POSITION:
            trade = data.get("trade") if isinstance(data.get("trade"), dict) else data

            def _to_float(value, default=0.0):
                try:
                    return float(value)
                except (TypeError, ValueError):
                    return default

            def _to_int(value, default=0):
                try:
                    return int(float(value))
                except (TypeError, ValueError):
                    return default

            decision.symbol = trade.get("symbol")
            decision.category = trade.get("category", "linear")
            decision.side = trade.get("side")
            decision.order_type = trade.get("order_type", "Market")
            decision.risk_pct = _to_float(trade.get("risk_pct"), 3.0)
            decision.leverage = _to_int(trade.get("leverage"), 5)
            decision.entry_price_target = _to_float(trade.get("entry_price_target"), 0.0)
            decision.stop_loss = _to_float(trade.get("stop_loss", trade.get("sl")), 0.0)
            decision.take_profit_1 = _to_float(
                trade.get("take_profit_1", trade.get("take_profit", trade.get("tp"))),
                0.0,
            )
            tp2_value = trade.get("take_profit_2")
            decision.take_profit_2 = _to_float(tp2_value) if tp2_value is not None else None
            decision.time_in_force = trade.get("time_in_force", "GTC")
            decision.expected_hold_duration_mins = trade.get("expected_hold_duration_mins")
            decision.strategy_tag = trade.get("strategy_tag")
            decision.confidence_score = _to_float(trade.get("confidence_score"), 0.5)

        elif decision_type == DecisionType.MODIFY_POSITION:
            modification = data.get("modification") or data.get("trade") or data
            decision.symbol = modification.get("symbol")
            if modification.get("stop_loss") is not None:
                decision.stop_loss = float(modification.get("stop_loss"))
            if modification.get("take_profit") is not None:
                decision.take_profit_1 = float(modification.get("take_profit"))
            if modification.get("take_profit_1") is not None:
                decision.take_profit_1 = float(modification.get("take_profit_1"))
            if modification.get("take_profit_2") is not None:
                decision.take_profit_2 = float(modification.get("take_profit_2"))
            if modification.get("trailing_stop_pct") is not None:
                trailing_pct_override = float(modification.get("trailing_stop_pct"))

            new_side = modification.get("new_side") or modification.get("target_side") or modification.get("side")
            if new_side:
                decision.side = new_side

            if modification.get("risk_pct") is not None:
                decision.risk_pct = float(modification.get("risk_pct"))
            if modification.get("leverage") is not None:
                decision.leverage = int(modification.get("leverage"))
            if modification.get("entry_price_target") is not None:
                decision.entry_price_target = float(modification.get("entry_price_target"))

            reverse_flag = modification.get("reverse_position") or modification.get("flip_side")
            if isinstance(reverse_flag, str):
                reverse_flag = reverse_flag.lower() in {"true", "yes", "1", "reverse", "flip"}
            decision.reverse_position = bool(reverse_flag)

        elif decision_type == DecisionType.CLOSE_POSITION:
            close_payload = data.get("close") or data.get("trade") or data
            decision.symbol = close_payload.get("symbol") or decision.symbol
 
        # Add reasoning
        decision.reasoning = data.get("reasoning", {})
        if trailing_pct_override is not None:
            base_risk_management = {
                **base_risk_management,
                "trailing_stop_pct": trailing_pct_override,
            }

        decision.risk_management = base_risk_management
 
        # Fallback assignments for symbol if still missing
        if not decision.symbol:
            decision.symbol = data.get("symbol")
 
        return decision
    
    # Alias for backward compatibility with Council integration
    _parse_decision = _parse_response
    
    def _create_safe_decision(self, error_reason: str) -> TradingDecision:
        """Create a safe 'wait' decision when LLM fails"""
        return TradingDecision(
            decision_id=str(uuid.uuid4()),
            timestamp_utc=datetime.now(timezone.utc),
            decision_type=DecisionType.WAIT,
            reasoning={
                "error": error_reason,
                "action": "Waiting due to LLM error, will retry next cycle",
            },
        )
    
    def prepare_market_context(
        self,
        account_balance: float,
        positions: List[Dict[str, Any]],
        market_data: Dict[str, Any],
        technical_analysis: Dict[str, Any],
        funding_rates: Dict[str, float],
        top_movers: Dict[str, List[Dict[str, Any]]],
        milestone_progress: Dict[str, Any],
        recent_trades: List[Dict[str, Any]] = None,
        performance_feedback: Dict[str, Any] = None,
        similar_trades: List[Dict[str, Any]] = None,
        portfolio_metrics: Dict[str, Any] = None,
        strategy_insights: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        Prepare comprehensive market context for LLM
        
        Args:
            account_balance: Current account balance
            positions: Open positions
            market_data: Current market prices and volumes
            technical_analysis: TA for multiple symbols/timeframes
            funding_rates: Current funding rates
            top_movers: Top gaining/losing symbols
            milestone_progress: Current milestone status
            recent_trades: Recent trade history
        
        Returns:
            Formatted context dictionary
        """
        # Calculate daily P&L
        daily_pnl = 0
        if recent_trades:
            today_trades = [
                t for t in recent_trades
                if datetime.fromisoformat(t["timestamp"]).date() == datetime.now(timezone.utc).date()
            ]
            daily_pnl = sum(t.get("pnl", 0) for t in today_trades)
        
        # Determine risk mode
        if account_balance >= 1000:
            risk_mode = "maximum_aggressive"
            allowed_risk_range = [3.5, 6.0]
        elif account_balance >= 500:
            risk_mode = "moderate_aggressive"
            allowed_risk_range = [3.0, 5.0]
        elif account_balance >= 150:
            risk_mode = "moderate"
            allowed_risk_range = [2.5, 4.0]
        elif account_balance >= 50:
            risk_mode = "conservative"
            allowed_risk_range = [2.0, 3.0]
        else:
            risk_mode = "capital_preservation"
            allowed_risk_range = [1.5, 2.5]
        
        current_stage = None
        next_target_value = None
        if isinstance(milestone_progress, dict):
            current_stage = milestone_progress.get("current_milestone")
            next_stage_str = milestone_progress.get("next_milestone") or milestone_progress.get("next")
            if isinstance(next_stage_str, str):
                try:
                    next_target_value = float(next_stage_str.replace("$", "").replace(",", "").split(" ")[0])
                except ValueError:
                    next_target_value = None

        context = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "current_balance_usd": account_balance,
            "milestone_progress": milestone_progress,
            "growth_objectives": {
                "current_stage": current_stage or milestone_progress,
                "next_target": next_target_value,
                "scaling_guidance": "Increase position size, risk %, and leverage gradually as each milestone is secured."
            },
            
            "account_state": {
                "available_balance": account_balance,
                "margin_used": sum(p.get("margin", 0) for p in positions),
                "unrealized_pnl": sum(p.get("unrealized_pnl", 0) for p in positions),
                "open_positions": positions,
                "daily_pnl": daily_pnl,
                "daily_trades": len(recent_trades) if recent_trades else 0,
                "current_drawdown_pct": 0,  # Calculate if needed
            },
            
            "market_overview": {
                "top_gainers_1h": top_movers.get("gainers", [])[:5],
                "top_losers_1h": top_movers.get("losers", [])[:5],
                "funding_rates": funding_rates,
                "market_regime": self._detect_market_regime(market_data),
            },
            
            "technical_analysis": technical_analysis,
            
            "performance_feedback": performance_feedback or {},
            "similar_trades": similar_trades or [],
            "portfolio_metrics": portfolio_metrics or {},

            "strategy_optimization": strategy_insights or {
                "recommended_strategies": [],
                "avoid_strategies": [],
                "performance_summary": {},
                "message": "No strategy data yet - learning from trades"
            },
            
            "portfolio_risk": portfolio_metrics or {
                "total_positions": 0,
                "total_position_value": 0,
                "portfolio_leverage": 0,
                "account_balance": account_balance,
                "portfolio_value": account_balance,
            },

            "system_state": {
                "risk_mode": risk_mode,
                "allowed_risk_range_pct": allowed_risk_range,
                "max_concurrent_positions": 2,
                "trading_enabled": True,
                "next_decision_window_seconds": 60,
            },
        }
        
        return context
    
    def _detect_market_regime(self, market_data: Dict[str, Any]) -> str:
        """Simple market regime detection"""
        # This would be more sophisticated in production
        btc_change = market_data.get("btc_24h_change", 0)
        
        if abs(btc_change) > 5:
            return "volatile_expansion"
        elif abs(btc_change) < 1:
            return "low_volatility"
        elif btc_change > 2:
            return "trending_up"
        elif btc_change < -2:
            return "trending_down"
        else:
            return "ranging"
    
    def _estimate_fear_greed(self, market_data: Dict[str, Any]) -> int:
        """Estimate fear & greed index (0-100)"""
        # Simplified estimation based on price action
        btc_change = market_data.get("btc_24h_change", 0)
        
        # Base score
        score = 50
        
        # Adjust based on BTC movement
        score += btc_change * 5  # +/-25 points for +/-5% move
        
        # Clamp to 0-100
        return max(0, min(100, int(score)))
    
    def _analyze_funding_sentiment(self, funding_rates: Dict[str, float]) -> str:
        """Analyze overall funding rate sentiment"""
        if not funding_rates:
            return "neutral"
        
        avg_funding = sum(funding_rates.values()) / len(funding_rates)
        
        if avg_funding > 0.01:  # 0.01% = 1 basis point
            return "bullish"
        elif avg_funding < -0.01:
            return "bearish"
        else:
            return "neutral"

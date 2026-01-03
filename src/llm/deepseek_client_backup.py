"""
DeepSeek LLM Client - Handles AI trading decisions
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
    """Combines comprehensive system prompt with clean data formatting"""

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

Execution:
  - Entry: $69,520 (confirmed break)
  - Stop: $69,050 (below consolidation, -0.68%)
  - Target: $70,800 (+1.84%, R:R = 2.7:1)
  - Leverage: 10x, Risk: 15%

Result: ✅ Hit target in 2.5 hours, Profit: +$2.80 (+18.7%)

Strategy Tag: "volume_breakout_continuation"

Why It Worked: Multi-timeframe alignment + volume confirmation

Key Lesson: Wait for consolidation THEN breakout, not chase

✅ WINNING TRADE #2: ETH Oversold Reversal at Support

Setup: ETH dumped -8% to major support at $3,200 (tested 3x in past week)

Catalyst: RSI hit 24 (extreme oversold), bullish divergence on 15m

Entry Trigger: Reversal candlestick (hammer, engulfing)

Execution:
  - Entry: $3,215 (after reversal candle close)
  - Stop: $3,160 (below support, -1.71%)
  - Target: $3,340 (+3.89%, R:R = 2.3:1)
  - Leverage: 12x, Risk: 18%

Result: ✅ Hit target in 8 hours, Profit: +$3.20 (+21.3%)

Strategy Tag: "oversold_support_reversal"

Why It Worked: Extreme RSI + major support + divergence = high probability

Key Lesson: Don't catch falling knife - wait for reversal confirmation

✅ WINNING TRADE #3: Funding Arbitrage + Range Fade

Setup: ARBUSDT funding rate at +0.18% (8 hours = 0.54% daily!)

Market: Ranging between $1.80-$1.95 for 2 days, currently at $1.93

Strategy: Short to collect funding + fade range top

Execution:
  - Entry: $1.928 (SHORT at range top)
  - Stop: $1.968 (above range high, -2.07%)
  - Target: $1.850 (range low, +4.05%, R:R = 2:1)
  - Leverage: 8x, Risk: 12%

  - Held: 16 hours (collected 2x funding payments = 0.36%)

Result: ✅ Target hit + funding collected, Profit: +$2.90 (+19.3%)

Strategy Tag: "funding_arb_range_fade"

Why It Worked: Dual profit source (funding + range trade)

Bonus: Combine with range trading for dual profit

✅ WINNING TRADE #4: SOL Momentum Continuation

Setup: SOL broke out from $145 → $158 (+9%) with massive volume

Pullback: Retraced to $155 (38.2% Fibonacci), found support

Catalyst: 1h still bullish, 15m forming higher low, momentum intact

Execution:
  - Entry: $156.20 (on 15m higher low)
  - Stop: $153.80 (below Fib 50%, -1.54%)
  - Target: $163.50 (+4.67%, R:R = 3:1)
  - Leverage: 15x, Risk: 20%

Result: ✅ Hit target in 4 hours, Profit: +$4.20 (+28%)

Strategy Tag: "momentum_continuation_fib_entry"

Why It Worked: Riding strong trend, entered on healthy pullback

Key Lesson: In strong trends, buy dips not breakouts

✅ WINNING TRADE #5: BTC Short Squeeze Hunt

Setup: BTC funding -0.12% (very negative = many shorts)

Technical: Liquidation map shows $5M shorts at $68,000

Price: Approaching $68,000, shorts getting nervous

Execution:
  - Entry: $67,950 (LONG just before liquidation level)
  - Stop: $67,500 (below liquidation cluster, -0.66%)
  - Target: $68,850 (after liquidations trigger, +1.33%, R:R = 2:1)
  - Leverage: 20x, Risk: 12%

Result: ✅ Shorts liquidated at $68k, squeeze to $68,920, Profit: +$2.60 (+17.3%)

Strategy Tag: "short_squeeze_liquidation_hunt"

Why It Worked: Asymmetric setup - liquidations provide fuel

Key Lesson: Negative funding + liquidation cluster = squeeze potential

═══════════════════════════════════════════════════════════════

PART 2: LOSING TRADES (Learn what to AVOID!)

═══════════════════════════════════════════════════════════════

❌ LOSING TRADE #1: FOMO into Pump

Mistake: Saw PEPE up +15% in 1 hour, jumped in without plan

Entry: $0.000089 (after already pumped)

Stop: None (greed mode)

Result: ❌ Reversed -8%, panic sold, Loss: -$1.80 (-12%)

Why It Failed: Entered after move was done, no edge

LESSON: NEVER chase pumps. Wait for pullback or skip.

❌ LOSING TRADE #2: Tight Stop in Volatile Market

Setup: BTC looked bullish, entered correctly at $69,500

Problem: Set stop at $69,300 (only 0.29%, less than 1 ATR)

Result: ❌ Normal volatility whipsawed stop, then rallied without us

Loss: -$1.20 (-8%)

Why It Failed: Stop too tight for market volatility (ATR was $450)

LESSON: Use minimum 1 ATR for stops, 1.5-2 ATR in volatile markets

❌ LOSING TRADE #3: Trading During News Event

Mistake: Entered ETH long 10 mins before Fed announcement

Entry: $3,450, Stop: $3,400

Result: ❌ Fed news caused -5% instant dump, stopped out

Loss: -$1.50 (-10%)

Why It Failed: High-impact news = unpredictable volatility

LESSON: NO trading 30 mins before/after major economic events

❌ LOSING TRADE #4: Ignoring Trend

Mistake: Tried to short BTC during strong uptrend because "RSI overbought"

Setup: BTC at $71,000, RSI 78, seemed "due for correction"

Reality: Strong uptrend continued, RSI stayed >70 for 3 days

Result: ❌ Stopped out at $72,400, Loss: -$1.95 (-13%)

Why It Failed: "Trend is your friend" - don't fight momentum

LESSON: RSI can stay overbought for extended periods in strong trends

❌ LOSING TRADE #5: No Clear Invalidation

Mistake: Entered based on "feeling" without clear stop plan

Entry: LINK at $14.50, vague idea of support at $14

Price: Slowly bled to $14.10, then $13.90, kept hoping

Result: ❌ Finally closed at $13.60, Loss: -$2.20 (-14.7%)

Why It Failed: No predefined stop = emotional exit = large loss

LESSON: ALWAYS have clear invalidation level BEFORE entry

═══════════════════════════════════════════════════════════════

PART 3: STRATEGY ENCYCLOPEDIA (Use, combine, or create your own!)

═══════════════════════════════════════════════════════════════

🚨 CRITICAL TREND RULE: In STRONG TRENDS (ADX > 25), only trade WITH the trend!
- Downtrend = Only SHORT positions
- Uptrend = Only LONG positions
- Counter-trend trades have <40% win rate in strong trends
- Wait for pullbacks in trends, don't fight momentum

STRATEGY #1: BREAKOUT TRADING

When: Consolidation (tight range) → Volume expansion

Best In: Trending markets (ADX > 25)

Entry: Break of range with 1.5x+ volume

Stop: Below consolidation low (or above high for shorts)

Target: 2-3x the range height

Win Rate: 70-75% in trends, 50% in ranging markets

Example: BTC $69k-69.5k for 6hrs → Break to $70.5k

STRATEGY #2: MEAN REVERSION

When: RSI <30 or >70 at major support/resistance

Best In: Ranging markets (ADX < 20) ONLY!

Entry: Reversal candlestick (hammer, engulfing)

Stop: Beyond the extreme (stop below low for longs)

Target: Return to 20-period EMA or opposite band

Win Rate: 65-70% in ranges, 40% in strong trends

🚨 CRITICAL WARNING: DON'T USE IN STRONG TRENDS!
- In downtrends (ADX > 25), buying oversold is suicide
- In uptrends (ADX > 25), shorting overbought is dangerous
- Only use when ADX < 20 and market is clearly ranging

STRATEGY #3: MOMENTUM CONTINUATION

When: Strong trend + healthy pullback to support

Best In: Trending markets with clear structure

Entry: Pullback to 38.2% or 50% Fibonacci

Stop: Below pullback low (or above for shorts)

Target: New highs (or new lows)

Win Rate: 75-80% in confirmed trends

Example: SOL rallies $145→$158, pulls to $155, resumes to $165

STRATEGY #4: FUNDING RATE ARBITRAGE

When: Extreme funding (>0.10% or <-0.05%)

Best In: Range-bound markets

Entry: Short when funding positive, long when negative

Stop: Tight (0.5-1% from entry)

Target: Hold 8-24 hours to collect funding

Win Rate: 70-80% (low risk, consistent income)

Bonus: Combine with range trading for dual profit

STRATEGY #5: SHORT SQUEEZE HUNTING

When: Negative funding + liquidation cluster nearby

Best In: After extended downtrend, shorts piling up

Entry: Just before liquidation level, going LONG

Stop: Below liquidation cluster (tight)

Target: Through liquidations (cascading buying)

Win Rate: 65-70%, high R:R (2.5:1+)

Risk: Timing critical, can fail if no liquidations

STRATEGY #6: DIVERGENCE TRADING

When: Price makes new high/low but RSI doesn't (divergence)

Best In: Late trend stages, exhaustion setups

Entry: After divergence + reversal confirmation

Stop: Beyond the divergence extreme

Target: Opposite band or key moving average

Win Rate: 60-65% (powerful but requires patience)

Types: Regular divergence (trend reversal), Hidden (trend continuation)

STRATEGY #7: VOLATILITY EXPANSION

When: Bollinger Bands very tight (squeeze), ATR low

Best In: Before major news, after consolidation

Entry: Breakout of squeeze in either direction

Stop: Opposite side of range (relatively tight)

Target: 2-3x recent ATR

Win Rate: 55-60% (direction unclear but move likely)

Note: Often combines with breakout strategy

STRATEGY #8: VOLUME SPIKE REVERSAL

When: Massive volume spike (3x+ average) at extreme

Best In: Panic selling or euphoric buying

Entry: After volume spike subsides, reversal starts

Stop: Beyond volume spike extreme

Target: Return to pre-spike level

Win Rate: 60-65%

Example: Flash crash to $67k on huge volume → bounce to $69k

STRATEGY #9: CORRELATION PLAY

When: BTC and altcoins diverge (BTC up, alts lagging)

Best In: After BTC establishes direction

Entry: Long strong alts that are "catching up"

Stop: Below recent support

Target: Align with BTC performance

Win Rate: 55-60% (requires correct correlation read)

Example: BTC +5%, but ETH only +1% → ETH likely to catch up

STRATEGY #10: FAILED BREAKOUT FADE

When: Breakout occurs but fails to sustain (false breakout)

Best In: Ranging markets, low volume breakouts

Entry: When price returns inside range, trade back to opposite side

Stop: Beyond the failed breakout point

Target: Opposite side of range

Win Rate: 65-70% (fakeouts often retrace fully)

Note: Fakeouts with volume are often real breakouts

═══════════════════════════════════════════════════════════════

PART 4: MARKET REGIME ADAPTATION

═══════════════════════════════════════════════════════════════

📈 TRENDING UP (ADX > 25, Price > EMAs):

✅ Best: Breakouts, Momentum continuation, Pullback longs

✅ Risk: 15-25% (trends can run far)

✅ Leverage: 10-20x

❌ Avoid: Fading the trend, shorting oversold

📉 TRENDING DOWN:

✅ Best: Breakdown shorts, Momentum continuation (short), Rally fades

✅ Risk: 15-25%

✅ Leverage: 10-20x

❌ Avoid: Catching falling knives, buying "cheap"

↔️ RANGING (ADX < 20):

✅ Best: Mean reversion, Range extremes, Funding arb

✅ Risk: 10-18% (ranges break eventually)

✅ Leverage: 8-15x

❌ Avoid: Breakout entries (many are false)

💥 VOLATILE EXPANSION (BB Width > 6%, ATR > 80th percentile):

✅ Best: Shorter timeframes, wider stops, quick profits

✅ Risk: 10-15% (volatility = danger)

✅ Leverage: 5-12x (reduce leverage in chaos)

❌ Avoid: Holding overnight, tight stops

😴 LOW VOLATILITY (BB Width < 2%, ATR < 50th percentile):

✅ Best: Funding arbitrage, wait for squeeze breakout

✅ Risk: 8-12% (compressed moves coming)

✅ Leverage: 5-10x

❌ Avoid: Forcing trades in dead markets

═══════════════════════════════════════════════════════════════

PART 5: ADVANCED TECHNIQUES

═══════════════════════════════════════════════════════════════

🎯 LIQUIDATION HUNTING:

- Check liquidation heatmap

- Large liquidation clusters = magnets

- Enter BEFORE price reaches cluster

- Exit AFTER liquidations trigger cascade

- Example: $5M longs liquidate at $68k → Price dips to $67.8k → Rallies to $69k

📊 FUNDING RATE PLAYS:

- Funding > 0.15% = Longs overcrowded → Potential dump

- Funding < -0.08% = Shorts overcrowded → Potential squeeze

- Funding extremes + contrary technical = high-prob reversal

- Can hold positions just to collect funding (0.5-1% per day!)

🔄 DIVERGENCE SIGNALS:

- Price higher highs, RSI lower highs = Bearish divergence (top forming)

- Price lower lows, RSI higher lows = Bullish divergence (bottom forming)

- Hidden divergence = Trend continuation signal

- Regular divergence = Reversal signal

- Combine with other confluence for best results

🌊 VOLUME ANALYSIS:

- Volume precedes price (accumulation/distribution)

- Breakouts without volume = likely fail

- Volume spike at support = buying interest

- Volume drying up = move ending or breakout imminent

- Compare to 20-period average

📏 FIBONACCI MAGIC:

- Pullbacks often respect 38.2%, 50%, 61.8% retracements

- Extensions at 127.2%, 161.8% for targets

- Use with other confluence (support, moving averages)

- Strong trends rarely retrace beyond 38.2%

- Weak trends retrace to 61.8% or more

⚡ MULTI-TIMEFRAME CONFLUENCE:

- Best setups: All timeframes aligned

- 1h trend + 15m entry + 5m confirmation = highest probability

- If 1h bullish but 15m bearish = WAIT for alignment

- Higher timeframe = trend direction, lower TF = entry timing

═══════════════════════════════════════════════════════════════

PART 6: YOUR DECISION-MAKING FRAMEWORK

═══════════════════════════════════════════════════════════════

🧠 THINK STEP-BY-STEP (Chain-of-Thought):

Step 1 - REGIME IDENTIFICATION:

Ask: "What type of market is this?"

- Trending up/down? (check ADX, EMAs, price structure)

- Ranging? (check support/resistance bounces)

- Volatile or quiet? (check ATR, BB width)

Step 2 - TREND RESPECT:

Ask: "Am I trading WITH or AGAINST the trend?"

- STRONG TREND (ADX > 25): Only trade WITH the trend, never against it

- MODERATE TREND (ADX 20-25): Prefer with-trend, avoid counter-trend

- NO TREND (ADX < 20): Can use mean reversion, but still respect broader market

- COUNTER-TREND TRADES: Require extreme conditions + very high confidence

Step 3 - OPPORTUNITY SCANNING:

Ask: "Which asset has the clearest setup?"

- In TRENDING markets: Look for pullbacks to support (longs) or rallies to resistance (shorts)

- In RANGING markets: Look for extreme RSI levels, divergences

- Check volume confirmation and timeframe alignment

- Review funding rates and order flow

Step 4 - STRATEGY SELECTION:

Ask: "What strategy fits this setup?"

- Trending → Momentum/Breakout

- Ranging → Mean reversion/Funding arb

- Unsure → WAIT (greeds kill accounts)

- Can combine multiple strategies!

Step 5 - RISK-REWARD CALCULATION:

Ask: "Is this trade worth it?"

- Calculate exact entry, stop, target using key_levels data

- TREND TRADES: R:R minimum 2:1 (trends can run far)

- COUNTER-TREND: R:R minimum 3:1 (higher risk needs higher reward)

- Factor in fees, slippage (~0.1%)

- If R:R poor, adjust or skip

Step 6 - CONFIDENCE ASSESSMENT:

Ask: "How confident am I?"

- High confidence (80%+) → Risk 20-30%

- Medium confidence (65-79%) → Risk 15-20%

- Low confidence (<65%) → Risk 10-15% or WAIT

- No confidence → Absolutely WAIT

- In strong trends: Higher confidence for with-trend trades

- Counter-trend trades need 85%+ confidence minimum

Step 7 - EXECUTION DECISION:

- WITH TREND + Good R:R + High Confidence → TRADE

- COUNTER TREND + Excellent Setup + Very High Confidence → TRADE

- ANY doubt → WAIT (patience is a position)

- In strong trends: Be patient, wait for perfect setup

- Counter-trend trades: Only when conditions are extreme and conviction is maximum

═══════════════════════════════════════════════════════════════

PART 7: TECHNICAL STOP LOSS & TAKE PROFIT CALCULATION

═══════════════════════════════════════════════════════════════

🚨 CRITICAL: Use the provided key_levels and ATR data for precision!

✅ STOP LOSS PLACEMENT:

FOR LONGS:

1. Find the lowest support: MIN(immediate_support, major_support, s1)

2. Add buffer: support - (ATR × 0.5 to 1.0)

3. Validate: Must be at least 1 ATR from entry

4. Consider: Volatility (use 1.5-2 ATR in high volatility)

FOR SHORTS:

1. Find the highest resistance: MAX(immediate_resistance, major_resistance, r1)

2. Add buffer: resistance + (ATR × 0.5 to 1.0)

3. Validate: Must be at least 1 ATR from entry

4. Consider: Volatility (use 1.5-2 ATR in high volatility)

✅ TAKE PROFIT PLACEMENT:

FOR LONGS:

1. Primary TP: Next major resistance level

2. Secondary TP: R2 or Fibonacci extension (1.618)

3. Validate: Ensures R:R ≥ 2:1

FOR SHORTS:

1. Primary TP: Next major support level

2. Secondary TP: S2 or Fibonacci extension (1.618)

3. Validate: Ensures R:R ≥ 2:1

✅ EXAMPLE CALCULATION:

Long Setup on BTC:

- Entry: $69,200

- immediate_support: $68,800

- major_support: $68,500

- ATR (1h): $200

- Stop Loss: MIN($68,800, $68,500) - ($200 × 0.5) = $68,400

- immediate_resistance: $70,000

- major_resistance: $70,500

- Take Profit: $70,000 (first target)

- R:R: ($70,000 - $69,200) / ($69,200 - $68,400) = $800 / $800 = 1:1 (adjust!)

- Better TP: $70,600 → R:R = $1,400 / $800 = 1.75:1 ✅

═══════════════════════════════════════════════════════════════

PART 8: OUTPUT FORMAT

═══════════════════════════════════════════════════════════════

You MUST respond with valid JSON:

{

  "decision_type": "open_position|hold|wait|close_position",

  "trade": {

    "symbol": "BTCUSDT",

    "category": "linear",

    "side": "Buy|Sell",

    "order_type": "Market|Limit",

    "risk_pct": 10-30,

    "leverage": 5-25,

    "entry_price_target": current_price,

    "stop_loss": calculated_using_key_levels,

    "take_profit_1": calculated_using_key_levels,

    "take_profit_2": optional_secondary_target,

    "strategy_tag": "your_strategy_name",

    "confidence_score": 0.0-1.0,

    "expected_hold_duration_mins": estimated_time

  },

  "reasoning": {

    "market_assessment": "regime + conditions",

    "technical_confluence": ["point1", "point2", ...],

    "entry_rationale": "why_here",

    "stop_rationale": "key_levels_calculation",

    "target_rationale": "key_levels_calculation",

    "risk_reward_calculation": "math",

    "what_could_go_wrong": "invalidation",

    "alternatives_considered": ["option1", "option2"]

  },

  "risk_management": {

    "position_size_usdt": calculated,

    "risk_amount_usdt": max_loss,

    "potential_profit_tp1": profit_at_tp1,

    "risk_reward_ratio": final_r_r,

    "portfolio_impact": "exposure_effect"

  }

}

═══════════════════════════════════════════════════════════════

PART 9: LEARNING & ADAPTATION (NEW!)

═══════════════════════════════════════════════════════════════

🔄 CONTINUOUS LEARNING PROTOCOL:

1. **Analyze Every Trade Result:**
   - Win/Loss + P&L percentage
   - Strategy effectiveness
   - Market conditions at entry/exit

2. **Strategy Performance Tracking:**
   - Which strategies work best in current regime?
   - Update win rates and adjust expectations
   - Identify patterns in successful trades

3. **Dynamic Strategy Selection:**
   - Prioritize strategies with >65% win rate in last 10 trades
   - Reduce risk on strategies with <50% win rate
   - Adapt to changing market conditions

4. **Risk Adjustment:**
   - Increase risk after 3+ consecutive wins
   - Decrease risk after 2+ consecutive losses
   - Consider portfolio volatility

5. **Market Regime Adaptation:**
   - Track which strategies work in trending vs ranging markets
   - Adjust leverage and risk based on regime
   - Learn from past regime transitions

LEARNING EXAMPLES:

✅ After 3 momentum wins: "Momentum working well, can increase risk to 25%"

❌ After 2 mean reversion losses: "Avoiding mean reversion in this trend"

📊 Performance Pattern: "BTC breakouts successful in uptrends, avoid in downtrends"

═══════════════════════════════════════════════════════════════

FINAL REMINDERS

═══════════════════════════════════════════════════════════════

🎯 Mission: $15 → $100,000

🧠 Strategy: Learn from examples, adapt to conditions

📊 Data: Use key_levels and ATR for technical precision

⚖️ Balance: Aggressive growth + capital preservation

🚀 Innovation: Create new strategies when you see opportunities

Quality > Quantity. One great trade beats five mediocre ones.

When in doubt, wait it out.

NOW ANALYZE THE MARKET AND MAKE YOUR DECISION:"""

    # Clean data formatting approach
    FOCUSED_USER_PROMPT_TEMPLATE = """# Current Market Snapshot

Timestamp: {timestamp}

Account Balance: ${balance:.2f}

Daily P&L: ${daily_pnl:+.2f}

Open Positions: {num_positions}

# Account State

{account_state}

# Market Overview

{market_overview}

# Technical Analysis (with key_levels and ATR)

{technical_analysis}

# Sentiment Indicators

{sentiment_data}

# Order Flow Analysis

{order_flow_data}

# On-Chain Metrics

{onchain_data}

# Your Recent Performance

{performance_feedback}

# Strategy Insights (Learn from History)

{strategy_insights}

# Portfolio Risk Metrics

{portfolio_metrics}

---

**Your Task:**

1. Analyze the above data using your decision framework (Part 6)

2. Identify the best opportunity using strategies from Part 3

3. Calculate stops/targets using key_levels and ATR (Part 7)

4. Make a decision with detailed reasoning

5. Output valid JSON as specified in Part 8

Think step-by-step. Show your calculations. Be precise.

**LEARNING DIRECTIVE:** Review your performance data and adapt your strategy selection based on what has worked recently."""

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
            onchain_data=onchain_data,
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
        for p in positions:
            lines.append(
                f"  - {p.get('symbol', 'N/A')}: {p.get('side', 'N/A')} "
                f"{p.get('size', 'N/A')} @ ${p.get('entry_price', 0):.2f} "
                f"(PnL: ${p.get('unrealized_pnl', 0):+.2f})"
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
            lines.append("\nTop Gainers (1h):")
            for g in gainers:
                lines.append(f"  {g.get('symbol', 'N/A')}: +{g.get('change_pct', 0):.2f}%")

        if losers:
            lines.append("\nTop Losers (1h):")
            for l in losers:
                lines.append(f"  {l.get('symbol', 'N/A')}: {l.get('change_pct', 0):.2f}%")

        if funding:
            lines.append("\nKey Funding Rates:")
            for symbol, rate in list(funding.items())[:5]:
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
                        lines.append(f"    Key Levels:")
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
        sentiment = context.get("sentiment_data", {})

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
        flow = context.get("order_flow_data", {})

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
        onchain = context.get("onchain_data", {})

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

            lines = [f"Performance Summary:"]
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
        insights = context.get("strategy_insights", {})

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
        metrics = context.get("portfolio_metrics", {})

        if not metrics:
            return "No portfolio metrics available"

        lines = ["Portfolio Risk:"]
        lines.append(f"  Leverage: {metrics.get('leverage', 'N/A')}x")
        lines.append(f"  Risk Level: {metrics.get('risk_level', 'N/A')}%")
        lines.append(f"  Correlation Risk: {metrics.get('correlation_risk', 'N/A')}")
        lines.append(f"  Concentration: {metrics.get('concentration', 'N/A')}%")

        return "\n".join(lines)


# Move enums and dataclasses back to top level
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
        
        # Note: OpenAI client is now created per-request in _call_deepseek()
        # to support new openai>=1.0.0 API
        
        logger.info(f"DeepSeek client initialized with model: {model}")
    
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
        start_time = datetime.utcnow()

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
                    (datetime.utcnow() - start_time).total_seconds() * 1000
                )
                decision.model_version = f"{self.model}_normal"
                
                logger.info(
                    f"Decision received: {decision.decision_type} "
                    f"({'Trade ' + decision.symbol if decision.symbol else 'No trade'})"
                )
                
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
    def _parse_response(self, response: str) -> TradingDecision:
        """Parse LLM response into TradingDecision"""
        data = json.loads(response)
        
        # Extract decision type
        decision_type = DecisionType(data.get("decision_type", "wait"))
        
        # Create base decision with guaranteed unique ID
        decision = TradingDecision(
            decision_id=str(uuid.uuid4()),  # Always generate unique ID
            timestamp_utc=datetime.utcnow(),  # Use current time
            decision_type=decision_type,
        )
        
        # Add trade details if opening position
        if decision_type == DecisionType.OPEN_POSITION and "trade" in data:
            trade = data["trade"]
            decision.symbol = trade.get("symbol")
            decision.category = trade.get("category", "linear")
            decision.side = trade.get("side")
            decision.order_type = trade.get("order_type", "Market")
            decision.risk_pct = float(trade.get("risk_pct", 15))
            decision.leverage = int(trade.get("leverage", 10))
            decision.entry_price_target = float(trade.get("entry_price_target", 0))
            decision.stop_loss = float(trade.get("stop_loss", 0))
            decision.take_profit_1 = float(trade.get("take_profit_1", 0))
            decision.take_profit_2 = float(trade.get("take_profit_2", 0)) if trade.get("take_profit_2") else None
            decision.time_in_force = trade.get("time_in_force", "GTC")
            decision.expected_hold_duration_mins = trade.get("expected_hold_duration_mins")
            decision.strategy_tag = trade.get("strategy_tag")
            decision.confidence_score = float(trade.get("confidence_score", 0.5))
        
        # Add reasoning
        decision.reasoning = data.get("reasoning", {})
        decision.risk_management = data.get("risk_management", {})
        
        return decision
    
    def _create_safe_decision(self, error_reason: str) -> TradingDecision:
        """Create a safe 'wait' decision when LLM fails"""
        return TradingDecision(
            decision_id=str(uuid.uuid4()),
            timestamp_utc=datetime.utcnow(),
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
        top_movers: List[Dict[str, Any]],
        milestone_progress: Dict[str, Any],
        recent_trades: List[Dict[str, Any]] = None,
        performance_feedback: Dict[str, Any] = None,
        position_monitor: Dict[str, Any] = None,
        order_flow_data: Dict[str, Any] = None,
        sentiment_data: Dict[str, Any] = None,
        portfolio_metrics: Dict[str, Any] = None,
        strategy_insights: Dict[str, Any] = None,
        onchain_data: Dict[str, Any] = None,
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
                if datetime.fromisoformat(t["timestamp"]).date() == datetime.utcnow().date()
            ]
            daily_pnl = sum(t.get("pnl", 0) for t in today_trades)
        
        # Determine risk mode
        risk_mode = "conservative"
        allowed_risk_range = [10, 18]
        
        if account_balance >= 500:
            risk_mode = "maximum_aggressive"
            allowed_risk_range = [22, 30]
        elif account_balance >= 200:
            risk_mode = "moderate_aggressive"
            allowed_risk_range = [18, 25]
        elif account_balance >= 50:
            risk_mode = "moderate"
            allowed_risk_range = [15, 22]
        
        context = {
            "timestamp_utc": datetime.utcnow().isoformat(),
            "current_balance_usd": account_balance,
            "milestone_progress": milestone_progress,
            
            "account_state": {
                "available_balance": account_balance,
                "margin_used": sum(p.get("margin", 0) for p in positions),
                "unrealized_pnl": sum(p.get("unrealized_pnl", 0) for p in positions),
                "open_positions": positions,
                "position_monitor": position_monitor or {"total_positions": 0, "positions": []},
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
            
            "order_flow_analysis": order_flow_data or {},
            
            "sentiment_signals": sentiment_data or {
                "fear_greed_index": {"value": self._estimate_fear_greed(market_data), "classification": "Estimated"},
                "funding_analysis": {"sentiment": self._analyze_funding_sentiment(funding_rates)},
                "long_short_sentiment": "Estimated from funding",
            },
            
            "performance_feedback": performance_feedback or {
                "total_trades": 0,
                "message": "No trades yet - make your first decision!",
            },

            "strategy_optimization": strategy_insights or {
                "recommended_strategies": [],
                "avoid_strategies": [],
                "performance_summary": {},
                "message": "No strategy data yet - learning from trades"
            },

            "onchain_analysis": onchain_data or {},
            
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
                "max_concurrent_positions": 3,
                "trading_enabled": True,
                "next_decision_window_seconds": 120,
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

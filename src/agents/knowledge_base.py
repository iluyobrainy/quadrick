"""
Strategy Knowledge Base
Professional trading principles and archetypes for Autonomous AI Trading.
Used as "Static Memory" to educate the Strategist Agent.
"""

TRADING_KNOWLEDGE = {
    "MARKET_STRUCTURE_PRINCIPLES": [
        "Trade with the Daily trend, but enter on the 15m/1h pullback.",
        "Liquidity is located above old highs (Buy Side Liquidity) and below old lows (Sell Side Liquidity).",
        "A 'Liquidity Grab' occurs when price sweeps an old high/low and then immediately reverses. This is a high-conviction entry.",
        "Fair Value Gaps (FVG) are imbalances in price that act as magnets. Price often returns to fill at least 50% of the gap."
    ],
    
    "STRATEGY_ARCHETYPES": {
        "LIQUIDITY_SWEEP": {
            "setup": "Identify a 'Major Resistance' or 'Major Support' from the Key Levels.",
            "trigger": "Price crosses the level but fails to close beyond it (Wick sweep) + RSI reversal.",
            "execution": "Enter in the opposite direction of the sweep. SL just above/below the wick. TP at the nearest FVG or EMA 50."
        },
        "IMBALANCE_FILL": {
            "setup": "Identify a large candle that created a price gap (FVG).",
            "trigger": "Price retraces into the gap + 5m/15m volume spike in the direction of the original gap.",
            "execution": "Enter as price touches the 50% fill level (Mean Threshold). SL at the origin of the gap. TP at the gap's full completion."
        },
        "TREND_CONTINUATION_EMA": {
            "setup": "HTF (4H/1D) shows clear trend + ADX > 25.",
            "trigger": "Price pullbacks to touch the EMA 21 or EMA 50 on the 1H timeframe.",
            "execution": "Look for a bullish/bearish engulfing candle at the EMA. SL below/above the EMA. TP at the recent swing high/low (extended)."
        },
        "VOLATILITY_SQUEEZE_BREAKOUT": {
            "setup": "Bollinger Band Width < 0.02 (Console/Squeeze).",
            "trigger": "15m candle close outside the bands + Volume > 2x average.",
            "execution": "Enter in the direction of the close. SL at the middle BB. TP at a 2.5:1 Risk/Reward ratio."
        }
    },
    
    "RISK_MANAGEMENT_RULES": [
        "Never risk more than 1% of the total account balance on a single trade.",
        "Stop Loss must ALWAYS be placed at a structural invalidation point (e.g. below a swing low).",
        "If the market is 'Range/Chop', reduce position size by 50%.",
        "Always target a minimum 2:1 Reward-to-Risk ratio."
    ]
}

def get_knowledge_summary() -> str:
    """Returns a string summary for the LLM prompt."""
    summary = "### PROFESSIONAL TRADING KNOWLEDGE BASE\n"
    
    summary += "\n**Market Principles:**\n"
    for p in TRADING_KNOWLEDGE["MARKET_STRUCTURE_PRINCIPLES"]:
        summary += f"- {p}\n"
        
    summary += "\n**Strategy Archetypes:**\n"
    for name, details in TRADING_KNOWLEDGE["STRATEGY_ARCHETYPES"].items():
        summary += f"- {name}: {details['setup']} -> Trigger: {details['trigger']}\n"
        
    summary += "\n**Risk Rules:**\n"
    for r in TRADING_KNOWLEDGE["RISK_MANAGEMENT_RULES"]:
        summary += f"- {r}\n"
        
    return summary

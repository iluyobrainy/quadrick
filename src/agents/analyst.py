from typing import Dict, Any, List, Tuple
from dataclasses import dataclass
from loguru import logger
import json

@dataclass
class MarketRegime:
    regime: str  # "bull_trend", "bear_trend", "range_chop", "volatile_breakout"
    confidence: float
    recommended_strategy: str
    reasoning: str

class AnalystAgent:
    """
    Agent 1: The Analyst
    Role: Identifies Market Regime & Selects High-Level Strategy
    Input: Technical Data (Timeframes, Indicators)
    Output: MarketRegime (Trend, Chop, etc.)
    """
    
    def __init__(self, llm_client):
        self.llm = llm_client
        
    async def analyze_market(self, symbol: str, market_data: Dict[str, Any]) -> Tuple[MarketRegime, str, str]:
        """
        Analyze market structure and determine regime.
        Returns: (MarketRegime, prompt, response)
        """
        prompt = self._build_analyst_prompt(symbol, market_data)
        
        try:
            response = await self.llm.get_completion(
                prompt=prompt,
                system_prompt="You are an elite Technical Analyst. Your ONLY job is to identify the Market Regime and recommend a Strategy class. You do NOT execute trades.",
                json_mode=True
            )
            
            data = json.loads(response)
            regime = MarketRegime(
                regime=data.get("regime", "unknown"),
                confidence=data.get("confidence", 0.0),
                recommended_strategy=data.get("recommended_strategy", "wait"),
                reasoning=data.get("reasoning", "")
            )
            return regime, prompt, response
            
        except Exception as e:
            logger.error(f"Analyst Agent failed for {symbol}: {e}")
            return MarketRegime("unknown", 0.0, "wait", f"Error: {e}"), prompt, str(e)

    def _build_analyst_prompt(self, symbol: str, data: Dict[str, Any]) -> str:
        # Extract key metrics for all relevant timeframes
        tf_analysis = data.get("timeframe_analysis", {})
        
        # Helper to extract data from TimeframeAnalysis objects or dicts
        def get_tf_metrics(tf_key: str) -> Dict[str, Any]:
            tf = tf_analysis.get(tf_key)
            if not tf:
                return {}
            
            # If it's a dataclass (TimeframeAnalysis)
            if hasattr(tf, 'indicators'):
                inds = tf.indicators
                struct = tf.structure
                return {
                    'trend': struct.trend.value if hasattr(struct.trend, 'value') else str(struct.trend),
                    'rsi': round(inds.rsi, 2),
                    'macd_signal': round(inds.macd_signal, 4),
                    'adx': round(inds.adx, 2) if inds.adx else "N/A",
                    'atr': round(inds.atr, 4),
                    'bb_position': "N/A", # Placeholder if not specific in struct
                    'bb_width': round(inds.bb_width, 4),
                    'volume_ratio': round(inds.volume_ratio, 2)
                }
            # Fallback for dict
            return tf if isinstance(tf, dict) else {}

        tf_4h = get_tf_metrics("4h")
        tf_1h = get_tf_metrics("1h")
        tf_15m = get_tf_metrics("15m")
        tf_5m = get_tf_metrics("5m")
        tf_daily = get_tf_metrics("1d") # main.py uses "1d"
        
        # Order flow context (if available)
        order_flow = data.get("order_flow", {})
        bid_ask_imbalance = order_flow.get("imbalance", "N/A")
        large_orders = order_flow.get("large_orders", "N/A")
        
        # Funding rate (for perps)
        funding_rate = data.get("funding_rate", "N/A")
        
        # Volume anomalies
        volume_spike = tf_15m.get("volume_ratio", 1.0) >= 2.0
        volume_context = "🔥 VOLUME SPIKE DETECTED" if volume_spike else "Normal volume"
        
        # BB squeeze detection
        bb_width = tf_15m.get("bb_width", 0.05)
        squeeze_alert = "⚠️ BB SQUEEZE" if bb_width < 0.015 else ""
        
        return f"""
ANALYZE MARKET REGIME FOR: {symbol}

DATA - MULTI-TIMEFRAME ANALYSIS:

[DAILY Timeframe - Overall Bias]
- Trend: {tf_daily.get('trend', 'N/A')}
- RSI: {tf_daily.get('rsi', 'N/A')}

[4H Timeframe - Swing Direction]
- Trend: {tf_4h.get('trend', 'N/A')}
- RSI: {tf_4h.get('rsi', 'N/A')}
- MACD: {tf_4h.get('macd_signal', 'N/A')}
- ADX: {tf_4h.get('adx', 'N/A')}

[1H Timeframe - Primary Setup]
- Trend: {tf_1h.get('trend', 'N/A')}
- RSI: {tf_1h.get('rsi', 'N/A')}
- MACD: {tf_1h.get('macd_signal', 'N/A')}
- ADX: {tf_1h.get('adx', 'N/A')} (Trend Strength)
- Volatility (ATR): {tf_1h.get('atr', 'N/A')}

[15M Timeframe - Entry Timing]
- Trend: {tf_15m.get('trend', 'N/A')}
- RSI: {tf_15m.get('rsi', 'N/A')}
- BB Width: {bb_width:.4f} {squeeze_alert}
- Volume: {tf_15m.get('volume_ratio', 1.0):.1f}x average

[5M Timeframe - Scalping Context]
- Trend: {tf_5m.get('trend', 'N/A')}
- RSI: {tf_5m.get('rsi', 'N/A')}
- MACD: {tf_5m.get('macd_signal', 'N/A')}

[ORDER FLOW & LIQUIDITY]
- Bid/Ask Imbalance: {bid_ask_imbalance}
- Large Orders Detected: {large_orders}
- Funding Rate: {funding_rate}
- Volume Status: {volume_context}

TASK:
1. Determine the Market Regime (Bull Trend, Bear Trend, Range/Chop, Volatile).
2. Check HTF (4H/Daily) ALIGNMENT before recommending directional trades.
3. Recommend the BEST Strategy Class:
   - "TrendFollowing" (Strong Trend + ADX > 25 + HTF alignment)
   - "MeanReversion" (Range/Chop + RSI Extremes + Low ADX)
   - "Breakout" (BB Squeeze + Building Volume + Consolidation)
   - "ScalpingMomentum" (High Volume Spike + Short-term Momentum)
   - "Wait" (Conflicting signals, HTF misalignment, or low confidence)

OUTPUT JSON:
{{
    "regime": "bull_trend" | "bear_trend" | "range_chop" | "volatile",
    "confidence": 0.0 to 1.0,
    "recommended_strategy": "TrendFollowing" | "MeanReversion" | "Breakout" | "ScalpingMomentum" | "Wait",
    "reasoning": "Brief explanation including HTF alignment check..."
}}
"""


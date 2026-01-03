from typing import Dict, Any
from .base_strategy import BaseStrategy, StrategySignal

class TrendFollowingStrategy(BaseStrategy):
    def __init__(self):
        super().__init__(
            name="TrendFollowing",
            description="Classic trend following strategy using EMA alignment and MACD momentum."
        )

    def analyze(self, symbol_data: Dict[str, Any]) -> StrategySignal:
        """
        Analyze a single symbol's data for trend following setup.
        Expects symbol_data to match the structure sent to LLM (ta_summary[symbol]).
        """
        tf_analysis = symbol_data.get("timeframe_analysis", {})
        h1_data = tf_analysis.get("1h", {})
        m15_data = tf_analysis.get("15m", {})
        
        if not h1_data or not m15_data:
            return StrategySignal("neutral", "none", 0.0, reasoning="Insufficient data")

        # Extract indicators
        trend_h1 = h1_data.get("trend", "neutral")
        rsi_h1 = h1_data.get("rsi", 50)
        macd_signal_h1 = h1_data.get("macd_signal", "neutral")
        
        # Logic
        signal_type = "neutral"
        direction = "none"
        confidence = 0.0
        reasoning = []

        # Long Setup
        if trend_h1 == "uptrend" and macd_signal_h1 == "bullish":
            if 40 <= rsi_h1 <= 70:  # Healthy RSI
                signal_type = "buy"
                direction = "long"
                confidence = 0.7
                reasoning.append("H1 Uptrend + Bullish MACD + Healthy RSI")
        
        # Short Setup
        elif trend_h1 == "downtrend" and macd_signal_h1 == "bearish":
            if 30 <= rsi_h1 <= 60:  # Healthy RSI
                signal_type = "sell"
                direction = "short"
                confidence = 0.7
                reasoning.append("H1 Downtrend + Bearish MACD + Healthy RSI")

        # Calculate SL/TP if signal exists
        stop_loss = None
        take_profit = None
        
        if signal_type != "neutral":
            current_price = symbol_data.get("current_price", 0)
            atr = h1_data.get("atr", 0)
            
            if current_price > 0 and atr > 0:
                if direction == "long":
                    stop_loss = current_price - (2 * atr)
                    take_profit = current_price + (3 * atr)
                else:
                    stop_loss = current_price + (2 * atr)
                    take_profit = current_price - (3 * atr)

        return StrategySignal(
            signal_type=signal_type,
            direction=direction,
            confidence=confidence,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reasoning="; ".join(reasoning)
        )

    def validate_entry(self, signal: StrategySignal, symbol_data: Dict[str, Any]) -> bool:
        """
        Strict validation for Trend Following.
        """
        if signal.signal_type == "neutral":
            return False
            
        tf_analysis = symbol_data.get("timeframe_analysis", {})
        h1_data = tf_analysis.get("1h", {})
        
        # Guardrail 1: Don't buy overbought / Don't sell oversold
        rsi = h1_data.get("rsi", 50)
        if signal.direction == "long" and rsi > 75:
            return False
        if signal.direction == "short" and rsi < 25:
            return False
            
        # Guardrail 2: Must have volume support (if available)
        # volume_ratio = h1_data.get("volume_ratio", 1.0)
        # if volume_ratio < 0.8:
        #     return False
            
        return True

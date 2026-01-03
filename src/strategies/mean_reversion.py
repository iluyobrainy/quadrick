from typing import Dict, Any
from .base_strategy import BaseStrategy, StrategySignal

class MeanReversionStrategy(BaseStrategy):
    def __init__(self):
        super().__init__(
            name="MeanReversion",
            description="Counter-trend strategy targeting RSI extremes and Bollinger Band reversals."
        )

    def analyze(self, symbol_data: Dict[str, Any]) -> StrategySignal:
        """
        Analyze a single symbol's data for mean reversion setup.
        """
        tf_analysis = symbol_data.get("timeframe_analysis", {})
        h1_data = tf_analysis.get("1h", {})
        m15_data = tf_analysis.get("15m", {})
        
        if not m15_data: # Mean reversion often works better on shorter frames like 15m
            return StrategySignal("neutral", "none", 0.0, reasoning="Insufficient data")

        # Extract indicators (focus on 15m for mean reversion entries)
        rsi_15m = m15_data.get("rsi", 50)
        bb_position = m15_data.get("bb_position", "middle")
        
        # Logic
        signal_type = "neutral"
        direction = "none"
        confidence = 0.0
        reasoning = []

        # Long Setup (Oversold)
        if rsi_15m < 30:
            if bb_position == "lower":
                signal_type = "buy"
                direction = "long"
                confidence = 0.65
                reasoning.append("15m RSI Oversold (<30) + Price at Lower BB")
        
        # Short Setup (Overbought)
        elif rsi_15m > 70:
            if bb_position == "upper":
                signal_type = "sell"
                direction = "short"
                confidence = 0.65
                reasoning.append("15m RSI Overbought (>70) + Price at Upper BB")

        # Calculate SL/TP
        stop_loss = None
        take_profit = None
        
        if signal_type != "neutral":
            current_price = symbol_data.get("current_price", 0)
            atr = m15_data.get("atr", 0)
            
            if current_price > 0 and atr > 0:
                if direction == "long":
                    stop_loss = current_price - (1.5 * atr) # Tighter stop for mean reversion
                    take_profit = current_price + (2 * atr)
                else:
                    stop_loss = current_price + (1.5 * atr)
                    take_profit = current_price - (2 * atr)

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
        Strict validation for Mean Reversion.
        """
        if signal.signal_type == "neutral":
            return False
            
        # Guardrail: Don't catch a falling knife (check higher timeframe trend?)
        # For pure mean reversion, we might actually want to trade against the trend, 
        # but we should ensure we aren't in a massive breakout.
        
        tf_analysis = symbol_data.get("timeframe_analysis", {})
        h1_data = tf_analysis.get("1h", {})
        
        # If H1 ADX is super high (>50), maybe avoid mean reversion as trend is too strong
        # (Assuming we had ADX, but we don't strictly have it in the summary dict yet, 
        # so we'll skip that check for now or use RSI extremes on H1 too)
        
        rsi_h1 = h1_data.get("rsi", 50)
        
        # If 15m is oversold but H1 is still crashing hard (e.g. RSI < 20), wait.
        if signal.direction == "long" and rsi_h1 < 25:
            return False # Too bearish on H1
            
        if signal.direction == "short" and rsi_h1 > 75:
            return False # Too bullish on H1
            
        return True

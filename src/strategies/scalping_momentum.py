"""
Scalping Momentum Strategy - Quick trades on volume spikes with momentum
"""
from typing import Dict, Any
from .base_strategy import BaseStrategy, StrategySignal


class ScalpingMomentumStrategy(BaseStrategy):
    """
    High-frequency momentum scalping strategy.
    
    Entry Conditions:
    - Volume spike (3x+ average) indicating institutional activity
    - RSI aligned with momentum direction (not extreme)
    - Clear price momentum in last 3-5 candles
    - MACD confirming direction
    
    Exit Strategy:
    - Quick profit target (1.5-2% for crypto)
    - Tight stop loss (0.8-1%)
    - Time-based exit (max hold 30-45 minutes)
    """
    
    def __init__(self):
        super().__init__(
            name="ScalpingMomentum",
            description="Quick momentum trades on volume spikes with tight risk management."
        )
    
    def analyze(self, symbol_data: Dict[str, Any]) -> StrategySignal:
        """
        Analyze for scalping momentum setup.
        Focuses on 5m and 15m timeframes for quick entries.
        """
        tf_analysis = symbol_data.get("timeframe_analysis", {})
        m5_data = tf_analysis.get("5m", {})
        m15_data = tf_analysis.get("15m", {})
        h1_data = tf_analysis.get("1h", {})
        
        # Need short-term data
        if not m5_data and not m15_data:
            return StrategySignal("neutral", "none", 0.0, reasoning="No short-term data available")
        
        # Use 5m if available, else 15m
        primary_tf = m5_data if m5_data else m15_data
        tf_name = "5m" if m5_data else "15m"
        
        # Extract indicators
        volume_ratio = primary_tf.get("volume_ratio", 1.0)
        rsi = primary_tf.get("rsi", 50)
        macd_signal = primary_tf.get("macd_signal", "neutral")
        trend = primary_tf.get("trend", "neutral")
        
        # 1H context for direction bias
        h1_trend = h1_data.get("trend", "neutral")
        h1_rsi = h1_data.get("rsi", 50)
        
        # Initialize
        signal_type = "neutral"
        direction = "none"
        confidence = 0.0
        reasoning = []
        
        # VOLUME SPIKE DETECTION (Primary trigger)
        has_major_volume = volume_ratio >= 3.0
        has_moderate_volume = volume_ratio >= 2.0
        
        if not has_moderate_volume:
            return StrategySignal("neutral", "none", 0.0, 
                                  reasoning=f"Volume ratio {volume_ratio:.1f}x insufficient for scalp")
        
        reasoning.append(f"Volume spike: {volume_ratio:.1f}x average on {tf_name}")
        
        # BULLISH MOMENTUM SCALP
        if macd_signal == "bullish" and trend in ["uptrend", "bullish"]:
            # RSI should show momentum but not overbought
            if 45 <= rsi <= 75:
                signal_type = "buy"
                direction = "long"
                confidence = 0.70
                reasoning.append(f"Bullish momentum: RSI {rsi:.0f}, MACD bullish")
                
                # Confidence boosts
                if has_major_volume:
                    confidence += 0.08
                    reasoning.append("Major volume spike (3x+)")
                
                if h1_trend in ["uptrend", "bullish"]:
                    confidence += 0.07
                    reasoning.append("1H trend alignment")
                
                if 50 <= rsi <= 65:  # Ideal RSI zone
                    confidence += 0.05
                    reasoning.append("RSI in optimal zone")
        
        # BEARISH MOMENTUM SCALP
        elif macd_signal == "bearish" and trend in ["downtrend", "bearish"]:
            # RSI should show momentum but not oversold
            if 25 <= rsi <= 55:
                signal_type = "sell"
                direction = "short"
                confidence = 0.70
                reasoning.append(f"Bearish momentum: RSI {rsi:.0f}, MACD bearish")
                
                # Confidence boosts
                if has_major_volume:
                    confidence += 0.08
                    reasoning.append("Major volume spike (3x+)")
                
                if h1_trend in ["downtrend", "bearish"]:
                    confidence += 0.07
                    reasoning.append("1H trend alignment")
                
                if 35 <= rsi <= 50:  # Ideal RSI zone
                    confidence += 0.05
                    reasoning.append("RSI in optimal zone")
        
        # Calculate SL/TP (tighter than other strategies)
        stop_loss = None
        take_profit = None
        
        if signal_type != "neutral":
            current_price = symbol_data.get("current_price", 0)
            atr = primary_tf.get("atr", 0)
            
            if current_price > 0 and atr > 0:
                # Scalping: Tight stop (1x ATR), Quick profit (1.5x ATR)
                # Minimum 1.5:1 R:R
                if direction == "long":
                    stop_loss = current_price - (1.0 * atr)
                    take_profit = current_price + (1.5 * atr)
                else:  # short
                    stop_loss = current_price + (1.0 * atr)
                    take_profit = current_price - (1.5 * atr)
            elif current_price > 0:
                # Fallback: percentage-based
                if direction == "long":
                    stop_loss = current_price * 0.992  # 0.8% stop
                    take_profit = current_price * 1.015  # 1.5% target
                else:
                    stop_loss = current_price * 1.008
                    take_profit = current_price * 0.985
        
        # Cap confidence
        confidence = min(confidence, 0.92)
        
        return StrategySignal(
            signal_type=signal_type,
            direction=direction,
            confidence=confidence,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reasoning="; ".join(reasoning) if reasoning else "No scalping setup detected"
        )
    
    def validate_entry(self, signal: StrategySignal, symbol_data: Dict[str, Any]) -> bool:
        """
        Strict validation for Scalping Momentum.
        """
        if signal.signal_type == "neutral":
            return False
        
        tf_analysis = symbol_data.get("timeframe_analysis", {})
        m5_data = tf_analysis.get("5m", {})
        m15_data = tf_analysis.get("15m", {})
        primary_tf = m5_data if m5_data else m15_data
        
        # Guardrail 1: Volume must still be elevated
        volume_ratio = primary_tf.get("volume_ratio", 1.0)
        if volume_ratio < 1.5:
            return False
        
        # Guardrail 2: Don't scalp at RSI extremes
        rsi = primary_tf.get("rsi", 50)
        if signal.direction == "long" and rsi > 80:
            return False
        if signal.direction == "short" and rsi < 20:
            return False
        
        # Guardrail 3: Require momentum alignment
        macd_signal = primary_tf.get("macd_signal", "neutral")
        if signal.direction == "long" and macd_signal == "bearish":
            return False
        if signal.direction == "short" and macd_signal == "bullish":
            return False
        
        return True

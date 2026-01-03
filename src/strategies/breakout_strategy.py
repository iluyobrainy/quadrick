"""
Breakout Strategy - Captures consolidation breakouts with volume confirmation
"""
from typing import Dict, Any
from .base_strategy import BaseStrategy, StrategySignal


class BreakoutStrategy(BaseStrategy):
    """
    Breakout strategy targeting consolidation breakouts.
    
    Entry Conditions:
    - Bollinger Band squeeze (bb_width < 0.025) indicating consolidation
    - Volume spike (2x+ average) confirming breakout
    - Price breaking above/below Bollinger Bands
    
    Exit Strategy:
    - ATR-based trailing stop (wider than trend following)
    - 2:1 risk/reward minimum
    """
    
    def __init__(self):
        super().__init__(
            name="Breakout",
            description="Consolidation breakout strategy with volume confirmation and ATR-based exits."
        )
    
    def analyze(self, symbol_data: Dict[str, Any]) -> StrategySignal:
        """
        Analyze a single symbol's data for breakout setup.
        Looks for BB squeeze + volume spike + directional break.
        """
        tf_analysis = symbol_data.get("timeframe_analysis", {})
        h1_data = tf_analysis.get("1h", {})
        m15_data = tf_analysis.get("15m", {})
        m5_data = tf_analysis.get("5m", {})
        
        # Need at least 15m data for breakout detection
        if not m15_data:
            return StrategySignal("neutral", "none", 0.0, reasoning="Insufficient data for breakout analysis")
        
        # Extract key indicators
        bb_width_15m = m15_data.get("bb_width", 0.05)
        bb_position_15m = m15_data.get("bb_position", "middle")
        volume_ratio_15m = m15_data.get("volume_ratio", 1.0)
        rsi_15m = m15_data.get("rsi", 50)
        
        # 1H context for confirmation
        trend_1h = h1_data.get("trend", "neutral")
        rsi_1h = h1_data.get("rsi", 50)
        
        # Initialize
        signal_type = "neutral"
        direction = "none"
        confidence = 0.0
        reasoning = []
        
        # SQUEEZE DETECTION: BB width < 2.5% indicates consolidation
        is_squeeze = bb_width_15m < 0.025
        
        # VOLUME CONFIRMATION: Need 2x+ average volume for valid breakout
        has_volume = volume_ratio_15m >= 2.0
        
        # BREAKOUT CONDITIONS
        if is_squeeze and has_volume:
            reasoning.append(f"BB Squeeze detected (width: {bb_width_15m:.3f})")
            reasoning.append(f"Volume spike: {volume_ratio_15m:.1f}x average")
            
            # BULLISH BREAKOUT
            if bb_position_15m == "upper":
                # Additional confirmation: RSI not extremely overbought
                if rsi_15m < 80:
                    signal_type = "buy"
                    direction = "long"
                    confidence = 0.72
                    reasoning.append("Price breaking above upper BB")
                    
                    # Boost confidence if 1H aligns
                    if trend_1h in ["uptrend", "bullish"]:
                        confidence += 0.08
                        reasoning.append("1H trend confirms bullish breakout")
                    
                    # Boost for strong volume
                    if volume_ratio_15m >= 3.0:
                        confidence += 0.05
                        reasoning.append("Exceptional volume (3x+)")
            
            # BEARISH BREAKOUT
            elif bb_position_15m == "lower":
                # Additional confirmation: RSI not extremely oversold
                if rsi_15m > 20:
                    signal_type = "sell"
                    direction = "short"
                    confidence = 0.72
                    reasoning.append("Price breaking below lower BB")
                    
                    # Boost confidence if 1H aligns
                    if trend_1h in ["downtrend", "bearish"]:
                        confidence += 0.08
                        reasoning.append("1H trend confirms bearish breakout")
                    
                    # Boost for strong volume
                    if volume_ratio_15m >= 3.0:
                        confidence += 0.05
                        reasoning.append("Exceptional volume (3x+)")
        
        # Check for early squeeze setup (alert, not trade yet)
        elif is_squeeze and not has_volume:
            reasoning.append(f"Squeeze forming (width: {bb_width_15m:.3f}) - waiting for volume confirmation")
        
        # Calculate SL/TP based on ATR
        stop_loss = None
        take_profit = None
        
        if signal_type != "neutral":
            current_price = symbol_data.get("current_price", 0)
            atr = m15_data.get("atr", 0) or h1_data.get("atr", 0)
            
            if current_price > 0 and atr > 0:
                # Breakouts need slightly wider stops than trend following
                # Use 2.5x ATR for stop, 5x ATR for take profit (2:1 R:R)
                if direction == "long":
                    stop_loss = current_price - (2.5 * atr)
                    take_profit = current_price + (5.0 * atr)
                else:  # short
                    stop_loss = current_price + (2.5 * atr)
                    take_profit = current_price - (5.0 * atr)
        
        # Cap confidence at 0.95
        confidence = min(confidence, 0.95)
        
        return StrategySignal(
            signal_type=signal_type,
            direction=direction,
            confidence=confidence,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reasoning="; ".join(reasoning) if reasoning else "No breakout setup detected"
        )
    
    def validate_entry(self, signal: StrategySignal, symbol_data: Dict[str, Any]) -> bool:
        """
        Strict validation for Breakout strategy.
        """
        if signal.signal_type == "neutral":
            return False
        
        tf_analysis = symbol_data.get("timeframe_analysis", {})
        m15_data = tf_analysis.get("15m", {})
        h1_data = tf_analysis.get("1h", {})
        
        # Guardrail 1: Require volume confirmation
        volume_ratio = m15_data.get("volume_ratio", 1.0)
        if volume_ratio < 1.8:  # Slightly lower than setup requirement
            return False
        
        # Guardrail 2: Don't buy at extreme RSI for breakouts
        rsi_15m = m15_data.get("rsi", 50)
        if signal.direction == "long" and rsi_15m > 85:
            return False
        if signal.direction == "short" and rsi_15m < 15:
            return False
        
        # Guardrail 3: Check 1H isn't strongly counter-trend
        rsi_1h = h1_data.get("rsi", 50)
        if signal.direction == "long" and rsi_1h < 30:
            return False  # Don't buy into 1H oversold reversal
        if signal.direction == "short" and rsi_1h > 70:
            return False  # Don't short into 1H overbought reversal
        
        return True

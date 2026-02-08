"""
Counter-Trend Validator - Algorithmic gate for counter-trend trades
Part of the hybrid LLM + Algorithm system

LLMs can articulate WHY a reversal might happen but can't reliably predict WHEN.
This module provides deterministic gates that must be passed before allowing
counter-trend trades.

Usage:
    validator = CounterTrendValidator()
    result = validator.validate(symbol, analysis, llm_reasoning)
    if result.allowed:
        # Proceed with trade
    else:
        # Block and log reason
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class CounterTrendValidation:
    """Result of counter-trend validation"""
    allowed: bool
    score: int  # 0-100
    reasons: List[str]
    requirements_met: Dict[str, bool]
    suggested_rr: float  # Minimum R:R to compensate for lower win rate


class CounterTrendValidator:
    """
    Deterministic validation gate for counter-trend trades.
    
    LLM spots potential reversal → Algorithm validates conditions → Trade or Block
    
    Required conditions for counter-trend (must meet threshold score):
    - ADX < 15 (weak trend = ranging market)
    - Volume spike confirming reversal attempt
    - RSI divergence present
    - Price at major support/resistance (within ATR)
    - Minimum R:R >= 2:1 (compensate for lower win rate)
    """
    
    def __init__(
        self,
        adx_threshold: float = 15.0,
        volume_spike_ratio: float = 1.5,
        min_rr_ratio: float = 2.0,
        min_score_to_allow: int = 70,
    ):
        """
        Initialize counter-trend validator.
        
        Args:
            adx_threshold: Maximum ADX for "weak trend" (default 15)
            volume_spike_ratio: Minimum volume ratio to confirm reversal
            min_rr_ratio: Minimum R:R ratio for counter-trend trades
            min_score_to_allow: Minimum validation score to allow trade
        """
        self.adx_threshold = adx_threshold
        self.volume_spike_ratio = volume_spike_ratio
        self.min_rr_ratio = min_rr_ratio
        self.min_score_to_allow = min_score_to_allow
    
    def validate(
        self,
        symbol: str,
        analysis: Dict[str, Any],
        proposed_side: str,
        proposed_sl: float,
        proposed_tp: float,
    ) -> CounterTrendValidation:
        """
        Validate a counter-trend trade proposal.
        
        Args:
            symbol: Trading pair
            analysis: Market analysis data
            proposed_side: "Buy" or "Sell"
            proposed_sl: Proposed stop loss price
            proposed_tp: Proposed take profit price
            
        Returns:
            CounterTrendValidation with allowed status and requirements
        """
        reasons = []
        requirements = {}
        score = 0
        
        # Extract analysis data
        tf_15m = analysis.get("timeframe_analysis", {}).get("15m", {})
        tf_1h = analysis.get("timeframe_analysis", {}).get("1h", {})
        
        current_price = analysis.get("current_price", 0)
        adx = tf_15m.get("adx", 25)
        volume_ratio = tf_15m.get("volume_ratio", 1.0)
        rsi = tf_15m.get("rsi", 50)
        trend_1h = tf_1h.get("trend", "neutral")
        atr = tf_15m.get("atr", current_price * 0.005)
        
        # Support/resistance levels
        support = analysis.get("support_level", current_price * 0.95)
        resistance = analysis.get("resistance_level", current_price * 1.05)
        
        # ==========================================
        # REQUIREMENT CHECKS (each adds to score)
        # ==========================================
        
        # 1. WEAK TREND (ADX < threshold) - Most important
        weak_trend = adx < self.adx_threshold
        requirements["weak_trend"] = weak_trend
        if weak_trend:
            score += 30
            reasons.append(f"ADX={adx:.1f} (weak trend, ranging market)")
        else:
            reasons.append(f"ADX={adx:.1f} (strong trend, risky counter)")
        
        # 2. VOLUME CONFIRMATION
        volume_confirmed = volume_ratio >= self.volume_spike_ratio
        requirements["volume_spike"] = volume_confirmed
        if volume_confirmed:
            score += 25
            reasons.append(f"Volume spike {volume_ratio:.1f}x confirms reversal attempt")
        else:
            reasons.append(f"Volume {volume_ratio:.1f}x below threshold")
        
        # 3. RSI DIVERGENCE (extreme levels)
        rsi_supports = False
        if proposed_side == "Buy" and rsi < 30:  # Oversold for long
            rsi_supports = True
            score += 20
            reasons.append(f"RSI={rsi:.0f} oversold (supports long)")
        elif proposed_side == "Sell" and rsi > 70:  # Overbought for short
            rsi_supports = True
            score += 20
            reasons.append(f"RSI={rsi:.0f} overbought (supports short)")
        requirements["rsi_divergence"] = rsi_supports
        
        # 4. KEY LEVEL PROXIMITY (within 1 ATR of support for longs, resistance for shorts)
        at_key_level = False
        if proposed_side == "Buy":
            distance_to_support = abs(current_price - support)
            if distance_to_support <= atr:
                at_key_level = True
                score += 15
                reasons.append(f"Near support ${support:.2f} (within ATR)")
        else:
            distance_to_resistance = abs(current_price - resistance)
            if distance_to_resistance <= atr:
                at_key_level = True
                score += 15
                reasons.append(f"Near resistance ${resistance:.2f} (within ATR)")
        requirements["key_level"] = at_key_level
        
        # 5. R:R RATIO CHECK (minimum 2:1 for counter-trend)
        if current_price > 0 and proposed_sl > 0 and proposed_tp > 0:
            risk = abs(current_price - proposed_sl)
            reward = abs(proposed_tp - current_price)
            rr = reward / risk if risk > 0 else 0
            
            adequate_rr = rr >= self.min_rr_ratio
            requirements["min_rr"] = adequate_rr
            if adequate_rr:
                score += 10
                reasons.append(f"R:R={rr:.2f}:1 (adequate for counter-trend)")
            else:
                reasons.append(f"R:R={rr:.2f}:1 (below {self.min_rr_ratio}:1 minimum)")
        else:
            requirements["min_rr"] = False
            reasons.append("Cannot calculate R:R")
        
        # ==========================================
        # FINAL DECISION
        # ==========================================
        allowed = score >= self.min_score_to_allow
        
        if allowed:
            logger.info(
                f"✅ Counter-trend {proposed_side} on {symbol} ALLOWED (score={score}): "
                f"{', '.join(reasons)}"
            )
        else:
            logger.warning(
                f"🚫 Counter-trend {proposed_side} on {symbol} BLOCKED (score={score}): "
                f"{', '.join(reasons)}"
            )
        
        return CounterTrendValidation(
            allowed=allowed,
            score=score,
            reasons=reasons,
            requirements_met=requirements,
            suggested_rr=self.min_rr_ratio,
        )
    
    def detect_counter_trend(
        self,
        proposed_side: str,
        trend_1h: str,
    ) -> bool:
        """
        Detect if a trade is counter-trend.
        
        Args:
            proposed_side: "Buy" or "Sell"
            trend_1h: 1H trend direction
            
        Returns:
            True if counter-trend, False otherwise
        """
        if proposed_side == "Buy" and trend_1h == "trending_down":
            return True
        if proposed_side == "Sell" and trend_1h == "trending_up":
            return True
        return False

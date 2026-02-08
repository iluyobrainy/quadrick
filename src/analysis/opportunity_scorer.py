"""
Opportunity Scorer - Deterministic pre-LLM opportunity detection for scalping
Part of the hybrid LLM + Algorithm system

This module scores trading opportunities BEFORE asking the LLM, helping focus
the LLM's attention on high-probability setups rather than noise.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class OpportunityScore:
    """Represents a scored trading opportunity"""
    symbol: str
    direction: str  # "long", "short", or "neutral"
    score: float  # 0-100
    signals: List[str]
    suggested_sl: float
    suggested_tp: float
    counter_trend: bool = False


class OpportunityScorer:
    """
    Deterministic opportunity scoring for scalping setups.
    
    Scores based on:
    - Volume spikes (institutional activity)
    - RSI extremes (momentum)
    - BB squeeze (volatility contraction -> pending breakout)
    - MACD direction (momentum confirmation)
    - HTF trend alignment (1H trend)
    
    Minimum 3/4 directional signals + volume >= 1.3x for high-quality setups.
    """
    
    def __init__(
        self,
        min_volume_ratio: float = 1.3,
        rsi_oversold: float = 25,
        rsi_overbought: float = 75,
        bb_squeeze_threshold: float = 0.02,
        min_score_threshold: int = 65,
    ):
        self.min_volume_ratio = min_volume_ratio
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.bb_squeeze_threshold = bb_squeeze_threshold
        self.min_score_threshold = min_score_threshold
    
    def score_symbol(self, symbol: str, analysis: Dict[str, Any]) -> OpportunityScore:
        """
        Score a symbol for scalping opportunity.
        
        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            analysis: Market analysis dict from market_analyzer
            
        Returns:
            OpportunityScore with direction, score, and signals
        """
        signals = []
        score = 50  # Neutral baseline
        direction = "neutral"
        counter_trend = False
        
        # Extract timeframe data
        tf_15m = analysis.get("timeframe_analysis", {}).get("15m", {})
        tf_1h = analysis.get("timeframe_analysis", {}).get("1h", {})
        
        # Core indicators
        rsi = tf_15m.get("rsi", 50)
        macd_signal = tf_15m.get("macd_signal", "neutral")
        volume_ratio = tf_15m.get("volume_ratio", 1.0)
        bb_width = tf_15m.get("bb_width", 0.03)
        trend_1h = tf_1h.get("trend", "neutral")
        adx = tf_15m.get("adx", 25)  # For counter-trend detection
        
        # Get price info for SL/TP calculation
        current_price = analysis.get("current_price", 0)
        atr = tf_15m.get("atr", current_price * 0.005)
        
        # ==========================================
        # SCORING LOGIC
        # ==========================================
        
        # 1. VOLUME SPIKE (Institutional activity) - High weight
        if volume_ratio >= 2.0:
            score += 18
            signals.append(f"volume_spike_{volume_ratio:.1f}x")
        elif volume_ratio >= self.min_volume_ratio:
            score += 10
            signals.append(f"elevated_volume_{volume_ratio:.1f}x")
        elif volume_ratio < 1.0:
            score -= 5  # Low volume = weak moves
        
        # 2. BB SQUEEZE (Volatility contraction = pending breakout)
        if bb_width < self.bb_squeeze_threshold:
            score += 14
            signals.append("bb_squeeze")
        
        # 3. RSI EXTREMES with reversal potential
        if rsi < self.rsi_oversold:
            if trend_1h != "trending_down":
                score += 12
                direction = "long"
                signals.append(f"rsi_oversold_{rsi:.0f}")
            else:
                score += 5  # Less confident in downtrend
                direction = "long"
                counter_trend = True
                signals.append(f"rsi_oversold_counter_{rsi:.0f}")
        elif rsi > self.rsi_overbought:
            if trend_1h != "trending_up":
                score += 12
                direction = "short"
                signals.append(f"rsi_overbought_{rsi:.0f}")
            else:
                score += 5
                direction = "short"
                counter_trend = True
                signals.append(f"rsi_overbought_counter_{rsi:.0f}")
        
        # 4. MACD DIRECTION
        if macd_signal == "bullish":
            score += 10
            if direction == "neutral":
                direction = "long"
            signals.append("macd_bullish")
        elif macd_signal == "bearish":
            score += 10
            if direction == "neutral":
                direction = "short"
            signals.append("macd_bearish")
        
        # 5. HTF ALIGNMENT BONUS (most important for scalping)
        if trend_1h == "trending_up" and direction == "long":
            score += 18
            signals.append("htf_aligned_long")
        elif trend_1h == "trending_down" and direction == "short":
            score += 18
            signals.append("htf_aligned_short")
        elif trend_1h != "neutral" and direction != "neutral":
            # Counter-trend detected
            if adx < 15:
                # Weak trend = ranging, counter-trend more acceptable
                score += 5
                signals.append("weak_trend_reversal_ok")
            else:
                score -= 15  # Strong trend counter = penalty
                counter_trend = True
                signals.append("counter_trend_warning")
        
        # Calculate suggested SL/TP based on ATR
        suggested_sl = 0.0
        suggested_tp = 0.0
        
        if current_price > 0 and direction != "neutral":
            if direction == "long":
                suggested_sl = current_price - (atr * 1.2)  # 1.2x ATR below
                suggested_tp = current_price + (atr * 1.8)  # 1.5:1 R:R
            else:
                suggested_sl = current_price + (atr * 1.2)  # 1.2x ATR above
                suggested_tp = current_price - (atr * 1.8)  # 1.5:1 R:R
        
        return OpportunityScore(
            symbol=symbol,
            direction=direction,
            score=min(100, max(0, score)),
            signals=signals,
            suggested_sl=suggested_sl,
            suggested_tp=suggested_tp,
            counter_trend=counter_trend,
        )
    
    def get_top_opportunities(
        self, 
        analyses: Dict[str, Dict[str, Any]], 
        min_score: Optional[int] = None,
        max_results: int = 5,
    ) -> List[OpportunityScore]:
        """
        Get top trading opportunities sorted by score.
        
        Args:
            analyses: Dict of symbol -> analysis data
            min_score: Minimum score to include (default: self.min_score_threshold)
            max_results: Maximum number of opportunities to return
            
        Returns:
            List of OpportunityScore objects, sorted by score descending
        """
        if min_score is None:
            min_score = self.min_score_threshold
        
        opportunities = []
        for symbol, analysis in analyses.items():
            try:
                opp = self.score_symbol(symbol, analysis)
                if opp.score >= min_score:
                    opportunities.append(opp)
            except Exception as e:
                logger.warning(f"Failed to score {symbol}: {e}")
                continue
        
        # Sort by score descending
        opportunities.sort(key=lambda x: x.score, reverse=True)
        
        return opportunities[:max_results]
    
    def filter_counter_trend(
        self,
        opportunities: List[OpportunityScore],
        allow_weak_trend: bool = True,
    ) -> List[OpportunityScore]:
        """
        Filter out counter-trend opportunities unless they meet strict criteria.
        
        Args:
            opportunities: List of scored opportunities
            allow_weak_trend: If True, allow counter-trend when ADX < 15
            
        Returns:
            Filtered list excluding risky counter-trend setups
        """
        filtered = []
        for opp in opportunities:
            if not opp.counter_trend:
                filtered.append(opp)
            elif allow_weak_trend and "weak_trend_reversal_ok" in opp.signals:
                filtered.append(opp)
            else:
                logger.info(f"Filtered counter-trend opportunity: {opp.symbol} ({opp.direction})")
        
        return filtered

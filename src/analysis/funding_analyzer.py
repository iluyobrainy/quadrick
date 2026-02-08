"""
Funding Rate Analyzer Module

Detects crowded trades based on extreme funding rates and adjusts
position sizing accordingly.
"""
from typing import Dict, Any
from dataclasses import dataclass
from loguru import logger


@dataclass
class FundingAnalysis:
    """Result of funding rate analysis"""
    is_crowded: bool         # True if trade goes with extreme funding
    is_contrarian: bool      # True if trade fades extreme funding
    funding_rate: float      # The current funding rate
    position_multiplier: float  # 1.0 = full size, 0.6 = 40% reduction
    reason: str              # Human-readable explanation


class FundingAnalyzer:
    """
    Analyze funding rates to detect crowded trades.
    
    When funding is extreme:
    - Going WITH the crowd is risky (long when funding very positive)
    - Going AGAINST the crowd can be profitable (short when funding very positive)
    """
    
    # Funding rate thresholds (8-hour rate)
    EXTREME_THRESHOLD = 0.0005    # 0.05% = Moderate extreme
    VERY_EXTREME_THRESHOLD = 0.001  # 0.1% = Very extreme
    
    # Position reduction for crowded trades
    CROWDED_REDUCTION = 0.6  # Reduce position to 60% (40% reduction)
    
    def __init__(
        self,
        extreme_threshold: float = 0.0005,
        crowded_reduction: float = 0.6,
    ):
        """
        Initialize funding analyzer.
        
        Args:
            extreme_threshold: Funding rate considered extreme
            crowded_reduction: Position multiplier for crowded trades (0.6 = 40% reduction)
        """
        self.extreme_threshold = extreme_threshold
        self.crowded_reduction = crowded_reduction
    
    def analyze(self, symbol: str, side: str, funding_rate: float) -> FundingAnalysis:
        """
        Analyze whether a trade is crowded based on funding rate.
        
        Args:
            symbol: Trading symbol
            side: "Buy" or "Sell"
            funding_rate: Current funding rate (e.g., 0.0003 = 0.03%)
            
        Returns:
            FundingAnalysis with position sizing recommendation
        """
        is_positive_funding = funding_rate > self.extreme_threshold
        is_negative_funding = funding_rate < -self.extreme_threshold
        is_very_extreme = abs(funding_rate) > self.VERY_EXTREME_THRESHOLD
        
        # Determine if this is a crowded or contrarian trade
        is_crowded = False
        is_contrarian = False
        reason = "Normal funding"
        
        if is_positive_funding:
            # Positive funding = longs pay shorts = everyone is long
            if side == "Buy":
                is_crowded = True
                reason = f"Crowded long (funding={funding_rate:.4%}, longs pay shorts)"
            else:
                is_contrarian = True
                reason = f"Contrarian short (fading crowded longs, funding={funding_rate:.4%})"
        
        elif is_negative_funding:
            # Negative funding = shorts pay longs = everyone is short
            if side == "Sell":
                is_crowded = True
                reason = f"Crowded short (funding={funding_rate:.4%}, shorts pay longs)"
            else:
                is_contrarian = True
                reason = f"Contrarian long (fading crowded shorts, funding={funding_rate:.4%})"
        
        # Calculate position multiplier
        if is_crowded:
            if is_very_extreme:
                multiplier = self.crowded_reduction * 0.8  # Even more reduction for very extreme
            else:
                multiplier = self.crowded_reduction
        else:
            multiplier = 1.0  # Full position size
        
        return FundingAnalysis(
            is_crowded=is_crowded,
            is_contrarian=is_contrarian,
            funding_rate=funding_rate,
            position_multiplier=multiplier,
            reason=reason,
        )
    
    def should_reduce_position(self, symbol: str, side: str, funding_rate: float) -> bool:
        """
        Quick check if position should be reduced.
        
        Args:
            symbol: Trading symbol
            side: "Buy" or "Sell"
            funding_rate: Current funding rate
            
        Returns:
            True if position should be reduced
        """
        analysis = self.analyze(symbol, side, funding_rate)
        return analysis.is_crowded
    
    def get_position_multiplier(self, symbol: str, side: str, funding_rate: float) -> float:
        """
        Get position size multiplier based on funding.
        
        Args:
            symbol: Trading symbol
            side: "Buy" or "Sell"
            funding_rate: Current funding rate
            
        Returns:
            Multiplier (1.0 = full size, 0.6 = 40% reduction)
        """
        analysis = self.analyze(symbol, side, funding_rate)
        return analysis.position_multiplier

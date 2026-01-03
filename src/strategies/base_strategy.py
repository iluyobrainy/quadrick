from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class StrategySignal:
    """Standardized output for strategy signals"""
    signal_type: str  # "buy", "sell", "neutral"
    direction: str    # "long", "short", "none"
    confidence: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    reasoning: str = ""
    metadata: Optional[Dict[str, Any]] = None

class BaseStrategy(ABC):
    """Abstract base class for all trading strategies"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    @abstractmethod
    def analyze(self, market_data: Dict[str, Any]) -> StrategySignal:
        """
        Analyze market data and return a trading signal.
        
        Args:
            market_data: Dictionary containing candles, indicators, etc.
            
        Returns:
            StrategySignal object
        """
        pass

    @abstractmethod
    def validate_entry(self, signal: StrategySignal, market_data: Dict[str, Any]) -> bool:
        """
        Validate if the signal meets strict entry criteria (guardrails).
        """
        pass

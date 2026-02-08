"""
Symbol-Specific Configuration Module

Per-symbol parameters for trailing stops, risk limits, and R:R ratios
based on volatility classification.
"""
from typing import Dict, Any
from dataclasses import dataclass


@dataclass
class SymbolConfig:
    """Configuration for a specific trading symbol"""
    trail_mult: float      # ATR multiplier for trailing stops (0.8-1.2)
    min_rr: float          # Minimum risk:reward ratio
    max_risk: float        # Maximum risk percentage per trade
    vol_class: str         # Volatility classification: low, medium, high
    

# Symbol configurations based on typical volatility profiles
# High volatility = wider stops, lower risk
# Low volatility = tighter stops, can risk more
SYMBOL_CONFIGS: Dict[str, SymbolConfig] = {
    # =========================================
    # HIGH VOLATILITY - Wider stops, lower risk
    # =========================================
    "1000PEPEUSDT": SymbolConfig(trail_mult=1.2, min_rr=1.5, max_risk=12, vol_class="high"),
    "DOGEUSDT": SymbolConfig(trail_mult=1.1, min_rr=1.5, max_risk=12, vol_class="high"),
    "SOLUSDT": SymbolConfig(trail_mult=1.0, min_rr=1.4, max_risk=15, vol_class="high"),
    
    # =========================================
    # MEDIUM VOLATILITY - Balanced settings
    # =========================================
    "AVAXUSDT": SymbolConfig(trail_mult=0.9, min_rr=1.3, max_risk=18, vol_class="medium"),
    "ARBUSDT": SymbolConfig(trail_mult=0.9, min_rr=1.3, max_risk=18, vol_class="medium"),
    "OPUSDT": SymbolConfig(trail_mult=0.9, min_rr=1.3, max_risk=18, vol_class="medium"),
    "LINKUSDT": SymbolConfig(trail_mult=0.9, min_rr=1.3, max_risk=18, vol_class="medium"),
    "DOTUSDT": SymbolConfig(trail_mult=0.9, min_rr=1.3, max_risk=18, vol_class="medium"),
    "ADAUSDT": SymbolConfig(trail_mult=0.9, min_rr=1.3, max_risk=18, vol_class="medium"),
    "XRPUSDT": SymbolConfig(trail_mult=0.9, min_rr=1.3, max_risk=18, vol_class="medium"),
    
    # =========================================
    # LOW VOLATILITY - Tighter stops, more risk
    # =========================================
    "BTCUSDT": SymbolConfig(trail_mult=0.8, min_rr=1.3, max_risk=20, vol_class="low"),
    "ETHUSDT": SymbolConfig(trail_mult=0.8, min_rr=1.3, max_risk=20, vol_class="low"),
}

# Default configuration for unknown symbols
DEFAULT_CONFIG = SymbolConfig(trail_mult=0.8, min_rr=1.3, max_risk=15, vol_class="medium")


def get_symbol_config(symbol: str) -> SymbolConfig:
    """
    Get configuration for a specific symbol.
    
    Args:
        symbol: Trading symbol (e.g., "BTCUSDT")
        
    Returns:
        SymbolConfig with per-symbol parameters
    """
    return SYMBOL_CONFIGS.get(symbol, DEFAULT_CONFIG)


def get_trail_distance(symbol: str, atr: float, price: float) -> float:
    """
    Calculate symbol-specific trailing stop distance.
    
    Args:
        symbol: Trading symbol
        atr: Current ATR value
        price: Current price
        
    Returns:
        Trail distance as percentage, bounded 0.3%-2.0%
    """
    config = get_symbol_config(symbol)
    atr_pct = (atr / price) * 100 if price > 0 else 0.5
    
    # Apply symbol-specific multiplier and bound
    trail_pct = atr_pct * config.trail_mult
    
    # Bounds vary by volatility class
    if config.vol_class == "high":
        return max(0.4, min(trail_pct, 2.0))  # Wider bounds for high vol
    elif config.vol_class == "low":
        return max(0.3, min(trail_pct, 1.2))  # Tighter bounds for low vol
    else:
        return max(0.3, min(trail_pct, 1.5))  # Medium bounds


def get_adjusted_risk(symbol: str, requested_risk: float) -> float:
    """
    Cap risk to symbol-specific maximum.
    
    Args:
        symbol: Trading symbol
        requested_risk: Risk percentage requested by LLM
        
    Returns:
        Risk percentage capped to symbol max
    """
    config = get_symbol_config(symbol)
    return min(requested_risk, config.max_risk)


def get_min_rr_ratio(symbol: str) -> float:
    """
    Get minimum R:R ratio for a symbol.
    
    Args:
        symbol: Trading symbol
        
    Returns:
        Minimum R:R ratio
    """
    config = get_symbol_config(symbol)
    return config.min_rr

"""
Order Flow Analysis - Detect whale orders and market depth
"""
from typing import Dict, List, Tuple, Any
from loguru import logger


class OrderFlowAnalyzer:
    """Analyze orderbook depth and flow"""
    
    def __init__(self):
        """Initialize order flow analyzer"""
        logger.info("Order flow analyzer initialized")
    
    def analyze_orderbook(
        self,
        symbol: str,
        bids: List[Tuple[float, float]],
        asks: List[Tuple[float, float]],
        current_price: float,
    ) -> Dict[str, Any]:
        """
        Analyze orderbook for trading signals
        
        Args:
            symbol: Trading symbol
            bids: List of (price, quantity) tuples
            asks: List of (price, quantity) tuples
            current_price: Current market price
        
        Returns:
            Order flow analysis dict
        """
        if not bids or not asks:
            return self._empty_analysis()
        
        # Calculate depths
        bid_depth_10 = sum(price * qty for price, qty in bids[:10])
        ask_depth_10 = sum(price * qty for price, qty in asks[:10])
        
        bid_depth_25 = sum(price * qty for price, qty in bids[:25])
        ask_depth_25 = sum(price * qty for price, qty in asks[:25])
        
        # Imbalance ratio
        imbalance = bid_depth_10 / ask_depth_10 if ask_depth_10 > 0 else 1.0
        
        # Detect large orders (whales)
        whale_threshold_usd = 50000  # $50k+ = whale order
        
        large_bids = []
        for price, qty in bids:
            value = price * qty
            if value >= whale_threshold_usd:
                distance_pct = ((current_price - price) / current_price) * 100
                large_bids.append({
                    "price": price,
                    "size_usd": value,
                    "distance_pct": round(distance_pct, 2),
                    "note": f"${value:,.0f} support wall" if value < 100000 else f"${value:,.0f} MASSIVE support",
                })
        
        large_asks = []
        for price, qty in asks:
            value = price * qty
            if value >= whale_threshold_usd:
                distance_pct = ((price - current_price) / current_price) * 100
                large_asks.append({
                    "price": price,
                    "size_usd": value,
                    "distance_pct": round(distance_pct, 2),
                    "note": f"${value:,.0f} resistance wall" if value < 100000 else f"${value:,.0f} MASSIVE resistance",
                })
        
        # Determine pressure
        if imbalance > 1.5:
            pressure = "strong_buying"
        elif imbalance > 1.2:
            pressure = "moderate_buying"
        elif imbalance < 0.67:
            pressure = "strong_selling"
        elif imbalance < 0.83:
            pressure = "moderate_selling"
        else:
            pressure = "balanced"
        
        # Calculate spread
        best_bid = bids[0][0] if bids else current_price
        best_ask = asks[0][0] if asks else current_price
        spread = best_ask - best_bid
        spread_pct = (spread / current_price) * 100
        
        return {
            "symbol": symbol,
            "bid_depth_10_levels_usd": round(bid_depth_10, 0),
            "ask_depth_10_levels_usd": round(ask_depth_10, 0),
            "bid_ask_imbalance": round(imbalance, 2),
            "market_pressure": pressure,
            "spread": round(spread, 2),
            "spread_pct": round(spread_pct, 4),
            "large_bids": large_bids[:5],  # Top 5 whale buy walls
            "large_asks": large_asks[:5],  # Top 5 whale sell walls
            "interpretation": self._interpret_orderbook(
                imbalance, large_bids, large_asks, pressure
            ),
        }
    
    def _interpret_orderbook(
        self,
        imbalance: float,
        large_bids: List[Dict],
        large_asks: List[Dict],
        pressure: str,
    ) -> str:
        """Generate human-readable interpretation"""
        interpretations = []
        
        # Imbalance
        if imbalance > 1.5:
            interpretations.append(f"Strong buying pressure (imbalance: {imbalance:.2f}x)")
        elif imbalance < 0.67:
            interpretations.append(f"Strong selling pressure (imbalance: {imbalance:.2f}x)")
        else:
            interpretations.append("Balanced orderbook")
        
        # Whale orders
        if large_bids:
            closest_support = min(large_bids, key=lambda x: x["distance_pct"])
            interpretations.append(f"Major support: ${closest_support['size_usd']:,.0f} wall at {closest_support['distance_pct']:.1f}% below")
        
        if large_asks:
            closest_resistance = min(large_asks, key=lambda x: x["distance_pct"])
            interpretations.append(f"Major resistance: ${closest_resistance['size_usd']:,.0f} wall at {closest_resistance['distance_pct']:.1f}% above")
        
        return " | ".join(interpretations)
    
    def _empty_analysis(self) -> Dict[str, Any]:
        """Return empty analysis when no data"""
        return {
            "bid_depth_10_levels_usd": 0,
            "ask_depth_10_levels_usd": 0,
            "bid_ask_imbalance": 1.0,
            "market_pressure": "unknown",
            "large_bids": [],
            "large_asks": [],
            "interpretation": "No orderbook data available",
        }

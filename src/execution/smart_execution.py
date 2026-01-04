"""
Smart Execution Module - Partial profits, trailing stops, advanced order management
"""
from typing import Dict, List, Any, Optional
from datetime import datetime
from loguru import logger
from dataclasses import dataclass


@dataclass
class TrailingStop:
    """Trailing stop configuration"""
    symbol: str
    initial_stop: float
    current_stop: float
    trail_distance_pct: float
    highest_price: float  # For longs
    lowest_price: float   # For shorts
    entry_price: float
    activated: bool = False


class SmartExecutionManager:
    """Manages advanced order execution strategies"""
    
    def __init__(self):
        """Initialize smart execution manager"""
        self.trailing_stops: Dict[str, TrailingStop] = {}
        self.partial_profit_targets: Dict[str, List[Dict]] = {}
        self.position_entry_times: Dict[str, datetime] = {}  # Track entry times
        logger.info("Smart execution manager initialized")
    
    def setup_partial_take_profits(
        self,
        symbol: str,
        position_size: float,
        tp_levels: List[Dict[str, Any]],
    ):
        """
        Setup partial take profit levels
        
        Args:
            symbol: Trading symbol
            position_size: Total position size
            tp_levels: List of {"price": float, "percentage": float}
        
        Example:
            tp_levels = [
                {"price": 71000, "percentage": 50},  # Take 50% profit at 71000
                {"price": 72000, "percentage": 50},  # Take remaining 50% at 72000
            ]
        """
        self.partial_profit_targets[symbol] = []
        
        remaining = position_size
        for level in tp_levels:
            amount = position_size * (level["percentage"] / 100)
            self.partial_profit_targets[symbol].append({
                "price": level["price"],
                "amount": amount,
                "percentage": level["percentage"],
                "executed": False,
            })
            remaining -= amount
        
        logger.info(f"Partial TPs set for {symbol}: {len(tp_levels)} levels")
    
    def setup_trailing_stop(
        self,
        symbol: str,
        side: str,
        initial_stop: float,
        trail_distance_pct: float,
        current_price: float,
        entry_price: float,
    ):
        """
        Setup trailing stop for a position
        
        Args:
            symbol: Trading symbol
            side: "Buy" or "Sell"
            initial_stop: Initial stop loss price
            trail_distance_pct: Trail distance as percentage
            current_price: Current market price
            entry_price: Position entry price (used to respect exchange stop-loss rules)
        """
        self.trailing_stops[symbol] = TrailingStop(
            symbol=symbol,
            initial_stop=initial_stop,
            current_stop=initial_stop,
            trail_distance_pct=trail_distance_pct,
            highest_price=current_price if side == "Buy" else 0,
            lowest_price=current_price if side == "Sell" else float('inf'),
            entry_price=entry_price,
            activated=False,
        )
        
        logger.info(f"Trailing stop set for {symbol}: {trail_distance_pct}% trail")
    
    def update_trailing_stop(
        self,
        symbol: str,
        current_price: float,
        side: str,
    ) -> Optional[Dict[str, Optional[float]]]:
        """
        Update trailing stop based on current price
        
        Args:
            symbol: Trading symbol
            current_price: Current market price
            side: "Buy" or "Sell"
        
        Returns:
            Dictionary containing updated `stop_loss`/`take_profit` levels, or None if no change
        """
        if symbol not in self.trailing_stops:
            return None
        
        trailing = self.trailing_stops[symbol]
        
        entry_price = trailing.entry_price

        if side == "Buy":  # Long position
            # Update highest price
            if current_price > trailing.highest_price:
                trailing.highest_price = current_price
                
                # Calculate new trailing stop
                new_stop = current_price * (1 - trailing.trail_distance_pct / 100)
                
                # Only update if new stop is higher than current
                if new_stop > trailing.current_stop:
                    old_stop = trailing.current_stop
                    trailing.current_stop = new_stop
                    trailing.activated = True
                    
                    logger.info(f"{symbol} trailing stop updated: ${old_stop:.2f} → ${new_stop:.2f}")
                    return {"stop_loss": new_stop, "take_profit": None}
        
        else:  # Short position
            # Update lowest price
            if current_price < trailing.lowest_price:
                trailing.lowest_price = current_price
                
                # Calculate new trailing stop
                new_stop = current_price * (1 + trailing.trail_distance_pct / 100)

                # Bybit requires stop loss for shorts to remain above entry price
                min_short_stop = entry_price * 1.0005  # small buffer above entry
                adjusted_stop = max(new_stop, min_short_stop)
                
                # Only update if new stop is lower than current
                if adjusted_stop < trailing.current_stop:
                    old_stop = trailing.current_stop
                    trailing.current_stop = adjusted_stop
                    trailing.activated = True

                    if adjusted_stop != new_stop:
                        logger.info(
                            f"{symbol} trailing stop adjusted to stay above entry: ${old_stop:.2f} → ${adjusted_stop:.2f}"
                        )
                    else:
                        logger.info(
                            f"{symbol} trailing stop updated: ${old_stop:.2f} → ${adjusted_stop:.2f}"
                        )

                    update_payload: Dict[str, Optional[float]] = {"stop_loss": adjusted_stop, "take_profit": None}

                    # If the adjusted stop cannot trail further due to exchange rules, bring take-profit closer
                    if adjusted_stop == min_short_stop:
                        # Aim to lock in profit by nudging take profit closer (avoid immediate fill)
                        trail_fraction = max(trailing.trail_distance_pct / 200, 0.1)
                        suggested_tp = current_price * (1 - trail_fraction / 100)

                        if suggested_tp < trailing.lowest_price:
                            update_payload["take_profit"] = suggested_tp

                    return update_payload
        
        return None
    
    def get_current_stop(self, symbol: str) -> Optional[float]:
        """Get current stop price for a symbol"""
        if symbol in self.trailing_stops:
            return self.trailing_stops[symbol].current_stop
        return None
    
    def should_take_partial_profit(
        self,
        symbol: str,
        current_price: float,
    ) -> Optional[Dict[str, Any]]:
        """
        Check if partial profit should be taken
        
        Args:
            symbol: Trading symbol
            current_price: Current market price
        
        Returns:
            Partial profit instruction dict or None
        """
        if symbol not in self.partial_profit_targets:
            return None
        
        for target in self.partial_profit_targets[symbol]:
            if target["executed"]:
                continue
            
            # Check if price hit target
            if current_price >= target["price"]:
                target["executed"] = True
                
                logger.info(
                    f"{symbol} partial profit triggered: "
                    f"Take {target['percentage']}% at ${target['price']}"
                )
                
                return {
                    "symbol": symbol,
                    "amount": target["amount"],
                    "price": target["price"],
                    "percentage": target["percentage"],
                }
        
        return None
    
    def move_stop_to_breakeven(
        self,
        symbol: str,
        entry_price: float,
        current_price: float,
        target_price: float,
        side: str,
    ) -> Optional[float]:
        """
        Move stop to break-even when 50% to target
        
        Args:
            symbol: Trading symbol
            entry_price: Entry price
            current_price: Current price
            target_price: Take profit target
            side: "Buy" or "Sell"
        
        Returns:
            Break-even stop price if should move, None otherwise
        """
        if side == "Buy":
            progress = (current_price - entry_price) / (target_price - entry_price)
        else:
            progress = (entry_price - current_price) / (entry_price - target_price)
        
        # Move to break-even at 50% progress
        if progress >= 0.5:
            logger.info(f"{symbol} reached 50% to target - moving stop to break-even")
            return entry_price
        
        return None
    
    def clear_position(self, symbol: str):
        """Clear execution data for a symbol"""
        if symbol in self.trailing_stops:
            del self.trailing_stops[symbol]
        if symbol in self.partial_profit_targets:
            del self.partial_profit_targets[symbol]
        if symbol in self.position_entry_times:
            del self.position_entry_times[symbol]
        logger.debug(f"Cleared execution data for {symbol}")
    
    def calculate_dynamic_trail_distance(
        self,
        current_atr: float,
        entry_atr: float,
        base_trail_pct: float = 1.5,
        min_trail_pct: float = 0.8,
        max_trail_pct: float = 3.0,
    ) -> float:
        """
        Calculate dynamic trailing stop distance based on volatility changes.
        
        If volatility (ATR) increases since entry, widen the trailing stop.
        If volatility decreases, tighten the trailing stop to lock in profits.
        
        Args:
            current_atr: Current ATR value
            entry_atr: ATR at position entry
            base_trail_pct: Base trailing distance percentage
            min_trail_pct: Minimum allowed trail distance
            max_trail_pct: Maximum allowed trail distance
            
        Returns:
            Adjusted trailing stop percentage
        """
        if entry_atr <= 0 or current_atr <= 0:
            return base_trail_pct
        
        # Volatility ratio: >1 means volatility increased, <1 means decreased
        volatility_ratio = current_atr / entry_atr
        
        # Adjust trail distance proportionally to volatility change
        # More volatility = wider trail (to avoid noise stops)
        # Less volatility = tighter trail (lock in profits)
        adjusted_trail = base_trail_pct * volatility_ratio
        
        # Clamp to min/max bounds
        adjusted_trail = max(min_trail_pct, min(adjusted_trail, max_trail_pct))
        
        logger.debug(
            f"Dynamic trail: ATR ratio={volatility_ratio:.2f}, "
            f"base={base_trail_pct}%, adjusted={adjusted_trail:.2f}%"
        )
        
        return adjusted_trail
    
    def update_trail_distance_from_atr(
        self,
        symbol: str,
        current_atr: float,
        entry_atr: float,
    ) -> bool:
        """
        Update an existing trailing stop's distance based on current volatility.
        
        Args:
            symbol: Trading symbol
            current_atr: Current ATR value
            entry_atr: ATR at position entry
            
        Returns:
            True if trail distance was updated, False otherwise
        """
        if symbol not in self.trailing_stops:
            return False
        
        trailing = self.trailing_stops[symbol]
        new_trail_pct = self.calculate_dynamic_trail_distance(
            current_atr=current_atr,
            entry_atr=entry_atr,
            base_trail_pct=trailing.trail_distance_pct,
        )
        
        if abs(new_trail_pct - trailing.trail_distance_pct) > 0.1:
            old_trail = trailing.trail_distance_pct
            trailing.trail_distance_pct = new_trail_pct
            logger.info(
                f"{symbol} trail distance adjusted: {old_trail:.2f}% → {new_trail_pct:.2f}% "
                f"(volatility change)"
            )
            return True
        
        return False

        return False

    def check_reversal_guard(
        self,
        symbol: str,
        side: str,
        current_price: float,
        entry_price: float,
        pnl_pct: float,
        market_data: Dict[str, Any]
    ) -> Optional[str]:
        """
        Smart Reversal Guard: Checks if a profitable position is showing signs of reversal.
        Returns a reason string if the trade should be closed immediately.
        """
        # 1. Only guard if we have decent profit (e.g. > 0.3%)
        # We don't want to panic exit small fluctuations around breakeven
        if pnl_pct < 0.3:
            return None

        # Extract indicators
        tf_analysis = market_data.get("timeframe_analysis", {})
        # Look at 1m or 5m for immediate reversal signs
        m1_data = tf_analysis.get("1m", {})
        m5_data = tf_analysis.get("5m", {})
        
        # Use available LTF data
        ltf_data = m1_data if m1_data else m5_data
        if not ltf_data:
            return None

        rsi = ltf_data.get("indicators", {}).get("rsi", 50)
        volume_ratio = ltf_data.get("indicators", {}).get("volume_ratio", 1.0)
        
        # 2. RSI Momentum Decay Check
        # If we are Long and RSI drops hard from overbought (e.g. was 75, now 60)
        # Or simply if RSI shows weakness while price is stalling
        if side == "Buy":
            if rsi < 50: 
                return f"Momentum Death (Long): RSI dropped to {rsi:.1f} while in profit"
        else: # Sell
            if rsi > 50:
                return f"Momentum Death (Short): RSI rose to {rsi:.1f} while in profit"

        # 3. Volume Exhaustion Check
        # If price is stalling (implied by this check running) and volume dies
        if volume_ratio < 0.5:
             return f"Volume Exhaustion: Volume ratio {volume_ratio:.2f} too low to sustain move"

        return None

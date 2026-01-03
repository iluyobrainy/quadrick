"""
Position Monitoring - Real-time tracking and management
"""
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from loguru import logger
from dataclasses import dataclass, field


@dataclass
class MonitoredPosition:
    """Position being monitored"""
    symbol: str
    side: str
    size: float
    entry_price: float
    entry_time: datetime
    stop_loss: float
    take_profit: float
    unrealized_pnl: float = 0.0
    current_price: float = 0.0
    last_update: datetime = field(default_factory=datetime.utcnow)


class PositionMonitor:
    """Monitor open positions in real-time"""
    
    def __init__(self):
        """Initialize position monitor"""
        self.monitored_positions: Dict[str, MonitoredPosition] = {}
        self.position_updates: List[Dict[str, Any]] = []
        logger.info("Position monitor initialized")
    
    def add_position(
        self,
        symbol: str,
        side: str,
        size: float,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
    ):
        """Add a position to monitor"""
        self.monitored_positions[symbol] = MonitoredPosition(
            symbol=symbol,
            side=side,
            size=size,
            entry_price=entry_price,
            entry_time=datetime.utcnow(),
            stop_loss=stop_loss,
            take_profit=take_profit,
        )
        
        logger.info(f"Now monitoring position: {symbol} {side}")
    
    def update_position(
        self,
        symbol: str,
        current_price: float,
        unrealized_pnl: float,
    ) -> Dict[str, Any]:
        """
        Update position with current market data
        
        Returns:
            Position update dict for DeepSeek
        """
        if symbol not in self.monitored_positions:
            return {}
        
        position = self.monitored_positions[symbol]
        position.current_price = current_price
        position.unrealized_pnl = unrealized_pnl
        position.last_update = datetime.utcnow()
        
        # Calculate metrics
        time_in_trade = (datetime.utcnow() - position.entry_time).total_seconds() / 60  # minutes
        price_change_pct = ((current_price - position.entry_price) / position.entry_price) * 100
        
        if position.side == "Sell":  # Short position
            price_change_pct = -price_change_pct
        
        # Distance to targets
        if position.side == "Buy":
            distance_to_stop_pct = ((position.stop_loss - current_price) / current_price) * 100
            distance_to_target_pct = ((position.take_profit - current_price) / current_price) * 100
        else:
            distance_to_stop_pct = ((current_price - position.stop_loss) / current_price) * 100
            distance_to_target_pct = ((current_price - position.take_profit) / current_price) * 100
        
        # Progress to target
        total_distance = abs(position.take_profit - position.entry_price)
        traveled = abs(current_price - position.entry_price)
        progress_to_target = (traveled / total_distance * 100) if total_distance > 0 else 0
        
        update = {
            "symbol": symbol,
            "side": position.side,
            "entry_price": position.entry_price,
            "current_price": current_price,
            "price_change_pct": round(price_change_pct, 2),
            "unrealized_pnl": round(unrealized_pnl, 2),
            "time_in_trade_mins": round(time_in_trade, 1),
            "stop_loss": position.stop_loss,
            "take_profit": position.take_profit,
            "distance_to_stop_pct": round(distance_to_stop_pct, 2),
            "distance_to_target_pct": round(distance_to_target_pct, 2),
            "progress_to_target_pct": round(progress_to_target, 1),
            "should_consider_closing": self._should_consider_closing(position, time_in_trade, progress_to_target),
        }
        
        self.position_updates.append(update)
        
        return update
    
    def _should_consider_closing(
        self,
        position: MonitoredPosition,
        time_in_trade: float,
        progress: float,
    ) -> str:
        """Determine if position should be reconsidered"""
        reasons = []
        
        # Been in trade too long with no progress
        if time_in_trade > 480 and progress < 20:  # 8 hours, <20% to target
            reasons.append("Long hold with minimal progress")
        
        # Close to break-even after long time
        pnl_pct = (position.unrealized_pnl / (position.entry_price * position.size)) * 100
        if time_in_trade > 240 and -5 < pnl_pct < 5:  # 4 hours, near break-even
            reasons.append("Stuck near break-even for 4+ hours")
        
        # Significant unrealized profit
        if progress > 75:
            reasons.append("Very close to target (75%+) - consider locking profit")
        
        return " | ".join(reasons) if reasons else "Position looks fine, let it run"
    
    def get_position_summary(self) -> Dict[str, Any]:
        """Get summary of all monitored positions"""
        if not self.monitored_positions:
            return {"total_positions": 0, "positions": []}
        
        summaries = []
        total_pnl = 0
        
        for symbol, pos in self.monitored_positions.items():
            time_in_trade = (datetime.utcnow() - pos.entry_time).total_seconds() / 60
            
            summaries.append({
                "symbol": symbol,
                "side": pos.side,
                "unrealized_pnl": round(pos.unrealized_pnl, 2),
                "price_change_pct": round(((pos.current_price - pos.entry_price) / pos.entry_price) * 100, 2),
                "time_in_trade_mins": round(time_in_trade, 1),
            })
            
            total_pnl += pos.unrealized_pnl
        
        return {
            "total_positions": len(self.monitored_positions),
            "positions": summaries,
            "total_unrealized_pnl": round(total_pnl, 2),
        }
    
    def remove_position(self, symbol: str):
        """Remove position from monitoring"""
        if symbol in self.monitored_positions:
            del self.monitored_positions[symbol]
            logger.info(f"Stopped monitoring {symbol}")
    
    def clear_all(self):
        """Clear all monitored positions"""
        self.monitored_positions.clear()
        self.position_updates.clear()
        logger.info("Cleared all monitored positions")

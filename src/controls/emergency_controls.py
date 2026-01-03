"""
Emergency Controls - Kill switch, pause, safety overrides
Shared state via bot_state.json for IPC between API and Bot
"""
import json
import os
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from loguru import logger
from enum import Enum
from pathlib import Path


class TradingMode(str, Enum):
    """Trading mode states"""
    ACTIVE = "active"
    PAUSED = "paused"
    EMERGENCY_STOP = "emergency_stop"
    COOLDOWN = "cooldown"


class EmergencyControls:
    """Emergency controls and safety mechanisms"""
    
    def __init__(self):
        """Initialize emergency controls"""
        self.mode = TradingMode.ACTIVE
        self.pause_reason: Optional[str] = None
        self.pause_until: Optional[datetime] = None
        self.consecutive_losses = 0
        self.cooldown_active = False
        self.cooldown_until: Optional[datetime] = None
        self.max_consecutive_losses = 4
        self.cooldown_duration_minutes = 60
        
        self.state_file = Path(__file__).parent.parent.parent / "bot_state.json"
        self._load_state()
        logger.info("Emergency controls initialized (Shared State)")

    def _load_state(self):
        """Load state from file"""
        if self.state_file.exists():
            try:
                with open(self.state_file, "r") as f:
                    data = json.load(f)
                    self.mode = TradingMode(data.get("mode", "active"))
                    self.pause_reason = data.get("pause_reason")
                    self.pause_until = datetime.fromisoformat(data["pause_until"]) if data.get("pause_until") else None
                    self.consecutive_losses = data.get("consecutive_losses", 0)
                    self.cooldown_active = data.get("cooldown_active", False)
                    self.cooldown_until = datetime.fromisoformat(data["cooldown_until"]) if data.get("cooldown_until") else None
            except Exception as e:
                logger.error(f"Failed to load bot state: {e}")
    
    def _save_state(self):
        """Save state to file"""
        try:
            data = {
                "mode": self.mode.value,
                "pause_reason": self.pause_reason,
                "pause_until": self.pause_until.isoformat() if self.pause_until else None,
                "consecutive_losses": self.consecutive_losses,
                "cooldown_active": self.cooldown_active,
                "cooldown_until": self.cooldown_until.isoformat() if self.cooldown_until else None,
                "last_updated": datetime.utcnow().isoformat()
            }
            with open(self.state_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save bot state: {e}")
    
    def emergency_stop(self, reason: str):
        """
        Trigger emergency stop - halts all trading
        
        Args:
            reason: Reason for emergency stop
        """
        self.mode = TradingMode.EMERGENCY_STOP
        self.pause_reason = reason
        self.pause_until = None  # Indefinite until manually resumed
        
        logger.critical(f"🚨 EMERGENCY STOP TRIGGERED: {reason}")
        logger.critical("Trading halted indefinitely. Manual intervention required.")
        self._save_state()
    
    def pause_trading(self, reason: str, duration_minutes: int = 60):
        """
        Pause trading temporarily
        
        Args:
            reason: Reason for pause
            duration_minutes: How long to pause
        """
        self.mode = TradingMode.PAUSED
        self.pause_reason = reason
        self.pause_until = datetime.utcnow() + timedelta(minutes=duration_minutes)
        
        logger.warning(f"⏸️  Trading paused: {reason}")
        logger.warning(f"Will resume at: {self.pause_until}")
        self._save_state()
    
    def resume_trading(self):
        """Resume trading after pause or emergency stop (manual override)"""
        # We now allow manual resume even from EMERGENCY_STOP via this method
        # which is called by the dashboard API.
        
        self.mode = TradingMode.ACTIVE
        self.pause_reason = None
        self.pause_until = None
        self.consecutive_losses = 0
        self.cooldown_active = False
        self.cooldown_until = None
        
        logger.info("✅ Trading resumed (Manual Override)")
        self._save_state()
        return True
    
    def check_status(self) -> Dict[str, Any]:
        """
        Check current status and auto-resume if applicable
        
        Returns:
            Status dict
        """
        # Always reload state first to catch updates from other processes
        self._load_state()

        # Check if pause should be lifted (AUTO-RESUME)
        if self.mode == TradingMode.PAUSED and self.pause_until:
            if datetime.utcnow() >= self.pause_until:
                # Use a specific internal method for auto-resume if we want to keep it separate,
                # but here we just update mode directly to avoid the "Override" log for auto-actions
                self.mode = TradingMode.ACTIVE
                self.pause_reason = None
                self.pause_until = None
                logger.info("⏰ Pause duration expired - Trading auto-resumed")
                self._save_state()
        
        # Check cooldown (AUTO-LIFT)
        if self.cooldown_active and self.cooldown_until:
            if datetime.utcnow() >= self.cooldown_until:
                self.cooldown_active = False
                self.cooldown_until = None
                self.consecutive_losses = 0
                logger.info("⏰ Cooldown period ended - Trading auto-resumed")
                self._save_state()
        
        return {
            "mode": self.mode.value,
            "trading_allowed": self.mode == TradingMode.ACTIVE and not self.cooldown_active,
            "pause_reason": self.pause_reason,
            "pause_until": self.pause_until.isoformat() if self.pause_until else None,
            "consecutive_losses": self.consecutive_losses,
            "cooldown_active": self.cooldown_active,
            "cooldown_until": self.cooldown_until.isoformat() if self.cooldown_until else None,
        }

    def is_trading_allowed(self) -> tuple[bool, Optional[str]]:
        """
        Check if trading is allowed - used by the main bot loop
        
        Returns:
            (allowed, reason) tuple
        """
        # This reloads state and checks auto-resume
        status = self.check_status()
        
        if not status["trading_allowed"]:
            if self.mode == TradingMode.EMERGENCY_STOP:
                return False, f"Emergency stop: {self.pause_reason}"
            elif self.mode == TradingMode.PAUSED:
                return False, f"Paused: {self.pause_reason} (until {self.pause_until})"
            elif self.cooldown_active:
                return False, f"Cooldown active: {self.consecutive_losses} consecutive losses (until {self.cooldown_until})"
        
        return True, None

    def record_trade_result(self, won: bool):
        """
        Record trade result and trigger cooldown if needed
        
        Args:
            won: True if trade was profitable
        """
        if won:
            self.consecutive_losses = 0
            logger.debug("Win recorded - consecutive losses reset")
        else:
            self.consecutive_losses += 1
            logger.debug(f"Loss recorded - consecutive losses: {self.consecutive_losses}")
            
            # Trigger cooldown after max consecutive losses
            if self.consecutive_losses >= self.max_consecutive_losses:
                self.trigger_cooldown()
        self._save_state()
    
    def trigger_cooldown(self):
        """Trigger cooldown period after consecutive losses"""
        self.cooldown_active = True
        self.cooldown_until = datetime.utcnow() + timedelta(minutes=self.cooldown_duration_minutes)
        
        logger.warning(f"🛑 COOLDOWN TRIGGERED: {self.consecutive_losses} consecutive losses")
        logger.warning(f"Trading paused until: {self.cooldown_until}")
        self._save_state()
    
    def manual_kill_switch(self):
        """Manual kill switch - immediate stop"""
        self.emergency_stop("Manual kill switch activated")
        logger.critical("🔴 MANUAL KILL SWITCH ACTIVATED")

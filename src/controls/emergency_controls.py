"""
Emergency Controls - Kill switch, pause, safety overrides
Shared state via bot_state.json for IPC between API and Bot
"""

import json
import time
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

from loguru import logger


class TradingMode(str, Enum):
    """Trading mode states."""

    ACTIVE = "active"
    PAUSED = "paused"
    EMERGENCY_STOP = "emergency_stop"
    COOLDOWN = "cooldown"


class EmergencyControls:
    """Emergency controls and safety mechanisms."""

    def __init__(self):
        """Initialize emergency controls."""
        self.mode = TradingMode.ACTIVE
        self.pause_reason: Optional[str] = None
        self.pause_until: Optional[datetime] = None
        self.consecutive_losses = 0
        self.cooldown_active = False
        self.cooldown_until: Optional[datetime] = None
        self.max_consecutive_losses = 4
        self.cooldown_duration_minutes = 60
        self.flat_trade_tolerance_pct = 0.01

        self.state_file = Path(__file__).parent.parent.parent / "bot_state.json"
        self._load_state()
        logger.info("Emergency controls initialized (Shared State)")

    @staticmethod
    def _utc_now() -> datetime:
        """Get current timezone-aware UTC timestamp."""
        return datetime.now(timezone.utc)

    @staticmethod
    def _parse_utc_datetime(value: Optional[str]) -> Optional[datetime]:
        """Parse persisted datetime into timezone-aware UTC."""
        if not value:
            return None

        parsed = datetime.fromisoformat(value)
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    @staticmethod
    def _serialize_utc_datetime(value: Optional[datetime]) -> Optional[str]:
        """Serialize datetime consistently in UTC."""
        if value is None:
            return None
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).isoformat()

    def _load_state(self):
        """Load state from file."""
        if self.state_file.exists():
            attempts = 3
            for attempt in range(1, attempts + 1):
                try:
                    with open(self.state_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    self.mode = TradingMode(data.get("mode", "active"))
                    self.pause_reason = data.get("pause_reason")
                    self.pause_until = self._parse_utc_datetime(data.get("pause_until"))
                    self.consecutive_losses = data.get("consecutive_losses", 0)
                    self.cooldown_active = data.get("cooldown_active", False)
                    self.cooldown_until = self._parse_utc_datetime(data.get("cooldown_until"))
                    return
                except json.JSONDecodeError as e:
                    if attempt == attempts:
                        logger.error(f"Failed to load bot state after {attempts} attempts: {e}")
                    else:
                        # State writes are tiny and can be read mid-write; quick retry avoids stale state lock-in.
                        time.sleep(0.05)
                except Exception as e:
                    logger.error(f"Failed to load bot state: {e}")
                    return

    def _save_state(self):
        """Save state to file."""
        try:
            data = {
                "mode": self.mode.value,
                "pause_reason": self.pause_reason,
                "pause_until": self._serialize_utc_datetime(self.pause_until),
                "consecutive_losses": self.consecutive_losses,
                "cooldown_active": self.cooldown_active,
                "cooldown_until": self._serialize_utc_datetime(self.cooldown_until),
                "last_updated": self._utc_now().isoformat(),
            }
            with open(self.state_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save bot state: {e}")

    def emergency_stop(self, reason: str):
        """
        Trigger emergency stop - halts all trading.

        Args:
            reason: Reason for emergency stop
        """
        self.mode = TradingMode.EMERGENCY_STOP
        self.pause_reason = reason
        self.pause_until = None

        logger.critical(f"EMERGENCY STOP TRIGGERED: {reason}")
        logger.critical("Trading halted indefinitely. Manual intervention required.")
        self._save_state()

    def pause_trading(self, reason: str, duration_minutes: int = 60):
        """
        Pause trading temporarily.

        Args:
            reason: Reason for pause
            duration_minutes: How long to pause
        """
        self.mode = TradingMode.PAUSED
        self.pause_reason = reason
        self.pause_until = self._utc_now() + timedelta(minutes=duration_minutes)

        logger.warning(f"Trading paused: {reason}")
        logger.warning(f"Will resume at: {self.pause_until}")
        self._save_state()

    def resume_trading(self):
        """Resume trading after pause or emergency stop (manual override)."""
        self.mode = TradingMode.ACTIVE
        self.pause_reason = None
        self.pause_until = None
        self.consecutive_losses = 0
        self.cooldown_active = False
        self.cooldown_until = None

        logger.info("Trading resumed (Manual Override)")
        self._save_state()
        return True

    def check_status(self) -> Dict[str, Any]:
        """
        Check current status and auto-resume if applicable.

        Returns:
            Status dict
        """
        self._load_state()
        now = self._utc_now()

        if self.mode == TradingMode.PAUSED and self.pause_until and now >= self.pause_until:
            self.mode = TradingMode.ACTIVE
            self.pause_reason = None
            self.pause_until = None
            logger.info("Pause duration expired - trading auto-resumed")
            self._save_state()

        if self.cooldown_active and not self.cooldown_until:
            self.cooldown_active = False
            self.consecutive_losses = 0
            logger.warning("Cooldown active without cooldown_until; auto-clearing stale cooldown state")
            self._save_state()
        elif self.cooldown_active and self.cooldown_until and now >= self.cooldown_until:
            self.cooldown_active = False
            self.cooldown_until = None
            self.consecutive_losses = 0
            logger.info("Cooldown period ended - trading auto-resumed")
            self._save_state()

        return {
            "mode": self.mode.value,
            "trading_allowed": self.mode == TradingMode.ACTIVE and not self.cooldown_active,
            "pause_reason": self.pause_reason,
            "pause_until": self._serialize_utc_datetime(self.pause_until),
            "consecutive_losses": self.consecutive_losses,
            "cooldown_active": self.cooldown_active,
            "cooldown_until": self._serialize_utc_datetime(self.cooldown_until),
        }

    def is_trading_allowed(self) -> tuple[bool, Optional[str]]:
        """
        Check if trading is allowed - used by the main bot loop.

        Returns:
            (allowed, reason) tuple
        """
        status = self.check_status()

        if status["trading_allowed"]:
            return True, None

        mode = status.get("mode")
        if mode == TradingMode.EMERGENCY_STOP.value:
            return False, f"Emergency stop: {self.pause_reason}"
        if mode == TradingMode.PAUSED.value:
            return False, f"Paused: {self.pause_reason} (until {status.get('pause_until')})"
        if status.get("cooldown_active"):
            return (
                False,
                "Cooldown active: "
                f"{status.get('consecutive_losses', 0)} consecutive losses "
                f"(until {status.get('cooldown_until')})",
            )

        return False, "Trading disabled by controls state"

    def record_trade_result(self, won: bool, pnl_pct: Optional[float] = None):
        """
        Record trade result and trigger cooldown if needed.

        Args:
            won: True if trade was profitable
            pnl_pct: Optional realized percent return used to avoid counting flat closes as losses
        """
        if pnl_pct is not None and abs(float(pnl_pct)) <= self.flat_trade_tolerance_pct:
            logger.debug(
                f"Flat trade recorded ({float(pnl_pct):+.4f}%) - consecutive losses unchanged ({self.consecutive_losses})"
            )
            self._save_state()
            return

        if won:
            self.consecutive_losses = 0
            logger.debug("Win recorded - consecutive losses reset")
        else:
            self.consecutive_losses += 1
            logger.debug(f"Loss recorded - consecutive losses: {self.consecutive_losses}")

            if self.consecutive_losses >= self.max_consecutive_losses:
                self.trigger_cooldown()

        self._save_state()

    def trigger_cooldown(self):
        """Trigger cooldown period after consecutive losses."""
        self.cooldown_active = True
        self.cooldown_until = self._utc_now() + timedelta(minutes=self.cooldown_duration_minutes)

        logger.warning(f"COOLDOWN TRIGGERED: {self.consecutive_losses} consecutive losses")
        logger.warning(f"Trading paused until (UTC): {self.cooldown_until.isoformat()}")
        self._save_state()

    def manual_kill_switch(self):
        """Manual kill switch - immediate stop."""
        self.emergency_stop("Manual kill switch activated")
        logger.critical("MANUAL KILL SWITCH ACTIVATED")

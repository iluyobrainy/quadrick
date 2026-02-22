"""
Dashboard bridge for pushing bot state to the local FastAPI dashboard.
"""
from __future__ import annotations

import json
import os
import threading
from datetime import date, datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import requests
from loguru import logger


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


class DashboardBridge:
    """Non-blocking bridge that mirrors bot state to `/internal/*` endpoints."""

    def __init__(self, enabled: Optional[bool] = None, port: Optional[str] = None):
        self.enabled = enabled if enabled is not None else _env_bool("DASHBOARD_BRIDGE_ENABLED", True)
        self.port = str(
            port
            or os.getenv("DASHBOARD_INTERNAL_PORT")
            or os.getenv("PORT")
            or "8001"
        )
        self.base_url = f"http://localhost:{self.port}/internal"
        self._connection_failed = False

        if self.enabled:
            self._check_connection()

    def _check_connection(self) -> bool:
        if not self.enabled:
            return False

        try:
            response = requests.get(f"http://localhost:{self.port}/", timeout=1.0)
            ok = response.status_code == 200
            if ok and self._connection_failed:
                logger.info("Dashboard API connectivity restored")
                self._connection_failed = False
            return ok
        except Exception:
            pass

        if not self._connection_failed:
            logger.warning(
                f"Dashboard API not reachable on port {self.port}; continuing in headless mode"
            )
            self._connection_failed = True
        return False

    def _serialize(self, obj: Any) -> Any:
        """Robust JSON serialization for dataclasses, enums, numpy and nested objects."""
        try:
            import numpy as np  # Optional at runtime
        except Exception:  # pragma: no cover - fallback when numpy unavailable
            np = None

        if isinstance(obj, Enum):
            return obj.value

        if isinstance(obj, (datetime, date)):
            return obj.isoformat()

        if isinstance(obj, dict):
            return {str(k): self._serialize(v) for k, v in obj.items()}

        if isinstance(obj, (list, tuple, set)):
            return [self._serialize(v) for v in obj]

        if np is not None and isinstance(obj, np.floating):
            if np.isnan(obj) or np.isinf(obj):
                return 0.0
            return float(obj)

        if np is not None and isinstance(obj, np.integer):
            return int(obj)

        if hasattr(obj, "__dict__") and not isinstance(obj, (type, Exception)):
            return self._serialize(obj.__dict__)

        return obj

    def _send_async(self, endpoint: str, data: Any) -> None:
        if not self.enabled:
            return

        payload = self._serialize(data)

        def _send() -> None:
            try:
                response = requests.post(
                    f"{self.base_url}/{endpoint}",
                    json=payload,
                    timeout=5.0,
                )
                if response.status_code == 200:
                    if self._connection_failed:
                        logger.info("Dashboard bridge connectivity restored")
                        self._connection_failed = False
                elif endpoint != "log":
                    logger.debug(
                        f"Dashboard bridge endpoint {endpoint} returned HTTP {response.status_code}"
                    )
            except Exception as exc:
                self._connection_failed = True
                if endpoint != "log":
                    logger.debug(f"Dashboard bridge send failed for {endpoint}: {exc}")

        threading.Thread(target=_send, daemon=True).start()

    def clear_logs(self) -> None:
        if not self.enabled:
            return
        try:
            requests.post(f"{self.base_url}/logs/clear", timeout=2.0)
        except Exception:
            pass

    def update_status(self, mode: str, trading_allowed: bool) -> None:
        self._send_async(
            "status",
            {
                "mode": mode,
                "trading_allowed": trading_allowed,
            },
        )

    def update_balance(self, balance_info: Dict[str, Any]) -> None:
        self._send_async("balance", balance_info)

    def update_positions(self, positions: List[Dict[str, Any]]) -> None:
        self._send_async("positions", positions)

    def update_market_context(self, context: Dict[str, Any]) -> None:
        self._send_async("market-context", context)

    def update_raw_market_data(self, raw_data: Dict[str, Any]) -> None:
        self._send_async("market-raw", raw_data)

    def update_ai_insights(self, insight: Dict[str, Any]) -> None:
        self._send_async("ai-insights", insight)

    def send_decision(self, decision: Dict[str, Any]) -> None:
        clean_decision = json.loads(json.dumps(decision, default=str))
        self._send_async("decision", clean_decision)

    def update_quant_monitor(self, monitor: Dict[str, Any]) -> None:
        self._send_async("quant-monitor", monitor)

    def update_quant_alert(self, alert: Dict[str, Any]) -> None:
        self._send_async("quant-alert", alert)

    def send_log(self, record: Dict[str, Any]) -> None:
        self._send_async(
            "log",
            {
                "timestamp": record["time"].isoformat(),
                "level": record["level"].name,
                "message": record["message"],
            },
        )


bridge = DashboardBridge()


def dashboard_sink(message: Any) -> None:
    """Loguru sink that forwards records to dashboard log stream."""
    bridge.send_log(message.record)

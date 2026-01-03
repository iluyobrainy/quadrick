"""
Dashboard Bridge Module
Handles communication between the main trading bot and the valid dashboard API server.
"""
import requests
import json
import threading
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from loguru import logger

DASHBOARD_API_URL = "http://localhost:8001/internal"

class DashboardBridge:
    """Sends updates to the local dashboard API"""
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.base_url = DASHBOARD_API_URL
        self._connection_failed = False
        self._check_connection()
        
    def _check_connection(self):
        """Check if dashboard API is reachable (fail silently after first log)"""
        if not self.enabled:
            return False # Return False if disabled, as no connection is made

        if self._connection_failed:
             return False

        try:
            # Check the root URL for general connectivity, not the internal API endpoint
            # The dashboard server runs on port 8000, the internal API on 8001
            response = requests.get("http://localhost:8000/", timeout=0.5)
            if response.status_code == 200:
                if self._connection_failed: # If it failed before but now works
                    logger.info("✓ Reconnected to Dashboard API")
                    self._connection_failed = False
                else: # First successful connection
                    logger.info("✓ Connected to Dashboard API")
                return True
        except requests.exceptions.RequestException:
            pass
            
        # Only log once to avoid spamming
        if not self._connection_failed:
             logger.warning("⚠️ Dashboard API not reachable (running in headless mode)")
             self._connection_failed = True
        return False

    def _send_async(self, endpoint: str, data: Dict[str, Any]):
        """Send request in a separate thread to avoid blocking main loop"""
        if not self.enabled:
            return
            
        # Do not attempt to send if connection is known to be down
        if self._connection_failed:
            return

        def _send():
            try:
                # Verbose logging for user confirmation
                if endpoint == "balance":
                    val = data.get('total', 0)
                    print(f"DEBUG: \033[96mPush Balance -> Dashboard\033[0m ... (${val:.2f})")
                elif endpoint == "status":
                    val = data.get('mode', 'unknown')
                    print(f"DEBUG: \033[96mPush Status -> Dashboard\033[0m ... ({val})")
                elif endpoint == "positions":
                    count = len(data)
                    print(f"DEBUG: \033[96mPush Positions -> Dashboard\033[0m ... ({count} active)")
                elif endpoint == "market-context":
                    count = len(data)
                    print(f"DEBUG: \033[96mPush Market Context -> Dashboard\033[0m ... ({count} symbols)")
                elif endpoint == "market-raw":
                    count = len(data.get('tickers', {}))
                    print(f"DEBUG: \033[96mPush Raw Market Data -> Dashboard\033[0m ... ({count} symbols)")
                elif endpoint == "ai-insights":
                    agent = data.get('agent', 'Unknown')
                    symbol = data.get('symbol', 'All')
                    print(f"DEBUG: \033[95mPush AI Insights -> Dashboard\033[0m ... ({agent} for {symbol})")
                elif endpoint == "decision":
                    symbol = data.get('symbol', 'unknown')
                    decision_type = data.get('type', 'unknown')
                    print(f"DEBUG: \033[96mPush Decision -> Dashboard\033[0m ... ({decision_type} {symbol})")

                response = requests.post(f"{DASHBOARD_API_URL}/{endpoint}", json=data, timeout=5.0)
                if response.status_code != 200:
                    print(f"DEBUG: \033[91mDashboard Error ({endpoint})\033[0m: HTTP {response.status_code}")
            except Exception as e:
                # Print connection errors to help user debug
                print(f"DEBUG: \033[91mDashboard Connection Failed ({endpoint})\033[0m: {str(e)[:100]}")
                
        threading.Thread(target=_send, daemon=True).start()

    def clear_logs(self):
        """Clear the dashboard logs on startup"""
        if not self.enabled:
            return
        try:
            requests.post(f"{DASHBOARD_API_URL}/logs/clear", timeout=2.0)
            logger.info("🧹 Dashboard log buffer cleared")
        except:
            pass

    def update_raw_market_data(self, raw_data: Dict[str, Any]):
        """Update raw tickers and funding rates"""
        self._send_async("market-raw", raw_data)

    def update_market_context(self, context: Dict[str, Any]):
        """Update rich technical market data"""
        if not self.enabled:
            return

        import numpy as np
        from dataclasses import is_dataclass, asdict

    def _serialize(self, obj):
        """Robust JSON serialization helper for complex bot data"""
        import numpy as np
        from datetime import datetime, date
        from dataclasses import is_dataclass, asdict
        from enum import Enum
        
        # Handle Enum
        if isinstance(obj, Enum):
            return obj.value
            
        # Handle EnumType or other type objects
        if isinstance(obj, type) or "EnumType" in str(type(obj)):
            return str(obj)

        # Handle datetime/date
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
            
        # Handle dataclasses
        if is_dataclass(obj):
            d = asdict(obj)
            # Remove large/non-serializable fields immediately
            if "candles" in d: del d["candles"]
            return {k: self._serialize(v) for k, v in d.items()}
            
        # Handle dicts and MappingProxy (Bybit internals)
        if hasattr(obj, "items") or "mappingproxy" in str(type(obj)):
            return {str(k): self._serialize(v) for k, v in dict(obj).items()}
            
        # Handle lists/sequences
        if isinstance(obj, (list, tuple, set)) or "deque" in str(type(obj)):
            return [self._serialize(i) for i in obj]
            
        # Handle floating point (NaN/Inf)
        if isinstance(obj, (float, np.floating)):
            if np.isnan(obj) or np.isinf(obj):
                return 0.0
            return float(obj)
            
        # Handle numpy integers
        if isinstance(obj, (np.integer)):
            return int(obj)
            
        # Handle objects with __dict__ (excluding some types)
        if hasattr(obj, "__dict__") and not isinstance(obj, (type, Exception)):
            return self._serialize(obj.__dict__)
            
        # Fallback to string for unknown complex types (like DataFrames or specific classes)
        if "pd.DataFrame" in str(type(obj)) or "pd.Series" in str(type(obj)):
            return "DataFrame/Series omitted"
            
        return obj

    def update_raw_market_data(self, raw_data: Dict[str, Any]):
        """Update raw tickers and funding rates"""
        clean_raw = self._serialize(raw_data)
        self._send_async("market-raw", clean_raw)

    def update_ai_insights(self, insight: Dict[str, Any]):
        """Push raw LLM prompt/response for transparency"""
        clean_insight = self._serialize(insight)
        self._send_async("ai-insights", clean_insight)

    def update_market_context(self, context: Dict[str, Any]):
        """Update rich technical market data"""
        if not self.enabled:
            return

        try:
            # Deep clean the context using the robust serializer
            clean_context = {}
            for symbol, analysis in context.items():
                # Use the new timeframe_analysis key (matches main.py and frontend)
                tf_data = analysis.get("timeframe_analysis", analysis.get("timeframes", {}))
                
                clean_context[symbol] = {
                    "timeframe_analysis": {tf: self._serialize(val) for tf, val in tf_data.items()},
                    "sentiment": self._serialize(analysis.get("sentiment", {}))
                }
            
            self._send_async("market-context", clean_context)
        except Exception as e:
            print(f"DEBUG: \033[91mFailed to serialize market context\033[0m: {e}")

    def update_status(self, mode: str, trading_allowed: bool):
        """Update bot status"""
        self._send_async("status", {
            "mode": mode,
            "trading_allowed": trading_allowed
        })

    def update_balance(self, balance_info: Dict[str, Any]):
        """Update account balance"""
        self._send_async("balance", balance_info)

    def update_positions(self, positions: List[Dict[str, Any]]):
        """Update active positions"""
        self._send_async("positions", positions)

    def send_decision(self, decision: Dict[str, Any]):
        """Send trading decision"""
        # Ensure decision is JSON serializable
        clean_decision = json.loads(json.dumps(decision, default=str))
        self._send_async("decision", clean_decision)

    def send_log(self, record: Dict[str, Any]):
        """Send log entry"""
        self._send_async("log", {
            "timestamp": record["time"].strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "level": record["level"].name,
            "message": record["message"]
        })

# Global instance for log sink
bridge = DashboardBridge()

def dashboard_sink(message):
    """Loguru sink for dashboard"""
    bridge.send_log(message.record)

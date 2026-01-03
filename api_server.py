"""
Quadrick Dashboard API Server
FastAPI backend for the Next.js dashboard - INTEGRATED with real Quadrick bot
"""
import asyncio
import json
import os
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from pathlib import Path
import sys

# Ensure we're using the correct path for Quadrick modules
QUADRICK_ROOT = Path(__file__).parent
sys.path.insert(0, str(QUADRICK_ROOT))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from loguru import logger
import uvicorn
from dotenv import load_dotenv

# Load environment variables
load_dotenv(QUADRICK_ROOT / ".env")

# Import Quadrick modules - these should work when running from quadrick directory
bybit_client = None
settings_instance = None
emergency_controls = None

try:
    from config.settings import Settings
    from src.exchange.bybit_client import BybitClient
    from src.controls.emergency_controls import EmergencyControls, TradingMode
    from src.analytics.performance_tracker import PerformanceTracker
    
    settings_instance = Settings()
    bybit_client = BybitClient(
        api_key=settings_instance.bybit.api_key,
        api_secret=settings_instance.bybit.api_secret,
        testnet=settings_instance.bybit.testnet
    )
    emergency_controls = EmergencyControls()
    
    logger.info("✅ Quadrick modules loaded successfully")
    logger.info(f"   Bybit: {'Testnet' if settings_instance.bybit.testnet else 'Mainnet'}")
    
except ImportError as e:
    logger.error(f"❌ Could not import Quadrick modules: {e}")
    logger.error("   Make sure you're running from the quadrick directory with the virtual environment activated")
    logger.error("   Run: pip install -r requirements.txt")

app = FastAPI(
    title="Quadrick Dashboard API",
    description="Backend API for Quadrick AI Trading Dashboard",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared state between dashboard and bot
bot_state = {
    "status": {
        "mode": "stopped",
        "trading_allowed": False,
        "pause_reason": None,
        "consecutive_losses": 0,
        "cooldown_active": False,
    },
    "decisions": [],
    "daily_starting_balance": None,
    "last_balance_update": {
        "total": 0.0,
        "available": 0.0,
        "unrealized_pnl": 0.0,
        "daily_pnl": 0.0,
        "daily_pnl_pct": 0.0,
        "daily_pnl_pct": 0.0,
    },
    "positions": [],
    "market_context": {},
    "raw_market_data": {},
    "ai_insights": []
}

# Logs buffer for streaming

# Logs buffer for streaming
log_buffer: List[Dict[str, Any]] = []


class DashboardLogSink:
    """Custom loguru sink to capture logs for dashboard"""
    def __call__(self, message):
        record = message.record
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record["level"].name,
            "message": record["message"],
        }
        log_buffer.append(log_entry)
        # Keep only last 500 logs
        if len(log_buffer) > 500:
            log_buffer.pop(0)


# Add custom sink to loguru
logger.add(DashboardLogSink(), format="{message}", level="INFO")


# --- Pydantic Models ---

class StatusResponse(BaseModel):
    mode: str
    trading_allowed: bool
    pause_reason: Optional[str] = None
    consecutive_losses: int
    cooldown_active: bool
    connected: bool


class BalanceResponse(BaseModel):
    total: float
    available: float
    unrealized_pnl: float
    daily_pnl: float
    daily_pnl_pct: float


class PositionResponse(BaseModel):
    symbol: str
    side: str
    size: float
    entry_price: float
    current_price: float
    pnl: float
    pnl_pct: float
    leverage: int
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None


class DecisionResponse(BaseModel):
    id: str
    timestamp: str
    symbol: str
    type: str
    confidence: float
    strategy: Optional[str] = "N/A"
    reasoning: Any
    htf_aligned: Optional[bool] = None


class ControlAction(BaseModel):
    action: str  # "start", "stop", "pause"


class SettingsUpdate(BaseModel):
    bybit_api_key: Optional[str] = None
    bybit_api_secret: Optional[str] = None
    bybit_testnet: Optional[bool] = None
    deepseek_api_key: Optional[str] = None
    supabase_url: Optional[str] = None
    supabase_key: Optional[str] = None
    telegram_bot_token: Optional[str] = None
    telegram_chat_id: Optional[str] = None
    min_risk_pct: Optional[float] = None
    max_risk_pct: Optional[float] = None
    max_leverage: Optional[int] = None
    max_concurrent_positions: Optional[int] = None
    decision_interval_seconds: Optional[int] = None


class LogEntry(BaseModel):
    timestamp: str
    level: str
    message: str


# --- API Endpoints ---

@app.get("/")
async def root():
    return {
        "message": "Quadrick Dashboard API",
        "version": "1.0.0",
        "connected": bybit_client is not None
    }


@app.get("/api/status", response_model=StatusResponse)
async def get_status():
    """Get current bot status"""
    if emergency_controls:
        status = emergency_controls.check_status()
        return {
            "mode": status["mode"],
            "trading_allowed": status["trading_allowed"],
            "pause_reason": status.get("pause_reason"),
            "consecutive_losses": status["consecutive_losses"],
            "cooldown_active": status["cooldown_active"],
            "connected": bybit_client is not None
        }
    return {**bot_state["status"], "connected": False}


@app.get("/api/balance", response_model=BalanceResponse)
async def get_balance():
    """Get current account balance (from bot state preferred, with fallback)"""
    # Prefer data pushed from main bot
    if bot_state["last_balance_update"]["total"] > 0:
        return bot_state["last_balance_update"]
        
    # Fallback to direct API call if bot hasn't reported yet
    if not bybit_client:
        return {
            "total": 0.0,
            "available": 0.0,
            "unrealized_pnl": 0.0,
            "daily_pnl": 0.0,
            "daily_pnl_pct": 0.0,
        }
    
    try:
        balance = bybit_client.get_account_balance()
        if balance:
            total = balance.total_equity
            
            # Calculate daily PnL
            if bot_state["daily_starting_balance"] is None:
                bot_state["daily_starting_balance"] = total
            
            daily_pnl = total - bot_state["daily_starting_balance"]
            daily_pnl_pct = (daily_pnl / bot_state["daily_starting_balance"] * 100) if bot_state["daily_starting_balance"] > 0 else 0
            
            return {
                "total": total,
                "available": balance.available_balance,
                "unrealized_pnl": balance.unrealized_pnl,
                "daily_pnl": daily_pnl,
                "daily_pnl_pct": daily_pnl_pct,
            }
    except Exception as e:
        logger.error(f"Failed to fetch balance locally: {e}")
    
    return bot_state["last_balance_update"]


@app.get("/api/positions", response_model=List[PositionResponse])
async def get_positions():
    """Get active positions (cached from bot)"""
    return bot_state["positions"]


@app.get("/api/market-context")
async def get_market_context():
    """Get rich technical market data for trader review"""
    from fastapi import Response
    return Response(
        content=json.dumps(bot_state["market_context"]),
        media_type="application/json",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"}
    )


@app.get("/api/market-raw")
async def get_market_raw():
    """Get raw market data (tickers, funding) for review"""
    from fastapi import Response
    return Response(
        content=json.dumps(bot_state["raw_market_data"]),
        media_type="application/json",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"}
    )


@app.get("/api/ai-insights")
async def get_ai_insights():
    """Get raw LLM prompts and responses for professional review"""
    from fastapi import Response
    return Response(
        content=json.dumps(bot_state["ai_insights"]),
        media_type="application/json",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"}
    )


@app.get("/api/decisions", response_model=List[DecisionResponse])
async def get_decisions(limit: int = 20):
    """Get recent trading decisions"""
    from fastapi import Response
    return Response(
        content=json.dumps(bot_state["decisions"][-limit:]),
        media_type="application/json",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"}
    )


@app.get("/api/logs")
async def get_logs(limit: int = 100):
    """Get recent logs"""
    from fastapi import Response
    return Response(
        content=json.dumps(log_buffer[-limit:]),
        media_type="application/json",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"}
    )


@app.get("/api/logs/stream")
async def stream_logs():
    """Stream logs via Server-Sent Events"""
    async def event_generator():
        last_index = len(log_buffer)
        while True:
            if len(log_buffer) > last_index:
                for log in log_buffer[last_index:]:
                    yield f"data: {json.dumps(log)}\n\n"
                last_index = len(log_buffer)
            await asyncio.sleep(0.5)
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"
    )


@app.post("/api/control")
async def control_bot(action: ControlAction):
    """Control the trading bot via IPC"""
    if not emergency_controls:
        raise HTTPException(status_code=503, detail="Emergency controls not initialized")
    
    try:
        if action.action == "start":
            emergency_controls.resume_trading()
            logger.info("▶ Received START command from dashboard")
            return {"success": True, "message": "Bot mode set to ACTIVE"}
            
        elif action.action == "stop":
            emergency_controls.emergency_stop("Manual stop from dashboard")
            logger.info("⏹ Received STOP command from dashboard")
            return {"success": True, "message": "Bot stopped via emergency switch"}
            
        elif action.action == "pause":
            emergency_controls.pause_trading("Manual pause from dashboard")
            logger.info("⏸ Received PAUSE command from dashboard")
            return {"success": True, "message": "Bot paused temporarily"}
            
        else:
            raise HTTPException(status_code=400, detail="Invalid action")
            
    except Exception as e:
        logger.error(f"Control action failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/settings")
async def get_settings():
    """Get current settings (masked secrets)"""
    if not settings_instance:
        return {}
    
    try:
        return {
            "bybit_api_key": "****" + settings_instance.bybit.api_key[-4:] if settings_instance.bybit.api_key else "",
            "bybit_testnet": settings_instance.bybit.testnet,
            "deepseek_api_key": "****" + settings_instance.llm.deepseek_api_key[-4:] if settings_instance.llm.deepseek_api_key else "",
            "supabase_url": settings_instance.database.supabase_url if hasattr(settings_instance, 'database') else "",
            "telegram_chat_id": settings_instance.notifications.chat_id if hasattr(settings_instance, 'notifications') else "",
            "min_risk_pct": settings_instance.trading.min_risk_pct,
            "max_risk_pct": settings_instance.trading.max_risk_pct,
            "max_leverage": settings_instance.trading.max_leverage,
            "max_concurrent_positions": settings_instance.trading.max_concurrent_positions,
            "decision_interval_seconds": settings_instance.trading.decision_interval_seconds,
        }
    except Exception as e:
        logger.error(f"Could not load settings: {e}")
        return {}


@app.post("/api/settings")
async def update_settings(settings_update: SettingsUpdate):
    """Update settings (writes to .env file)"""
    env_path = QUADRICK_ROOT / ".env"
    
    if not env_path.exists():
        raise HTTPException(status_code=404, detail=".env file not found")
    
    try:
        # Read existing .env
        with open(env_path, "r") as f:
            lines = f.readlines()
        
        # Update values
        updates = settings_update.dict(exclude_none=True)
        env_mapping = {
            "bybit_api_key": "BYBIT_API_KEY",
            "bybit_api_secret": "BYBIT_API_SECRET",
            "bybit_testnet": "BYBIT_TESTNET",
            "deepseek_api_key": "DEEPSEEK_API_KEY",
            "supabase_url": "SUPABASE_URL",
            "supabase_key": "SUPABASE_KEY",
            "telegram_bot_token": "TELEGRAM_BOT_TOKEN",
            "telegram_chat_id": "TELEGRAM_CHAT_ID",
            "min_risk_pct": "MIN_RISK_PCT",
            "max_risk_pct": "MAX_RISK_PCT",
            "max_leverage": "MAX_LEVERAGE",
            "max_concurrent_positions": "MAX_CONCURRENT_POSITIONS",
            "decision_interval_seconds": "DECISION_INTERVAL_SECONDS",
        }
        
        for key, value in updates.items():
            env_key = env_mapping.get(key)
            if env_key:
                # Find and update the line
                found = False
                for i, line in enumerate(lines):
                    if line.startswith(f"{env_key}="):
                        lines[i] = f"{env_key}={value}\n"
                        found = True
                        break
                if not found:
                    lines.append(f"{env_key}={value}\n")
        
        # Write back
        with open(env_path, "w") as f:
            f.writelines(lines)
        
        logger.info("⚙️ Settings updated via dashboard")
        return {"success": True, "message": "Settings saved. Restart bot to apply changes."}
    
    except Exception as e:
        logger.error(f"Failed to update settings: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/performance")
async def get_performance():
    """Get performance metrics"""
    try:
        tracker = PerformanceTracker()
        perf = tracker.get_recent_performance()
        return perf
    except:
        return {
            "total_trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "total_pnl": 0.0,
            "best_strategy": None,
        }


# --- Internal Webhooks for Main Bot ---

@app.post("/internal/status")
async def update_internal_status(status: Dict[str, Any]):
    """Receive status update from main bot"""
    if "mode" in status:
        bot_state["status"]["mode"] = status["mode"]
    if "trading_allowed" in status:
        bot_state["status"]["trading_allowed"] = status["trading_allowed"]
    return {"success": True}


@app.post("/internal/positions")
async def update_internal_positions(positions_data: List[Dict[str, Any]]):
    """Receive positions update from main bot"""
    bot_state["positions"] = positions_data
    return {"success": True}


@app.post("/internal/market-context")
async def update_internal_market_context(context_data: Dict[str, Any]):
    """Receive rich technical data from main bot"""
    bot_state["market_context"] = context_data
    return {"success": True}


@app.post("/internal/market-raw")
async def update_internal_market_raw(raw_data: Dict[str, Any]):
    """Receive raw market data from main bot"""
    bot_state["raw_market_data"] = raw_data
    return {"success": True}


@app.post("/internal/ai-insights")
async def update_internal_ai_insights(insight_data: Dict[str, Any]):
    """Receive raw LLM interactions from main bot"""
    # insight_data should be: {"symbol": "BTCUSDT", "agent": "Analyst", "prompt": "...", "response": "..."}
    bot_state["ai_insights"].insert(0, insight_data)
    # Keep last 20 insights
    bot_state["ai_insights"] = bot_state["ai_insights"][:20]
    return {"success": True}


@app.post("/internal/balance")
async def update_internal_balance(balance_data: Dict[str, Any]):
    """Receive balance update from main bot"""
    bot_state["last_balance_update"] = balance_data
    return {"success": True}


@app.post("/internal/decision")
async def update_internal_decision(decision: Dict[str, Any]):
    """Receive decision from main bot"""
    add_decision(decision)
    return {"success": True}


@app.post("/internal/log")
async def update_internal_log(log: LogEntry):
    """Receive log from main bot"""
    log_entry = {
        "timestamp": log.timestamp,
        "level": log.level,
        "message": log.message,
    }
    log_buffer.append(log_entry)
    if len(log_buffer) > 500:
        log_buffer.pop(0)
    
    return {"success": True}


@app.post("/internal/logs/clear")
async def clear_internal_logs():
    """Clear the log buffer (called on bot restart)"""
    log_buffer.clear()
    logger.info("🧹 Dashboard log buffer cleared")
    return {"success": True}


# --- Helper functions for bot integration ---

def add_decision(decision: Dict[str, Any]):
    """Called by main bot to log a decision"""
    decision["id"] = str(len(bot_state["decisions"]) + 1)
    decision["timestamp"] = datetime.now(timezone.utc).isoformat()
    bot_state["decisions"].append(decision)
    # Keep only last 100 decisions
    if len(bot_state["decisions"]) > 100:
        bot_state["decisions"].pop(0)


def get_emergency_controls():
    """Get the emergency controls instance for bot integration"""
    return emergency_controls


def is_dashboard_running():
    """Check if dashboard API is active"""
    return True


# --- Run server ---

if __name__ == "__main__":
    print("\n" + "="*60)
    print("QUADRICK DASHBOARD API SERVER")
    print("="*60)
    
    if bybit_client:
        print("✅ Connected to Bybit")
        try:
            balance = bybit_client.get_account_balance()
            if balance:
                print(f"   Balance: ${balance.total_equity:.2f}")
        except:
            pass
    else:
        print("❌ Not connected to Bybit - run from quadrick directory with venv activated")
    
    # Use PORT from environment (Railway/Vercel) or default to 8000
    port = int(os.getenv("PORT", 8001))
    
    print(f"\n🌐 Starting server on http://0.0.0.0:{port}")
    print("📊 Dashboard: http://localhost:3000")
    print("="*60 + "\n")
    
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=port,
        reload=False,  # Disable reload in production
        log_level="info"
    )

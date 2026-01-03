"""
Quadrick Dashboard API - Vercel Serverless Functions
This is the main API endpoint for the Vercel deployment.
All data is stored in Supabase.
"""
import os
import sys
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from datetime import datetime

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.database.supabase_bridge import SupabaseBridge

app = FastAPI(title="Quadrick Cloud API")

# Configure CORS - allow all origins (restrict in production if needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, you can restrict to your Vercel domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Bridge
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_ANON_KEY") or os.getenv("SUPABASE_KEY")
bridge = SupabaseBridge(SUPABASE_URL, SUPABASE_KEY) if SUPABASE_URL and SUPABASE_KEY else None

# Pydantic Models (Matching api_server.py)
class BotStatus(BaseModel):
    mode: str
    trading_allowed: bool
    pause_reason: Optional[str] = None
    consecutive_losses: int = 0
    cooldown_active: bool = False
    connected: bool = True

class Balance(BaseModel):
    total: float
    available: float
    unrealized_pnl: float
    daily_pnl: float
    daily_pnl_pct: float

class Position(BaseModel):
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

class Decision(BaseModel):
    id: str
    timestamp: str
    symbol: str
    type: str
    confidence: float
    strategy: str
    reasoning: str
    htf_aligned: Optional[bool] = None

class LogEntry(BaseModel):
    timestamp: str
    level: str
    message: str

class ControlAction(BaseModel):
    action: str  # "start", "stop", "pause"

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "Quadrick Dashboard API",
        "version": "1.0.0",
        "environment": "Production (Vercel)",
        "connected": bridge is not None
    }

# Status endpoint
@app.get("/api/status", response_model=BotStatus)
async def get_status():
    """Get current bot status"""
    if not bridge:
        return {
            "mode": "error",
            "trading_allowed": False,
            "pause_reason": "Supabase not configured",
            "consecutive_losses": 0,
            "cooldown_active": False,
            "connected": False
        }
    
    try:
        status = bridge.get_status()
        return {
            "mode": status.get("mode", "stopped"),
            "trading_allowed": status.get("trading_allowed", False),
            "pause_reason": status.get("pause_reason"),
            "consecutive_losses": status.get("consecutive_losses", 0),
            "cooldown_active": status.get("cooldown_active", False),
            "connected": True
        }
    except Exception as e:
        return {
            "mode": "error",
            "trading_allowed": False,
            "pause_reason": str(e),
            "consecutive_losses": 0,
            "cooldown_active": False,
            "connected": False
        }

# Balance endpoint
@app.get("/api/balance", response_model=Balance)
async def get_balance():
    """Get account balance"""
    if not bridge:
        return {
            "total": 0.0,
            "available": 0.0,
            "unrealized_pnl": 0.0,
            "daily_pnl": 0.0,
            "daily_pnl_pct": 0.0
        }
    
    try:
        status = bridge.get_status()
        return {
            "total": status.get("total_balance", 0.0),
            "available": status.get("available_balance", 0.0),
            "unrealized_pnl": status.get("unrealized_pnl", 0.0),
            "daily_pnl": status.get("daily_pnl", 0.0),
            "daily_pnl_pct": status.get("daily_pnl_pct", 0.0)
        }
    except Exception as e:
        return {
            "total": 0.0,
            "available": 0.0,
            "unrealized_pnl": 0.0,
            "daily_pnl": 0.0,
            "daily_pnl_pct": 0.0
        }

# Positions endpoint
@app.get("/api/positions", response_model=List[Position])
async def get_positions():
    """Get active positions"""
    if not bridge:
        return []
    
    try:
        positions = bridge.get_positions()
        # Transform to match expected format
        result = []
        for pos in positions:
            result.append({
                "symbol": pos.get("symbol", ""),
                "side": pos.get("side", "Buy"),
                "size": float(pos.get("size", 0)),
                "entry_price": float(pos.get("entry_price", 0)),
                "current_price": float(pos.get("current_price", 0)),
                "pnl": float(pos.get("pnl", 0)),
                "pnl_pct": float(pos.get("pnl_pct", 0)),
                "leverage": int(pos.get("leverage", 1)),
                "stop_loss": pos.get("stop_loss"),
                "take_profit": pos.get("take_profit")
            })
        return result
    except Exception as e:
        return []

# Decisions endpoint
@app.get("/api/decisions", response_model=List[Decision])
async def get_decisions(limit: int = 20):
    """Get recent trading decisions"""
    if not bridge:
        return []
    
    try:
        # Fetch from Supabase decisions table
        # Query decisions from Supabase, ordered by created_at descending
        res = bridge.client.table("decisions").select("*").order("created_at", desc=True).limit(limit).execute()
        decisions = res.data if res.data else []
        
        # Transform to match expected format
        result = []
        for dec in decisions:
            result.append({
                "id": str(dec.get("id", "")),
                "timestamp": dec.get("created_at", dec.get("timestamp", datetime.now().isoformat())),
                "symbol": dec.get("symbol", ""),
                "type": dec.get("type", dec.get("decision_type", "WAIT")),
                "confidence": float(dec.get("confidence", dec.get("confidence_score", 0.0))),
                "strategy": dec.get("strategy", dec.get("strategy_tag", "unknown")),
                "reasoning": dec.get("reasoning", dec.get("reasoning_text", "")),
                "htf_aligned": dec.get("htf_aligned")
            })
        return result
    except Exception as e:
        return []

# Logs endpoint
@app.get("/api/logs", response_model=List[LogEntry])
async def get_logs(limit: int = 100):
    """Get recent logs"""
    if not bridge:
        return []
    
    try:
        logs = bridge.get_logs(limit)
        # Transform to match expected format
        result = []
        for log in logs:
            result.append({
                "timestamp": log.get("timestamp", datetime.now().isoformat()),
                "level": log.get("level", "INFO"),
                "message": log.get("message", "")
            })
        return result
    except Exception as e:
        return []

# Control endpoint
@app.post("/api/control")
async def control_bot(action: ControlAction):
    """Control the trading bot"""
    if not bridge:
        raise HTTPException(status_code=500, detail="Supabase bridge not initialized")
    
    try:
        if action.action == "start":
            bridge.update_status({"mode": "active", "trading_allowed": True})
            return {"success": True, "message": "Bot mode set to ACTIVE"}
        elif action.action == "stop":
            bridge.update_status({"mode": "stopped", "trading_allowed": False})
            return {"success": True, "message": "Bot stopped via emergency switch"}
        elif action.action == "pause":
            bridge.update_status({"mode": "paused", "trading_allowed": False, "pause_reason": "Manual pause from dashboard"})
            return {"success": True, "message": "Bot paused temporarily"}
        else:
            raise HTTPException(status_code=400, detail="Invalid action. Use 'start', 'stop', or 'pause'")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Settings endpoint (GET)
@app.get("/api/settings")
async def get_settings():
    """Get current settings (read-only for Vercel deployment)"""
    # Settings are managed via environment variables in Vercel
    # Return a placeholder response
    return {
        "message": "Settings are managed via Vercel environment variables",
        "note": "Update settings in Vercel dashboard → Settings → Environment Variables"
    }

# Settings endpoint (POST) - disabled for Vercel
@app.post("/api/settings")
async def update_settings():
    """Update settings (disabled for Vercel deployment)"""
    raise HTTPException(
        status_code=501,
        detail="Settings updates are not supported in Vercel deployment. Update environment variables in Vercel dashboard."
    )

# Performance endpoint
@app.get("/api/performance")
async def get_performance():
    """Get performance metrics"""
    # This would need to query Supabase for trade history
    # For now, return placeholder
    return {
        "total_trades": 0,
        "wins": 0,
        "losses": 0,
        "win_rate": 0.0,
        "total_pnl": 0.0,
        "best_strategy": None
    }

# For Vercel, we need to export the handler
# Vercel Python runtime supports ASGI apps directly
try:
    from mangum import Mangum
    handler = Mangum(app)
except ImportError:
    # Fallback if mangum is not available
    # Vercel's @vercel/python should handle FastAPI apps directly
    handler = app

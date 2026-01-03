import os
import sys
import asyncio
import uuid
import json
from datetime import datetime, timezone
from loguru import logger

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import QuadrickTradingBot
from src.database.supabase_bridge import SupabaseBridge
from src.integration.dashboard_bridge import bridge

async def handler(request):
    """
    Vercel Cron / API Handler for the Trading Pulse
    Triggered every minute by vercel.json cron config
    """
    logger.info("--- Starting Trading Pulse ---")
    
    # 1. Initialize Bridge & Check Status
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_KEY")
    
    if not SUPABASE_URL or not SUPABASE_KEY:
        return {"success": False, "error": "Supabase credentials missing"}
        
    supabase = SupabaseBridge(SUPABASE_URL, SUPABASE_KEY)
    
    status = supabase.get_status()
    if status.get("mode") not in ["active", "initialized"]:
        logger.info(f"Bot mode is {status.get('mode')}. Skipping pulse.")
        return {"success": True, "message": f"Bot is in {status.get('mode')} mode"}

    try:
        # 2. Initialize Bot
        bot = QuadrickTradingBot()
        
        # 3. Perform One Cycle
        # Initialize (fetches balance and positions)
        await asyncio.to_thread(bot.initialize)
        
        # A. Update Account State
        await bot._update_account_state()
        
        # B. Sync Status and Positions to Supabase
        supabase.update_status({
            "total_balance": float(bot.account_balance),
            "available_balance": float(bot.account_balance), # Re-calc logic if needed
            "unrealized_pnl": sum(float(p.unrealized_pnl) for p in bot.positions) if bot.positions else 0,
            "updated_at": datetime.now(timezone.utc).isoformat()
        })
        
        # Sync Positions
        positions_data = []
        for pos in bot.positions:
             positions_data.append({
                "symbol": pos.symbol,
                "side": pos.side,
                "size": float(pos.size),
                "entry_price": float(pos.avg_price),
                "mark_price": float(pos.mark_price),
                "unrealized_pnl": float(pos.unrealized_pnl),
                "leverage": float(pos.leverage),
                "stop_loss": float(pos.stop_loss) if pos.stop_loss else None,
                "take_profit": float(pos.take_profit) if pos.take_profit else None
            })
        supabase.update_positions(positions_data)
        
        # C. Market Data & Analysis
        market_data = await bot._fetch_market_data()
        analysis = await bot._analyze_markets(market_data)
        
        # D. Council Decision & Execution
        logger.info("Requesting decisions for watchlist...")
        for symbol in bot.watchlist:
            if symbol in analysis:
                decision_data = await bot.council.make_decision(
                    symbol=symbol,
                    market_data=analysis[symbol],
                    account_balance=bot.account_balance
                )
                
                # Convert dict to TradingDecision object for bot handlers
                from src.agents.council import DecisionType, TradingDecision
                
                if decision_data.get("decision_type") != "wait":
                    # Execute decision
                    # Note: We create a proper decision object here
                    decision = bot.deepseek._parse_decision(json.dumps(decision_data))
                    await bot._execute_decision(decision, market_data, analysis)
        
        # E. Logging
        supabase.push_log("INFO", f"Pulse complete. Balance: ${bot.account_balance:.2f} | Positions: {len(bot.positions)}")
        
        logger.info("--- Pulse Complete ---")
        return {"success": True, "message": "Pulse completed successfully"}

    except Exception as e:
        error_msg = f"Pulse failed: {str(e)}"
        logger.error(error_msg)
        if supabase:
            supabase.push_log("ERROR", error_msg)
        return {"success": False, "error": error_msg}

    except Exception as e:
        logger.error(f"Pulse failed: {e}")
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    # Local test
    asyncio.run(handler(None))

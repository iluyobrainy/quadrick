import asyncio
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from loguru import logger
from supabase import create_client, Client

class SupabaseBridge:
    """Bridge for shared state between Dashboard and Bot on Supabase"""
    
    def __init__(self, supabase_url: str, supabase_key: str):
        self.url = supabase_url
        self.key = supabase_key
        self.client: Client = create_client(supabase_url, supabase_key)

    def get_status(self) -> Dict[str, Any]:
        """Fetch bot status from Supabase"""
        try:
            res = self.client.table("bot_status").select("*").eq("id", 1).execute()
            if res.data:
                return res.data[0]
            return {"mode": "stopped", "trading_allowed": False}
        except Exception as e:
            logger.error(f"Failed to fetch status: {e}")
            return {"mode": "error", "trading_allowed": False}

    def update_status(self, status_data: Dict[str, Any]):
        """Update bot status in Supabase"""
        try:
            status_data["updated_at"] = datetime.now(timezone.utc).isoformat()
            self.client.table("bot_status").upsert({"id": 1, **status_data}).execute()
        except Exception as e:
            logger.error(f"Failed to update status: {e}")

    def get_positions(self) -> List[Dict[str, Any]]:
        """Fetch active positions from Supabase"""
        try:
            res = self.client.table("active_positions").select("*").execute()
            return res.data if res.data else []
        except Exception as e:
            logger.error(f"Failed to fetch positions: {e}")
            return []

    def update_positions(self, positions: List[Dict[str, Any]]):
        """Update active positions in Supabase"""
        try:
            # 1. Clear existing positions (simplified sync)
            self.client.table("active_positions").delete().neq("symbol", "NONE").execute()
            # 2. Insert new positions
            if positions:
                for pos in positions:
                    pos["updated_at"] = datetime.now(timezone.utc).isoformat()
                self.client.table("active_positions").insert(positions).execute()
        except Exception as e:
            logger.error(f"Failed to update positions: {e}")

    def push_log(self, level: str, message: str):
        """Push a system log to Supabase"""
        try:
            log_entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "level": level,
                "message": message
            }
            self.client.table("system_logs").insert(log_entry).execute()
        except Exception as e:
            # Don't use logger here to avoid recursion
            pass

    def get_logs(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Fetch recent logs from Supabase"""
        try:
            res = self.client.table("system_logs").select("*").order("timestamp", desc=True).limit(limit).execute()
            return res.data if res.data else []
        except Exception as e:
            logger.error(f"Failed to fetch logs: {e}")
            return []

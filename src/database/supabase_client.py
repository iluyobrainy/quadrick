import asyncio
import json
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from loguru import logger
from supabase import create_client, Client

class SupabaseClient:
    """Supabase client for RAG memory and trade logging"""

    def __init__(self, supabase_url: str, supabase_key: str, enabled: bool = True):
        self.enabled = enabled
        self.client: Optional[Client] = None
        
        if enabled and supabase_url and supabase_key:
            try:
                self.client = create_client(supabase_url, supabase_key)
                logger.info("Supabase client initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Supabase client: {e}")
                self.enabled = False

    async def save_trade(self, trade_data: Dict[str, Any]):
        """Save trade execution to database"""
        if not self.enabled or not self.client:
            return

        try:
            # Run in executor to avoid blocking
            await asyncio.to_thread(
                lambda: self.client.table("trades").insert(trade_data).execute()
            )
            logger.debug(f"Trade {trade_data.get('trade_id')} saved to Supabase")
        except Exception as e:
            logger.error(f"Failed to save trade to Supabase: {e}")

    async def save_decision(self, decision_data: Dict[str, Any]):
        """Save LLM decision to database"""
        if not self.enabled or not self.client:
            return

        try:
            # Run in executor
            await asyncio.to_thread(
                lambda: self.client.table("decisions").insert(decision_data).execute()
            )
        except Exception as e:
            logger.error(f"Failed to save decision to Supabase: {e}")

    async def save_active_context(self, symbol: str, context: Dict[str, Any]):
        """Persist trade entry context so it survives restarts"""
        if not self.enabled or not self.client:
            return
        try:
            data = {
                "symbol": symbol,
                "context": context,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            await asyncio.to_thread(
                lambda: self.client.table("active_contexts").upsert(data, on_conflict="symbol").execute()
            )
        except Exception as e:
            logger.error(f"Failed to save active context for {symbol}: {e}")

    async def get_active_context(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Retrieve entry context for a trade opened before restart"""
        if not self.enabled or not self.client:
            return None
        try:
            response = await asyncio.to_thread(
                lambda: self.client.table("active_contexts").select("context").eq("symbol", symbol).execute()
            )
            if response.data:
                return response.data[0].get("context")
        except Exception as e:
            logger.error(f"Failed to get active context for {symbol}: {e}")
        return None

    async def delete_active_context(self, symbol: str):
        """Clear context after trade is closed and saved to memory"""
        if not self.enabled or not self.client:
            return
        try:
            await asyncio.to_thread(
                lambda: self.client.table("active_contexts").delete().eq("symbol", symbol).execute()
            )
        except Exception as e:
            logger.error(f"Failed to delete active context for {symbol}: {e}")

    async def save_strategy_stats(self, stats: Dict[str, Any]):
        """Save strategy optimizer learning data"""
        if not self.enabled or not self.client:
            return
        try:
            payload = {
                "id": "global_stats", # Single record for optimization data
                "data": stats,
                "updated_at": datetime.now(timezone.utc).isoformat()
            }
            await asyncio.to_thread(
                lambda: self.client.table("strategy_learning").upsert(payload, on_conflict="id").execute()
            )
        except Exception as e:
            logger.error(f"Failed to save strategy stats: {e}")

    async def load_strategy_stats(self) -> Optional[Dict[str, Any]]:
        """Load strategy optimizer learning data"""
        if not self.enabled or not self.client:
            return None
        try:
            response = await asyncio.to_thread(
                lambda: self.client.table("strategy_learning").select("data").eq("id", "global_stats").execute()
            )
            if response.data:
                return response.data[0].get("data")
        except Exception as e:
            logger.error(f"Failed to load strategy stats: {e}")
        return None

    async def save_trade_memory(self, trade_result: Dict[str, Any], market_context: Dict[str, Any]):
        """
        Save a completed trade with its market context vector for future retrieval.
        
        Args:
            trade_result: Dict with 'pnl_pct', 'win' (bool), 'strategy', 'symbol'
            market_context: Dict with technical indicators (rsi, trend, volatility, etc.)
        """
        if not self.enabled or not self.client:
            return

        try:
            # 1. Vectorize the context
            embedding = self._vectorize_context(market_context)
            
            # 2. Prepare record
            memory_record = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "symbol": trade_result.get("symbol"),
                "strategy": trade_result.get("strategy"),
                "pnl_pct": trade_result.get("pnl_pct"),
                "win": trade_result.get("win"),
                "market_vector": embedding, # pgvector column
                "market_context_json": market_context # Store raw context for debugging
            }

            # 3. Insert
            await asyncio.to_thread(
                lambda: self.client.table("trade_memories").insert(memory_record).execute()
            )
            logger.info(f"Trade memory saved for {trade_result.get('symbol')} (Win: {trade_result.get('win')})")

        except Exception as e:
            logger.error(f"Failed to save trade memory: {e}")

    async def find_similar_trades(self, symbol: str, market_data: Dict[str, Any], limit: int = 5) -> List[Dict[str, Any]]:
        """
        Find similar past trades using vector similarity search.
        
        Args:
            symbol: Trading symbol (e.g., "BTCUSDT")
            market_data: Market analysis data with timeframe_analysis structure
            limit: Maximum number of similar trades to return
        """
        if not self.enabled or not self.client:
            return []

        try:
            # Extract the symbol's 1h timeframe data
            # Handle both 'timeframe_analysis' (new format) and 'timeframes' (old format)
            tf_data = market_data.get("timeframe_analysis", {}).get("1h") or market_data.get("timeframes", {}).get("1h")
            
            if not tf_data:
                logger.warning(f"No 1h timeframe data for {symbol}")
                return []
            
            # Convert TimeframeAnalysis object to dict if needed
            if hasattr(tf_data, 'indicators'):
                # It's a TimeframeAnalysis object, extract what we need
                tf_dict = {
                    "rsi": getattr(tf_data.indicators, 'rsi', 50),
                    "trend": getattr(tf_data.structure, 'trend', 'neutral').value if hasattr(getattr(tf_data.structure, 'trend', None), 'value') else str(getattr(tf_data.structure, 'trend', 'neutral')),
                    "atr": getattr(tf_data.indicators, 'atr', 0),
                    "close": getattr(tf_data.indicators, 'close', [0])[-1] if hasattr(getattr(tf_data.indicators, 'close', []), '__getitem__') else 0,
                    "macd_signal": getattr(tf_data.indicators, 'macd_signal', 'neutral').value if hasattr(getattr(tf_data.indicators, 'macd_signal', None), 'value') else str(getattr(tf_data.indicators, 'macd_signal', 'neutral')),
                    "current_price": market_data.get("current_price", 0)
                }
                tf_data = tf_dict
            
            # Add current price if missing
            if "current_price" not in tf_data:
                tf_data["current_price"] = market_data.get("current_price", 0)
            
            # Vectorize current context
            query_vector = self._vectorize_context(tf_data)
            logger.debug(f"Query vector for {symbol}: {query_vector}")
            
            # Call Supabase RPC function
            response = await asyncio.to_thread(
                lambda: self.client.rpc(
                    "match_trade_memories",
                    {
                        "query_embedding": query_vector,
                        "match_threshold": 0.7,
                        "match_count": limit
                    }
                ).execute()
            )
            
            return response.data if response.data else []

        except Exception as e:
            logger.error(f"Failed to find similar trades: {e}")
            return []

    def _vectorize_context(self, context: Dict[str, Any]) -> List[float]:
        """
        Convert market context into a normalized 8-dimensional float vector.
        
        Vector Dimensions (8D for enhanced similarity matching):
        [0]: RSI / 100 (0-1)
        [1]: Trend (-1, 0, 1)
        [2]: Volatility (ATR% normalized)
        [3]: Momentum (MACD signal)
        [4]: ADX / 100 (trend strength, 0-1)
        [5]: Volume Ratio (normalized, 0-1)
        [6]: BB Position (-1 lower, 0 middle, 1 upper)
        [7]: Time of Day factor (0-1, hour / 24)
        """
        try:
            # [0] RSI normalized
            rsi = float(context.get("rsi", 50))
            
            # [1] Trend direction
            trend_str = str(context.get("trend", "neutral")).lower()
            trend_map = {
                "uptrend": 1.0, "bullish": 1.0, "trending_up": 1.0, "up": 1.0,
                "downtrend": -1.0, "bearish": -1.0, "trending_down": -1.0, "down": -1.0,
                "neutral": 0.0, "ranging": 0.0, "sideways": 0.0
            }
            trend = trend_map.get(trend_str, 0.0)
            
            # [2] Volatility (ATR as % of price)
            atr = float(context.get("atr", 0))
            price = float(context.get("close", context.get("current_price", 1)))
            volatility_pct = (atr / price * 100) if price > 0 else 0
            
            # [3] Momentum (MACD signal)
            macd_signal = str(context.get("macd_signal", "neutral")).lower()
            momentum = 1.0 if macd_signal == "bullish" else (-1.0 if macd_signal == "bearish" else 0.0)
            
            # [4] ADX (trend strength) - NEW
            adx = float(context.get("adx", 25))
            
            # [5] Volume ratio (normalized) - NEW
            volume_ratio = float(context.get("volume_ratio", 1.0))
            
            # [6] Bollinger Band position - NEW
            bb_position_str = str(context.get("bb_position", "middle")).lower()
            bb_position_map = {"upper": 1.0, "lower": -1.0, "middle": 0.0}
            bb_position = bb_position_map.get(bb_position_str, 0.0)
            
            # [7] Time of day factor - NEW
            from datetime import datetime
            hour = datetime.utcnow().hour
            time_factor = hour / 24.0
            
            # Build 8-dimension vector (normalized)
            vec = [
                rsi / 100.0,                           # [0] RSI: 0-1
                trend,                                  # [1] Trend: -1 to 1
                min(volatility_pct / 5.0, 1.0),        # [2] Volatility: 0-1
                momentum,                               # [3] Momentum: -1 to 1
                min(adx / 100.0, 1.0),                 # [4] ADX: 0-1
                min(volume_ratio / 5.0, 1.0),          # [5] Volume: 0-1 (capped at 5x)
                bb_position,                            # [6] BB Position: -1 to 1
                time_factor                             # [7] Time: 0-1
            ]
            
            logger.debug(
                f"Vectorized (8D): RSI={rsi:.1f}, Trend={trend_str}, Vol={volatility_pct:.2f}%, "
                f"MACD={macd_signal}, ADX={adx:.1f}, VolumeRatio={volume_ratio:.1f}x"
            )
            return vec
            
        except Exception as e:
            logger.warning(f"Vectorization failed: {e}")
            return [0.0] * 8  # Return 8-dimension zero vector



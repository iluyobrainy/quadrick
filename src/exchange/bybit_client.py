"""
Bybit Exchange Client - Handles all interactions with Bybit API
"""
import asyncio
import json
import time
import hmac
import hashlib
from typing import Dict, List, Optional, Any, Literal
from datetime import datetime, timedelta
import aiohttp
import websockets
import requests
from loguru import logger
from pybit import _helpers
from pybit.unified_trading import HTTP, WebSocket
import os
from dataclasses import dataclass, field
from enum import Enum


# PYBIT_TIMESTAMP_OFFSET_MS = 0 # DEPRECATED
# def _generate_timestamp_with_offset() -> int:
#     """Override pybit timestamp generation to include server offset."""
#     return int(time.time() * 1000 + PYBIT_TIMESTAMP_OFFSET_MS)

# _helpers.generate_timestamp = _generate_timestamp_with_offset


class OrderType(str, Enum):
    MARKET = "Market"
    LIMIT = "Limit"
    MARKET_IF_TOUCHED = "MarketIfTouched"
    LIMIT_IF_TOUCHED = "LimitIfTouched"


class OrderSide(str, Enum):
    BUY = "Buy"
    SELL = "Sell"


class TimeInForce(str, Enum):
    GTC = "GTC"  # Good Till Cancel
    IOC = "IOC"  # Immediate or Cancel
    FOK = "FOK"  # Fill or Kill
    POST_ONLY = "PostOnly"


@dataclass
class MarketData:
    """Market data structure"""
    symbol: str
    last_price: float
    bid_price: float
    ask_price: float
    volume_24h: float
    turnover_24h: float
    price_24h_change: float
    high_24h: float
    low_24h: float
    timestamp: datetime


@dataclass
class OrderBook:
    """Order book data structure"""
    symbol: str
    bids: List[tuple[float, float]]  # [(price, quantity), ...]
    asks: List[tuple[float, float]]
    timestamp: datetime


@dataclass
class Position:
    """Position data structure"""
    symbol: str
    side: str
    size: float
    avg_price: float
    mark_price: float
    unrealized_pnl: float
    realized_pnl: float
    leverage: int
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class AccountBalance:
    """Account balance structure"""
    total_equity: float
    available_balance: float
    used_margin: float
    unrealized_pnl: float
    realized_pnl: float
    timestamp: datetime


@dataclass
class FundingRate:
    """Funding rate data"""
    symbol: str
    funding_rate: float
    funding_rate_timestamp: datetime
    next_funding_time: datetime


class BybitClient:
    """Bybit Exchange Client"""
    
    def __init__(self, api_key: str, api_secret: str, testnet: bool = True):
        """Initialize Bybit client"""
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        
        # API Configuration
        self.BYBIT_API_URL = os.getenv("BYBIT_BASE_URL_MAINNET", "https://api.bybit.com")
        self.BYBIT_TESTNET_API_URL = os.getenv("BYBIT_BASE_URL_TESTNET", "https://api-testnet.bybit.com")
        
        # WebSocket Configuration
        self.BYBIT_WS_URL = os.getenv("BYBIT_WS_URL_MAINNET", "wss://stream.bybit.com/v5/public")
        self.BYBIT_TESTNET_WS_URL = os.getenv("BYBIT_WS_URL_TESTNET", "wss://stream-testnet.bybit.com/v5/public")

        # Determine the base domain for HTTP requests
        # pybit expects just the domain stem (e.g., "bybit" or "bytick")
        if self.testnet:
            self.domain = self.BYBIT_TESTNET_API_URL.replace("https://", "").split("/")[0]
        else:
            self.domain = self.BYBIT_API_URL.replace("https://", "").split("/")[0]

        # Extract domain stem for pybit (e.g., "bytick" from "api.bytick.com")
        # pybit constructs URL as: https://{subdomain}.{domain}.{tld}
        if "bytick" in self.domain:
            # For bytick, we need to be careful as pybit might expect standard bybit.com
            # But we are passing the full URL manually to HTTP/WebSocket sessions below
            pybit_domain = "bytick"
        elif "bybit" in self.domain:
            pybit_domain = "bybit"
        else:
            # Fallback/Custom
            pybit_domain = "bytick" if not testnet else "bybit"

        
        logger.info(f"Using Bybit API URL: {self.BYBIT_API_URL if not testnet else self.BYBIT_TESTNET_API_URL}")
        
        # Initialize HTTP Session
        self.http = HTTP(
            testnet=testnet,
            api_key=api_key,
            api_secret=api_secret,
            recv_window=20000,  # Increased recv_window to reduce timeouts
            max_retries=5,
            retry_delay=2,
            force_retry=True,
            domain=pybit_domain  # Pass stem only
        )
        
        # FORCE override the base URL in the internal session
        # This is critical because pybit might default to bytick or bybit based on testnet flag
        # We want to strictly use what's in the env vars
        if testnet:
            self.http.endpoint = self.BYBIT_TESTNET_API_URL
        else:
             self.http.endpoint = self.BYBIT_API_URL
             
        # Log the final endpoint to be 100% sure
        logger.info(f"Final HTTP Session Endpoint: {self.http.endpoint}")

        # WebSocket clients will be initialized when needed
        self.ws_public = None
        self.ws_private = None
        
        # Market data cache
        self.market_data_cache: Dict[str, MarketData] = {}
        self.orderbook_cache: Dict[str, OrderBook] = {}
        self.funding_rates_cache: Dict[str, FundingRate] = {}
        
        # Position and balance cache
        self.positions_cache: Dict[str, Position] = {}
        self.account_balance: Optional[AccountBalance] = None

        self._last_time_sync: float = 0.0
        self._symbol_rules_cache: Dict[str, Dict] = {}
        self._sync_time(initial=True)
        
        logger.info(f"Bybit client initialized ({'Testnet' if testnet else 'Mainnet'})")
    
    def _sync_time(self, force: bool = False, initial: bool = False) -> None:
        """Sync local timestamp offset with Bybit server."""
        global PYBIT_TIMESTAMP_OFFSET_MS

        # Avoid frequent calls unless forced or initialisation
        now = time.time()
        if not force and not initial and (now - self._last_time_sync) < 60:
            return

        base_url = f"https://{self.domain}"
        try:
            response = requests.get(f"{base_url}/v5/market/time", timeout=5)
            response.raise_for_status()
            data = response.json()

            if data.get("retCode") != 0:
                raise ValueError(data.get("retMsg", "Unknown error"))

            result = data.get("result", {})
            server_time_ms = int(result.get("timeNano", 0)) // 1_000_000
            if server_time_ms == 0:
                server_time_ms = int(result.get("timeSecond", 0)) * 1000

            local_time_ms = int(time.time() * 1000)
            offset = server_time_ms - local_time_ms

            PYBIT_TIMESTAMP_OFFSET_MS = offset
            self._last_time_sync = now

            label = "initial" if initial else "forced" if force else "periodic"
            logger.info(f"Bybit time sync ({label}): offset {offset} ms")

        except Exception as exc:
            if initial:
                logger.warning(f"Failed to sync Bybit time during initialization: {exc}")
            elif force:
                logger.warning(f"Failed to resync Bybit time: {exc}")
            else:
                logger.debug(f"Time sync skipped: {exc}")

    def _execute_with_time_sync(self, func, *args, **kwargs):
        """Execute a pybit call with time-sync retry on timestamp errors."""
        self._sync_time()
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            message = str(exc)
            if "Retryable error" in message or "10002" in message or "timestamp" in message.lower():
                logger.warning("Timestamp mismatch detected, resyncing with Bybit and retrying...")
                self._sync_time(force=True)
                time.sleep(0.1)
                return func(*args, **kwargs)
            raise

    def initialize_websockets(self):
        """Initialize WebSocket connections (synchronous)"""
        try:
            # Public WebSocket for market data
            self.ws_public = WebSocket(
                testnet=self.testnet,
                channel_type="linear",
            )

            # Private WebSocket for account data
            self.ws_private = WebSocket(
                testnet=self.testnet,
                channel_type="private",
                api_key=self.api_key,
                api_secret=self.api_secret,
            )

            logger.info("WebSocket connections initialized")

        except Exception as e:
            logger.error(f"Failed to initialize WebSockets: {e}")
            raise
    
    def get_account_info(self) -> Dict[str, Any]:
        """Get account information"""
        try:
            response = self._execute_with_time_sync(
                self.http.get_wallet_balance,
                accountType="UNIFIED",
                coin="USDT",
            )
            return response["result"]
        except Exception as e:
            logger.error(f"Failed to get account info: {e}")
            raise
    
    def get_positions(self) -> List[Position]:
        """Get all open positions"""
        try:
            response = self._execute_with_time_sync(
                self.http.get_positions,
                category="linear",
                settleCoin="USDT",
            )

            # Check for API errors
            if response.get("retCode") != 0:
                logger.error(f"API error getting positions: {response.get('retMsg', 'Unknown error')}")
                return []

            positions = []
            result = response.get("result", {})
            positions_list = result.get("list", []) if isinstance(result, dict) else []

            for pos_data in positions_list:
                try:
                    if float(pos_data.get("size", 0)) > 0:
                        position = Position(
                            symbol=pos_data["symbol"],
                            side=pos_data["side"],
                            size=float(pos_data["size"]),
                            avg_price=float(pos_data["avgPrice"]),
                            mark_price=float(pos_data["markPrice"]),
                            unrealized_pnl=float(pos_data["unrealisedPnl"]),
                            realized_pnl=float(pos_data.get("realisedPnl", 0)),
                            leverage=int(pos_data["leverage"]),
                            stop_loss=float(pos_data["stopLoss"]) if pos_data.get("stopLoss") else None,
                            take_profit=float(pos_data["takeProfit"]) if pos_data.get("takeProfit") else None,
                        )
                        positions.append(position)
                except (KeyError, ValueError, TypeError) as e:
                    logger.warning(f"Skipping invalid position data: {e}")
                    continue

            return positions
            
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            raise
    
    def place_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        qty: float,
        price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        leverage: int = 1,
        time_in_force: TimeInForce = TimeInForce.GTC,
        reduce_only: bool = False,
    ) -> Dict[str, Any]:
        """Place an order on Bybit"""
        try:
            # Try to set leverage (might already be set)
            try:
                self.http.set_leverage(
                    category="linear",
                    symbol=symbol,
                    buyLeverage=str(leverage),
                    sellLeverage=str(leverage),
                )
                logger.debug(f"Leverage set to {leverage}x for {symbol}")
            except Exception as e:
                # Leverage already set or can't be changed - continue anyway
                logger.debug(f"Leverage setting skipped: {e}")
            
            # Build order parameters
            order_params = {
                "category": "linear",
                "symbol": symbol,
                "side": side.value,
                "orderType": order_type.value,
                "qty": str(qty),
                "timeInForce": time_in_force.value,
                "reduceOnly": reduce_only,
            }
            
            # Add price for limit orders
            if order_type == OrderType.LIMIT and price:
                order_params["price"] = str(price)
            
            # Add stop loss
            if stop_loss:
                order_params["stopLoss"] = str(stop_loss)
                order_params["slTriggerBy"] = "MarkPrice"
            
            # Add take profit
            if take_profit:
                order_params["takeProfit"] = str(take_profit)
                order_params["tpTriggerBy"] = "MarkPrice"
            
            # Place the order
            response = self._execute_with_time_sync(self.http.place_order, **order_params)

            if response["retCode"] == 0:
                logger.info(f"Order placed successfully: {response['result']}")
                return response  # Return full response instead of just result
            else:
                logger.error(f"Order failed: {response}")
                raise Exception(f"Order failed: {response['retMsg']}")
            
        except Exception as e:
            logger.error(f"Failed to place order: {e}")
            raise
    
    def update_position_protection(
        self,
        symbol: str,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        trailing_stop: Optional[float] = None,
        sl_trigger: str = "MarkPrice",
        tp_trigger: str = "MarkPrice",
    ) -> Dict[str, Any]:
        """Update stop loss / take profit / trailing stop for an open position."""

        if stop_loss is None and take_profit is None and trailing_stop is None:
            logger.debug("No protection updates requested")
            return {"status": "noop"}

        params: Dict[str, Any] = {
            "category": "linear",
            "symbol": symbol,
        }

        if stop_loss is not None:
            params["stopLoss"] = str(stop_loss)
            params["slTriggerBy"] = sl_trigger

        if take_profit is not None:
            params["takeProfit"] = str(take_profit)
            params["tpTriggerBy"] = tp_trigger

        if trailing_stop is not None:
            params["trailingStop"] = str(trailing_stop)

        try:
            response = self._execute_with_time_sync(self.http.set_trading_stop, **params)

            if response.get("retCode") == 0:
                logger.info(
                    f"Updated position protection for {symbol}: "
                    f"stop={stop_loss}, tp={take_profit}, trail={trailing_stop}"
                )
                return response

            logger.error(f"Failed to update position protection: {response}")
            raise Exception(response.get("retMsg", "Unknown error"))
        except Exception as exc:
            logger.error(f"Failed to update position protection for {symbol}: {exc}")
            raise

    def cancel_order(self, symbol: str, order_id: str) -> bool:
        """Cancel an order"""
        try:
            response = self.http.cancel_order(
                category="linear",
                symbol=symbol,
                orderId=order_id,
            )
            
            if response["retCode"] == 0:
                logger.info(f"Order {order_id} cancelled successfully")
                return True
            else:
                logger.error(f"Failed to cancel order: {response}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to cancel order: {e}")
            return False
    
    def close_position(self, symbol: str) -> Dict[str, Any]:
        """Close a position"""
        try:
            # Get current position
            positions = self.get_positions()
            position = next((p for p in positions if p.symbol == symbol), None)

            if not position:
                logger.warning(f"No position found for {symbol}")
                return {"status": "no_position"}

            # Place opposite order to close
            close_side = OrderSide.SELL if position.side == "Buy" else OrderSide.BUY

            result = self.place_order(
                symbol=symbol,
                side=close_side,
                order_type=OrderType.MARKET,
                qty=position.size,
                reduce_only=True,
            )

            logger.info(f"Position closed for {symbol}")
            return result

        except Exception as e:
            logger.error(f"Failed to close position: {e}")
            raise

    def place_bracket_order(
        self,
        symbol: str,
        side: OrderSide,
        qty: float,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        leverage: int = 10,
        account_balance: float = 0,
    ) -> Dict[str, Any]:
        """
        Place a bracket order (entry + stop loss + take profit)
        For small accounts, only place entry order to avoid margin issues

        Args:
            symbol: Trading symbol
            side: Buy or Sell
            qty: Quantity
            entry_price: Entry price (limit order)
            stop_loss: Stop loss price
            take_profit: Take profit price
            leverage: Leverage to use
            account_balance: Current account balance to check if bracket is feasible

        Returns:
            Order result
        """
        try:
            # For very small accounts (< $50), don't use bracket orders
            # They require margin for all legs even if unfilled
            if account_balance < 50:
                logger.info(f"Small account (${account_balance:.2f}) - skipping bracket order, using standard entry only")
                return {
                    "bracket_status": "skipped_small_account",
                    "reason": "insufficient_balance_for_bracket"
                }

            # Set leverage first (using the same logic as place_order)
            try:
                self.http.set_leverage(
                    category="linear",
                    symbol=symbol,
                    buyLeverage=str(leverage),
                    sellLeverage=str(leverage),
                )
                logger.debug(f"Bracket order leverage set to {leverage}x for {symbol}")
            except Exception as e:
                # Leverage already set or can't be changed - continue anyway
                logger.debug(f"Bracket order leverage setting skipped: {e}")

            # Place limit entry order
            entry_result = self.place_order(
                symbol=symbol,
                side=side,
                order_type=OrderType.LIMIT,
                qty=qty,
                price=entry_price,
            )

            if entry_result["retCode"] != 0:
                logger.error(f"Failed to place entry order: {entry_result}")
                return entry_result

            # Try to place OCO order for exit (stop loss + take profit)
            # This might fail for small accounts
            try:
                oco_result = self.place_oco_order(
                    symbol=symbol,
                    side=OrderSide.SELL if side == OrderSide.BUY else OrderSide.BUY,
                    qty=qty,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                )

                return {
                    "entry_order": entry_result,
                    "oco_order": oco_result,
                    "bracket_status": "placed"
                }
            except Exception as oco_error:
                logger.warning(f"OCO order failed, using entry only: {oco_error}")
                return {
                    "entry_order": entry_result,
                    "oco_order": {"error": str(oco_error)},
                    "bracket_status": "entry_only"
                }

        except Exception as e:
            logger.error(f"Failed to place bracket order: {e}")
            raise

    def place_oco_order(
        self,
        symbol: str,
        side: OrderSide,
        qty: float,
        stop_loss: float,
        take_profit: float,
    ) -> Dict[str, Any]:
        """
        Place One-Cancels-Other (OCO) order

        Args:
            symbol: Trading symbol
            side: Sell (for long positions) or Buy (for short positions)
            qty: Quantity to close
            stop_loss: Stop loss price
            take_profit: Take profit price

        Returns:
            OCO order result
        """
        try:
            # Bybit doesn't have native OCO, so we'll place two separate orders
            # In production, you'd want to use Bybit's advanced order features

            # Place take profit limit order
            tp_result = self.place_order(
                symbol=symbol,
                side=side,
                order_type=OrderType.LIMIT,
                qty=qty,
                price=take_profit,
                reduce_only=True,
            )

            # Place stop loss order (could be market or limit)
            sl_result = self.place_order(
                symbol=symbol,
                side=side,
                order_type=OrderType.MARKET,  # Use market for guaranteed execution
                qty=qty,
                stop_loss=stop_loss,
                reduce_only=True,
            )

            return {
                "take_profit_order": tp_result,
                "stop_loss_order": sl_result,
                "oco_status": "placed"
            }

        except Exception as e:
            logger.error(f"Failed to place OCO order: {e}")
            raise

    def place_trailing_stop_order(
        self,
        symbol: str,
        side: OrderSide,
        qty: float,
        trailing_distance: float,
        activation_price: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Place a trailing stop order

        Args:
            symbol: Trading symbol
            side: Order side
            qty: Quantity
            trailing_distance: Trailing distance in percentage
            activation_price: Price at which to activate trailing stop

        Returns:
            Trailing stop order result
        """
        try:
            # Convert percentage to price distance
            # This is a simplified implementation
            current_price = float(self.get_ticker(symbol).last_price)

            if side == OrderSide.BUY:
                # For long positions, trail below current price
                trail_price = current_price * (1 - trailing_distance / 100)
            else:
                # For short positions, trail above current price
                trail_price = current_price * (1 + trailing_distance / 100)

            result = self.place_order(
                symbol=symbol,
                side=side,
                order_type=OrderType.MARKET,  # Could be LIMIT in production
                qty=qty,
                stop_loss=trail_price,
                reduce_only=True,
            )

            return result

        except Exception as e:
            logger.error(f"Failed to place trailing stop: {e}")
            raise
    
    def get_ticker(self, symbol: str) -> MarketData:
        """Get ticker data for a symbol"""
        try:
            response = self._execute_with_time_sync(
                self.http.get_tickers,
                category="linear",
                symbol=symbol,
            )

            # Check for API errors
            if response.get("retCode") != 0:
                logger.error(f"API error getting ticker: {response.get('retMsg', 'Unknown error')}")
                raise Exception(f"API error: {response.get('retMsg', 'Unknown error')}")

            result = response.get("result", {})
            ticker_list = result.get("list", []) if isinstance(result, dict) else []

            if ticker_list:
                ticker = ticker_list[0]
                return MarketData(
                    symbol=ticker["symbol"],
                    last_price=float(ticker["lastPrice"]),
                    bid_price=float(ticker["bid1Price"]),
                    ask_price=float(ticker["ask1Price"]),
                    volume_24h=float(ticker["volume24h"]),
                    turnover_24h=float(ticker["turnover24h"]),
                    price_24h_change=float(ticker["price24hPcnt"]),
                    high_24h=float(ticker["highPrice24h"]),
                    low_24h=float(ticker["lowPrice24h"]),
                    timestamp=datetime.utcnow(),
                )
            else:
                raise Exception(f"Failed to get ticker: {response}")
                
        except Exception as e:
            logger.error(f"Failed to get ticker for {symbol}: {e}")
            raise
    
    def get_orderbook(self, symbol: str, limit: int = 50) -> OrderBook:
        """Get order book for a symbol"""
        try:
            response = self._execute_with_time_sync(
                self.http.get_orderbook,
                category="linear",
                symbol=symbol,
                limit=limit,
            )

            # Check for API errors
            if response.get("retCode") != 0:
                logger.error(f"API error getting orderbook: {response.get('retMsg', 'Unknown error')}")
                raise Exception(f"API error: {response.get('retMsg', 'Unknown error')}")

            if response["retCode"] == 0:
                result = response["result"]
                
                bids = [(float(bid[0]), float(bid[1])) for bid in result["b"]]
                asks = [(float(ask[0]), float(ask[1])) for ask in result["a"]]
                
                return OrderBook(
                    symbol=symbol,
                    bids=bids,
                    asks=asks,
                    timestamp=datetime.utcnow(),
                )
            else:
                raise Exception(f"Failed to get orderbook: {response}")
                
        except Exception as e:
            logger.error(f"Failed to get orderbook for {symbol}: {e}")
            raise
    
    def get_klines(
        self,
        symbol: str,
        interval: str,
        limit: int = 200,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Get historical klines/candlestick data"""
        try:
            params = {
                "category": "linear",
                "symbol": symbol,
                "interval": interval,  # 1, 3, 5, 15, 30, 60, 120, 240, 360, 720, D, W, M
                "limit": limit,
            }

            if start_time:
                params["start"] = start_time
            if end_time:
                params["end"] = end_time

            response = self._execute_with_time_sync(self.http.get_kline, **params)

            # Check for API errors
            if response.get("retCode") != 0:
                logger.error(f"API error getting klines: {response.get('retMsg', 'Unknown error')}")
                return []

            result = response.get("result", {})
            klines_list = result.get("list", []) if isinstance(result, dict) else []

            klines = []
            for kline in klines_list:
                try:
                    if len(kline) >= 7:  # Ensure we have all required fields
                        klines.append({
                            "timestamp": int(kline[0]),
                            "open": float(kline[1]),
                            "high": float(kline[2]),
                            "low": float(kline[3]),
                            "close": float(kline[4]),
                            "volume": float(kline[5]),
                            "turnover": float(kline[6]),
                        })
                except (IndexError, ValueError, TypeError) as e:
                    logger.warning(f"Skipping invalid kline data: {e}")
                    continue
            return klines
                
        except Exception as e:
            logger.error(f"Failed to get klines for {symbol}: {e}")
            raise
    
    def get_funding_rate(self, symbol: str) -> FundingRate:
        """Get funding rate for a perpetual contract"""
        try:
            response = self._execute_with_time_sync(
                self.http.get_funding_rate_history,
                category="linear",
                symbol=symbol,
                limit=1,
            )
            
            if response["retCode"] == 0 and response["result"]["list"]:
                funding = response["result"]["list"][0]
                return FundingRate(
                    symbol=symbol,
                    funding_rate=float(funding["fundingRate"]),
                    funding_rate_timestamp=datetime.fromtimestamp(
                        int(funding["fundingRateTimestamp"]) / 1000
                    ),
                    next_funding_time=datetime.fromtimestamp(
                        int(funding["fundingRateTimestamp"]) / 1000
                    ) + timedelta(hours=8),  # Funding every 8 hours
                )
            else:
                raise Exception(f"Failed to get funding rate: {response}")
                
        except Exception as e:
            logger.error(f"Failed to get funding rate for {symbol}: {e}")
            raise
    
    def get_instruments_info(self, category: str = "linear") -> List[Dict[str, Any]]:
        """Get all available instruments"""
        try:
            response = self._execute_with_time_sync(
                self.http.get_instruments_info,
                category=category,
            )
            
            if response["retCode"] == 0:
                return response["result"]["list"]
            else:
                raise Exception(f"Failed to get instruments: {response}")
                
        except Exception as e:
            logger.error(f"Failed to get instruments info: {e}")
            raise

    def get_symbol_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Fetch and cache symbol-specific rules (min order qty, qty step, etc.)"""
        if symbol in self._symbol_rules_cache:
            return self._symbol_rules_cache[symbol]

        try:
            # Note: Fetching all instruments once and caching is more efficient
            instruments = self.get_instruments_info()
            for inst in instruments:
                self._symbol_rules_cache[inst['symbol']] = inst
            
            return self._symbol_rules_cache.get(symbol)
        except Exception as e:
            logger.warning(f"Failed to fetch info for {symbol}: {e}")
            return None

    def round_quantity(self, symbol: str, qty: float) -> float:
        """Round quantity to the correct step for the symbol"""
        info = self.get_symbol_info(symbol)
        if not info:
            # Safe fallbacks based on common symbol patterns
            if "BTC" in symbol: return round(qty, 3)
            if "ETH" in symbol: return round(qty, 2)
            return round(qty, 1)
            
        lot_size = info.get('lotSizeFilter', {})
        qty_step = float(lot_size.get('qtyStep', '0.1'))
        
        # Calculate precision based on the raw string from API
        step_str = lot_size.get('qtyStep', '0.1')
        if '.' in step_str:
            precision = len(step_str.split('.')[1])
        else:
            precision = 0
            
        # Standard rounding to step
        rounded = round(qty / qty_step) * qty_step
        return round(rounded, precision)
    
    def get_account_balance(self) -> AccountBalance:
        """Get account balance details"""
        try:
            response = self._execute_with_time_sync(
                self.http.get_wallet_balance,
                accountType="UNIFIED",
            )

            # Check for API errors
            if response.get("retCode") != 0:
                logger.error(f"API error getting balance: {response.get('retMsg', 'Unknown error')}")
                raise Exception(f"API error: {response.get('retMsg', 'Unknown error')}")

            if response["retCode"] == 0:
                result_list = response.get("result", {}).get("list", [])
                if not result_list:
                    raise Exception("Empty balance list in Bybit response")
                
                account_data = result_list[0]
                
                # Helper for safe float conversion
                def safe_float(value, default=0.0):
                    try:
                        return float(value) if value and value != '' else default
                    except (ValueError, TypeError):
                        return default

                # For Unified Trading Account, we should primarily use ACCOUNT LEVEL totals
                # these fields are provided at the list[0] level
                total_equity = safe_float(account_data.get("totalEquity"))
                available_balance = safe_float(account_data.get("totalAvailableBalance"))
                unrealized_pnl = safe_float(account_data.get("totalPerpUPL"))
                
                # If these are 0, try to fall back to USDT coin-specific data
                if total_equity == 0:
                    usdt_info = None
                    for coin in account_data.get("coin", []):
                        if coin["coin"] == "USDT":
                            usdt_info = coin
                            break
                    
                    if usdt_info:
                        total_equity = safe_float(usdt_info.get("equity"))
                        available_balance = safe_float(usdt_info.get("availableToWithdraw"))
                        unrealized_pnl = safe_float(usdt_info.get("unrealisedPnl"))
                
                return AccountBalance(
                    total_equity=total_equity,
                    available_balance=available_balance,
                    used_margin=safe_float(account_data.get("totalMarginBalance", 0)) - available_balance,
                    unrealized_pnl=unrealized_pnl,
                    realized_pnl=0.0, # aggregate realized pnl is harder to get from this endpoint
                    timestamp=datetime.utcnow(),
                )
            else:
                raise Exception(f"Failed to get balance: {response}")
                
        except Exception as e:
            logger.error(f"Failed to get account balance: {e}")
            raise
    
    def subscribe_market_data(self, symbols: List[str]):
        """Subscribe to market data via WebSocket"""
        if not self.ws_public:
            self.initialize_websockets()

        # Subscribe to tickers
        for symbol in symbols:
            self.ws_public.ticker_stream(
                symbol=symbol,
                callback=self._handle_ticker_update
            )

            # Subscribe to orderbook
            self.ws_public.orderbook_stream(
                depth=50,
                symbol=symbol,
                callback=self._handle_orderbook_update
            )
    
    def _handle_ticker_update(self, message):
        """Handle ticker updates from WebSocket"""
        try:
            data = message["data"]
            symbol = data["symbol"]
            
            self.market_data_cache[symbol] = MarketData(
                symbol=symbol,
                last_price=float(data["lastPrice"]),
                bid_price=float(data["bid1Price"]),
                ask_price=float(data["ask1Price"]),
                volume_24h=float(data["volume24h"]),
                turnover_24h=float(data["turnover24h"]),
                price_24h_change=float(data["price24hPcnt"]),
                high_24h=float(data["highPrice24h"]),
                low_24h=float(data["lowPrice24h"]),
                timestamp=datetime.utcnow(),
            )
            
        except Exception as e:
            logger.error(f"Failed to handle ticker update: {e}")
    
    def _handle_orderbook_update(self, message):
        """Handle orderbook updates from WebSocket"""
        try:
            data = message["data"]
            symbol = data["s"]
            
            bids = [(float(bid[0]), float(bid[1])) for bid in data["b"]]
            asks = [(float(ask[0]), float(ask[1])) for ask in data["a"]]
            
            self.orderbook_cache[symbol] = OrderBook(
                symbol=symbol,
                bids=bids,
                asks=asks,
                timestamp=datetime.utcnow(),
            )
            
        except Exception as e:
            logger.error(f"Failed to handle orderbook update: {e}")

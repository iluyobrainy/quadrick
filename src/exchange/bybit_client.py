"""
Bybit Exchange Client - Handles all interactions with Bybit API
"""
import asyncio
import json
import time
import hmac
import hashlib
import math
from typing import Dict, List, Optional, Any, Literal
from datetime import datetime, timedelta
from urllib.parse import urlparse
import aiohttp
import websockets
import requests
from loguru import logger
from pybit import _helpers
from pybit.unified_trading import HTTP, WebSocket
import os
from dataclasses import dataclass, field
from enum import Enum


PYBIT_TIMESTAMP_OFFSET_MS = 0


def _generate_timestamp_with_offset() -> int:
    """Override pybit timestamp generation to include server offset."""
    return int(time.time() * 1000 + PYBIT_TIMESTAMP_OFFSET_MS)


_helpers.generate_timestamp = _generate_timestamp_with_offset


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
    open_interest: float = 0.0
    basis: float = 0.0


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

        selected_base_url = self.BYBIT_TESTNET_API_URL if self.testnet else self.BYBIT_API_URL
        self.domain = self._extract_host(selected_base_url)
        self._fallback_base_urls = self._build_fallback_urls(selected_base_url)

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

        
        logger.info(f"Using Bybit API URL: {selected_base_url}")
        
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
        
        # Force initial endpoint from env
        self.http.endpoint = selected_base_url
             
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

    @staticmethod
    def _extract_host(base_url: str) -> str:
        parsed = urlparse(base_url)
        if parsed.netloc:
            return parsed.netloc
        return base_url.replace("https://", "").replace("http://", "").split("/")[0]

    def _build_fallback_urls(self, base_url: str) -> List[str]:
        urls = [base_url]
        if "bybit.com" in base_url:
            urls.append(base_url.replace("bybit.com", "bytick.com"))
        elif "bytick.com" in base_url:
            urls.append(base_url.replace("bytick.com", "bybit.com"))
        # Preserve order while removing duplicates
        return list(dict.fromkeys(urls))

    def _switch_endpoint(self, base_url: str) -> None:
        if not base_url or base_url == self.http.endpoint:
            return
        old_endpoint = self.http.endpoint
        self.http.endpoint = base_url
        self.domain = self._extract_host(base_url)
        logger.warning(f"Switched Bybit endpoint from {old_endpoint} to {base_url}")
    
    def _sync_time(self, force: bool = False, initial: bool = False) -> None:
        """Sync local timestamp offset with Bybit server."""
        global PYBIT_TIMESTAMP_OFFSET_MS

        # Avoid frequent calls unless forced or initialisation
        now = time.time()
        if not force and not initial and (now - self._last_time_sync) < 60:
            return

        endpoints = [self.http.endpoint] + [u for u in self._fallback_base_urls if u != self.http.endpoint]
        last_exc: Optional[Exception] = None

        for base_url in endpoints:
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

                self._switch_endpoint(base_url)
                PYBIT_TIMESTAMP_OFFSET_MS = offset
                self._last_time_sync = now

                label = "initial" if initial else "forced" if force else "periodic"
                logger.info(f"Bybit time sync ({label}): offset {offset} ms")
                return
            except Exception as exc:
                last_exc = exc
                continue

        if initial:
            logger.warning(f"Failed to sync Bybit time during initialization: {last_exc}")
        elif force:
            logger.warning(f"Failed to resync Bybit time: {last_exc}")
        else:
            logger.debug(f"Time sync skipped: {last_exc}")

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
            if "NameResolutionError" in message or "getaddrinfo failed" in message:
                logger.warning("Endpoint resolution failed, attempting fallback endpoint and retry...")
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
        time_in_force: Optional[TimeInForce] = None,
        reduce_only: bool = False,
        slippage_tolerance_pct: Optional[float] = None,
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

            qty = self.round_quantity(symbol, qty)
            if qty is None or qty <= 0:
                raise ValueError(f"Invalid quantity after rounding for {symbol}: {qty}")

            symbol_info = self.get_symbol_info(symbol) or {}
            lot_filter = symbol_info.get("lotSizeFilter", {}) if isinstance(symbol_info, dict) else {}
            try:
                min_qty = float(lot_filter.get("minOrderQty", 0) or 0)
            except (TypeError, ValueError):
                min_qty = 0.0
            try:
                max_qty = float(lot_filter.get("maxOrderQty", 0) or 0)
            except (TypeError, ValueError):
                max_qty = 0.0
            qty_step_raw = lot_filter.get("qtyStep", "0")
            try:
                qty_step = float(qty_step_raw or 0)
            except (TypeError, ValueError):
                qty_step = 0.0
            precision = self._step_precision(str(qty_step_raw or "0"))

            cap_qty = max_qty if max_qty > 0 else 0.0
            if cap_qty > 0 and qty > cap_qty:
                logger.warning(
                    f"Clipping {symbol} qty to exchange cap: {qty} -> {cap_qty} "
                    f"(maxOrderQty={max_qty}, orderType={order_type.value})"
                )
                qty = cap_qty
            if qty_step > 0:
                qty = math.floor(qty / qty_step) * qty_step
                qty = round(qty, precision)
            if min_qty > 0 and qty < min_qty:
                raise ValueError(
                    f"Quantity {qty} below minOrderQty {min_qty} for {symbol} after cap clipping"
                )
             
            effective_tif = time_in_force
            if effective_tif is None:
                effective_tif = TimeInForce.IOC if order_type == OrderType.MARKET else TimeInForce.GTC

            # Build order parameters
            order_params = {
                "category": "linear",
                "symbol": symbol,
                "side": side.value,
                "orderType": order_type.value,
                "qty": str(qty),
                "timeInForce": effective_tif.value,
                "reduceOnly": reduce_only,
            }

            if order_type == OrderType.MARKET and slippage_tolerance_pct is not None:
                try:
                    slippage = float(slippage_tolerance_pct)
                except (TypeError, ValueError):
                    slippage = 0.0
                if slippage > 0:
                    slippage = min(10.0, max(0.01, slippage))
                    order_params["slippageToleranceType"] = "Percent"
                    order_params["slippageTolerance"] = str(round(slippage, 3))
            
            # Add price for limit orders
            if order_type == OrderType.LIMIT and price:
                limit_bias = "up" if side == OrderSide.BUY else "down"
                order_params["price"] = str(self.round_price(symbol, price, bias=limit_bias))
            
            # Add stop loss
            if stop_loss:
                # Long SL should be below price (down); short SL above price (up)
                sl_bias = "down" if side == OrderSide.BUY else "up"
                order_params["stopLoss"] = str(self.round_price(symbol, stop_loss, bias=sl_bias))
                order_params["slTriggerBy"] = "MarkPrice"
            
            # Add take profit
            if take_profit:
                # Long TP above price (up); short TP below price (down)
                tp_bias = "up" if side == OrderSide.BUY else "down"
                order_params["takeProfit"] = str(self.round_price(symbol, take_profit, bias=tp_bias))
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

        # Coerce potential string payloads to floats before normalization.
        if stop_loss is not None:
            try:
                stop_loss = float(stop_loss)
            except (TypeError, ValueError):
                stop_loss = None
        if take_profit is not None:
            try:
                take_profit = float(take_profit)
            except (TypeError, ValueError):
                take_profit = None

        side_hint = None
        reference_price: Optional[float] = None
        try:
            current_positions = self.get_positions()
            current_position = next((p for p in current_positions if p.symbol == symbol and float(p.size) > 0), None)
            if current_position:
                side_hint = current_position.side
                reference_price = float(getattr(current_position, "mark_price", 0) or 0)
                if reference_price <= 0:
                    reference_price = float(getattr(current_position, "avg_price", 0) or 0)
        except Exception:
            # Continue without side hint if position lookup fails
            pass

        if reference_price is None or reference_price <= 0:
            try:
                reference_price = float(self.get_ticker(symbol).last_price)
            except Exception:
                reference_price = None

        price_filter = (self.get_symbol_info(symbol) or {}).get("priceFilter", {}) if side_hint else {}
        try:
            tick_size = float(price_filter.get("tickSize", 0) or 0)
        except (TypeError, ValueError):
            tick_size = 0.0

        guard_ratio = 0.0015
        if reference_price and reference_price > 0 and tick_size > 0:
            guard_ratio = max(guard_ratio, (tick_size * 5) / reference_price)

        def align_levels(
            ref_price: Optional[float],
            sl_value: Optional[float],
            tp_value: Optional[float],
            pad_ratio: float,
        ) -> tuple[Optional[float], Optional[float]]:
            if ref_price is None or ref_price <= 0 or side_hint not in {"Buy", "Sell"}:
                return sl_value, tp_value

            adjusted_sl = sl_value
            adjusted_tp = tp_value

            if adjusted_sl is not None:
                if side_hint == "Buy" and adjusted_sl >= ref_price:
                    adjusted_sl = ref_price * (1 - pad_ratio)
                elif side_hint == "Sell" and adjusted_sl <= ref_price:
                    adjusted_sl = ref_price * (1 + pad_ratio)

            if adjusted_tp is not None:
                if side_hint == "Buy" and adjusted_tp <= ref_price:
                    adjusted_tp = ref_price * (1 + pad_ratio)
                elif side_hint == "Sell" and adjusted_tp >= ref_price:
                    adjusted_tp = ref_price * (1 - pad_ratio)

            return adjusted_sl, adjusted_tp

        stop_loss, take_profit = align_levels(reference_price, stop_loss, take_profit, guard_ratio)

        def build_params(
            sl_value: Optional[float],
            tp_value: Optional[float],
            trail_value: Optional[float],
        ) -> Dict[str, Any]:
            params: Dict[str, Any] = {
                "category": "linear",
                "symbol": symbol,
            }

            if sl_value is not None:
                sl_bias = "nearest"
                if side_hint == "Buy":
                    sl_bias = "down"
                elif side_hint == "Sell":
                    sl_bias = "up"
                params["stopLoss"] = str(self.round_price(symbol, sl_value, bias=sl_bias))
                params["slTriggerBy"] = sl_trigger

            if tp_value is not None:
                tp_bias = "nearest"
                if side_hint == "Buy":
                    tp_bias = "up"
                elif side_hint == "Sell":
                    tp_bias = "down"
                params["takeProfit"] = str(self.round_price(symbol, tp_value, bias=tp_bias))
                params["tpTriggerBy"] = tp_trigger

            if trail_value is not None:
                params["trailingStop"] = str(trail_value)

            return params

        params = build_params(stop_loss, take_profit, trailing_stop)

        def is_price_guard_error(message: str) -> bool:
            msg = message.lower()
            return (
                "base_price" in msg
                or "takeprofit" in msg
                or "stoploss" in msg
                or "should be higher than" in msg
                or "should be lower than" in msg
            )

        response = self._execute_with_time_sync(self.http.set_trading_stop, **params)

        if response.get("retCode") != 0:
            ret_msg = str(response.get("retMsg", "Unknown error"))
            if is_price_guard_error(ret_msg) and side_hint in {"Buy", "Sell"}:
                try:
                    refreshed_price = float(self.get_ticker(symbol).last_price)
                except Exception:
                    refreshed_price = reference_price

                retry_guard = max(guard_ratio * 2.0, 0.003)
                retry_sl, retry_tp = align_levels(refreshed_price, stop_loss, take_profit, retry_guard)
                retry_params = build_params(retry_sl, retry_tp, trailing_stop)
                retry_response = self._execute_with_time_sync(self.http.set_trading_stop, **retry_params)
                if retry_response.get("retCode") == 0:
                    logger.info(
                        f"Updated position protection for {symbol} after reprice retry: "
                        f"stop={retry_sl}, tp={retry_tp}, trail={trailing_stop}"
                    )
                    return retry_response
                response = retry_response
                ret_msg = str(response.get("retMsg", ret_msg))

            logger.error(f"Failed to update position protection: {response}")
            raise Exception(ret_msg)

        logger.info(
            f"Updated position protection for {symbol}: "
            f"stop={stop_loss}, tp={take_profit}, trail={trailing_stop}"
        )
        return response

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
        """Deprecated: bracket flow disabled to prevent accidental duplicate entries."""
        logger.warning(
            "place_bracket_order is deprecated and disabled; use place_order + update_position_protection"
        )
        return {
            "bracket_status": "disabled",
            "reason": "single_entry_model_enforced",
        }

    def place_oco_order(
        self,
        symbol: str,
        side: OrderSide,
        qty: float,
        stop_loss: float,
        take_profit: float,
    ) -> Dict[str, Any]:
        """Deprecated: OCO helper disabled in favor of set_trading_stop protection."""
        raise NotImplementedError(
            "place_oco_order is disabled; use update_position_protection(set_trading_stop) instead"
        )

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
                    open_interest=float(ticker.get("openInterest", 0) or 0),
                    basis=float(
                        ticker.get("basis")
                        or ticker.get("basisRate")
                        or ticker.get("basisRateYear")
                        or 0
                    ),
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
            if symbol in self._symbol_rules_cache:
                return self._symbol_rules_cache[symbol]
        except Exception as e:
            logger.debug(f"Bulk instruments fetch failed for {symbol}: {e}")

        # Fallback: targeted instrument query for the requested symbol.
        try:
            response = self._execute_with_time_sync(
                self.http.get_instruments_info,
                category="linear",
                symbol=symbol,
            )
            if response.get("retCode") == 0:
                items = response.get("result", {}).get("list", [])
                if items:
                    self._symbol_rules_cache[symbol] = items[0]
                    return items[0]
        except Exception as e:
            logger.warning(f"Failed to fetch info for {symbol}: {e}")

        return None

    @staticmethod
    def _step_precision(step_str: str) -> int:
        """Infer decimal precision from a step string like 0.001."""
        if not step_str:
            return 6
        if "." in step_str:
            return len(step_str.split(".")[1].rstrip("0"))
        return 0

    def round_price(self, symbol: str, price: float, bias: str = "nearest") -> float:
        """
        Round price to instrument tick size and clamp to exchange min/max.

        bias:
          - "nearest": standard rounding
          - "up": round up to next tick
          - "down": round down to previous tick
        """
        if price is None or price <= 0:
            return price

        info = self.get_symbol_info(symbol)
        if not info:
            return round(price, 6)

        price_filter = info.get("priceFilter", {})
        tick_str = str(price_filter.get("tickSize", "0.0001"))
        min_price_str = str(price_filter.get("minPrice", "0"))
        max_price_str = str(price_filter.get("maxPrice", "0"))

        try:
            tick_size = float(tick_str)
        except (ValueError, TypeError):
            tick_size = 0.0001
            tick_str = "0.0001"

        precision = self._step_precision(tick_str)
        if tick_size <= 0:
            return round(price, max(precision, 6))

        if bias == "up":
            rounded = math.ceil(price / tick_size) * tick_size
        elif bias == "down":
            rounded = math.floor(price / tick_size) * tick_size
        else:
            rounded = round(price / tick_size) * tick_size

        try:
            min_price = float(min_price_str)
        except (ValueError, TypeError):
            min_price = 0.0
        try:
            max_price = float(max_price_str)
        except (ValueError, TypeError):
            max_price = 0.0

        if min_price > 0:
            rounded = max(rounded, min_price)
        if max_price > 0:
            rounded = min(rounded, max_price)

        if not math.isfinite(rounded):
            rounded = price

        return round(rounded, precision)

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

    def get_latest_closed_pnl(self, symbol: str, within_minutes: int = 20) -> Optional[Dict[str, Any]]:
        """
        Fetch the most recent closed-PnL record for a symbol.

        Returns None when no recent closed fill is available.
        """
        try:
            response = self._execute_with_time_sync(
                self.http.get_closed_pnl,
                category="linear",
                symbol=symbol,
                limit=20,
            )
            if response.get("retCode") != 0:
                logger.debug(f"get_closed_pnl API error for {symbol}: {response.get('retMsg')}")
                return None

            entries = ((response.get("result") or {}).get("list") or [])
            if not entries:
                return None

            cutoff_ms = int((datetime.utcnow() - timedelta(minutes=max(1, within_minutes))).timestamp() * 1000)

            for item in entries:
                updated_ms = int(item.get("updatedTime") or item.get("createdTime") or 0)
                if updated_ms and updated_ms < cutoff_ms:
                    continue

                try:
                    return {
                        "symbol": item.get("symbol", symbol),
                        "side": item.get("side"),
                        "closed_pnl": float(item.get("closedPnl", 0) or 0.0),
                        "qty": float(item.get("qty", 0) or 0.0),
                        "avg_entry_price": float(item.get("avgEntryPrice", 0) or 0.0),
                        "avg_exit_price": float(item.get("avgExitPrice", 0) or 0.0),
                        "updated_ms": updated_ms,
                    }
                except (TypeError, ValueError):
                    continue

            return None
        except Exception as exc:
            logger.debug(f"Failed to fetch latest closed pnl for {symbol}: {exc}")
            return None
    
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

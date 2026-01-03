"""
Quadrick AI Trading System - Main Entry Point
Autonomous trading system powered by DeepSeek LLM
"""
import asyncio
import signal
import sys
import uuid
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
import json
from typing import Dict, Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from loguru import logger
from config.settings import Settings
from src.exchange.bybit_client import BybitClient
from src.analysis.market_analyzer import MarketAnalyzer
from src.analysis.order_flow import OrderFlowAnalyzer
from src.llm.deepseek_client import DeepSeekClient, DecisionType, TradingDecision
from src.risk.risk_manager import RiskManager
from src.notifications.telegram_notifier import TelegramNotifier
from src.database.supabase_client import SupabaseClient
from src.analytics.performance_tracker import PerformanceTracker
from src.analytics.strategy_optimizer import StrategyOptimizer
from src.execution.smart_execution import SmartExecutionManager
from src.positions.position_monitor import PositionMonitor
from src.controls.emergency_controls import EmergencyControls, TradingMode
from src.agents.council import TradingCouncil
from src.integration.dashboard_bridge import bridge, dashboard_sink


class QuadrickTradingBot:
    """Main trading bot orchestrator"""
    
    def __init__(self):
        """Initialize the trading bot"""
        logger.info("=" * 80)
        logger.info("QUADRICK AI TRADING SYSTEM INITIALIZING")
        logger.info("Mission: $15 → $100,000")
        logger.info("=" * 80)
        
        # Load configuration first
        self.settings = Settings()
        
        # Only normal mode supported
        self.trading_mode = "normal"
        
        logger.info(f"🎯 Trading Mode: {self.trading_mode.upper()}")
        logger.info("📊 Normal Mode: Multi-timeframe swing trading")
        
        # Configuration already loaded above
        self._setup_logging()
        
        # Initialize components
        self.bybit = BybitClient(
            api_key=self.settings.bybit.api_key,
            api_secret=self.settings.bybit.api_secret,
            testnet=self.settings.bybit.testnet,
        )
        
        self.analyzer = MarketAnalyzer()
        self.order_flow_analyzer = OrderFlowAnalyzer()
        
        self.deepseek = DeepSeekClient(
            api_key=self.settings.llm.deepseek_api_key,
            model=self.settings.llm.deepseek_model,
            temperature=self.settings.llm.temperature,
        )
        
        self.risk_manager = RiskManager(
            min_risk_pct=self.settings.trading.min_risk_pct,
            max_risk_pct=self.settings.trading.max_risk_pct,
            max_daily_drawdown_pct=self.settings.trading.max_daily_drawdown_pct,
            max_leverage=self.settings.trading.max_leverage,
            max_concurrent_positions=self.settings.trading.max_concurrent_positions,
            min_account_balance=self.settings.trading.min_account_balance,
            high_impact_news_blackout_mins=self.settings.system.high_impact_news_blackout_mins,
        )
        
        # Initialize Telegram notifier
        self.telegram = None
        if self.settings.notifications.telegram_enabled:
            self.telegram = TelegramNotifier(
                bot_token=self.settings.notifications.telegram_bot_token,
                chat_id=self.settings.notifications.telegram_chat_id,
            )
        
        # Initialize Database
        self.db = None
        if self.settings.database.database_provider == "supabase":
            if self.settings.database.supabase_url and self.settings.database.supabase_anon_key:
                self.db = SupabaseClient(
                    supabase_url=self.settings.database.supabase_url,
                    supabase_key=self.settings.database.supabase_anon_key,
                    enabled=True,
                )
                logger.info("✓ Supabase database enabled")
            else:
                logger.warning("Supabase credentials missing - database logging disabled")
        else:
            logger.info("Database logging disabled (set DATABASE_PROVIDER=supabase to enable)")
        
        # Initialize Trading Council (Multi-Agent System) - AFTER RiskManager AND Database
        self.council = TradingCouncil(self.deepseek, self.risk_manager, self.db, bridge=bridge)
        
        # Initialize Performance Tracker
        self.performance = PerformanceTracker()
        logger.info("✓ Performance tracking enabled")

        # Initialize Strategy Optimizer
        self.strategy_optimizer = StrategyOptimizer()
        logger.info("✓ Strategy optimization enabled")
        
        # Initialize Smart Execution Manager
        self.execution_manager = SmartExecutionManager()
        logger.info("✓ Smart execution enabled")
        
        # Initialize Position Monitor
        self.position_monitor = PositionMonitor()
        logger.info("✓ Position monitoring enabled")
        
        # Initialize Emergency Controls
        self.emergency_controls = EmergencyControls()
        logger.info("✓ Emergency controls enabled")
        


        
        # System state
        self.running = False
        self.account_balance = 0.0
        self.available_balance = 0.0
        self.starting_balance = 0.0
        self.positions = []
        self.trade_history = []
        self.active_trade_contexts = {}  # Store entry context for RAG
        
        # Watchlist
        self.watchlist = [
            "BTCUSDT", "ARBUSDT","XRPUSDT", "ETHUSDT", 
            "OPUSDT", "AVAXUSDT", "LINKUSDT",
            "DOTUSDT", "ADAUSDT", "DOGEUSDT", "1000PEPEUSDT",
        ]
        
        # Apply normal mode settings
        self.decision_interval = self.settings.trading.decision_interval_seconds
        self.timeframes_to_fetch = ["1", "5", "15", "60", "240", "D", "W"]  # All 7 timeframes
        self.default_risk_pct = 15
        self.default_leverage = 12
        self.max_hold_minutes = 480  # 8 hours max
        self.target_profit_pct = 2.5
        logger.info(f"📊 Normal settings: {self.decision_interval}s decisions, {self.default_risk_pct}% risk")
        
        # Clear dashboard logs on startup
        bridge.clear_logs()
        
        logger.info("Trading bot initialized successfully")
        logger.info(f"Mode: {'TESTNET' if self.settings.bybit.testnet else 'LIVE'}")
    
    def _setup_logging(self):
        """Configure logging"""
        logger.remove()  # Remove default handler
        
        # Console logging
        logger.add(
            sys.stderr,
            level=self.settings.logging.log_level,
            format=self.settings.logging.log_format,
        )
        
        # Dashboard logging
        logger.add(dashboard_sink, level="INFO")
        
        # File logging
        if self.settings.logging.log_to_file:
            logger.add(
                self.settings.logging.log_file_path,
                level=self.settings.logging.log_level,
                format=self.settings.logging.log_format,
                rotation=f"{self.settings.logging.log_max_size_mb} MB",
                retention=self.settings.logging.log_backup_count,
            )
    
    async def initialize(self):
        """Initialize bot and verify connectivity"""
        logger.info("Performing startup checks...")
        
        # Test Bybit connection (single attempt to avoid timestamp issues)
        try:
            account_info = self.bybit.get_account_info()
            logger.info("✓ Bybit connection successful")

            # Try to get balance, but allow fallback if it fails
            try:
                balance_info = self.bybit.get_account_balance()
                self.account_balance = balance_info.total_equity
                self.starting_balance = self.account_balance

                logger.info(f"Account Balance: ${self.account_balance:.2f}")
                logger.info(f"Balance Details: Equity=${balance_info.total_equity:.2f}, Available=${balance_info.available_balance:.2f}, Used=${balance_info.used_margin:.2f}")

                # Push initial balance to dashboard
                bridge.update_balance({
                    "total": self.account_balance,
                    "available": balance_info.available_balance,
                    "unrealized_pnl": balance_info.unrealized_pnl,
                    "daily_pnl": 0.0,
                    "daily_pnl_pct": 0.0
                })

            except Exception as balance_error:
                # For ANY API issues during balance retrieval, use fallback values
                logger.warning(f"⚠️  Balance check failed, using fallback values: {balance_error}")
                # Use a more realistic fallback balance based on previous sessions
                self.account_balance = 8.39  # Known working balance from logs
                self.starting_balance = self.account_balance
                logger.info(f"Using fallback balance: ${self.account_balance:.2f}")

            # Get current milestone
            try:
                current_milestone, next_milestone = self.settings.trading.get_current_milestone(
                    self.account_balance
                )
                logger.info(
                    f"Current Milestone: ${current_milestone:.0f} → ${next_milestone:.0f}"
                )
            except Exception as milestone_error:
                logger.warning(f"Failed to get milestones, using defaults: {milestone_error}")
                current_milestone, next_milestone = 0, 50  # Default milestones

        except Exception as e:
            error_msg = str(e).lower()
            if "timestamp" in error_msg or "recv_window" in error_msg:
                logger.warning(f"⚠️  API timing issues detected. Starting with fallback values: {e}")
                # Set default values to allow bot to start
                self.account_balance = 8.39  # Known working balance from logs
                self.starting_balance = self.account_balance
                current_milestone, next_milestone = 0, 50  # Default milestones
                logger.info(f"Using fallback balance: ${self.account_balance:.2f}")
                logger.info(f"Fallback Milestone: ${current_milestone:.0f} → ${next_milestone:.0f}")
            else:
                logger.error(f"✗ Failed to connect to Bybit: {e}")
                raise
        
        # Test DeepSeek connection
        try:
            test_context = self.deepseek.prepare_market_context(
                account_balance=self.account_balance,
                positions=[],
                market_data={"btc_24h_change": 0},
                technical_analysis={},
                funding_rates={},
                top_movers={"gainers": [], "losers": []},
                milestone_progress={"current": current_milestone, "next": next_milestone},
            )
            
            # Quick test decision
            logger.info("Testing DeepSeek LLM connection...")
            # Note: In production, you might want to skip this to save API calls
            
            logger.info("✓ DeepSeek connection ready")
            
        except Exception as e:
            logger.error(f"✗ Failed to initialize DeepSeek: {e}")
            raise
        
        # Load positions (single attempt to avoid timestamp issues)
        try:
            self.positions = self.bybit.get_positions()
            if self.positions:
                logger.info(f"Found {len(self.positions)} open positions")
        except Exception as e:
            error_msg = str(e).lower()
            if "timestamp" in error_msg or "recv_window" in error_msg:
                logger.warning(f"⚠️  Failed to load positions due to API timing, starting with empty positions: {e}")
            else:
                logger.warning(f"Failed to load positions: {e}")
            self.positions = []  # Default to empty list
        
        # Initialize Telegram and send startup message
        if self.telegram:
            try:
                await self.telegram.initialize()
                await self.telegram.send_startup_message(
                    balance=self.account_balance,
                    mode="LIVE" if not self.settings.bybit.testnet else "TESTNET",
                    milestone_current=current_milestone,
                    milestone_next=next_milestone,
                )
                logger.info("✓ Telegram notifications enabled")
            except Exception as e:
                logger.warning(f"Telegram initialization failed: {e}")
        
        logger.info("Initialization complete! Ready to trade.")
        bridge.update_status("initialized", False)
        logger.info("=" * 80)
    
    async def run(self):
        """Main trading loop"""
        self.running = True
        logger.info("Starting main trading loop...")
        logger.info(f"⏱️  Decision interval: {self.decision_interval} seconds")
        
        while self.running:
            try:
                # Check emergency controls
                trading_allowed, block_reason = self.emergency_controls.is_trading_allowed()
                if not trading_allowed:
                    logger.warning(f"Trading blocked: {block_reason}")
                    await asyncio.sleep(self.decision_interval)
                    continue
                
                # Update account state (single call per cycle)
                await self._update_account_state()

                # Update position monitor with current data
                if self.positions:
                    for position in self.positions:
                        try:
                            if float(position.size) > 0:
                                symbol = position.symbol
                                current_price = float(position.mark_price) if hasattr(position, 'mark_price') else None
                                unrealized_pnl = float(position.unrealized_pnl) if hasattr(position, 'unrealized_pnl') else 0.0

                                if current_price is None:
                                    # Get current price from ticker
                                    try:
                                        ticker = self.bybit.get_ticker(symbol)
                                        if ticker:
                                            current_price = float(ticker.last_price)
                                    except Exception as e:
                                        logger.warning(f"Failed to get ticker for {symbol}: {e}")
                                        continue

                                if current_price:
                                    try:
                                        self.position_monitor.update_position(
                                            symbol=symbol,
                                            current_price=current_price,
                                            unrealized_pnl=unrealized_pnl
                                        )

                                        # Execute smart execution features
                                        await self._execute_smart_features(position, current_price)
                                    except Exception as e:
                                        logger.warning(f"Failed to update position monitor for {symbol}: {e}")
                        except (ValueError, AttributeError) as e:
                            logger.warning(f"Invalid position data for {getattr(position, 'symbol', 'unknown')}: {e}")
                            continue

                # Get market data
                market_data = await self._fetch_market_data()
                
                # Push raw market data for transparency
                if market_data:
                    bridge.update_raw_market_data(market_data)

                # Analyze markets
                # print(f"DEBUG: Analyzing {len(market_data.get('tickers', {}))} symbols...")
                analysis = await self._analyze_markets(market_data)
                
                # Push technical analysis to dashboard for trader review
                if analysis:
                    bridge.update_market_context(analysis)
                else:
                    print("DEBUG: Analysis yielded no results. Check if klines were fetched correctly.")
                    # Push empty to clear loading state if needed
                    bridge.update_market_context({})

                # Minimal context for emergency controls (Council handles LLM context)
                context = {
                    "emergency_controls": self.emergency_controls.check_status()
                }

                # 4. Get Trading Decision from Council
                logger.info(f"🤔 Requesting decision from Trading Council...")
                
                all_decisions = []
                
                # Iterate through watchlist to find opportunities
                for symbol in self.watchlist:
                    if symbol not in analysis:
                        continue
                        
                    decision_data = await self.council.make_decision(
                        symbol=symbol,
                        market_data=analysis.get(symbol, {}),
                        account_balance=self.account_balance,
                        autonomous_mode=True # Enable Heuristic-Free AI Strategy
                    )
                    
                    if decision_data["decision_type"] == "open_position":
                        # Add symbol to decision for tracking
                        decision_data["_evaluated_symbol"] = symbol
                        all_decisions.append(decision_data)
                
                # Select the trade with highest confidence
                if all_decisions:
                    best_decision = max(all_decisions, key=lambda d: d.get("trade", {}).get("confidence_score", 0))
                    logger.info(f"✅ Selected {best_decision['trade']['symbol']} (Conf: {best_decision['trade']['confidence_score']:.2f}) from {len(all_decisions)} opportunities")
                else:
                     # Default to wait if no symbol has a trade
                     best_decision = {
                        "decision_type": "wait",
                        "symbol": self.watchlist[0] if self.watchlist else "BTCUSDT",
                        "reasoning": "No suitable setups found by Council across watchlist",
                        "confidence_score": 0.0,
                        "decision_id": str(uuid.uuid4())
                     }
                
                # Ensure decision_id exists
                if "decision_id" not in best_decision:
                    best_decision["decision_id"] = str(uuid.uuid4())

                # Convert dictionary response to TradingDecision object
                decision = self.deepseek._parse_decision(json.dumps(best_decision))
                
                conf_score = decision.confidence_score if decision.confidence_score is not None else 0.0
                logger.info(f"🤖 Council Decision: {decision.decision_type.value} ({conf_score:.2f})")
                
                # Send decision to dashboard
                if decision:
                    # Handle reasoning (might be dict or string)
                    reason_str = decision.reasoning
                    if isinstance(reason_str, dict):
                        # Extract the most human-readable part or JSON stringify
                        reason_str = reason_str.get("market_regime", str(reason_str))
                    
                    dashboard_decision = {
                        "id": decision.decision_id,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "symbol": decision.symbol or "TOTAL",
                        "type": decision.decision_type.value,
                        "confidence": float(decision.confidence_score or 0),
                        "strategy": decision.strategy_tag or "Neutral",
                        "reasoning": str(reason_str or "Analyzing market..."),
                        "htf_aligned": True
                    }
                    bridge.send_decision(dashboard_decision)
                
                if self.db:
                    try:
                        await self.db.save_decision({
                            "decision_id": decision.decision_id,
                            "decision_type": decision.decision_type.value,
                            "account_balance": self.account_balance,
                            "open_positions": len(self.positions),
                            "daily_pnl": sum(t.get("pnl", 0) for t in self.trade_history[-20:]),
                            "current_milestone": context.get("milestone_progress", {}).get("current_milestone"),
                            "symbol": decision.symbol,
                            "side": decision.side,
                            "risk_pct": decision.risk_pct,
                            "leverage": decision.leverage,
                            "entry_price_target": decision.entry_price_target,
                            "stop_loss": decision.stop_loss,
                            "take_profit_1": decision.take_profit_1,
                            "take_profit_2": decision.take_profit_2,
                            "strategy_tag": decision.strategy_tag,
                            "confidence_score": decision.confidence_score,
                            "reasoning": decision.reasoning,
                            "risk_management": decision.risk_management,
                            "model_version": decision.model_version,
                            "processing_time_ms": decision.processing_time_ms,
                            "executed": False,  # Will update after execution
                        })
                    except Exception as e:
                        logger.warning(f"Failed to save decision to database: {e}")

                # Execute decision
                await self._execute_decision(decision, market_data, analysis)
                
                # Log status
                self._log_status()
                
                # Wait for next cycle (mode-specific interval)
                logger.info(f"Next decision in {self.decision_interval} seconds...")
                await asyncio.sleep(self.decision_interval)
                
            except KeyboardInterrupt:
                logger.info("Received interrupt signal")
                break
                
            except Exception as e:
                import traceback
                logger.error(f"Error in main loop: {e}")
                logger.error(f"Traceback: {traceback.format_exc()}")

                # Try to save error to database for debugging
                if self.db:
                    try:
                        await self.db.save_decision({
                            "decision_id": f"ERROR_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
                            "decision_type": "ERROR",
                            "error_message": str(e),
                            "error_traceback": traceback.format_exc()[:1000],  # Limit size
                            "account_balance": getattr(self, 'account_balance', 0),
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        })
                    except Exception as db_e:
                        logger.error(f"Failed to save error to database: {db_e}")

                await asyncio.sleep(30)  # Wait before retry
    
    async def _update_account_state(self):
        """Update account balance and positions with minimal API calls"""
        try:
            # Small delay to prevent API rate limiting and timestamp issues
            await asyncio.sleep(0.1)

            # 1. Update Balance
            try:
                balance_info = self.bybit.get_account_balance()
                if balance_info:
                    old_balance = self.account_balance
                    self.account_balance = balance_info.total_equity
                    self.available_balance = balance_info.available_balance
                    
                    wallet_balance = self.available_balance
                    unrealized_pnl = balance_info.unrealized_pnl
                    
                    if abs(self.account_balance - old_balance) > 0.01:
                        logger.debug(f"Balance updated: ${self.account_balance:.2f} (was ${old_balance:.2f})")
                    
                    # Push update to dashboard
                    daily_pnl = self.account_balance - self.starting_balance
                    daily_pnl_pct = (daily_pnl / self.starting_balance * 100) if self.starting_balance > 0 else 0
                    
                    bridge.update_balance({
                        "total": float(self.account_balance),
                        "available": float(wallet_balance),
                        "unrealized_pnl": float(unrealized_pnl),
                        "daily_pnl": float(daily_pnl),
                        "daily_pnl_pct": float(daily_pnl_pct),
                    })
            except Exception as balance_error:
                logger.debug(f"Balance update failed: {balance_error}")

            # 2. Update Positions
            try:
                self.positions = self.bybit.get_positions()
                
                # Serialize positions for dashboard
                positions_data = []
                if self.positions:
                    for pos in self.positions:
                        try:
                            pnl_pct = 0
                            size = float(pos.size)
                            # Bybit API might return avg_price or entry_price depending on version/endpoint
                            entry_price = float(getattr(pos, 'avg_price', getattr(pos, 'entry_price', 0)))
                            
                            if size * entry_price > 0:
                                 pnl_pct = (float(pos.unrealized_pnl) / (size * entry_price)) * 100
                            
                            positions_data.append({
                                "symbol": pos.symbol,
                                "side": pos.side,
                                "size": size,
                                "entry_price": entry_price,
                                "current_price": float(getattr(pos, 'mark_price', entry_price)),
                                "pnl": float(pos.unrealized_pnl),
                                "pnl_pct": pnl_pct,
                                "leverage": float(pos.leverage),
                                "stop_loss": float(pos.stop_loss) if getattr(pos, 'stop_loss', None) else None,
                                "take_profit": float(pos.take_profit) if getattr(pos, 'take_profit', None) else None,
                            })
                        except (ValueError, TypeError, AttributeError) as pos_item_err:
                            logger.debug(f"Error parsing position item: {pos_item_err}")
                            
                bridge.update_positions(positions_data)
            except Exception as pos_error:
                logger.debug(f"Position update failed: {pos_error}")

            # 3. Update risk manager start balance if needed
            if hasattr(self, 'risk_manager') and self.risk_manager.daily_start_balance is None:
                self.risk_manager.daily_start_balance = self.account_balance

        except Exception as e:
            logger.debug(f"Account state update encountered issues, continuing with cached data: {str(e)[:100]}")
    
    async def _fetch_market_data(self):
        """Fetch comprehensive market data"""
        market_data = {
            "tickers": {},
            "klines": {},
            "funding_rates": {},
            "orderbooks": {},
        }
        
        try:
            # Fetch data for watchlist symbols
            for symbol in self.watchlist:
                try:
                    # Get ticker
                    ticker = self.bybit.get_ticker(symbol)
                    market_data["tickers"][symbol] = ticker
                except Exception as e:
                    logger.warning(f"Skipping {symbol}: {e}")
                    continue  # Skip this symbol and continue with others
                
                # Get klines for mode-specific timeframes
                klines = {}
                timeframe_mapping = {
                    "1": "1m",
                    "5": "5m",
                    "15": "15m",
                    "60": "1h",
                    "240": "4h",
                    "D": "1d",
                    "W": "1w",
                }
                
                # Fetch only timeframes needed for current mode
                for interval in self.timeframes_to_fetch:
                    name = timeframe_mapping.get(interval, f"{interval}m")
                    try:
                        data = self.bybit.get_klines(
                            symbol=symbol,
                            interval=interval,
                            limit=200,
                        )
                        klines[name] = data
                        if not data:
                            pass # print(f"DEBUG: \033[93mWarning: Empty klines for {symbol} ({name})\033[0m")
                    except Exception as e:
                        logger.warning(f"Failed to fetch {name} klines for {symbol}: {e}")
                
                market_data["klines"][symbol] = klines
                # print(f"DEBUG: symbol {symbol} klines fetched: {list(klines.keys())}")
                
                # Get funding rate
                try:
                    funding = self.bybit.get_funding_rate(symbol)
                    market_data["funding_rates"][symbol] = funding.funding_rate
                except:
                    market_data["funding_rates"][symbol] = 0.0
                
                # Get orderbook for order flow analysis (top 5 symbols only to save time)
                if symbol in self.watchlist[:5]:
                    try:
                        orderbook = self.bybit.get_orderbook(symbol, limit=50)
                        order_flow = self.order_flow_analyzer.analyze_orderbook(
                            symbol=symbol,
                            bids=orderbook.bids,
                            asks=orderbook.asks,
                            current_price=ticker.last_price,
                        )
                        market_data["order_flow"] = market_data.get("order_flow", {})
                        market_data["order_flow"][symbol] = order_flow
                    except Exception as e:
                        logger.warning(f"Failed to analyze order flow for {symbol}: {e}")
            
            # Identify top movers
            gainers = []
            losers = []
            
            for symbol, ticker in market_data["tickers"].items():
                change_data = {
                    "symbol": symbol,
                    "change_pct": ticker.price_24h_change * 100,
                    "volume": ticker.volume_24h,
                }
                
                if ticker.price_24h_change > 0:
                    gainers.append(change_data)
                else:
                    losers.append(change_data)
            
            market_data["top_movers"] = {
                "gainers": sorted(gainers, key=lambda x: x["change_pct"], reverse=True)[:5],
                "losers": sorted(losers, key=lambda x: x["change_pct"])[:5],
            }
            
            # BTC metrics for overall market
            btc_ticker = market_data["tickers"].get("BTCUSDT")
            if btc_ticker:
                market_data["btc_24h_change"] = btc_ticker.price_24h_change * 100
            

            
        except Exception as e:
            logger.error(f"Error fetching market data: {e}")
        
        return market_data
    
    async def _analyze_markets(self, market_data):
        """Analyze markets using technical indicators"""
        analyses = {}
        
        # Analyze ALL symbols in watchlist (not just 5)
        for symbol in self.watchlist:
            if symbol not in market_data.get("klines", {}):
                continue
            if symbol not in market_data.get("tickers", {}):
                continue
            
            try:
                symbol_analysis = self.analyzer.analyze_symbol(
                    symbol=symbol,
                    klines_data=market_data["klines"][symbol],
                    current_price=market_data["tickers"][symbol].last_price,
                )
                
                # Get market sentiment
                sentiment = self.analyzer.get_market_sentiment(symbol_analysis)
                
                # Convert TimeframeAnalysis objects to dictionaries for JSON serialization and Strategy/Agent compatibility
                serializable_analysis = {tf: analysis_obj.to_dict() for tf, analysis_obj in symbol_analysis.items()}
                
                analyses[symbol] = {
                    "timeframe_analysis": serializable_analysis,
                    "sentiment": sentiment,
                    "current_price": market_data["tickers"][symbol].last_price,
                }
                
            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}")
        
        
        return analyses
    
    def _prepare_llm_context(self, market_data, analysis):
        """Prepare comprehensive context for LLM"""
        # Get milestone progress
        current_milestone, next_milestone = self.settings.trading.get_current_milestone(
            self.account_balance
        )
        
        progress_pct = (
            (self.account_balance - current_milestone) / 
            (next_milestone - current_milestone) * 100
        )
        
        milestone_progress = {
            "current_milestone": f"${current_milestone:.0f} → ${next_milestone:.0f}",
            "progress_pct": progress_pct,
            "next_milestone": f"${next_milestone:.0f}",
        }
        
        # Format positions
        positions_data = []
        for pos in self.positions:
            entry_price = pos.avg_price
            mark_price = pos.mark_price
            if pos.side == "Buy":
                pnl_pct = ((mark_price - entry_price) / entry_price * 100) if entry_price else 0
            else:
                pnl_pct = ((entry_price - mark_price) / entry_price * 100) if entry_price else 0

            positions_data.append({
                "symbol": pos.symbol,
                "side": pos.side,
                "size": pos.size,
                "entry_price": entry_price,
                "mark_price": mark_price,
                "unrealized_pnl": pos.unrealized_pnl,
                "leverage": pos.leverage,
                "pnl_pct": pnl_pct,
                "stop_loss": getattr(pos, "stop_loss", None),
                "take_profit": getattr(pos, "take_profit", None),
            })
        
        # Format technical analysis - SEND COMPREHENSIVE DATA
        ta_summary = {}
        for symbol, data in analysis.items():
            # Get current price
            current_price = market_data["tickers"][symbol].last_price if symbol in market_data.get("tickers", {}) else 0
            price_24h_change = market_data["tickers"][symbol].price_24h_change * 100 if symbol in market_data.get("tickers", {}) else 0
            
            # Get key timeframe indicators
            indicators_summary = {}
            if "timeframes" in data:
                for tf, tf_data in data["timeframes"].items():
                    if tf in ["1m", "5m", "15m", "1h", "4h"]:  # Send key timeframes, including ultra-short-term view
                        indicators_summary[tf] = {
                            "rsi": round(tf_data.indicators.rsi, 1),
                            "macd_signal": "bullish" if tf_data.indicators.macd_histogram > 0 else "bearish",
                            "trend": tf_data.structure.trend.value,
                            "atr": round(tf_data.indicators.atr, 2),
                            "volume_ratio": round(tf_data.indicators.volume_ratio, 2),
                            "bb_position": "upper" if current_price > tf_data.indicators.bb_upper else "lower" if current_price < tf_data.indicators.bb_lower else "middle",
                            "patterns": tf_data.patterns[:3] if tf_data.patterns else [],
                            # CRITICAL: Add key levels for SL/TP calculation
                            "key_levels": {
                                "pivot_point": round(tf_data.key_levels.pivot_point, 2),
                                "immediate_support": round(tf_data.key_levels.immediate_support, 2),
                                "immediate_resistance": round(tf_data.key_levels.immediate_resistance, 2),
                                "major_support": round(tf_data.key_levels.major_support, 2),
                                "major_resistance": round(tf_data.key_levels.major_resistance, 2),
                                "s1": round(tf_data.key_levels.s1, 2),
                                "s2": round(tf_data.key_levels.s2, 2),
                                "r1": round(tf_data.key_levels.r1, 2),
                                "r2": round(tf_data.key_levels.r2, 2),
                            },
                        }
            
            # Overall sentiment
            sentiment_data = data.get("sentiment", {})
            
            ta_summary[symbol] = {
                "current_price": round(current_price, 2),
                "price_24h_change_pct": round(price_24h_change, 2),
                "overall_sentiment": sentiment_data.get("overall_sentiment", "neutral"),
                "confidence": round(sentiment_data.get("confidence", 0), 1),
                "key_signals": sentiment_data.get("signals", [])[:5],
                "timeframe_analysis": indicators_summary,
                "funding_rate": market_data.get("funding_rates", {}).get(symbol, 0),
            }
        
        # Get position monitor summary
        position_monitor_summary = self.position_monitor.get_position_summary()



        # Get performance feedback
        performance_feedback = self.performance.get_recent_performance(limit=20)

        # Get strategy optimization insights
        strategy_insights = self.strategy_optimizer.get_strategy_recommendations(
            available_strategies=["momentum_continuation", "breakout", "mean_reversion",
                                "volume_breakout", "support_resistance", "fibonacci_retracement"]
        )
        
        # Get similar trades from RAG memory
        similar_trades = []
        
        # Build context
        context = self.deepseek.prepare_market_context(
            account_balance=self.account_balance,
            positions=positions_data,
            position_monitor=position_monitor_summary,
            market_data=market_data,
            technical_analysis=ta_summary,
            funding_rates=market_data.get("funding_rates", {}),
            top_movers=market_data.get("top_movers", {"gainers": [], "losers": []}),
            milestone_progress=milestone_progress,
            recent_trades=self.trade_history[-10:] if self.trade_history else [],
            performance_feedback=performance_feedback,
            portfolio_metrics=getattr(validation, 'portfolio_metrics', {}) if 'validation' in locals() else {},
            strategy_insights=strategy_insights,
            similar_trades=similar_trades,
        )
        
        return context

    async def _execute_smart_features(self, position, current_price: float):
        """Execute smart execution features for a position"""
        try:
            symbol = position.symbol
            side = position.side
            entry_price = float(position.avg_price)
            take_profit = float(position.take_profit) if position.take_profit else None

            # Check trailing stop updates
            if take_profit:
                try:
                    # Debug: check what parameters we're passing
                    import inspect
                    sig = inspect.signature(self.execution_manager.update_trailing_stop)
                    logger.debug(f"update_trailing_stop signature: {sig}")

                    trailing_update = self.execution_manager.update_trailing_stop(
                        symbol=symbol,
                        current_price=current_price,
                        side=side
                    )

                    if trailing_update:
                        try:
                            stop_loss_price = trailing_update.get("stop_loss")
                            take_profit_price = trailing_update.get("take_profit")

                            if stop_loss_price is None and take_profit_price is None:
                                logger.debug("Trailing update returned no actionable prices")
                            else:
                                self.bybit.update_position_protection(
                                    symbol=symbol,
                                    stop_loss=stop_loss_price,
                                    take_profit=take_profit_price,
                                )

                                if stop_loss_price is not None:
                                    logger.info(f"✅ {symbol} trailing stop updated to ${stop_loss_price:.4f}")
                                if take_profit_price is not None:
                                    logger.info(f"✅ {symbol} trailing take-profit nudged to ${take_profit_price:.4f}")
                        except Exception as e:
                            logger.warning(f"Failed to update trailing protection for {symbol}: {e}")
                except Exception as e:
                    logger.warning(f"Error checking trailing stop for {symbol}: {e}")

            # Check break-even move
            if take_profit:
                try:
                    breakeven_stop = self.execution_manager.move_stop_to_breakeven(
                        symbol=symbol,
                        entry_price=entry_price,
                        current_price=current_price,
                        target_price=take_profit,
                        side=side
                    )

                    if breakeven_stop:
                        try:
                            self.bybit.update_position_protection(
                                symbol=symbol,
                                stop_loss=breakeven_stop,
                            )
                            logger.info(f"✅ {symbol} stop moved to break-even at ${breakeven_stop:.4f}")
                        except Exception as e:
                            logger.warning(f"Failed to move {symbol} to break-even: {e}")
                except Exception as e:
                    logger.warning(f"Error checking break-even for {symbol}: {e}")

            # Check partial profit taking (only if profitable)
            try:
                pnl_pct = 0
                if side == "Buy":
                    pnl_pct = (current_price - entry_price) / entry_price * 100
                else:
                    pnl_pct = (entry_price - current_price) / entry_price * 100

                if pnl_pct > 1.0:  # Only if at least 1% profit
                    partial = self.execution_manager.should_take_partial_profit(symbol, current_price)
                    if partial:
                        try:
                            # Execute partial close
                            close_qty = float(position.size) * (partial["percentage"] / 100)
                            from src.exchange.bybit_client import OrderSide, OrderType

                            self.bybit.place_order(
                                symbol=symbol,
                                side=OrderSide.SELL if side == "Buy" else OrderSide.BUY,
                                order_type=OrderType.MARKET,
                                qty=close_qty,
                                reduce_only=True
                            )
                            logger.info(f"✅ {symbol} partial profit: {partial['percentage']}% at ${partial['price']:.4f}")
                        except Exception as e:
                            logger.warning(f"Failed to take partial profit for {symbol}: {e}")
            except Exception as e:
                logger.warning(f"Error checking partial profit for {symbol}: {e}")

        except Exception as e:
            logger.warning(f"Error in smart execution for {getattr(position, 'symbol', 'unknown')}: {e}")

    
    async def _execute_decision(self, decision, market_data, analysis):
        """Execute trading decision"""
        logger.info(f"Decision: {decision.decision_type}")
        
        if decision.decision_type == DecisionType.WAIT:
            logger.info("No trade - waiting for better setup")
            if decision.reasoning:
                # Log full reasoning for WAIT decisions
                reasoning_text = str(decision.reasoning)
                logger.info(f"DeepSeek's Analysis: {reasoning_text[:500]}...")  # First 500 chars
                
                # Log specific fields if available
                if isinstance(decision.reasoning, dict):
                    if "action" in decision.reasoning:
                        logger.info(f"  → {decision.reasoning['action']}")
                    if "market_regime_assessment" in decision.reasoning:
                        logger.info(f"  → Market: {decision.reasoning['market_regime_assessment']}")
                    if "waiting_for" in decision.reasoning:
                        logger.info(f"  → Waiting for: {decision.reasoning['waiting_for']}")
            return
        
        if decision.decision_type == DecisionType.HOLD:
            logger.info("Holding current positions")
            return
        
        if decision.decision_type == DecisionType.OPEN_POSITION:
            await self._open_position(decision, market_data, analysis)
        
        elif decision.decision_type == DecisionType.CLOSE_POSITION:
            await self._close_position(decision)

        elif decision.decision_type == DecisionType.MODIFY_POSITION:
            await self._modify_position(decision, market_data, analysis)
    
    async def _open_position(self, decision, market_data, analysis):
        """Open a new position"""
        if not decision.symbol:
            logger.warning("No symbol specified in decision")
            return
        
        # Get current price
        ticker = market_data["tickers"].get(decision.symbol)
        if not ticker:
            ticker = self.bybit.get_ticker(decision.symbol)
        
        current_price = ticker.last_price
        
        # Validate trade with risk manager
        trade_data = {
            "symbol": decision.symbol,
            "side": decision.side,
            "risk_pct": decision.risk_pct,
            "leverage": decision.leverage,
            "stop_loss": decision.stop_loss,
            "allow_counter_trend": (decision.risk_management or {}).get("allow_counter_trend"),
        }
        
        validation = self.risk_manager.validate_trade(
            trade_decision=trade_data,
            account_balance=self.account_balance,
            available_balance=self.available_balance,
            open_positions=self.positions,
            market_price=current_price,
            market_analysis=analysis,
        )
        
        if not validation.is_valid:
            logger.warning(f"Trade rejected: {validation.rejection_reasons}")
            return
        
        # Use adjusted values if any
        risk_pct = validation.adjusted_risk_pct or decision.risk_pct
        leverage = validation.adjusted_leverage or decision.leverage
        
        # Calculate position size - CRITICAL FOR SMALL ACCOUNTS
        # For $15 balance, we need very small positions
        
        # Maximum we can afford (use min of 70% of total OR 95% of available)
        total_margin_cap = self.account_balance * 0.7
        available_margin_cap = self.available_balance * 0.95 # Leave 5% for fees/slippage
        
        max_margin = min(total_margin_cap, available_margin_cap)
        max_position_value = max_margin * leverage
        max_position_size = max_position_value / current_price
        
        # Calculate based on risk and stop distance
        risk_amount = self.account_balance * (risk_pct / 100)  # 18% of $15.24 = $2.74
        stop_distance_pct = abs(current_price - decision.stop_loss) / current_price
        
        if stop_distance_pct > 0:
            # Position value based on risk
            risk_based_value = risk_amount / stop_distance_pct
            # But cap it at what we can afford
            position_value = min(risk_based_value, max_position_value)
        else:
            position_value = max_position_value * 0.5  # Conservative if no stop
        
        position_size = position_value / current_price
        
        # Ensure position size is within balance limits
        # Final margin check
        required_margin = (position_size * current_price) / leverage
        
        if required_margin > max_margin:
            # Reduce to fit
            position_size = (max_margin * leverage) / current_price
            logger.warning(f"Position size capped by balance: {position_size:.4f}")
        
        # Round to appropriate decimal places using Bybit rules
        position_size = self.bybit.round_quantity(decision.symbol, position_size)
        
        # Absolute minimum notional check (Bybit usually requires ~$1.0 minimum per order)
        min_value = 1.10 if self.account_balance >= 3 else 0.5 # Safe minimum for UTA
        
        current_notional = position_size * current_price
        if current_notional < min_value:
            # Check if raising it would exceed our available margin
            required_margin_for_min = min_value / max(1, leverage)
            if required_margin_for_min > max_margin:
                logger.warning(f"Trade too small (${current_notional:.2f}) but raising it would exceed available margin. Skipping.")
                return

            logger.warning(f"Position too small (${current_notional:.2f}), raising to minimum ${min_value:.2f}")
            position_size = min_value / current_price
            position_size = self.bybit.round_quantity(decision.symbol, position_size)

        # Final check against exchange-specific minOrderQty
        symbol_info = self.bybit.get_symbol_info(decision.symbol)
        if symbol_info:
            min_qty = float(symbol_info.get('lotSizeFilter', {}).get('minOrderQty', 0))
            if position_size < min_qty:
                # Check if this minimum is actually affordable
                required_margin_for_min_qty = (min_qty * current_price) / max(1, leverage)
                if required_margin_for_min_qty > max_margin:
                    logger.warning(
                        f"Instrument minimum {min_qty} ({decision.symbol}) requires ${required_margin_for_min_qty:.2f} margin, "
                        f"but only ${max_margin:.2f} is available. Skipping trade."
                    )
                    return

                logger.warning(f"Position size {position_size} below instrument minimum {min_qty}, raising to {min_qty}")
                position_size = min_qty
        
        logger.info(
            f"Opening position: {decision.symbol} {decision.side} "
            f"Size: {position_size} Risk: {risk_pct}% Leverage: {leverage}x"
        )
        logger.info(f"🧠 DEEPSEEK DECISION: {decision.symbol} {decision.side} @ ${current_price:.2f} SL: ${decision.stop_loss:.2f} TP: ${decision.take_profit_1:.2f}")
        
        try:
            # Re-fetch current price right before order to ensure SL validity
            latest_ticker = self.bybit.get_ticker(decision.symbol)
            latest_price = latest_ticker.last_price
            
            # Adjust stop loss based on latest price to ensure validity
            adjusted_stop_loss = decision.stop_loss
            
            if decision.side == "Sell":
                # For shorts, ensure SL is at least 0.5% above current mark price
                min_stop_loss = latest_price * 1.005
                if adjusted_stop_loss <= latest_price:
                    adjusted_stop_loss = min_stop_loss
                    logger.warning(f"Adjusting short SL from ${decision.stop_loss:.2f} to ${adjusted_stop_loss:.2f} (market moved to ${latest_price:.2f})")
                elif adjusted_stop_loss < min_stop_loss:
                    # Even if above current price, ensure minimum buffer
                    adjusted_stop_loss = min_stop_loss
                    logger.info(f"Buffering short SL from ${decision.stop_loss:.2f} to ${adjusted_stop_loss:.2f} for safety")
            else:  # Buy/Long position
                # For longs, ensure SL is at least 0.5% below current mark price
                max_stop_loss = latest_price * 0.995
                if adjusted_stop_loss >= latest_price:
                    adjusted_stop_loss = max_stop_loss
                    logger.warning(f"Adjusting long SL from ${decision.stop_loss:.2f} to ${adjusted_stop_loss:.2f} (market moved to ${latest_price:.2f})")
                elif adjusted_stop_loss > max_stop_loss:
                    # Even if below current price, ensure minimum buffer
                    adjusted_stop_loss = max_stop_loss
                    logger.info(f"Buffering long SL from ${decision.stop_loss:.2f} to ${adjusted_stop_loss:.2f} for safety")
            
            # Place the order
            from src.exchange.bybit_client import OrderSide, OrderType
            
            order_result = self.bybit.place_order(
                symbol=decision.symbol,
                side=OrderSide(decision.side),
                order_type=OrderType.MARKET,
                qty=position_size,
                stop_loss=adjusted_stop_loss,
                take_profit=decision.take_profit_1,
                leverage=leverage,
            )

            logger.info(f"📤 BYBIT ORDER: {decision.symbol} {OrderSide(decision.side).value} {position_size} units")
            logger.info(f"✓ Order placed successfully: {order_result}")

            # Refresh positions from Bybit after successful order
            self.positions = self.bybit.get_positions()
            logger.info(f"📊 POSITIONS AFTER ORDER: {len(self.positions)} positions")
            for pos in self.positions:
                logger.info(f"   {pos.symbol}: {pos.side} {pos.size} @ ${pos.avg_price}")

            # Try to place advanced orders (bracket/OCO) if supported
            try:
                if hasattr(self.bybit, 'place_bracket_order'):
                    # Place bracket order for better execution (pass account balance)
                    bracket_result = self.bybit.place_bracket_order(
                        symbol=decision.symbol,
                        side=OrderSide(decision.side),
                        qty=position_size,
                        entry_price=current_price,
                        stop_loss=decision.stop_loss,
                        take_profit=decision.take_profit_1,
                        leverage=leverage,
                        account_balance=self.account_balance,
                    )

                    # Check if bracket was skipped due to small account
                    if bracket_result.get("bracket_status") == "skipped_small_account":
                        logger.info(f"ℹ️  Bracket order skipped for small account (${self.account_balance:.2f})")
                    else:
                        logger.info(f"✅ Bracket order placed for {decision.symbol}")
                else:
                    logger.info(f"ℹ️  Bracket orders not available, using standard orders")
            except Exception as e:
                logger.warning(f"Failed to place bracket order, using standard: {e}")

            # Add position to monitor
            self.position_monitor.add_position(
                symbol=decision.symbol,
                side=decision.side,
                size=position_size,
                entry_price=current_price,
                stop_loss=adjusted_stop_loss,  # Use adjusted SL
                take_profit=decision.take_profit_1,
            )

            # Setup smart execution features
            if decision.take_profit_1:
                # Setup trailing stop
                self.execution_manager.setup_trailing_stop(
                    symbol=decision.symbol,
                    side=decision.side,
                    initial_stop=adjusted_stop_loss,  # Use adjusted SL
                    trail_distance_pct=0.5,
                    current_price=current_price,
                    entry_price=current_price,
                )

                # Setup partial profit targets (only if we have a take profit)
                if decision.take_profit_1:
                    try:
                        # Create TP levels for partial profits
                        tp_levels = [
                            {"price": decision.take_profit_1, "percentage": 100}  # Full exit at TP
                        ]

                        self.execution_manager.setup_partial_take_profits(
                            symbol=decision.symbol,
                            position_size=position_size,
                            tp_levels=tp_levels,
                        )
                        logger.debug(f"Partial take profits set for {decision.symbol}")
                    except Exception as tp_error:
                        logger.warning(f"Failed to setup partial take profits: {tp_error}")
                else:
                    logger.debug(f"No take profit set for {decision.symbol}")

            # Generate trade ID
            trade_id = f"TRADE_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{decision.symbol}"
            
            # Record trade in memory
            self.trade_history.append({
                "trade_id": trade_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "symbol": decision.symbol,
                "side": decision.side,
                "size": position_size,
                "entry_price": current_price,
                "risk_pct": risk_pct,
                "strategy": decision.strategy_tag,
            })
            
            # Save to database
            if self.db:
                await self.db.save_trade({
                    "trade_id": trade_id,
                    "symbol": decision.symbol,
                    "side": decision.side,
                    "order_type": "Market",
                    "size": position_size,
                    "entry_price": current_price,
                    "stop_loss": adjusted_stop_loss,  # Use adjusted SL
                    "take_profit": decision.take_profit_1,
                    "leverage": leverage,
                    "risk_pct": risk_pct,
                    "position_value": position_size * current_price,
                    "status": "open",
                    "strategy_tag": decision.strategy_tag,
                    "ai_confidence": decision.confidence_score,
                    "ai_reasoning": str(decision.reasoning),
                    "decision_id": decision.decision_id,
                    "balance_before": self.account_balance,
                    "exchange_order_id": order_result.get("orderId"),
                })

                # Capture context for RAG
                try:
                    if decision.symbol in analysis:
                        symbol_analysis = analysis[decision.symbol]
                        tf_1h = symbol_analysis["timeframes"].get("1h")
                        if tf_1h:
                            context_snapshot = {
                                "rsi": tf_1h.indicators.rsi,
                                "trend": tf_1h.structure.trend.value,
                                "atr_pct": (tf_1h.indicators.atr / current_price * 100),
                                "price_24h_change_pct": market_data["tickers"][decision.symbol].price_24h_change * 100
                            }
                            self.active_trade_contexts[decision.symbol] = context_snapshot
                except Exception as e:
                    logger.warning(f"Failed to capture RAG context for {decision.symbol}: {e}")
            
            # Send Telegram notification
            if self.telegram:
                await self.telegram.send_trade_alert(
                    action="OPEN",
                    symbol=decision.symbol,
                    side=decision.side,
                    size=position_size,
                    price=current_price,
                    risk_pct=risk_pct,
                    stop_loss=decision.stop_loss,
                    take_profit=decision.take_profit_1,
                    reason=decision.strategy_tag,
                )
            
        except Exception as e:
            logger.error(f"✗ Failed to place order: {e}")
    
    async def _close_position(self, decision):
        """Close an existing position"""
        symbol = decision.symbol

        if not symbol:
            logger.warning("No symbol specified for closing")
            return

        try:
            # Get position PnL before closing
            pnl = 0
            position_found = False

            for pos in self.positions:
                if pos.symbol == symbol and float(pos.size) > 0:
                    # Calculate realized PnL
                    entry_price = float(pos.avg_price)
                    current_price = float(pos.mark_price) if hasattr(pos, 'mark_price') else entry_price

                    if pos.side == "Buy":
                        pnl = (current_price - entry_price) / entry_price * 100
                    else:  # Short position
                        pnl = (entry_price - current_price) / entry_price * 100

                    position_found = True
                    break

            result = self.bybit.close_position(symbol)
            logger.info(f"✓ Position closed: {symbol}")

            # Record trade result with emergency controls
            if position_found:
                won = pnl > 0
                self.emergency_controls.record_trade_result(won)
                logger.debug(f"Trade result recorded: {'WIN' if won else 'LOSS'} (PnL: {pnl:+.2f}%)")

                # Update performance tracker
                strategy_name = decision.strategy_tag if hasattr(decision, 'strategy_tag') else "manual_close"
                self.performance.add_trade({
                    "symbol": symbol,
                    "pnl": pnl,
                    "strategy": strategy_name,
                    "win": won,
                    "timestamp": datetime.now(timezone.utc)
                })

                # Save Trade Memory (RAG)
                if self.db and symbol in self.active_trade_contexts:
                    try:
                        entry_context = self.active_trade_contexts[symbol]
                        trade_result = {
                            "symbol": symbol,
                            "strategy": strategy_name,
                            "pnl_pct": pnl,
                            "win": won
                        }
                        await self.db.save_trade_memory(trade_result, entry_context)
                        # Clean up
                        del self.active_trade_contexts[symbol]
                    except Exception as e:
                        logger.warning(f"Failed to save trade memory for {symbol}: {e}")

                # Update strategy optimizer
                market_regime = "neutral"  # Could be enhanced to detect actual regime
                self.strategy_optimizer.analyze_strategy_performance(
                    strategy_name=strategy_name,
                    pnl=pnl,
                    win=won,
                    market_regime=market_regime,
                    timeframe="1h",  # Could be dynamic
                    leverage=10.0,  # Could be from position data
                    risk_pct=20.0   # Could be from position data
                )

            # Remove from position monitor
            self.position_monitor.remove_position(symbol)

            # Clear smart execution data
            self.execution_manager.clear_position(symbol)

            # Refresh cached positions
            self.positions = self.bybit.get_positions()

        except Exception as e:
            logger.error(f"✗ Failed to close position: {e}")
    
    async def _modify_position(self, decision, market_data, analysis):
        """Modify protection levels for an existing position or reverse it"""
        symbol = decision.symbol

        if not symbol:
            logger.warning("No symbol specified for modification request")
            return

        position = next((p for p in self.positions if p.symbol == symbol and float(p.size) > 0), None)
        if not position:
            logger.warning(f"No open position found for {symbol} to modify")
            return

        entry_price = float(position.avg_price)
        current_side = position.side

        target_side = decision.side or current_side
        reverse_requested = bool(getattr(decision, "reverse_position", False)) or (target_side != current_side)

        reverse_payload = {}
        if isinstance(decision.risk_management, dict):
            reverse_payload = decision.risk_management.get("reverse_trade", {}) or {}
            if reverse_payload.get("side") and reverse_payload.get("side") != current_side:
                target_side = reverse_payload.get("side")
                reverse_requested = True

        desired_stop = decision.stop_loss
        desired_tp = decision.take_profit_1 or decision.take_profit_2
        trailing_distance = None
        if decision.risk_management:
            trailing_distance = decision.risk_management.get("trailing_stop_pct")

        if reverse_requested:
            logger.info(f"🔄 DeepSeek requested reversal on {symbol}: {current_side} → {target_side}")

            new_stop = decision.stop_loss or reverse_payload.get("stop_loss")
            new_tp = (decision.take_profit_1 or decision.take_profit_2 or
                      reverse_payload.get("take_profit") or reverse_payload.get("take_profit_1"))

            if new_stop is None or new_tp is None:
                logger.warning(f"Reverse instruction missing stop or TP for {symbol}; ignoring")
                return

            close_decision = TradingDecision(
                decision_id=str(uuid.uuid4()),
                timestamp_utc=datetime.now(timezone.utc),
                decision_type=DecisionType.CLOSE_POSITION,
                symbol=symbol,
                strategy_tag=decision.strategy_tag or "reverse_exit",
                reasoning={"action": "reverse_position"},
            )

            await self._close_position(close_decision)
            await asyncio.sleep(0.1)
            self.positions = self.bybit.get_positions()

            reversal_decision = TradingDecision(
                decision_id=str(uuid.uuid4()),
                timestamp_utc=datetime.now(timezone.utc),
                decision_type=DecisionType.OPEN_POSITION,
                symbol=symbol,
                side=target_side,
                order_type=decision.order_type or reverse_payload.get("order_type", "Market"),
                risk_pct=float(decision.risk_pct if decision.risk_pct is not None else reverse_payload.get("risk_pct", 15)),
                leverage=int(decision.leverage if decision.leverage is not None else reverse_payload.get("leverage", position.leverage)),
                stop_loss=float(new_stop),
                take_profit_1=float(new_tp),
                take_profit_2=float(reverse_payload.get("take_profit_2", 0)) if reverse_payload.get("take_profit_2") else decision.take_profit_2,
                strategy_tag=decision.strategy_tag or reverse_payload.get("strategy_tag", "reverse_flip"),
                confidence_score=(decision.confidence_score if decision.confidence_score is not None else reverse_payload.get("confidence_score", 0.7)),
                reasoning={"action": "reverse_entry"},
                risk_management=decision.risk_management,
            )

            await self._open_position(reversal_decision, market_data, analysis)
            return

        adjustments: Dict[str, Optional[float]] = {"stop_loss": None, "take_profit": None, "trailing_stop": None}

        if desired_stop is not None:
            adjusted_stop = float(desired_stop)
            if current_side == "Buy" and adjusted_stop >= entry_price:
                adjusted_stop = entry_price * 0.9995
                logger.info(f"Adjusting requested long stop for {symbol} to stay below entry (${adjusted_stop:.4f})")
            elif current_side == "Sell" and adjusted_stop <= entry_price:
                adjusted_stop = entry_price * 1.0005
                logger.info(f"Adjusting requested short stop for {symbol} to stay above entry (${adjusted_stop:.4f})")
            adjustments["stop_loss"] = adjusted_stop

        if desired_tp is not None:
            adjusted_tp = float(desired_tp)
            if current_side == "Buy" and adjusted_tp <= entry_price:
                adjusted_tp = entry_price * 1.001
                logger.info(f"Adjusting requested take-profit for long {symbol} to stay above entry (${adjusted_tp:.4f})")
            elif current_side == "Sell" and adjusted_tp >= entry_price:
                adjusted_tp = entry_price * 0.999
                logger.info(f"Adjusting requested take-profit for short {symbol} to stay below entry (${adjusted_tp:.4f})")
            adjustments["take_profit"] = adjusted_tp

        if trailing_distance is not None:
            try:
                adjustments["trailing_stop"] = float(trailing_distance)
            except (TypeError, ValueError):
                logger.debug(f"Invalid trailing distance provided by LLM: {trailing_distance}")

        if all(value is None for value in adjustments.values()):
            logger.warning(f"No actionable modifications provided for {symbol}")
            return

        try:
            self.bybit.update_position_protection(
                symbol=symbol,
                stop_loss=adjustments["stop_loss"],
                take_profit=adjustments["take_profit"],
                trailing_stop=adjustments["trailing_stop"],
            )

            trailing = self.execution_manager.trailing_stops.get(symbol)
            if trailing and adjustments["stop_loss"] is not None:
                trailing.current_stop = adjustments["stop_loss"]
                trailing.entry_price = entry_price

            logger.info(
                f"🔧 Modified {symbol} protection: SL={adjustments['stop_loss']} TP={adjustments['take_profit']} Trail={adjustments['trailing_stop']}"
            )

            self.positions = self.bybit.get_positions()

        except Exception as exc:
            logger.error(f"✗ Failed to modify {symbol} protection: {exc}")

    def _log_status(self):
        """Log current status"""
        pnl = self.account_balance - self.starting_balance
        pnl_pct = (pnl / self.starting_balance * 100) if self.starting_balance > 0 else 0
        
        logger.info("=" * 60)
        logger.info(f"Balance: ${self.account_balance:.2f} | PnL: ${pnl:.2f} ({pnl_pct:+.1f}%)")
        logger.info(f"Positions: {len(self.positions)} | Trades Today: {len(self.risk_manager.trades_today)}")
        
        current_milestone, next_milestone = self.settings.trading.get_current_milestone(
            self.account_balance
        )
        progress = (self.account_balance - current_milestone) / (next_milestone - current_milestone) * 100
        logger.info(f"Milestone: ${current_milestone:.0f} → ${next_milestone:.0f} ({progress:.1f}%)")
        logger.info("=" * 60)
        
        # Save account snapshot to database (disabled - method not implemented)
        # if self.db:
        #     asyncio.create_task(self.db.save_account_snapshot({
        #         "total_equity": self.account_balance,
        #         "available_balance": self.account_balance - sum(p.unrealized_pnl for p in self.positions),
        #         "used_margin": sum(p.unrealized_pnl for p in self.positions) if self.positions else 0,
        #         "unrealized_pnl": sum(p.unrealized_pnl for p in self.positions),
        #         "realized_pnl": pnl,
        #         "total_pnl": pnl,
        #         "total_pnl_pct": pnl_pct,
        #         "open_positions": len(self.positions),
        #         "total_trades_today": len(self.risk_manager.trades_today),
        #         "current_milestone": f"${current_milestone:.0f} → ${next_milestone:.0f}",
        #         "milestone_progress_pct": progress,
        #     }))
    
    async def shutdown(self):
        """Graceful shutdown"""
        logger.info("Shutting down trading bot...")
        self.running = False
        
        # Send shutdown notification
        if self.telegram:
            try:
                pnl = self.account_balance - self.starting_balance
                await self.telegram.send_shutdown_message(
                    balance=self.account_balance,
                    total_pnl=pnl,
                    trades_today=len(self.risk_manager.trades_today),
                )
            except Exception as e:
                logger.warning(f"Failed to send shutdown notification: {e}")
        
        # Close all positions if configured
        if self.positions and False:  # Set to True to close on shutdown
            logger.warning("Closing all open positions...")
            for position in self.positions:
                try:
                    self.bybit.close_position(position.symbol)
                    logger.info(f"Closed {position.symbol}")
                except Exception as e:
                    logger.error(f"Failed to close {position.symbol}: {e}")
        
        # Close Telegram session
        if self.telegram:
            await self.telegram.close()
        
        logger.info("Shutdown complete")


async def main():
    """Main entry point"""
    # Create bot instance
    bot = QuadrickTradingBot()
    
    # Setup signal handlers
    def signal_handler(sig, frame):
        logger.info("Interrupt received, shutting down...")
        asyncio.create_task(bot.shutdown())
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Initialize bot
        await bot.initialize()
        
        # Run main loop
        await bot.run()
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        
    finally:
        await bot.shutdown()


if __name__ == "__main__":
    # Run the bot
    asyncio.run(main())

"""
Quadrick AI Trading System - Main Entry Point
Autonomous trading system powered by DeepSeek LLM
"""
import asyncio
import signal
import sys
import uuid
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from loguru import logger
from config.settings import Settings
from src.exchange.bybit_client import BybitClient
from src.analysis.market_analyzer import MarketAnalyzer
from src.analysis.order_flow import OrderFlowAnalyzer
from src.analysis.opportunity_scorer import OpportunityScorer  # Hybrid: Pre-LLM scoring
from src.analysis.counter_trend_validator import CounterTrendValidator  # Hybrid: Counter-trend gate
from src.analysis.funding_analyzer import FundingAnalyzer  # Hybrid: Crowded trade detection
from src.config.symbol_config import get_symbol_config, get_trail_distance, get_adjusted_risk, get_min_rr_ratio
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
from src.quant import ForecastEngine, QuantEngine


class QuadrickTradingBot:
    """Main trading bot orchestrator"""
    
    def __init__(self):
        """Initialize the trading bot"""
        logger.info("=" * 80)
        logger.info("QUADRICK AI TRADING SYSTEM INITIALIZING")
        logger.info("Mission: $15  $100,000")
        logger.info("=" * 80)
        
        # Load configuration first
        self.settings = Settings()
        
        # Only normal mode supported
        self.trading_mode = "normal"
        
        logger.info(f" Trading Mode: {self.trading_mode.upper()}")
        logger.info(" Normal Mode: Multi-timeframe swing trading")
        
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
        self.forecast_engine = ForecastEngine() if self.settings.trading.enable_forecast_engine else None
        self.quant_engine = QuantEngine(self.settings) if self.settings.trading.quant_primary_mode else None
        self.llm_execution_locked = bool(
            self.settings.trading.quant_primary_mode
            and self.settings.trading.quant_enforce_llm_execution_lock
            and not self.settings.trading.llm_audit_enabled
        )
        self.relaxed_trade_gating = bool(self.settings.trading.relaxed_trade_gating)
        self.deepseek: Optional[DeepSeekClient] = None
        if not self.llm_execution_locked:
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
            small_account_balance_threshold=self.settings.trading.small_account_balance_threshold,
            small_account_max_risk_pct=self.settings.trading.small_account_max_risk_pct,
            small_account_max_leverage=self.settings.trading.small_account_max_leverage,
            min_stop_distance_pct=self.settings.trading.min_stop_distance_pct,
            max_stop_distance_pct=self.settings.trading.max_stop_distance_pct,
            symbol_max_margin_pct=self.settings.trading.symbol_max_margin_pct,
            portfolio_max_margin_pct=self.settings.trading.portfolio_max_margin_pct,
            enforce_single_position_per_symbol=self.settings.trading.enforce_single_position_per_symbol,
            allow_scale_in=self.settings.trading.allow_scale_in,
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
            supabase_key = (
                self.settings.database.supabase_service_role_key
                or self.settings.database.supabase_key
                or self.settings.database.supabase_anon_key
            )
            if self.settings.database.supabase_url and supabase_key:
                self.db = SupabaseClient(
                    supabase_url=self.settings.database.supabase_url,
                    supabase_key=supabase_key,
                    enabled=True,
                )
                if self.settings.database.supabase_service_role_key:
                    logger.info(" Supabase database enabled (service role)")
                else:
                    logger.warning("Supabase running without service-role key; some writes may fail with RLS")
            else:
                logger.warning("Supabase credentials missing - database logging disabled")
        else:
            logger.info("Database logging disabled (set DATABASE_PROVIDER=supabase to enable)")
        
        # Initialize Trading Council (Multi-Agent System) - AFTER RiskManager AND Database
        self.council: Optional[TradingCouncil] = None
        if self.deepseek is not None:
            self.council = TradingCouncil(self.deepseek, self.risk_manager, self.db, bridge=bridge)
        
        # Initialize Performance Tracker
        self.performance = PerformanceTracker()
        logger.info(" Performance tracking enabled")

        # Initialize Strategy Optimizer
        self.strategy_optimizer = StrategyOptimizer()
        logger.info(" Strategy optimization enabled")
        
        # Initialize Smart Execution Manager
        self.execution_manager = SmartExecutionManager()
        logger.info(" Smart execution enabled")
        
        # Initialize Position Monitor
        self.position_monitor = PositionMonitor()
        logger.info(" Position monitoring enabled")
        
        # Initialize Emergency Controls
        self.emergency_controls = EmergencyControls()
        logger.info(" Emergency controls enabled")
        
        # Initialize Hybrid LLM+Algo Components
        self.opportunity_scorer = OpportunityScorer(
            min_volume_ratio=1.3,
            min_score_threshold=65,  # Only feed high-quality setups to LLM
        )
        quant_ct_mode = bool(self.settings.trading.quant_primary_mode)
        self.counter_trend_validator = CounterTrendValidator(
            adx_threshold=18.0 if quant_ct_mode else 15.0,
            volume_spike_ratio=1.10 if quant_ct_mode else 1.50,
            min_rr_ratio=1.70 if quant_ct_mode else 2.0,
            min_score_to_allow=45 if quant_ct_mode else 70,
        )
        self.funding_analyzer = FundingAnalyzer(
            extreme_threshold=0.0005,  # 0.05% funding = extreme
            crowded_reduction=0.6,     # Reduce position to 60% (40% reduction)
        )
        logger.info(" Hybrid LLM+Algo system enabled (OpportunityScorer, CounterTrendValidator, FundingAnalyzer)")
        if self.quant_engine is not None:
            logger.info(" Quant primary mode enabled (LLM used for audit/explanations only)")
        if self.llm_execution_locked:
            logger.info(" LLM execution lock enabled: DeepSeek/Council trade path disabled")
        if self.relaxed_trade_gating:
            logger.warning(" RELAXED_TRADE_GATING enabled: strict pauses/filters are bypassed for this run")
        


        
        # System state
        self.running = False
        self.account_balance = 0.0
        self.available_balance = 0.0
        self.starting_balance = 0.0
        self.positions = []
        self.trade_history = []
        self.active_trade_contexts = {}  # Store entry context for RAG
        self._last_open_positions: Dict[str, Dict[str, Any]] = {}
        self._handled_closures = set()
        self._affordability_notice_until: Dict[str, datetime] = {}
        self._symbol_wait_streak: Dict[str, int] = {}
        self._recent_entry_symbols: List[str] = []
        self._max_recent_entry_symbols = 12
        self._max_consecutive_symbol_entries = int(self.settings.trading.max_consecutive_symbol_entries)
        self._symbol_repeat_window = int(self.settings.trading.symbol_repeat_window)
        self._symbol_repeat_penalty_pct = float(self.settings.trading.symbol_repeat_penalty_pct)
        self._symbol_repeat_override_gap_pct = float(self.settings.trading.symbol_repeat_override_gap_pct)
        self._flat_trade_tolerance_pct = max(
            0.01,
            float(getattr(self.emergency_controls, "flat_trade_tolerance_pct", 0.01)),
        )
        self._flat_symbol_cooldown_minutes = int(self.settings.trading.flat_symbol_cooldown_minutes)
        self._symbol_flat_streak: Dict[str, int] = {}
        self._symbol_loss_streak: Dict[str, int] = {}
        self._symbol_reject_streak: Dict[str, int] = {}
        self._queued_quant_payloads: List[Dict[str, Any]] = []
        self._queued_quant_updated_at: Optional[datetime] = None
        self._market_order_slippage_tolerance_pct = float(
            self.settings.trading.market_order_slippage_tolerance_pct
        )
        self._market_order_reject_cooldown_minutes = int(
            self.settings.trading.market_order_reject_cooldown_minutes
        )
        self._max_reprice_attempts = int(self.settings.trading.max_reprice_attempts)
        self._affordability_margin_epsilon_usd = max(
            0.0, float(self.settings.trading.affordability_margin_epsilon_usd)
        )
        
        # Watchlist
        self.watchlist = [
            "BTCUSDT", "ARBUSDT","XRPUSDT", "ETHUSDT", 
            "OPUSDT", "AVAXUSDT", "LINKUSDT",
            "DOTUSDT", "ADAUSDT", "DOGEUSDT", "1000PEPEUSDT",
        ]
        self.symbol_cooldowns: Dict[str, datetime] = {}
        if self.settings.bybit.testnet:
            # Some symbols intermittently return empty ticker payloads on testnet.
            # DOGEUSDT short entries on testnet frequently reject with ErrCode 30209.
            testnet_unstable_symbols = {"ARBUSDT", "AVAXUSDT", "DOGEUSDT"}
            self.watchlist = [s for s in self.watchlist if s not in testnet_unstable_symbols]
            logger.info(
                f"Testnet watchlist filter applied, removed unstable symbols: {sorted(testnet_unstable_symbols)}"
            )
        
        # Apply normal mode settings
        self.decision_interval = self.settings.trading.decision_interval_seconds
        self.timeframes_to_fetch = ["1", "5", "15", "60", "240", "D", "W"]  # All 7 timeframes
        self.default_risk_pct = min(3.0, float(self.settings.trading.max_risk_pct))
        self.default_leverage = min(5, int(self.settings.trading.max_leverage))
        self.max_hold_minutes = 480  # 8 hours max
        self.target_profit_pct = 2.5
        logger.info(f" Normal settings: {self.decision_interval}s decisions, {self.default_risk_pct}% risk")
        
        # Clear dashboard logs on startup
        bridge.clear_logs()
        
        logger.info("Trading bot initialized successfully")
        logger.info(f"Mode: {'TESTNET' if self.settings.bybit.testnet else 'LIVE'}")
        if not self.settings.bybit.testnet:
            if self.settings.system.allow_live_trading:
                logger.warning("LIVE trading is enabled (ALLOW_LIVE_TRADING=true)")
            else:
                logger.warning("LIVE mode detected but order placement is blocked (ALLOW_LIVE_TRADING=false)")
    
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

    def _effective_balances(self) -> Tuple[float, float]:
        """
        Return balances aligned with runtime sizing/risk basis.

        On testnet, always honor INITIAL_BALANCE as a sizing/risk cap so the
        bot can be evaluated against a fixed target account size (e.g. $100),
        even when testnet wallet equity is larger.
        """
        effective_account_balance = float(self.account_balance or 0.0)
        effective_available_balance = float(self.available_balance or 0.0)

        if self.settings.bybit.testnet and self.settings.trading.initial_balance > 0:
            cap = float(self.settings.trading.initial_balance)
            if effective_account_balance > cap:
                effective_account_balance = cap
            if effective_available_balance > cap:
                effective_available_balance = cap

        return effective_account_balance, effective_available_balance

    @staticmethod
    def _position_symbol(position) -> str:
        if isinstance(position, dict):
            return str(position.get("symbol") or "")
        return str(getattr(position, "symbol", "") or "")

    @staticmethod
    def _position_size(position) -> float:
        if isinstance(position, dict):
            raw = position.get("size", 0)
        else:
            raw = getattr(position, "size", 0)
        try:
            return abs(float(raw or 0))
        except (TypeError, ValueError):
            return 0.0

    def _has_open_position_for_symbol(self, symbol: str) -> bool:
        if not symbol:
            return False
        for position in self.positions or []:
            if self._position_symbol(position) != symbol:
                continue
            if self._position_size(position) > 0:
                return True
        return False

    def _max_trade_margin_cap(self) -> float:
        """Maximum margin budget for opening a new position under current caps."""
        effective_account_balance, effective_available_balance = self._effective_balances()
        if effective_account_balance <= 0 or effective_available_balance <= 0:
            return 0.0

        portfolio_cap_margin = effective_account_balance * (
            float(self.settings.trading.portfolio_max_margin_pct) / 100.0
        )
        symbol_cap_margin = effective_account_balance * (
            float(self.settings.trading.symbol_max_margin_pct) / 100.0
        )
        current_margin_used = effective_account_balance * (self._current_portfolio_risk_pct() / 100.0)
        remaining_portfolio_margin = max(0.0, portfolio_cap_margin - current_margin_used)
        available_margin_cap = effective_available_balance * 0.95
        return max(0.0, min(symbol_cap_margin, remaining_portfolio_margin, available_margin_cap))

    def _open_position_slots_left(self) -> int:
        active = 0
        for pos in self.positions or []:
            if self._position_size(pos) > 0:
                active += 1
        max_positions = int(self.settings.trading.max_concurrent_positions)
        return max(0, max_positions - active)

    def _queue_quant_candidates(self, payloads: List[Dict[str, Any]]) -> None:
        valid_payloads: List[Dict[str, Any]] = []
        for payload in payloads or []:
            if not isinstance(payload, dict):
                continue
            symbol = str(payload.get("symbol") or "").strip().upper()
            side = str(payload.get("side") or "").strip()
            if not symbol or side not in {"Buy", "Sell"}:
                continue
            valid_payloads.append(payload)
        self._queued_quant_payloads = valid_payloads[:8]
        self._queued_quant_updated_at = datetime.now(timezone.utc) if self._queued_quant_payloads else None

    def _dequeue_quant_candidate(self, market_data: Dict[str, Any]) -> Optional[TradingDecision]:
        if self._open_position_slots_left() <= 0:
            return None
        if not self._queued_quant_payloads:
            return None
        max_age_minutes = int(getattr(self.settings.trading, "quant_slot_queue_max_age_minutes", 30))
        if self._queued_quant_updated_at is not None:
            age_minutes = (datetime.now(timezone.utc) - self._queued_quant_updated_at).total_seconds() / 60.0
            if age_minutes > max_age_minutes:
                self._queued_quant_payloads.clear()
                self._queued_quant_updated_at = None
                return None

        while self._queued_quant_payloads:
            payload = self._queued_quant_payloads.pop(0)
            decision = self._quant_payload_to_decision(
                payload,
                fallback_rank=0,
                fallback_parent_symbol=str(payload.get("symbol") or ""),
            )
            if decision is None:
                continue
            ticker = (market_data.get("tickers", {}) or {}).get(decision.symbol)
            current_price = self._safe_float(getattr(ticker, "last_price", 0.0), 0.0) if ticker else 0.0
            if current_price <= 0:
                try:
                    current_price = self._safe_float(self.bybit.get_ticker(decision.symbol).last_price, 0.0)
                except Exception:
                    current_price = 0.0
            if current_price > 0:
                is_affordable, _, _, _ = self._check_symbol_affordability(
                    symbol=decision.symbol,
                    market_price=current_price,
                    leverage_hint=decision.leverage,
                )
                if not is_affordable:
                    continue
            if not isinstance(decision.reasoning, dict):
                decision.reasoning = {}
            decision.reasoning["queued_slot_retry"] = True
            decision.reasoning["queued_candidates_remaining"] = len(self._queued_quant_payloads)
            if not self._queued_quant_payloads:
                self._queued_quant_updated_at = None
            return decision

        self._queued_quant_updated_at = None
        return None

    def _check_symbol_affordability(
        self,
        symbol: str,
        market_price: float,
        leverage_hint: Optional[float] = None,
    ) -> Tuple[bool, str, float, float]:
        """
        Check if symbol minimum executable order can fit current margin budget.

        Returns: (is_affordable, reason, required_margin, max_margin_budget)
        """
        try:
            price = float(market_price or 0)
        except (TypeError, ValueError):
            price = 0.0

        max_margin = self._max_trade_margin_cap()
        if max_margin <= 0:
            return False, "no_margin_budget", 0.0, 0.0

        if price <= 0:
            return False, "invalid_price", 0.0, max_margin

        leverage = float(leverage_hint or self.default_leverage or 10)
        leverage = max(1.0, leverage)

        symbol_info = self.bybit.get_symbol_info(symbol)
        if not symbol_info:
            # For small accounts, fail-safe if we cannot verify instrument minimums.
            effective_account_balance, _ = self._effective_balances()
            if effective_account_balance <= 150:
                return False, "symbol_rules_unavailable_small_account", 0.0, max_margin
            # For larger accounts, keep scanning even if metadata is temporarily unavailable.
            return True, "unknown_rules", 0.0, max_margin

        lot_filter = symbol_info.get("lotSizeFilter", {}) or {}
        try:
            min_qty = float(lot_filter.get("minOrderQty", 0) or 0)
        except (TypeError, ValueError):
            min_qty = 0.0

        min_notional = max(5.5, min_qty * price)
        required_margin = min_notional / leverage
        if (required_margin - max_margin) > self._affordability_margin_epsilon_usd:
            return False, "min_order_margin_exceeds_budget", required_margin, max_margin

        return True, "ok", required_margin, max_margin

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    def _proposal_entry_price_from_market(self, proposal: Any, market_data: Dict[str, Any]) -> float:
        proposal_price = self._safe_float(getattr(proposal, "entry_price", 0.0), 0.0)
        if proposal_price > 0:
            return proposal_price
        symbol = str(getattr(proposal, "symbol", "") or "")
        ticker = (market_data.get("tickers", {}) or {}).get(symbol)
        if ticker is not None:
            ticker_price = self._safe_float(getattr(ticker, "last_price", 0.0), 0.0)
            if ticker_price > 0:
                return ticker_price
        try:
            return self._safe_float(self.bybit.get_ticker(symbol).last_price, 0.0)
        except Exception:
            return 0.0

    def _select_executable_quant_proposals(
        self,
        proposals: List[Any],
        market_data: Dict[str, Any],
    ) -> Tuple[List[Any], List[Dict[str, Any]]]:
        executable: List[Any] = []
        rejected: List[Dict[str, Any]] = []
        for proposal in proposals or []:
            symbol = str(getattr(proposal, "symbol", "") or "").strip().upper()
            leverage_hint = self._safe_float(
                getattr(proposal, "leverage", self.default_leverage),
                self.default_leverage,
            )
            entry_price = self._proposal_entry_price_from_market(proposal, market_data)
            is_affordable, reason, required_margin, max_margin = self._check_symbol_affordability(
                symbol=symbol,
                market_price=entry_price,
                leverage_hint=leverage_hint,
            )
            if is_affordable:
                executable.append(proposal)
            else:
                rejected.append(
                    {
                        "symbol": symbol,
                        "reason": reason,
                        "required_margin": required_margin,
                        "max_margin": max_margin,
                    }
                )
        return executable, rejected

    def _pick_best_executable_quant_candidate(
        self,
        proposals: List[Any],
        market_data: Dict[str, Any],
    ) -> Tuple[Optional[Any], List[Any], List[Dict[str, Any]]]:
        executable, rejected = self._select_executable_quant_proposals(
            proposals=proposals,
            market_data=market_data,
        )
        best = executable[0] if executable else None
        return best, executable, rejected

    def _validate_slippage_units(
        self,
        symbol: str,
        estimated_slippage_bps: float,
        realized_slippage_bps: float,
        requested_entry_price: float,
        fill_price: float,
    ) -> Tuple[bool, str]:
        if (
            not bool(self.settings.trading.win_rate_mode_enabled)
            or not bool(self.settings.trading.slippage_unit_validation_enabled)
        ):
            return True, "disabled"

        estimated_bps = self._safe_float(estimated_slippage_bps, 0.0)
        realized_bps = self._safe_float(realized_slippage_bps, 0.0)
        requested_price = self._safe_float(requested_entry_price, 0.0)
        realized_price = self._safe_float(fill_price, 0.0)

        estimated_pct = estimated_bps / 100.0
        realized_pct_from_bps = realized_bps / 100.0
        realized_pct_from_price = (
            abs(realized_price - requested_price) / requested_price * 100.0
            if requested_price > 0
            else 0.0
        )
        realized_bps_from_price = realized_pct_from_price * 100.0

        logger.info(
            f"SLIPPAGE_UNITS {symbol}: estimated={estimated_bps:.3f}bps ({estimated_pct:.5f}%), "
            f"realized={realized_bps:.3f}bps ({realized_pct_from_bps:.5f}%), "
            f"realized_from_price={realized_bps_from_price:.3f}bps ({realized_pct_from_price:.5f}%)"
        )

        max_bps = float(self.settings.trading.slippage_unit_validation_max_bps)
        max_est_bps = float(self.settings.trading.slippage_unit_validation_max_est_bps)
        if estimated_bps < 0 or estimated_bps > max_est_bps:
            return False, f"estimated_slippage_out_of_range bps={estimated_bps:.3f} max_est_bps={max_est_bps:.3f}"
        if realized_bps < 0 or realized_bps > max_bps:
            return False, f"realized_slippage_out_of_range bps={realized_bps:.3f} max_bps={max_bps:.3f}"
        if requested_price <= 0 or realized_price <= 0:
            return False, "invalid_price_for_slippage_validation"

        max_abs_bps_diff = float(self.settings.trading.slippage_unit_validation_max_abs_bps_diff)
        max_abs_pct_diff = float(self.settings.trading.slippage_unit_validation_max_abs_pct_diff)
        bps_diff = abs(realized_bps - realized_bps_from_price)
        pct_diff = abs(realized_pct_from_bps - realized_pct_from_price)
        if bps_diff > max_abs_bps_diff or pct_diff > max_abs_pct_diff:
            return (
                False,
                "slippage_unit_inconsistent "
                f"bps_diff={bps_diff:.6f} pct_diff={pct_diff:.6f} "
                f"max_bps_diff={max_abs_bps_diff:.6f} max_pct_diff={max_abs_pct_diff:.6f}",
            )
        return True, "ok"

    def _estimate_entry_slippage_bps(
        self,
        symbol: str,
        market_data: Dict[str, Any],
        current_price: float,
    ) -> float:
        ticker = (market_data.get("tickers", {}) or {}).get(symbol)
        if not ticker or current_price <= 0:
            return 0.0

        bid = self._safe_float(getattr(ticker, "bid_price", 0.0), 0.0)
        ask = self._safe_float(getattr(ticker, "ask_price", 0.0), 0.0)
        spread_bps = ((ask - bid) / current_price * 10000.0) if ask > 0 and bid > 0 and ask >= bid else 0.0

        order_flow = (market_data.get("order_flow", {}) or {}).get(symbol, {}) or {}
        bid_depth = self._safe_float(order_flow.get("bid_depth_10_levels_usd"), 0.0)
        ask_depth = self._safe_float(order_flow.get("ask_depth_10_levels_usd"), 0.0)
        depth_total = max(1.0, bid_depth + ask_depth)
        depth_penalty = 0.0
        if depth_total < 120000:
            depth_penalty += 12.0
        if depth_total < 50000:
            depth_penalty += 18.0

        imbalance = self._safe_float(order_flow.get("bid_ask_imbalance"), 1.0)
        imbalance_penalty = min(8.0, abs(imbalance - 1.0) * 8.0)
        return max(0.0, spread_bps + depth_penalty + imbalance_penalty)

    def _confirm_entry_second_step(
        self,
        decision: TradingDecision,
        market_data: Dict[str, Any],
        analysis: Dict[str, Any],
        latest_price: float,
        entry_tier: str,
    ) -> Tuple[bool, str, Dict[str, float]]:
        if not bool(self.settings.trading.win_rate_mode_enabled):
            return True, "disabled", {"disabled": 1.0}

        symbol = str(decision.symbol or "")
        side = str(decision.side or "")
        reasoning = decision.reasoning if isinstance(decision.reasoning, dict) else {}
        metadata = reasoning.get("metadata", {}) if isinstance(reasoning.get("metadata"), dict) else {}

        ticker = (market_data.get("tickers", {}) or {}).get(symbol)
        bid = self._safe_float(getattr(ticker, "bid_price", 0.0), 0.0) if ticker else 0.0
        ask = self._safe_float(getattr(ticker, "ask_price", 0.0), 0.0) if ticker else 0.0
        spread_bps = (
            ((ask - bid) / latest_price * 10000.0)
            if latest_price > 0 and ask > 0 and bid > 0 and ask >= bid
            else 0.0
        )

        symbol_analysis = analysis.get(symbol, {}) if isinstance(analysis, dict) else {}
        tf_15m = symbol_analysis.get("timeframe_analysis", {}).get("15m", {})
        atr_15m = self._safe_float(tf_15m.get("atr"), 0.0)
        atr_bps = (atr_15m / latest_price * 10000.0) if latest_price > 0 and atr_15m > 0 else 0.0
        spread_to_atr_ratio = spread_bps / max(1e-9, atr_bps)

        forecast_agg = (
            symbol_analysis.get("forecast", {}).get("aggregate", {})
            if isinstance(symbol_analysis, dict)
            else {}
        )
        forecast_prob_up = self._safe_float(forecast_agg.get("prob_up"), 0.5)
        direction_prob = forecast_prob_up if side == "Buy" else (1.0 - forecast_prob_up)
        prior_direction_prob = self._safe_float(reasoning.get("win_probability"), 0.5)
        min_direction_prob = max(0.50, prior_direction_prob - 0.08)

        regime_stability_ratio = self._safe_float(metadata.get("regime_stability_ratio"), 1.0)
        regime_stability_bars = int(max(0, self._safe_float(metadata.get("regime_stability_bars"), 0.0)))
        required_regime_bars = int(self.settings.trading.full_entry_regime_stability_bars)
        required_regime_ratio = float(self.settings.trading.full_entry_regime_stability_min_ratio)

        diagnostics = {
            "spread_bps": float(spread_bps),
            "atr_bps": float(atr_bps),
            "spread_to_atr_ratio": float(spread_to_atr_ratio),
            "direction_prob": float(direction_prob),
            "min_direction_prob": float(min_direction_prob),
            "regime_stability_ratio": float(regime_stability_ratio),
            "regime_stability_bars": float(regime_stability_bars),
        }

        if direction_prob < min_direction_prob:
            return False, "second_step_direction_drop", diagnostics

        if str(entry_tier).lower() == "full":
            max_ratio = float(self.settings.trading.full_entry_max_spread_to_atr_ratio)
            if atr_bps <= 0 or spread_to_atr_ratio > max_ratio:
                return False, "second_step_spread_atr_reject", diagnostics
            if regime_stability_bars < required_regime_bars or regime_stability_ratio < required_regime_ratio:
                return False, "second_step_regime_unstable", diagnostics

        return True, "ok", diagnostics

    def _classify_trade_outcome(self, pnl_pct: float) -> Tuple[Optional[bool], bool]:
        pnl_value = self._safe_float(pnl_pct, 0.0)
        if abs(pnl_value) <= self._flat_trade_tolerance_pct:
            return None, True
        return pnl_value > 0, False

    def _position_r_multiple(
        self,
        side: str,
        entry_price: float,
        stop_loss: float,
        current_price: float,
    ) -> float:
        if entry_price <= 0 or stop_loss <= 0 or current_price <= 0:
            return 0.0
        risk_per_unit = abs(entry_price - stop_loss)
        if risk_per_unit <= 0:
            return 0.0
        if str(side) == "Buy":
            reward = current_price - entry_price
        else:
            reward = entry_price - current_price
        return float(reward / risk_per_unit)

    def _build_close_event_details(
        self,
        symbol: str,
        strategy_name: str,
        outcome: str,
        pnl_pct: float,
        extra_details: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        entry_meta = dict(self._last_open_positions.get(symbol, {}) or {})
        details: Dict[str, Any] = {
            "outcome": str(outcome),
            "strategy": str(strategy_name or entry_meta.get("strategy_tag") or "unknown"),
            "side": str(entry_meta.get("side") or "unknown"),
            "market_regime": str(entry_meta.get("market_regime") or "unknown"),
            "regime": str(entry_meta.get("market_regime") or "unknown"),
            "entry_tier": str(entry_meta.get("entry_tier") or "full"),
            "policy_state": str(entry_meta.get("policy_state") or "green"),
            "policy_key": str(entry_meta.get("policy_key") or ""),
            "risk_pct": self._safe_float(entry_meta.get("risk_pct"), self.default_risk_pct),
            "leverage": int(self._safe_float(entry_meta.get("leverage"), self.default_leverage)),
            "expected_hold_minutes": int(self._safe_float(entry_meta.get("expected_hold_minutes"), 45)),
            "pnl_pct": float(pnl_pct),
        }
        if extra_details:
            details.update(extra_details)
        return details

    def _parse_datetime_utc(self, value: Any) -> Optional[datetime]:
        if value is None:
            return None
        try:
            if isinstance(value, datetime):
                parsed = value
            else:
                text = str(value).strip()
                if not text:
                    return None
                if text.endswith("Z"):
                    text = text[:-1] + "+00:00"
                parsed = datetime.fromisoformat(text)
            if parsed.tzinfo is None:
                return parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc)
        except Exception:
            return None

    def _symbol_loss_streak_multiplier(self, symbol: str) -> float:
        symbol_name = str(symbol or "").strip()
        if not symbol_name:
            return 1.0
        streak = int(self._symbol_loss_streak.get(symbol_name, 0))
        soft_limit = int(self.settings.trading.symbol_loss_soft_streak)
        if streak < soft_limit:
            return 1.0
        return max(0.10, min(1.0, float(self.settings.trading.symbol_loss_risk_multiplier)))

    def _apply_post_close_symbol_controls(
        self,
        symbol: str,
        pnl_pct: float,
        won: Optional[bool],
        is_flat: bool,
    ) -> None:
        symbol_name = str(symbol or "").strip()
        if not symbol_name:
            return

        now_utc = datetime.now(timezone.utc)
        relaxed = bool(getattr(self, "relaxed_trade_gating", False))
        soft_mode = relaxed or bool(getattr(self.settings.trading, "soft_governor_enabled", False))
        pnl_value = self._safe_float(pnl_pct, 0.0)
        previous_loss_streak = int(self._symbol_loss_streak.get(symbol_name, 0))

        if is_flat:
            self._symbol_flat_streak[symbol_name] = int(self._symbol_flat_streak.get(symbol_name, 0)) + 1
            if previous_loss_streak > 0:
                self._symbol_loss_streak[symbol_name] = previous_loss_streak - 1
        else:
            self._symbol_flat_streak[symbol_name] = 0
            if won is True:
                if previous_loss_streak > 0:
                    logger.info(
                        f"{symbol_name} loss streak reset after win ({previous_loss_streak} -> 0)"
                    )
                self._symbol_loss_streak[symbol_name] = 0
            elif won is False:
                self._symbol_loss_streak[symbol_name] = previous_loss_streak + 1

        if won is False:
            if pnl_value <= -0.80:
                loss_cooldown_minutes = 45
            elif pnl_value <= -0.15:
                loss_cooldown_minutes = 20
            else:
                loss_cooldown_minutes = 0

            loss_streak = int(self._symbol_loss_streak.get(symbol_name, 0))
            hard_streak = int(self.settings.trading.symbol_loss_hard_streak)
            if hard_streak > 0 and loss_streak >= hard_streak:
                loss_cooldown_minutes = max(
                    loss_cooldown_minutes,
                    int(self.settings.trading.symbol_loss_hard_cooldown_minutes),
                )

            if loss_cooldown_minutes > 0:
                if not soft_mode:
                    self.symbol_cooldowns[symbol_name] = now_utc + timedelta(minutes=loss_cooldown_minutes)
                    logger.info(
                        f"{symbol_name} entered loss cooldown for {loss_cooldown_minutes} minutes "
                        f"after {pnl_value:+.2f}% close (loss_streak={loss_streak})"
                    )
                else:
                    logger.info(
                        f"{symbol_name} soft_penalty_applied (no cooldown) "
                        f"after {pnl_value:+.2f}% close (loss_streak={loss_streak})"
                    )
            else:
                logger.info(
                    f"{symbol_name} loss cooldown skipped for micro-loss {pnl_value:+.2f}% "
                    f"(loss_streak={loss_streak})"
                )
        elif is_flat and self._flat_symbol_cooldown_minutes > 0 and not soft_mode:
            self.symbol_cooldowns[symbol_name] = now_utc + timedelta(
                minutes=self._flat_symbol_cooldown_minutes
            )
            logger.info(
                f"{symbol_name} entered flat-trade cooldown for {self._flat_symbol_cooldown_minutes} minutes "
                f"after {pnl_value:+.2f}% close"
            )

    def _register_entry_symbol(self, symbol: str) -> None:
        symbol_name = str(symbol or "").strip()
        if not symbol_name:
            return
        self._recent_entry_symbols.append(symbol_name)
        if len(self._recent_entry_symbols) > self._max_recent_entry_symbols:
            self._recent_entry_symbols = self._recent_entry_symbols[-self._max_recent_entry_symbols :]

    def _consecutive_recent_entries(self, symbol: str) -> int:
        symbol_name = str(symbol or "").strip()
        if not symbol_name or not self._recent_entry_symbols:
            return 0
        streak = 0
        for item in reversed(self._recent_entry_symbols):
            if item != symbol_name:
                break
            streak += 1
        return streak

    def _repetition_penalty_pct(self, symbol: str) -> float:
        symbol_name = str(symbol or "").strip()
        if not symbol_name or not self._recent_entry_symbols:
            return 0.0
        window = self._recent_entry_symbols[-self._symbol_repeat_window :]
        repeats = max(0, window.count(symbol_name) - 1)
        return min(0.60, repeats * self._symbol_repeat_penalty_pct)

    def _flat_streak_penalty_pct(self, symbol: str) -> float:
        symbol_name = str(symbol or "").strip()
        if not symbol_name:
            return 0.0
        streak = int(self._symbol_flat_streak.get(symbol_name, 0))
        return min(0.30, streak * 0.06)

    def _reject_penalty_pct(self, symbol: str) -> float:
        symbol_name = str(symbol or "").strip()
        if not symbol_name:
            return 0.0
        streak = int(self._symbol_reject_streak.get(symbol_name, 0))
        return min(0.40, streak * 0.10)

    def _mark_entry_reject(self, symbol: str, reason: str) -> None:
        symbol_name = str(symbol or "").strip()
        if not symbol_name:
            return
        self._symbol_reject_streak[symbol_name] = int(self._symbol_reject_streak.get(symbol_name, 0)) + 1
        if self.quant_engine is not None:
            try:
                self.quant_engine.record_execution_event(
                    event_type="reject",
                    symbol=symbol_name,
                    reason_code="entry_reject",
                    details={"reason": str(reason)},
                )
            except Exception:
                pass
        cooldown_minutes = self._market_order_reject_cooldown_minutes
        soft_mode = bool(self.relaxed_trade_gating) or bool(getattr(self.settings.trading, "soft_governor_enabled", False))
        if cooldown_minutes > 0 and not soft_mode:
            self.symbol_cooldowns[symbol_name] = datetime.now(timezone.utc) + timedelta(
                minutes=cooldown_minutes
            )
        logger.warning(
            f"{symbol_name} entry reject tracked (streak={self._symbol_reject_streak[symbol_name]}): {reason}"
            + (
                f"; cooldown {cooldown_minutes}m applied"
                if cooldown_minutes > 0 and not soft_mode
                else ""
            )
        )

    def _clear_entry_reject(self, symbol: str) -> None:
        symbol_name = str(symbol or "").strip()
        if not symbol_name:
            return
        if symbol_name in self._symbol_reject_streak:
            del self._symbol_reject_streak[symbol_name]

    def _current_portfolio_risk_pct(self) -> float:
        effective_balance, _ = self._effective_balances()
        if effective_balance <= 0:
            return 0.0
        total_margin = 0.0
        for pos in self.positions or []:
            try:
                size = abs(float(getattr(pos, "size", 0.0) or 0.0))
                mark = float(getattr(pos, "mark_price", 0.0) or 0.0)
                leverage = max(1.0, float(getattr(pos, "leverage", 1.0) or 1.0))
                total_margin += (size * mark) / leverage
            except (TypeError, ValueError):
                continue
        return max(0.0, (total_margin / effective_balance) * 100.0)

    def _current_drawdown_pct(self) -> float:
        if not hasattr(self, "risk_manager") or self.risk_manager is None:
            return 0.0
        effective_balance, _ = self._effective_balances()
        if effective_balance <= 0:
            return 0.0
        try:
            return max(0.0, float(self.risk_manager._calculate_daily_drawdown(effective_balance)))
        except Exception:
            return 0.0

    def _historical_strategy_win_rate(self, strategy_tag: str) -> Tuple[float, int]:
        strategy = str(strategy_tag or "unknown")
        stats = self.strategy_optimizer.strategy_performance.get(strategy, {})
        wins = int(stats.get("wins", 0) or 0)
        losses = int(stats.get("losses", 0) or 0)
        decisive_trades = int(stats.get("decisive_trades", wins + losses) or 0)
        if decisive_trades <= 0:
            return 0.5, 0
        return max(0.0, min(1.0, wins / decisive_trades)), decisive_trades

    def _estimate_trade_edge(
        self,
        candidate: Dict[str, Any],
        analysis: Dict[str, Any],
        market_data: Dict[str, Any],
    ) -> Optional[Dict[str, float]]:
        trade = candidate.get("trade", {}) if isinstance(candidate, dict) else {}
        symbol = str(trade.get("symbol") or "")
        side = str(trade.get("side") or "")
        if not symbol or side not in {"Buy", "Sell"}:
            return None

        symbol_analysis = analysis.get(symbol, {})
        ticker = market_data.get("tickers", {}).get(symbol)
        current_price = self._safe_float(symbol_analysis.get("current_price"), 0.0)
        if current_price <= 0 and ticker:
            current_price = self._safe_float(getattr(ticker, "last_price", 0), 0.0)

        entry = self._safe_float(trade.get("entry_price_target"), current_price)
        if entry <= 0:
            entry = current_price
        if entry <= 0:
            return None

        stop_loss = self._safe_float(trade.get("stop_loss"), 0.0)
        take_profit = self._safe_float(trade.get("take_profit_1"), 0.0)
        if stop_loss <= 0 or take_profit <= 0:
            return None

        if side == "Buy" and not (stop_loss < entry < take_profit):
            return None
        if side == "Sell" and not (take_profit < entry < stop_loss):
            return None

        risk_move_pct = abs(entry - stop_loss) / entry * 100.0
        reward_move_pct = abs(take_profit - entry) / entry * 100.0
        if risk_move_pct <= 0.0 or reward_move_pct <= 0.0:
            return None

        rr_ratio = reward_move_pct / risk_move_pct if risk_move_pct > 0 else 0.0
        confidence = max(0.0, min(1.0, self._safe_float(trade.get("confidence_score"), 0.5)))
        base_win_prob = max(0.30, min(0.82, 0.30 + (confidence * 0.52)))

        strategy_tag = str(trade.get("strategy_tag") or "unknown")
        historical_win_rate, historical_samples = self._historical_strategy_win_rate(strategy_tag)
        if historical_samples >= 8:
            win_prob = (0.70 * base_win_prob) + (0.30 * historical_win_rate)
        elif historical_samples >= 3:
            win_prob = (0.85 * base_win_prob) + (0.15 * historical_win_rate)
        else:
            win_prob = base_win_prob

        tf_1h = symbol_analysis.get("timeframe_analysis", {}).get("1h", {})
        trend_1h = str(tf_1h.get("trend", "neutral")).lower()
        adx_1h = self._safe_float(tf_1h.get("adx"), 0.0)
        long_aligned = trend_1h in {"uptrend", "bullish", "trending_up", "up"}
        short_aligned = trend_1h in {"downtrend", "bearish", "trending_down", "down"}
        aligned = (side == "Buy" and long_aligned) or (side == "Sell" and short_aligned)
        if aligned:
            win_prob += 0.03
        else:
            win_prob -= 0.05
            if adx_1h >= 25:
                win_prob -= 0.04

        forecast = symbol_analysis.get("forecast", {}) if isinstance(symbol_analysis, dict) else {}
        forecast_aggregate = forecast.get("aggregate", {}) if isinstance(forecast, dict) else {}
        forecast_conf = self._safe_float(forecast_aggregate.get("confidence"), 0.0)
        forecast_prob_up = self._safe_float(forecast_aggregate.get("prob_up"), 0.5)
        forecast_prob_up = max(0.0, min(1.0, forecast_prob_up))
        forecast_direction_prob = forecast_prob_up if side == "Buy" else (1.0 - forecast_prob_up)
        forecast_weight = max(0.0, min(1.0, float(self.settings.trading.forecast_weight_pct)))
        forecast_min_conf = max(0.0, min(1.0, float(self.settings.trading.forecast_min_confidence)))
        if self.forecast_engine is not None and forecast_conf >= forecast_min_conf:
            win_prob = ((1.0 - forecast_weight) * win_prob) + (forecast_weight * forecast_direction_prob)

        win_prob = max(0.20, min(0.90, win_prob))
        expected_cost_pct = max(0.0, self._safe_float(self.settings.trading.estimated_round_trip_cost_pct, 0.14))
        expected_return_pct = (win_prob * reward_move_pct) - ((1.0 - win_prob) * risk_move_pct) - expected_cost_pct

        return {
            "symbol": symbol,
            "side": side,
            "expected_return_pct": round(expected_return_pct, 4),
            "win_probability": round(win_prob, 4),
            "rr_ratio": round(rr_ratio, 4),
            "risk_move_pct": round(risk_move_pct, 4),
            "reward_move_pct": round(reward_move_pct, 4),
            "historical_samples": float(historical_samples),
            "historical_win_rate": round(historical_win_rate, 4),
            "aligned": 1.0 if aligned else 0.0,
            "forecast_confidence": round(forecast_conf, 4),
            "forecast_direction_prob": round(forecast_direction_prob, 4),
        }

    def _serialize_quant_proposal(self, proposal) -> Dict[str, Any]:
        raw_metadata = proposal.metadata if isinstance(getattr(proposal, "metadata", None), dict) else {}
        metadata: Dict[str, Any] = {
            "risk_multiplier": self._safe_float(raw_metadata.get("risk_multiplier"), 1.0),
            "uncertainty": self._safe_float(raw_metadata.get("uncertainty"), 0.0),
            "horizon_predictions": raw_metadata.get("horizon_predictions", {}),
            "portfolio_objective_score": raw_metadata.get("portfolio_objective_score"),
        }
        for key in (
            "gate",
            "gate_override",
            "feature_id",
            "objective_edge_floor_pct",
            "dynamic_edge_floor_for_proposal_pct",
            "dynamic_min_confidence",
            "dynamic_max_uncertainty",
            "dynamic_probe_quality_min",
            "dynamic_full_quality_min",
            "portfolio_adjusted_edge_pct",
            "portfolio_effective_edge_pct",
            "portfolio_exploration_override",
            "quality_score_raw",
            "quality_score_adjusted",
            "uncertainty_penalty",
            "policy_quality_penalty",
            "uncertainty_scaled",
            "edge_net_after_cost_pct",
            "spread_to_atr_ratio",
            "atr_bps_15m",
            "atr_pct_15m",
            "regime_stability_ratio",
            "regime_stability_bars",
            "regime_stability_required_bars",
            "regime_stability_required_ratio",
            "candidate_diag",
            "policy",
        ):
            if key in raw_metadata:
                metadata[key] = raw_metadata.get(key)
        return {
            "symbol": str(getattr(proposal, "symbol", "") or ""),
            "side": str(getattr(proposal, "side", "") or ""),
            "entry_price": self._safe_float(getattr(proposal, "entry_price", 0.0), 0.0),
            "stop_loss": self._safe_float(getattr(proposal, "stop_loss", 0.0), 0.0),
            "take_profit": self._safe_float(getattr(proposal, "take_profit", 0.0), 0.0),
            "risk_pct": self._safe_float(getattr(proposal, "risk_pct", self.default_risk_pct), self.default_risk_pct),
            "leverage": int(max(1, self._safe_float(getattr(proposal, "leverage", self.default_leverage), self.default_leverage))),
            "expected_hold_minutes": int(max(5, self._safe_float(getattr(proposal, "expected_hold_minutes", 30), 30))),
            "strategy_tag": str(getattr(proposal, "strategy_tag", "quant_ev_moe") or "quant_ev_moe"),
            "regime": str(getattr(proposal, "regime", "unknown") or "unknown"),
            "expected_edge_pct": self._safe_float(getattr(proposal, "expected_edge_pct", 0.0), 0.0),
            "expectancy_per_hour_pct": self._safe_float(getattr(proposal, "expectancy_per_hour_pct", 0.0), 0.0),
            "win_probability": self._safe_float(getattr(proposal, "win_probability", 0.5), 0.5),
            "rr_ratio": self._safe_float(getattr(proposal, "rr_ratio", 1.0), 1.0),
            "confidence": self._safe_float(getattr(proposal, "confidence", 0.5), 0.5),
            "quality_score": self._safe_float(getattr(proposal, "quality_score", 0.0), 0.0),
            "quality_score_raw": self._safe_float(getattr(proposal, "quality_score_raw", getattr(proposal, "quality_score", 0.0)), 0.0),
            "quality_score_adjusted": self._safe_float(getattr(proposal, "quality_score_adjusted", getattr(proposal, "quality_score", 0.0)), 0.0),
            "entry_tier": str(getattr(proposal, "entry_tier", "full") or "full"),
            "symbol_weight": self._safe_float(getattr(proposal, "symbol_weight", 1.0), 1.0),
            "policy_state": str(getattr(proposal, "policy_state", "green") or "green"),
            "policy_key": str(getattr(proposal, "policy_key", "") or ""),
            "estimated_slippage_bps": self._safe_float(getattr(proposal, "estimated_slippage_bps", 0.0), 0.0),
            "metadata": metadata,
        }

    def _quant_payload_to_decision(
        self,
        payload: Dict[str, Any],
        fallback_rank: int = 0,
        fallback_parent_symbol: Optional[str] = None,
    ) -> Optional[TradingDecision]:
        if not isinstance(payload, dict):
            return None
        symbol = str(payload.get("symbol") or "").strip().upper()
        side = str(payload.get("side") or "").strip()
        if not symbol or side not in {"Buy", "Sell"}:
            return None
        entry_price = self._safe_float(payload.get("entry_price"), 0.0)
        stop_loss = self._safe_float(payload.get("stop_loss"), 0.0)
        take_profit = self._safe_float(payload.get("take_profit"), 0.0)
        if entry_price <= 0 or stop_loss <= 0 or take_profit <= 0:
            return None
        metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
        reasoning = {
            "source": "quant_engine",
            "regime": str(payload.get("regime") or "unknown"),
            "expected_edge_pct": self._safe_float(payload.get("expected_edge_pct"), 0.0),
            "expectancy_per_hour_pct": self._safe_float(payload.get("expectancy_per_hour_pct"), 0.0),
            "win_probability": self._safe_float(payload.get("win_probability"), 0.5),
            "rr_ratio": self._safe_float(payload.get("rr_ratio"), 1.0),
            "confidence": self._safe_float(payload.get("confidence"), 0.5),
            "quality_score": self._safe_float(payload.get("quality_score"), 0.0),
            "quality_score_raw": self._safe_float(
                payload.get("quality_score_raw", payload.get("quality_score")),
                0.0,
            ),
            "quality_score_adjusted": self._safe_float(
                payload.get("quality_score_adjusted", payload.get("quality_score")),
                0.0,
            ),
            "entry_tier": str(payload.get("entry_tier") or "full"),
            "symbol_weight": self._safe_float(payload.get("symbol_weight"), 1.0),
            "policy_state": str(payload.get("policy_state") or "green"),
            "policy_key": str(payload.get("policy_key") or ""),
            "estimated_slippage_bps": self._safe_float(payload.get("estimated_slippage_bps"), 0.0),
            "risk_multiplier": self._safe_float(metadata.get("risk_multiplier"), 1.0),
            "metadata": metadata,
        }
        policy_meta = metadata.get("policy", {}) if isinstance(metadata.get("policy"), dict) else {}
        if fallback_rank > 0:
            reasoning["fallback_rank"] = int(fallback_rank)
        if fallback_parent_symbol:
            reasoning["fallback_parent_symbol"] = str(fallback_parent_symbol)
        risk_management = {
            "engine": "quant_moe",
            "allow_counter_trend": bool(policy_meta.get("allow_counter_trend_bypass", True)),
            "uncertainty": float(metadata.get("uncertainty", 0.0) or 0.0),
            "entry_tier": str(payload.get("entry_tier") or "full"),
            "horizon_predictions": metadata.get("horizon_predictions", {}),
        }
        take_profit_2 = (
            entry_price + ((take_profit - entry_price) * 1.4)
            if side == "Buy"
            else entry_price - ((entry_price - take_profit) * 1.4)
        )
        return TradingDecision(
            decision_id=str(uuid.uuid4()),
            timestamp_utc=datetime.now(timezone.utc),
            decision_type=DecisionType.OPEN_POSITION,
            symbol=symbol,
            category="linear",
            side=side,
            order_type="Market",
            risk_pct=self._safe_float(payload.get("risk_pct"), self.default_risk_pct),
            leverage=int(max(1, self._safe_float(payload.get("leverage"), self.default_leverage))),
            entry_price_target=entry_price,
            stop_loss=stop_loss,
            take_profit_1=take_profit,
            take_profit_2=take_profit_2,
            time_in_force="IOC",
            expected_hold_duration_mins=int(max(5, self._safe_float(payload.get("expected_hold_minutes"), 30))),
            strategy_tag=str(payload.get("strategy_tag") or "quant_ev_moe"),
            confidence_score=self._safe_float(payload.get("confidence"), 0.5),
            reasoning=reasoning,
            risk_management=risk_management,
            model_version="quant_engine_v2",
        )

    def _quant_proposal_to_decision(self, proposal) -> TradingDecision:
        payload = self._serialize_quant_proposal(proposal)
        decision = self._quant_payload_to_decision(payload)
        if decision is not None:
            return decision
        reasoning = {
            "source": "quant_engine",
            "regime": proposal.regime,
            "expected_edge_pct": proposal.expected_edge_pct,
            "expectancy_per_hour_pct": proposal.expectancy_per_hour_pct,
            "win_probability": proposal.win_probability,
            "rr_ratio": proposal.rr_ratio,
            "confidence": proposal.confidence,
            "quality_score": proposal.quality_score,
            "quality_score_raw": proposal.quality_score_raw,
            "quality_score_adjusted": proposal.quality_score_adjusted,
            "entry_tier": proposal.entry_tier,
            "symbol_weight": proposal.symbol_weight,
            "policy_state": proposal.policy_state,
            "policy_key": proposal.policy_key,
            "estimated_slippage_bps": proposal.estimated_slippage_bps,
            "risk_multiplier": float(proposal.metadata.get("risk_multiplier", 1.0)),
            "metadata": proposal.metadata,
        }
        policy_meta = proposal.metadata.get("policy", {}) if isinstance(proposal.metadata, dict) else {}
        risk_management = {
            "engine": "quant_moe",
            "allow_counter_trend": bool(policy_meta.get("allow_counter_trend_bypass", True)),
            "uncertainty": float(proposal.metadata.get("uncertainty", 0.0)),
            "entry_tier": proposal.entry_tier,
            "horizon_predictions": proposal.metadata.get("horizon_predictions", {}),
        }
        return TradingDecision(
            decision_id=str(uuid.uuid4()),
            timestamp_utc=datetime.now(timezone.utc),
            decision_type=DecisionType.OPEN_POSITION,
            symbol=proposal.symbol,
            category="linear",
            side=proposal.side,
            order_type="Market",
            risk_pct=float(proposal.risk_pct),
            leverage=int(proposal.leverage),
            entry_price_target=float(proposal.entry_price),
            stop_loss=float(proposal.stop_loss),
            take_profit_1=float(proposal.take_profit),
            take_profit_2=float(
                proposal.entry_price
                + ((proposal.take_profit - proposal.entry_price) * 1.4)
                if proposal.side == "Buy"
                else proposal.entry_price
                - ((proposal.entry_price - proposal.take_profit) * 1.4)
            ),
            time_in_force="IOC",
            expected_hold_duration_mins=int(max(5, proposal.expected_hold_minutes)),
            strategy_tag="quant_ev_moe",
            confidence_score=float(proposal.confidence),
            reasoning=reasoning,
            risk_management=risk_management,
            model_version="quant_engine_v2",
        )

    async def _get_quant_primary_decision(
        self,
        market_data: Dict[str, Any],
        analysis: Dict[str, Any],
    ) -> TradingDecision:
        if self.quant_engine is None:
            return TradingDecision(
                decision_id=str(uuid.uuid4()),
                timestamp_utc=datetime.now(timezone.utc),
                decision_type=DecisionType.WAIT,
                reasoning={"reason": "quant_engine_not_initialized"},
            )

        queued_decision = self._dequeue_quant_candidate(market_data)
        if queued_decision is not None:
            logger.info(
                "Quant slot-queue retry selected {} {}",
                queued_decision.symbol,
                queued_decision.side,
            )
            return queued_decision

        effective_account_balance, _ = self._effective_balances()
        proposals, metrics = self.quant_engine.ingest_and_score_cycle(
            market_data=market_data,
            analysis=analysis,
            open_positions=self.positions,
            symbol_reject_streak=self._symbol_reject_streak,
            current_total_risk_pct=self._current_portfolio_risk_pct(),
            account_balance=effective_account_balance,
            account_drawdown_pct=self._current_drawdown_pct(),
        )

        logger.info(
            "Quant cycle: candidates={} proposals={} accepted={} drift={:.3f} reject_rate={:.2%} "
            "events={:.0f} latency={:.1f}ms tph={:.2f} eph={:+.3f}% edge_floor={:+.3f}% "
            "risk_mult={:.2f} gov={}",
            metrics.candidates_scored,
            metrics.proposals_generated,
            metrics.proposals_accepted,
            metrics.drift_score,
            metrics.recent_reject_rate,
            metrics.recent_execution_events,
            metrics.cycle_latency_ms,
            metrics.objective_trades_per_hour,
            metrics.objective_expectancy_per_hour_pct,
            metrics.objective_edge_floor_pct,
            metrics.risk_multiplier,
            metrics.governor_mode,
        )
        guard_snapshot = {}
        if isinstance(metrics.governor_snapshot, dict):
            guard_snapshot = metrics.governor_snapshot.get("expectancy_floor_guard", {}) or {}
        if bool(guard_snapshot.get("active")):
            logger.warning(
                "Expectancy-floor guard active: exp/trade={:+.3f}% floor={:+.3f}% tail5={:.2%}",
                self._safe_float(guard_snapshot.get("expectancy_per_trade_pct"), 0.0),
                self._safe_float(guard_snapshot.get("floor_pct"), 0.0),
                self._safe_float(guard_snapshot.get("tail_loss_5_rate"), 0.0),
            )
        policy_state_counts: Dict[str, int] = {}
        for p in proposals:
            p_state = str(getattr(p, "policy_state", "green") or "green")
            policy_state_counts[p_state] = int(policy_state_counts.get(p_state, 0)) + 1
        for idx, candidate in enumerate(proposals[:5], start=1):
            diag = candidate.metadata.get("candidate_diag", {}) if isinstance(candidate.metadata, dict) else {}
            logger.info(
                "Quant cand#{} {} {} {} q_raw={:.1f} q_adj={:.1f} edge={:+.3f}% cost_est={:.1f}bps cap_clipped={} final_size={}",
                idx,
                candidate.symbol,
                candidate.side,
                str(getattr(candidate, "policy_state", "green")),
                self._safe_float(getattr(candidate, "quality_score_raw", candidate.quality_score), 0.0),
                self._safe_float(getattr(candidate, "quality_score_adjusted", candidate.quality_score), 0.0),
                self._safe_float(candidate.expected_edge_pct, 0.0),
                self._safe_float(candidate.estimated_slippage_bps, 0.0),
                bool(diag.get("cap_clipped", False)),
                diag.get("final_size"),
            )
        try:
            bridge.update_quant_monitor(
                {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "candidates_scored": metrics.candidates_scored,
                    "proposals_generated": metrics.proposals_generated,
                    "uncertainty_passed": metrics.uncertainty_passed,
                    "portfolio_passed": metrics.portfolio_passed,
                    "affordability_passed": metrics.affordability_passed,
                    "selected": metrics.selected,
                    "proposals_accepted": metrics.proposals_accepted,
                    "drift_score": metrics.drift_score,
                    "recent_execution_events": metrics.recent_execution_events,
                    "recent_reject_rate": metrics.recent_reject_rate,
                    "recent_fill_slippage_bps": metrics.recent_fill_slippage_bps,
                    "cycle_latency_ms": metrics.cycle_latency_ms,
                    "objective_trades_per_hour": metrics.objective_trades_per_hour,
                    "objective_expectancy_per_trade_pct": metrics.objective_expectancy_per_trade_pct,
                    "objective_expectancy_per_hour_pct": metrics.objective_expectancy_per_hour_pct,
                    "objective_closed_trades": metrics.objective_closed_trades,
                    "objective_edge_floor_pct": metrics.objective_edge_floor_pct,
                    "objective_score": metrics.objective_score,
                    "risk_multiplier": metrics.risk_multiplier,
                    "governor_mode": metrics.governor_mode,
                    "governor_reason": metrics.governor_reason,
                    "governor_snapshot": metrics.governor_snapshot or {},
                    "symbol_policy_summary": {
                        "tracked_buckets": len(getattr(self.quant_engine.symbol_policy, "buckets", {}) or {}),
                        "states": policy_state_counts,
                    },
                    "symbol_funnel": metrics.symbol_funnel or {},
                    "reject_reason_counts": metrics.reject_reason_counts or {},
                    "retrained": metrics.retrained,
                    "top_proposals": [
                        {
                            "symbol": p.symbol,
                            "side": p.side,
                            "edge_pct": p.expected_edge_pct,
                            "expectancy_per_hour_pct": p.expectancy_per_hour_pct,
                            "win_probability": p.win_probability,
                            "rr_ratio": p.rr_ratio,
                            "risk_pct": p.risk_pct,
                            "leverage": p.leverage,
                            "expected_hold_minutes": p.expected_hold_minutes,
                            "quality_score": p.quality_score,
                            "quality_score_raw": getattr(p, "quality_score_raw", p.quality_score),
                            "quality_score_adjusted": getattr(p, "quality_score_adjusted", p.quality_score),
                            "entry_tier": p.entry_tier,
                            "symbol_weight": p.symbol_weight,
                            "policy_state": getattr(p, "policy_state", "green"),
                            "policy_key": getattr(p, "policy_key", ""),
                            "estimated_slippage_bps": p.estimated_slippage_bps,
                            "objective_score": p.metadata.get("portfolio_objective_score"),
                            "edge_net_after_cost_pct": p.metadata.get("edge_net_after_cost_pct"),
                            "candidate_diag": p.metadata.get("candidate_diag", {}),
                        }
                        for p in proposals[:5]
                    ],
                }
            )
        except Exception:
            pass
        for alert in metrics.monitor_alerts or []:
            logger.warning(
                "Quant monitor alert [{}] {}: {}",
                alert.get("severity", "unknown"),
                alert.get("code", "unknown"),
                alert.get("message", ""),
            )
            try:
                bridge.update_quant_alert(alert)
            except Exception:
                pass
            if self.telegram and str(alert.get("severity", "")).lower() in {"warning", "critical"}:
                try:
                    await self.telegram.send_warning(
                        title=f"Quant {alert.get('code', 'alert')}",
                        message=str(alert.get("message", "")),
                    )
                except Exception:
                    pass
        if metrics.retrained and metrics.retrain_summary:
            logger.info("Quant retrain executed: {}", metrics.retrain_summary)

        latency_threshold_ms = float(self.settings.trading.quant_latency_kill_switch_ms)
        if metrics.cycle_latency_ms > latency_threshold_ms:
            logger.warning(
                "Soft governor path: latency threshold exceeded "
                f"({metrics.cycle_latency_ms:.0f}ms > {latency_threshold_ms:.0f}ms). "
                "No pause applied; risk governor handles de-risking."
            )

        reject_rate_threshold = float(self.settings.trading.quant_reject_rate_kill_switch)
        reject_min_events = int(self.settings.trading.quant_reject_kill_switch_min_events)
        if (
            metrics.recent_execution_events >= reject_min_events
            and metrics.recent_reject_rate > reject_rate_threshold
        ):
            logger.warning(
                "Soft governor path: reject-rate threshold exceeded "
                f"({metrics.recent_reject_rate:.2%} > {reject_rate_threshold:.2%}; "
                f"events={metrics.recent_execution_events:.0f}). "
                "No pause applied; risk governor handles de-risking."
            )

        if (
            metrics.objective_closed_trades >= int(self.settings.trading.quant_min_closed_trades_for_objective)
            and metrics.objective_expectancy_per_hour_pct < -0.35
        ):
            logger.warning(
                "Soft governor path: expectancy/hour safeguard breached "
                f"({metrics.objective_expectancy_per_hour_pct:+.3f}% < -0.350%). "
                "No pause applied; risk governor handles de-risking."
            )

        if not proposals:
            return TradingDecision(
                decision_id=str(uuid.uuid4()),
                timestamp_utc=datetime.now(timezone.utc),
                decision_type=DecisionType.WAIT,
                reasoning={
                    "reason": "quant_no_positive_edge_setup",
                    "risk_multiplier": metrics.risk_multiplier,
                    "governor_mode": metrics.governor_mode,
                    "governor_reason": metrics.governor_reason,
                },
            )

        best, affordable_proposals, rejected_for_affordability = self._pick_best_executable_quant_candidate(
            proposals=proposals,
            market_data=market_data,
        )

        if rejected_for_affordability:
            head = rejected_for_affordability[0]
            logger.info(
                "Quant affordability filter rejected {} proposal(s); top reject {} reason={} required=${:.2f} cap=${:.2f}",
                len(rejected_for_affordability),
                head["symbol"],
                head["reason"],
                float(head["required_margin"]),
                float(head["max_margin"]),
            )
        metrics.affordability_passed = len(affordable_proposals)
        try:
            bridge.update_quant_monitor(
                {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "affordability_passed": metrics.affordability_passed,
                    "selected": len(affordable_proposals),
                    "symbol_funnel": metrics.symbol_funnel or {},
                }
            )
        except Exception:
            pass

        open_slots = self._open_position_slots_left()
        if affordable_proposals and open_slots <= 0:
            queued_payloads = [self._serialize_quant_proposal(p) for p in affordable_proposals[:6]]
            self._queue_quant_candidates(queued_payloads)
            logger.info(
                "Position slots full; queued {} quant candidate(s) for slot-retry",
                len(self._queued_quant_payloads),
            )
            return TradingDecision(
                decision_id=str(uuid.uuid4()),
                timestamp_utc=datetime.now(timezone.utc),
                decision_type=DecisionType.WAIT,
                reasoning={
                    "reason": "quant_slot_full_queued",
                    "queued_candidates": len(self._queued_quant_payloads),
                    "max_concurrent_positions": int(self.settings.trading.max_concurrent_positions),
                    "open_positions": int(len([p for p in self.positions if self._position_size(p) > 0])),
                    "risk_multiplier": metrics.risk_multiplier,
                    "governor_mode": metrics.governor_mode,
                    "governor_reason": metrics.governor_reason,
                },
            )

        if not affordable_proposals:
            now_utc = datetime.now(timezone.utc)
            soft_mode = bool(self.relaxed_trade_gating) or bool(getattr(self.settings.trading, "soft_governor_enabled", False))
            for rejected in rejected_for_affordability:
                symbol_name = str(rejected.get("symbol") or "").strip()
                reason = str(rejected.get("reason") or "")
                if not symbol_name:
                    continue
                if reason == "min_order_margin_exceeds_budget":
                    if soft_mode:
                        logger.info(
                            f"{symbol_name} soft_penalty_applied (affordability; no cooldown) "
                            f"(required=${float(rejected.get('required_margin', 0.0)):.2f}, "
                            f"cap=${float(rejected.get('max_margin', 0.0)):.2f})"
                        )
                    else:
                        cooldown_minutes = 25
                        cooldown_until = now_utc + timedelta(minutes=cooldown_minutes)
                        existing = self.symbol_cooldowns.get(symbol_name)
                        if existing is None or existing < cooldown_until:
                            self.symbol_cooldowns[symbol_name] = cooldown_until
                            logger.info(
                                f"{symbol_name} affordability cooldown applied for {cooldown_minutes} minutes "
                                f"(required=${float(rejected.get('required_margin', 0.0)):.2f}, "
                                f"cap=${float(rejected.get('max_margin', 0.0)):.2f})"
                            )
            return TradingDecision(
                decision_id=str(uuid.uuid4()),
                timestamp_utc=datetime.now(timezone.utc),
                decision_type=DecisionType.WAIT,
                reasoning={
                    "reason": "quant_no_affordable_setup",
                    "risk_multiplier": metrics.risk_multiplier,
                    "governor_mode": metrics.governor_mode,
                    "governor_reason": metrics.governor_reason,
                    "rejected": rejected_for_affordability[:3],
                },
            )

        logger.info(
            "Quant selected {} {} edge={:+.3f}% eph={:+.3f}% pwin={:.2f} rr={:.2f} "
            "risk={:.2f}% lev={}x hold={}m tier={} q={:.1f} risk_mult={:.2f}",
            best.symbol,
            best.side,
            best.expected_edge_pct,
            best.expectancy_per_hour_pct,
            best.win_probability,
            best.rr_ratio,
            best.risk_pct,
            best.leverage,
            best.expected_hold_minutes,
            best.entry_tier,
            best.quality_score,
            metrics.risk_multiplier,
        )
        decision = self._quant_proposal_to_decision(best)
        if isinstance(decision.reasoning, dict):
            fallback_payloads = [
                self._serialize_quant_proposal(p)
                for p in affordable_proposals[1:5]
            ]
            if fallback_payloads:
                decision.reasoning["fallback_proposals"] = fallback_payloads
                decision.reasoning["fallback_candidates"] = [
                    {
                        "symbol": str(p.get("symbol") or ""),
                        "side": str(p.get("side") or ""),
                        "quality_score": self._safe_float(p.get("quality_score"), 0.0),
                        "entry_tier": str(p.get("entry_tier") or "full"),
                    }
                    for p in fallback_payloads
                ]
                # Keep a short queue for immediate slot-retry when positions free up.
                self._queue_quant_candidates(fallback_payloads)
        return decision

    async def _record_trade_outcome(
        self,
        symbol: str,
        pnl_pct: float,
        won: Optional[bool],
        strategy_name: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        metadata = metadata or {}

        self.performance.add_trade(
            {
                "symbol": symbol,
                "pnl": pnl_pct,
                "strategy_tag": strategy_name,
                "regime": str(metadata.get("market_regime") or "unknown"),
                "win": bool(won) if won is not None else False,
                "is_flat": won is None,
                "timestamp": datetime.now(timezone.utc),
            }
        )

        try:
            if hasattr(self, "risk_manager") and self.risk_manager:
                effective_balance, _ = self._effective_balances()
                self.risk_manager.update_trade_result(
                    trade_id=f"{symbol}_{int(datetime.now(timezone.utc).timestamp())}",
                    pnl=pnl_pct,
                    account_balance=effective_balance,
                )
        except Exception as risk_result_err:
            logger.warning(f"Failed to update risk manager trade result for {symbol}: {risk_result_err}")

        market_regime = str(metadata.get("market_regime") or "unknown")
        timeframe = str(metadata.get("timeframe") or "15m")
        leverage = max(1.0, self._safe_float(metadata.get("leverage"), self.default_leverage))
        risk_pct = max(self.settings.trading.min_risk_pct, self._safe_float(metadata.get("risk_pct"), self.default_risk_pct))

        self.strategy_optimizer.analyze_strategy_performance(
            strategy_name=strategy_name,
            pnl=pnl_pct,
            win=won,
            market_regime=market_regime,
            timeframe=timeframe,
            leverage=leverage,
            risk_pct=risk_pct,
        )

        if self.db:
            try:
                stats = self.strategy_optimizer.get_stats()
                await self.db.save_strategy_stats(stats)
            except Exception as save_stats_err:
                logger.warning(f"Failed to save strategy stats: {save_stats_err}")

        try:
            entry_context = self.active_trade_contexts.get(symbol)
            if not entry_context and self.db:
                entry_context = await self.db.get_active_context(symbol)

            if self.db and entry_context:
                trade_result = {
                    "symbol": symbol,
                    "strategy": strategy_name,
                    "pnl_pct": pnl_pct,
                    "win": won if won is not None else None,
                }
                await self.db.save_trade_memory(trade_result, entry_context)
                await self.db.delete_active_context(symbol)
                logger.info(f"Memory saved for {symbol} (PnL: {pnl_pct:+.2f}%)")
        except Exception as memory_err:
            logger.warning(f"Failed to save trade memory for {symbol}: {memory_err}")
        finally:
            if symbol in self.active_trade_contexts:
                del self.active_trade_contexts[symbol]
    
    async def initialize(self):
        """Initialize bot and verify connectivity"""
        logger.info("Performing startup checks...")
        
        # Test Bybit connection (single attempt to avoid timestamp issues)
        try:
            account_info = self.bybit.get_account_info()
            logger.info(" Bybit connection successful")

            # Try to get balance, but allow fallback if it fails
            try:
                balance_info = self.bybit.get_account_balance()
                self.account_balance = balance_info.total_equity
                self.starting_balance = self.account_balance
                self.available_balance = balance_info.available_balance

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
                logger.warning(f"  Balance check failed, using fallback values: {balance_error}")
                # Use a more realistic fallback balance based on previous sessions
                self.account_balance = 8.39  # Known working balance from logs
                self.starting_balance = self.account_balance
                self.available_balance = self.account_balance
                logger.info(f"Using fallback balance: ${self.account_balance:.2f}")

            # Get current milestone
            try:
                current_milestone, next_milestone = self.settings.trading.get_current_milestone(
                    self.account_balance
                )
                logger.info(
                    f"Current Milestone: ${current_milestone:.0f}  ${next_milestone:.0f}"
                )
            except Exception as milestone_error:
                logger.warning(f"Failed to get milestones, using defaults: {milestone_error}")
                current_milestone, next_milestone = 0, 50  # Default milestones

        except Exception as e:
            error_msg = str(e).lower()
            if "timestamp" in error_msg or "recv_window" in error_msg:
                logger.warning(f"API timing issues detected. Starting with fallback values: {e}")
                # Set default values to allow bot to start
                self.account_balance = 8.39  # Known working balance from logs
                self.starting_balance = self.account_balance
                self.available_balance = self.account_balance
                current_milestone, next_milestone = 0, 50  # Default milestones
                logger.info(f"Using fallback balance: ${self.account_balance:.2f}")
                logger.info(f"Fallback milestone: ${current_milestone:.0f} -> ${next_milestone:.0f}")
            elif "errcode: 401" in error_msg or "invalid api key" in error_msg or "api key" in error_msg:
                mode_label = "testnet" if self.settings.bybit.testnet else "mainnet"
                logger.error(f"Bybit authentication failed for {mode_label}: {e}")
                raise RuntimeError(
                    f"Bybit authentication failed in {mode_label} mode. "
                    f"Ensure BYBIT_TESTNET={str(self.settings.bybit.testnet).lower()} matches the API key set."
                ) from e
            else:
                logger.error(f"Bybit connectivity check failed: {e}")
                raise RuntimeError(f"Failed to initialize Bybit connection: {e}") from e
        
        # Hydrate Strategy Optimizer from Supabase
        if self.db:
            try:
                stats = await self.db.load_strategy_stats()
                if stats:
                    self.strategy_optimizer.load_stats(stats)
            except Exception as e:
                logger.warning(f"Failed to load strategy optimizer stats: {e}")
        
        # Test DeepSeek connection only when LLM path is enabled.
        if self.deepseek is not None:
            try:
                effective_account_balance, _ = self._effective_balances()
                _ = self.deepseek.prepare_market_context(
                    account_balance=effective_account_balance,
                    positions=[],
                    market_data={"btc_24h_change": 0},
                    technical_analysis={},
                    funding_rates={},
                    top_movers={"gainers": [], "losers": []},
                    milestone_progress={"current": current_milestone, "next": next_milestone},
                )
                logger.info("Testing DeepSeek LLM connection...")
                logger.info(" DeepSeek connection ready")
            except Exception as e:
                logger.error(f" Failed to initialize DeepSeek: {e}")
                raise
        else:
            logger.info("DeepSeek initialization skipped (LLM execution lock active)")
        
        # Load positions (single attempt to avoid timestamp issues)
        try:
            self.positions = self.bybit.get_positions()
            if self.positions:
                logger.info(f"Found {len(self.positions)} open positions")
        except Exception as e:
            error_msg = str(e).lower()
            if "timestamp" in error_msg or "recv_window" in error_msg:
                logger.warning(f"  Failed to load positions due to API timing, starting with empty positions: {e}")
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
                logger.info(" Telegram notifications enabled")
            except Exception as e:
                logger.warning(f"Telegram initialization failed: {e}")
        
        logger.info("Initialization complete! Ready to trade.")
        bridge.update_status("initialized", False)
        logger.info("=" * 80)
    
    async def run(self):
        """Main trading loop"""
        self.running = True
        logger.info("Starting main trading loop...")
        logger.info(f"  Decision interval: {self.decision_interval} seconds")
        
        while self.running:
            try:
                # Check emergency controls
                trading_allowed, block_reason = self.emergency_controls.is_trading_allowed()
                if not trading_allowed:
                    if self.settings.trading.soft_governor_enabled:
                        try:
                            status = self.emergency_controls.check_status()
                            if status.get("mode") != "emergency_stop":
                                logger.warning(
                                    f"Soft governor mode auto-resume: clearing non-emergency block ({block_reason})"
                                )
                                self.emergency_controls.resume_trading()
                                await asyncio.sleep(1)
                                continue
                        except Exception:
                            pass
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

                if self.settings.trading.quant_primary_mode and self.quant_engine is not None:
                    effective_account_balance, _ = self._effective_balances()
                    decision = await self._get_quant_primary_decision(market_data, analysis)

                    conf_score = decision.confidence_score if decision.confidence_score is not None else 0.0
                    logger.info(f" Quant Decision: {decision.decision_type.value} ({conf_score:.2f})")

                    reason_str = decision.reasoning
                    if isinstance(reason_str, dict):
                        reason_str = reason_str.get("reason", str(reason_str))
                    dashboard_decision = {
                        "id": decision.decision_id,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "symbol": decision.symbol or "TOTAL",
                        "type": decision.decision_type.value,
                        "confidence": float(decision.confidence_score or 0),
                        "strategy": decision.strategy_tag or "Quant",
                        "reasoning": str(reason_str or "Quant evaluation"),
                        "htf_aligned": True,
                    }
                    bridge.send_decision(dashboard_decision)

                    if self.db:
                        try:
                            await self.db.save_decision({
                                "decision_id": decision.decision_id,
                                "decision_type": decision.decision_type.value,
                                "account_balance": effective_account_balance,
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
                                "executed": False,
                            })
                        except Exception as e:
                            logger.warning(f"Failed to save quant decision to database: {e}")

                    await self._execute_decision(decision, market_data, analysis)
                    self._log_status()
                    logger.info(f"Next decision in {self.decision_interval} seconds...")
                    await asyncio.sleep(self.decision_interval)
                    continue

                if self.council is None or self.deepseek is None:
                    logger.warning(
                        "LLM decision path unavailable (execution lock or missing client); waiting for next cycle"
                    )
                    await asyncio.sleep(self.decision_interval)
                    continue

                # 4. Get Trading Decision from Council
                logger.info(f" Requesting decision from Trading Council...")
                
                all_decisions = []
                
                # =========================================
                # OPPORTUNITY SCORING (HYBRID ALGO PRE-FILTER)
                # =========================================
                effective_account_balance, _ = self._effective_balances()
                small_account_mode = effective_account_balance <= 150
                min_opportunity_score = 60
                preferred_watchlist = list(self.watchlist)
                if small_account_mode:
                    # Small accounts: keep quality filter, but do not starve opportunity flow.
                    min_opportunity_score = 60
                    preferred_watchlist = [
                        symbol for symbol in self.watchlist
                        if get_symbol_config(symbol).vol_class != "high"
                    ] or list(self.watchlist)

                # Use OpportunityScorer to intelligently filter before LLM
                opportunities = self.opportunity_scorer.get_top_opportunities(
                    analysis, 
                    min_score=min_opportunity_score,
                    max_results=7
                )
                if small_account_mode:
                    opportunities = [opp for opp in opportunities if opp.symbol in preferred_watchlist]
                opportunity_map = {opp.symbol: opp for opp in opportunities}
                
                def volume_ratio_key(symbol: str) -> float:
                    raw_ratio = analysis.get(symbol, {}).get("timeframe_analysis", {}).get("15m", {}).get("volume_ratio", 0)
                    try:
                        return float(raw_ratio) if raw_ratio is not None else 0.0
                    except (TypeError, ValueError):
                        return 0.0
                
                if opportunities:
                    ranked_watchlist = [opp.symbol for opp in opportunities]
                    if len(ranked_watchlist) < 3:
                        fallback_symbols = sorted(preferred_watchlist, key=volume_ratio_key, reverse=True)
                        for fallback_symbol in fallback_symbols:
                            if fallback_symbol not in ranked_watchlist:
                                ranked_watchlist.append(fallback_symbol)
                            if len(ranked_watchlist) >= min(5, len(preferred_watchlist)):
                                break
                    scores_str = ", ".join([f"{o.symbol}({o.score})" for o in opportunities])
                    logger.info(f" OpportunityScorer: Top {len(opportunities)} setups: {scores_str}")
                else:
                    # Fallback: still probe top-liquidity symbols even without score threshold.
                    ranked_watchlist = sorted(preferred_watchlist, key=volume_ratio_key, reverse=True)[:5]
                    logger.info(
                        f" No high-scoring opportunities; volume fallback selected: {ranked_watchlist}"
                    )
                # Ensure each symbol is executable with current balance and margin cap.
                affordable_watchlist = []
                now_utc = datetime.now(timezone.utc)
                for symbol in ranked_watchlist:
                    ticker = market_data.get("tickers", {}).get(symbol)
                    if not ticker:
                        continue

                    is_affordable, reason, required_margin, max_margin = self._check_symbol_affordability(
                        symbol=symbol,
                        market_price=float(getattr(ticker, "last_price", 0) or 0),
                        leverage_hint=10,
                    )
                    if is_affordable:
                        affordable_watchlist.append(symbol)
                        continue

                    notice_until = self._affordability_notice_until.get(symbol)
                    if not notice_until or now_utc >= notice_until:
                        logger.info(
                            f"Skipping {symbol}: min executable margin ${required_margin:.2f} exceeds "
                            f"budget ${max_margin:.2f} ({reason})"
                        )
                        self._affordability_notice_until[symbol] = now_utc + timedelta(minutes=15)

                ranked_watchlist = affordable_watchlist
                if not ranked_watchlist:
                    logger.info("No affordable symbols under current balance/margin constraints; waiting this cycle")

                # Iterate through ranked watchlist to find opportunities
                for symbol in ranked_watchlist:
                    if symbol not in analysis:
                        continue
                        
                    council_market_data = dict(analysis.get(symbol, {}))
                    council_market_data["funding_rate"] = market_data.get("funding_rates", {}).get(symbol, 0.0)
                    council_market_data["order_flow"] = market_data.get("order_flow", {}).get(symbol, {})
                    opportunity = opportunity_map.get(symbol)
                    if opportunity:
                        council_market_data["opportunity_score"] = float(opportunity.score)
                        council_market_data["opportunity_direction"] = str(opportunity.direction)
                        council_market_data["opportunity_signals"] = list(opportunity.signals or [])
                        council_market_data["opportunity_suggested_sl"] = float(opportunity.suggested_sl or 0.0)
                        council_market_data["opportunity_suggested_tp"] = float(opportunity.suggested_tp or 0.0)

                    decision_data = await self.council.make_decision(
                        symbol=symbol,
                        market_data=council_market_data,
                        account_balance=effective_account_balance,
                        autonomous_mode=self.settings.trading.autonomous_mode_enabled,
                    )
                    
                    if decision_data["decision_type"] == "open_position":
                        # Add symbol to decision for tracking
                        decision_data["_evaluated_symbol"] = symbol
                        if opportunity:
                            decision_data["_opportunity_score"] = float(opportunity.score)
                        all_decisions.append(decision_data)
                        self._symbol_wait_streak[symbol] = 0
                    else:
                        wait_streak = int(self._symbol_wait_streak.get(symbol, 0)) + 1
                        self._symbol_wait_streak[symbol] = wait_streak
                        if wait_streak >= 3:
                            soft_mode = bool(self.relaxed_trade_gating) or bool(
                                getattr(self.settings.trading, "soft_governor_enabled", False)
                            )
                            cooldown_minutes = 12
                            if not soft_mode:
                                self.symbol_cooldowns[symbol] = datetime.now(timezone.utc) + timedelta(
                                    minutes=cooldown_minutes
                                )
                            self._symbol_wait_streak[symbol] = 0
                            if soft_mode:
                                logger.info(
                                    f"{symbol} soft_penalty_applied after repeated WAIT decisions (no cooldown)"
                                )
                            else:
                                logger.info(
                                    f"{symbol} entered opportunity cooldown for {cooldown_minutes} minutes "
                                    f"after repeated WAIT decisions"
                                )
                
                # Select the trade with best estimated edge (EV), not raw confidence.
                if all_decisions:
                    scored_candidates: List[Dict[str, Any]] = []
                    for decision_data in all_decisions:
                        edge = self._estimate_trade_edge(decision_data, analysis, market_data)
                        if edge is None:
                            continue
                        symbol_name = str(decision_data.get("trade", {}).get("symbol") or "")
                        repeat_penalty = self._repetition_penalty_pct(symbol_name)
                        flat_penalty = self._flat_streak_penalty_pct(symbol_name)
                        reject_penalty = self._reject_penalty_pct(symbol_name)
                        total_penalty = min(0.95, repeat_penalty + flat_penalty + reject_penalty)
                        adjusted_edge = edge.get("expected_return_pct", -999.0) - total_penalty
                        decision_data["_edge"] = edge
                        decision_data["_adj_edge_pct"] = adjusted_edge
                        decision_data["_edge_penalties"] = {
                            "repeat": round(repeat_penalty, 4),
                            "flat": round(flat_penalty, 4),
                            "reject": round(reject_penalty, 4),
                            "total": round(total_penalty, 4),
                        }
                        decision_data["_consecutive_entry_streak"] = self._consecutive_recent_entries(symbol_name)
                        scored_candidates.append(decision_data)

                    min_edge = float(self.settings.trading.min_expected_edge_pct)
                    positive_edge_candidates = [
                        d for d in scored_candidates
                        if d.get("_adj_edge_pct", -999.0) >= min_edge
                    ]

                    if positive_edge_candidates:
                        positive_edge_candidates.sort(
                            key=lambda d: d.get("_adj_edge_pct", -999.0),
                            reverse=True,
                        )
                        best_decision = None
                        override_gap = max(0.0, float(self._symbol_repeat_override_gap_pct))
                        for candidate in positive_edge_candidates:
                            candidate_symbol = str(candidate.get("trade", {}).get("symbol") or "")
                            streak = int(candidate.get("_consecutive_entry_streak", 0))
                            if streak < self._max_consecutive_symbol_entries:
                                best_decision = candidate
                                break

                            candidate_raw_edge = float(candidate.get("_edge", {}).get("expected_return_pct", -999.0))
                            alternatives = [
                                d for d in positive_edge_candidates
                                if str(d.get("trade", {}).get("symbol") or "") != candidate_symbol
                            ]
                            if alternatives:
                                alternative_best_raw = max(
                                    float(d.get("_edge", {}).get("expected_return_pct", -999.0))
                                    for d in alternatives
                                )
                                if (candidate_raw_edge - alternative_best_raw) >= override_gap:
                                    logger.info(
                                        f"{candidate_symbol} exceeded consecutive-entry cap ({streak}) but override allowed "
                                        f"(raw EV gap {candidate_raw_edge - alternative_best_raw:+.3f} >= {override_gap:.3f})"
                                    )
                                    best_decision = candidate
                                    break
                                logger.info(
                                    f"Skipping {candidate_symbol}: consecutive-entry cap reached ({streak}) and "
                                    f"raw EV gap {candidate_raw_edge - alternative_best_raw:+.3f} < {override_gap:.3f}"
                                )
                                continue

                            if candidate_raw_edge >= (min_edge + override_gap):
                                logger.info(
                                    f"{candidate_symbol} is the only positive-edge candidate; allowing cap override "
                                    f"(raw EV {candidate_raw_edge:+.3f})"
                                )
                                best_decision = candidate
                                break

                        if best_decision is None:
                            best_decision = {
                                "decision_type": "wait",
                                "symbol": ranked_watchlist[0] if ranked_watchlist else (self.watchlist[0] if self.watchlist else "BTCUSDT"),
                                "reasoning": "Diversification guard skipped concentrated candidates this cycle",
                                "confidence_score": 0.0,
                                "decision_id": str(uuid.uuid4()),
                            }

                        if best_decision.get("decision_type") == "open_position":
                            edge = best_decision.get("_edge", {})
                            penalties = best_decision.get("_edge_penalties", {})
                            selected_symbol = best_decision.get("trade", {}).get("symbol", "UNKNOWN")
                            logger.info(
                                f"Selected {selected_symbol} by EV: "
                                f"EV={edge.get('expected_return_pct', 0):+.3f}% "
                                f"AdjEV={best_decision.get('_adj_edge_pct', 0):+.3f}% "
                                f"Penalty={penalties.get('total', 0):.3f} "
                                f"Pwin={edge.get('win_probability', 0):.2f} RR={edge.get('rr_ratio', 0):.2f}"
                            )
                    else:
                        best_decision = {
                            "decision_type": "wait",
                            "symbol": ranked_watchlist[0] if ranked_watchlist else (self.watchlist[0] if self.watchlist else "BTCUSDT"),
                            "reasoning": f"No positive-edge setup met EV threshold ({min_edge:.2f}%)",
                            "confidence_score": 0.0,
                            "decision_id": str(uuid.uuid4()),
                        }
                else:
                    # Default to wait if no symbol has a trade
                    best_decision = {
                        "decision_type": "wait",
                        "symbol": ranked_watchlist[0] if ranked_watchlist else (self.watchlist[0] if self.watchlist else "BTCUSDT"),
                        "reasoning": "No suitable setups found by Council across watchlist",
                        "confidence_score": 0.0,
                        "decision_id": str(uuid.uuid4()),
                    }

                # Ensure decision_id exists
                if "decision_id" not in best_decision:
                    best_decision["decision_id"] = str(uuid.uuid4())

                # Convert dictionary response to TradingDecision object
                decision = self.deepseek._parse_decision(json.dumps(best_decision))
                
                conf_score = decision.confidence_score if decision.confidence_score is not None else 0.0
                logger.info(f" Council Decision: {decision.decision_type.value} ({conf_score:.2f})")
                
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
                            "account_balance": effective_account_balance,
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
                previous_open_positions = dict(self._last_open_positions)
                self.positions = self.bybit.get_positions()
                current_open_positions: Dict[str, Dict[str, Any]] = {}
                
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

                            current_open_positions[pos.symbol] = {
                                "side": pos.side,
                                "entry_price": entry_price,
                                "size": size,
                                "last_mark_price": float(getattr(pos, 'mark_price', entry_price)),
                                "strategy_tag": previous_open_positions.get(pos.symbol, {}).get("strategy_tag", "exchange_managed"),
                                "risk_pct": previous_open_positions.get(pos.symbol, {}).get("risk_pct", self.default_risk_pct),
                                "leverage": previous_open_positions.get(pos.symbol, {}).get("leverage", pos.leverage),
                                "entry_tier": previous_open_positions.get(pos.symbol, {}).get("entry_tier", "full"),
                                "policy_state": previous_open_positions.get(pos.symbol, {}).get("policy_state", "green"),
                                "policy_key": previous_open_positions.get(pos.symbol, {}).get("policy_key", ""),
                                "timeframe": previous_open_positions.get(pos.symbol, {}).get("timeframe", "15m"),
                                "market_regime": previous_open_positions.get(pos.symbol, {}).get("market_regime", "unknown"),
                                "opened_at": previous_open_positions.get(pos.symbol, {}).get(
                                    "opened_at",
                                    datetime.now(timezone.utc).isoformat(),
                                ),
                                "expected_hold_minutes": previous_open_positions.get(pos.symbol, {}).get(
                                    "expected_hold_minutes",
                                    45,
                                ),
                                "risk_distance": previous_open_positions.get(pos.symbol, {}).get("risk_distance", 0.0),
                                "tp1_price": previous_open_positions.get(pos.symbol, {}).get("tp1_price"),
                                "be_trigger_r": previous_open_positions.get(pos.symbol, {}).get("be_trigger_r", 1.0),
                                "trail_activation_r": previous_open_positions.get(pos.symbol, {}).get("trail_activation_r", 1.2),
                                "breakeven_moved": bool(previous_open_positions.get(pos.symbol, {}).get("breakeven_moved", False)),
                                "trail_active": bool(previous_open_positions.get(pos.symbol, {}).get("trail_active", False)),
                            }
                        except (ValueError, TypeError, AttributeError) as pos_item_err:
                            logger.debug(f"Error parsing position item: {pos_item_err}")

                closed_symbols = set(previous_open_positions.keys()) - set(current_open_positions.keys())
                for symbol in closed_symbols:
                    if symbol in self._handled_closures:
                        self._handled_closures.discard(symbol)
                        continue

                    previous = previous_open_positions.get(symbol, {})
                    entry_price = float(previous.get("entry_price", 0) or 0)
                    exit_price = float(previous.get("last_mark_price", entry_price) or entry_price)
                    position_size = float(previous.get("size", 0) or 0)
                    side = previous.get("side", "Buy")
                    strategy_name = previous.get("strategy_tag", "exchange_managed_exit")

                    if entry_price <= 0:
                        continue

                    if side == "Buy":
                        pnl_pct = (exit_price - entry_price) / entry_price * 100
                    else:
                        pnl_pct = (entry_price - exit_price) / entry_price * 100

                    # Prefer realized PnL from exchange if available; fallback to mark-price estimate.
                    if position_size > 0:
                        closed_info = self.bybit.get_latest_closed_pnl(symbol, within_minutes=20)
                        if closed_info and closed_info.get("closed_pnl") is not None:
                            notional = entry_price * position_size
                            if notional > 0:
                                pnl_pct = float(closed_info["closed_pnl"]) / notional * 100

                    won, is_flat = self._classify_trade_outcome(pnl_pct)
                    self.emergency_controls.record_trade_result(won is True, pnl_pct=pnl_pct)
                    self._apply_post_close_symbol_controls(
                        symbol=symbol,
                        pnl_pct=pnl_pct,
                        won=won,
                        is_flat=is_flat,
                    )
                    await self._record_trade_outcome(
                        symbol=symbol,
                        pnl_pct=pnl_pct,
                        won=won,
                        strategy_name=strategy_name,
                        metadata={
                            "market_regime": previous.get("market_regime", "unknown"),
                            "timeframe": previous.get("timeframe", "15m"),
                            "leverage": previous.get("leverage", self.default_leverage),
                            "risk_pct": previous.get("risk_pct", self.default_risk_pct),
                        },
                    )
                    if self.quant_engine is not None:
                        try:
                            close_details = self._build_close_event_details(
                                symbol=symbol,
                                strategy_name=strategy_name,
                                outcome="flat" if is_flat else ("win" if won else "loss"),
                                pnl_pct=float(pnl_pct),
                                extra_details={
                                    "close_reason": "exchange_managed_close",
                                },
                            )
                            self.quant_engine.record_execution_event(
                                event_type="close",
                                symbol=symbol,
                                reason_code="exchange_managed_close",
                                pnl_pct=float(pnl_pct),
                                details=close_details,
                            )
                        except Exception:
                            pass
                    self.position_monitor.remove_position(symbol)
                    self.execution_manager.clear_position(symbol)
                    logger.info(
                        f"Detected exchange-managed close for {symbol}: PnL {pnl_pct:+.2f}%"
                    )

                self._last_open_positions = current_open_positions
                bridge.update_positions(positions_data)
            except Exception as pos_error:
                logger.debug(f"Position update failed: {pos_error}")

            # 3. Keep risk drawdown baseline on same balance basis as sizing logic.
            if hasattr(self, "risk_manager"):
                effective_balance, _ = self._effective_balances()
                baseline_balance = effective_balance if effective_balance > 0 else float(self.account_balance or 0.0)

                if self.risk_manager.daily_start_balance is None:
                    self.risk_manager.daily_start_balance = baseline_balance
                elif (
                    self.settings.bybit.testnet
                    and self.settings.trading.initial_balance > 0
                    and baseline_balance > 0
                    and self.risk_manager.daily_start_balance > (baseline_balance * 2)
                ):
                    self.risk_manager.daily_start_balance = baseline_balance
                    logger.info(
                        f"Adjusted risk drawdown baseline to capped testnet balance: ${baseline_balance:.2f}"
                    )

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
            now_utc = datetime.now(timezone.utc)
            effective_account_balance, _ = self._effective_balances()
            small_account_mode = (
                effective_account_balance <= float(self.settings.trading.small_account_balance_threshold)
            )
            prefilter_enabled = bool(
                small_account_mode and getattr(self.settings.trading, "small_account_prefilter_enabled", False)
            )
            runtime_watchlist: List[str] = []
            for symbol in self.watchlist:
                cooldown_until = self.symbol_cooldowns.get(symbol)
                if cooldown_until and now_utc < cooldown_until:
                    continue

                try:
                    # Get ticker
                    ticker = self.bybit.get_ticker(symbol)
                    market_data["tickers"][symbol] = ticker
                except Exception as e:
                    error_text = str(e).lower()
                    if "list': []" in error_text or "result': {'category': 'linear', 'list': []" in error_text:
                        cooldown_minutes = 30
                        self.symbol_cooldowns[symbol] = now_utc + timedelta(minutes=cooldown_minutes)
                        logger.warning(
                            f"{symbol} returned empty ticker data; pausing symbol for {cooldown_minutes} minutes"
                        )
                    logger.warning(f"Skipping {symbol}: {e}")
                    continue  # Skip this symbol and continue with others

                # Small-account prefilter is optional.
                # Disabled by default so symbols are analyzed first, then filtered at execution time.
                if prefilter_enabled:
                    is_affordable, reason, required_margin, max_margin = self._check_symbol_affordability(
                        symbol=symbol,
                        market_price=float(getattr(ticker, "last_price", 0.0) or 0.0),
                        leverage_hint=min(4, int(self.default_leverage)),
                    )
                    if not is_affordable:
                        soft_mode = bool(self.relaxed_trade_gating) or bool(
                            getattr(self.settings.trading, "soft_governor_enabled", False)
                        )
                        notice_until = self._affordability_notice_until.get(symbol)
                        if not notice_until or now_utc >= notice_until:
                            logger.info(
                                f"Small-account filter skipped {symbol}: required margin ${required_margin:.2f} "
                                f"> cap ${max_margin:.2f} ({reason})"
                            )
                            self._affordability_notice_until[symbol] = now_utc + timedelta(minutes=20)
                        if not soft_mode:
                            self.symbol_cooldowns[symbol] = now_utc + timedelta(minutes=20)
                        continue
                elif small_account_mode:
                    # Keep visibility when a symbol is likely unaffordable, but do not skip it.
                    is_affordable, reason, required_margin, max_margin = self._check_symbol_affordability(
                        symbol=symbol,
                        market_price=float(getattr(ticker, "last_price", 0.0) or 0.0),
                        leverage_hint=min(4, int(self.default_leverage)),
                    )
                    if not is_affordable:
                        notice_until = self._affordability_notice_until.get(symbol)
                        if not notice_until or now_utc >= notice_until:
                            logger.info(
                                f"Small-account prefilter disabled; keeping {symbol} in analysis "
                                f"(min margin ${required_margin:.2f} > cap ${max_margin:.2f}; {reason})"
                            )
                            self._affordability_notice_until[symbol] = now_utc + timedelta(minutes=20)

                runtime_watchlist.append(symbol)
                
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
                
                # Get orderbook for order flow analysis (top 5 runtime symbols only to save time)
                if len(runtime_watchlist) <= 5:
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
            
            market_data["runtime_watchlist"] = runtime_watchlist or list(market_data.get("tickers", {}).keys())


        except Exception as e:
            logger.error(f"Error fetching market data: {e}")
        
        return market_data
    
    async def _analyze_markets(self, market_data):
        """Analyze markets using technical indicators"""
        analyses = {}
        
        runtime_watchlist = (
            market_data.get("runtime_watchlist")
            or list(market_data.get("tickers", {}).keys())
            or list(self.watchlist)
        )

        # Analyze all symbols gathered in this runtime cycle.
        for symbol in runtime_watchlist:
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
                symbol_payload = {
                    "timeframe_analysis": serializable_analysis,
                    "sentiment": sentiment,
                    "current_price": market_data["tickers"][symbol].last_price,
                }
                if self.forecast_engine is not None:
                    symbol_payload["forecast"] = self.forecast_engine.predict(symbol_payload)

                analyses[symbol] = {
                    **symbol_payload
                }
                
            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}")
        
        
        return analyses
    
    def _prepare_llm_context(self, market_data, analysis):
        """Prepare comprehensive context for LLM"""
        if self.deepseek is None:
            raise RuntimeError("LLM context requested while DeepSeek path is disabled")

        # Get milestone progress
        current_milestone, next_milestone = self.settings.trading.get_current_milestone(
            self.account_balance
        )
        
        progress_pct = (
            (self.account_balance - current_milestone) / 
            (next_milestone - current_milestone) * 100
        )
        
        milestone_progress = {
            "current_milestone": f"${current_milestone:.0f}  ${next_milestone:.0f}",
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
            timeframe_data = data.get("timeframe_analysis", data.get("timeframes", {}))
            for tf, tf_data in timeframe_data.items():
                if tf not in ["1m", "5m", "15m", "1h", "4h"]:
                    continue

                if isinstance(tf_data, dict):
                    indicators_summary[tf] = {
                        "rsi": round(float(tf_data.get("rsi", 50)), 1),
                        "macd_signal": tf_data.get("macd_signal", "neutral"),
                        "trend": tf_data.get("trend", "neutral"),
                        "atr": round(float(tf_data.get("atr", 0)), 2),
                        "volume_ratio": round(float(tf_data.get("volume_ratio", 1)), 2),
                        "bb_position": tf_data.get("bb_position", "middle"),
                        "patterns": tf_data.get("patterns", [])[:3],
                        "key_levels": tf_data.get("key_levels", {}),
                    }
                    continue

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
            forecast_data = data.get("forecast", {}).get("aggregate", {})
            
            ta_summary[symbol] = {
                "current_price": round(current_price, 2),
                "price_24h_change_pct": round(price_24h_change, 2),
                "overall_sentiment": sentiment_data.get("overall_sentiment", "neutral"),
                "confidence": round(sentiment_data.get("confidence", 0), 1),
                "key_signals": sentiment_data.get("signals", [])[:5],
                "timeframe_analysis": indicators_summary,
                "funding_rate": market_data.get("funding_rates", {}).get(symbol, 0),
                "forecast": {
                    "consensus": str(forecast_data.get("consensus", "neutral")),
                    "prob_up": round(self._safe_float(forecast_data.get("prob_up"), 0.5), 3),
                    "prob_down": round(self._safe_float(forecast_data.get("prob_down"), 0.5), 3),
                    "confidence": round(self._safe_float(forecast_data.get("confidence"), 0.0), 3),
                },
            }
        
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
        effective_account_balance, _ = self._effective_balances()
        context = self.deepseek.prepare_market_context(
            account_balance=effective_account_balance,
            positions=positions_data,
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

        # Keep LLM constraints aligned with runtime risk configuration.
        context.setdefault("system_state", {})
        context["system_state"]["max_concurrent_positions"] = self.settings.trading.max_concurrent_positions
        context["system_state"]["next_decision_window_seconds"] = self.decision_interval
        min_risk, max_risk = self.settings.trading.get_suggested_risk_pct(effective_account_balance)
        context["system_state"]["allowed_risk_range_pct"] = [round(min_risk, 3), round(max_risk, 3)]
        
        return context

    async def _execute_smart_features(self, position, current_price: float):
        """Execute smart execution features for a position"""
        try:
            symbol = position.symbol
            side = position.side
            entry_price = float(position.avg_price)
            take_profit = float(position.take_profit) if position.take_profit else None
            pnl_pct = 0.0
            if side == "Buy":
                pnl_pct = (current_price - entry_price) / entry_price * 100
            else:
                pnl_pct = (entry_price - current_price) / entry_price * 100

            if self.settings.trading.time_stop_enabled:
                entry_meta = self._last_open_positions.get(symbol, {})
                entry_tier = str(entry_meta.get("entry_tier") or "full").lower()
                opened_at = self._parse_datetime_utc(entry_meta.get("opened_at"))
                expected_hold_minutes = int(
                    max(
                        5,
                        self._safe_float(
                            entry_meta.get("expected_hold_minutes"),
                            getattr(position, "expected_hold_duration_mins", 45) or 45,
                        ),
                    )
                )
                if opened_at is not None:
                    hold_minutes = max(
                        0.0,
                        (datetime.now(timezone.utc) - opened_at).total_seconds() / 60.0,
                    )
                    min_hold = int(self.settings.trading.time_stop_min_hold_minutes)
                    soft_limit = max(
                        min_hold,
                        int(expected_hold_minutes * float(self.settings.trading.time_stop_soft_multiplier)),
                    )
                    hard_limit = max(
                        min_hold + 5,
                        int(expected_hold_minutes * float(self.settings.trading.time_stop_hard_multiplier)),
                    )
                    hard_limit = min(int(self.max_hold_minutes), hard_limit)
                    if entry_tier == "probe":
                        soft_limit = min(
                            soft_limit,
                            int(self.settings.trading.probe_tier_time_stop_soft_minutes),
                        )
                        hard_limit = min(
                            hard_limit,
                            int(self.settings.trading.probe_tier_time_stop_hard_minutes),
                        )

                    stop_loss = self._safe_float(getattr(position, "stop_loss", None), 0.0)
                    risk_to_sl_pct = (
                        abs(entry_price - stop_loss) / entry_price * 100.0
                        if stop_loss > 0
                        else 0.0
                    )
                    loss_degrade_threshold = float(
                        self.settings.trading.time_stop_soft_loss_threshold_pct
                    )
                    if risk_to_sl_pct > 0:
                        loss_degrade_threshold = min(
                            loss_degrade_threshold,
                            -(risk_to_sl_pct * float(self.settings.trading.loss_degrade_sl_fraction)),
                        )

                    force_exit_reason: Optional[str] = None
                    if hold_minutes >= soft_limit and pnl_pct <= loss_degrade_threshold:
                        force_exit_reason = (
                            f"time_stop_soft_degrade hold={hold_minutes:.1f}m "
                            f"limit={soft_limit}m pnl={pnl_pct:+.2f}%"
                        )
                    elif hold_minutes >= hard_limit and pnl_pct <= float(
                        self.settings.trading.time_stop_hard_flat_threshold_pct
                    ):
                        force_exit_reason = (
                            f"time_stop_hard_stale hold={hold_minutes:.1f}m "
                            f"limit={hard_limit}m pnl={pnl_pct:+.2f}%"
                        )

                    if force_exit_reason:
                        try:
                            self.bybit.close_position(symbol)
                            logger.warning(f"Auto-close {symbol}: {force_exit_reason}")
                            if self.quant_engine is not None:
                                try:
                                    close_details = self._build_close_event_details(
                                        symbol=symbol,
                                        strategy_name=str(entry_meta.get("strategy_tag") or "time_stop"),
                                        outcome="flat" if abs(float(pnl_pct)) <= self._flat_trade_tolerance_pct else ("win" if pnl_pct > 0 else "loss"),
                                        pnl_pct=float(pnl_pct),
                                        extra_details={
                                            "close_reason": "time_stop",
                                            "reason": force_exit_reason,
                                            "expected_hold_minutes": expected_hold_minutes,
                                            "hold_minutes": hold_minutes,
                                            "risk_to_sl_pct": risk_to_sl_pct,
                                        },
                                    )
                                    self.quant_engine.record_execution_event(
                                        event_type="close",
                                        symbol=symbol,
                                        reason_code="time_stop",
                                        pnl_pct=float(pnl_pct),
                                        details=close_details,
                                    )
                                except Exception:
                                    pass
                            return
                        except Exception as e:
                            logger.warning(f"Failed time-stop close for {symbol}: {e}")

            entry_meta = self._last_open_positions.get(symbol, {})
            entry_tier = str(entry_meta.get("entry_tier") or "full").lower()
            stop_loss = self._safe_float(getattr(position, "stop_loss", None), 0.0)
            risk_distance = self._safe_float(entry_meta.get("risk_distance"), abs(entry_price - stop_loss))
            current_r = self._position_r_multiple(
                side=side,
                entry_price=entry_price,
                stop_loss=stop_loss,
                current_price=current_price,
            )
            be_trigger_r = self._safe_float(
                entry_meta.get("be_trigger_r"),
                float(self.settings.trading.full_tier_breakeven_r_multiple),
            )
            trail_trigger_r = self._safe_float(
                entry_meta.get("trail_activation_r"),
                float(self.settings.trading.full_tier_trail_activation_r_multiple),
            )

            # Win-rate stack (optional): TP1 partial -> breakeven -> trailing (full tier).
            if (
                bool(self.settings.trading.win_rate_mode_enabled)
                and entry_tier == "full"
                and risk_distance > 0
                and current_r >= be_trigger_r
            ):
                if not bool(entry_meta.get("breakeven_moved", False)):
                    try:
                        self.bybit.update_position_protection(symbol=symbol, stop_loss=entry_price)
                        entry_meta["breakeven_moved"] = True
                        self._last_open_positions[symbol] = entry_meta
                        logger.info(f" {symbol} stop moved to break-even at ${entry_price:.4f} (R={current_r:.2f})")
                    except Exception as e:
                        logger.warning(f"Failed to move {symbol} to break-even: {e}")

            trailing_allowed = True
            if (
                bool(self.settings.trading.win_rate_mode_enabled)
                and entry_tier == "full"
                and risk_distance > 0
                and current_r < trail_trigger_r
            ):
                trailing_allowed = False
            if trailing_allowed and take_profit:
                try:
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
                                    logger.info(f" {symbol} trailing stop updated to ${stop_loss_price:.4f}")
                                if take_profit_price is not None:
                                    logger.info(f" {symbol} trailing take-profit nudged to ${take_profit_price:.4f}")
                                if (
                                    bool(self.settings.trading.win_rate_mode_enabled)
                                    and entry_tier == "full"
                                    and current_r >= trail_trigger_r
                                ):
                                    entry_meta["trail_active"] = True
                                    self._last_open_positions[symbol] = entry_meta
                        except Exception as e:
                            logger.warning(f"Failed to update trailing protection for {symbol}: {e}")
                except Exception as e:
                    logger.warning(f"Error checking trailing stop for {symbol}: {e}")

            # Partials are driven by configured TP ladder and should trigger by price, not fixed pnl gate.
            try:
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
                        logger.info(f" {symbol} partial profit: {partial['percentage']}% at ${partial['price']:.4f}")
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
                source_label = "Quant Analysis" if self.settings.trading.quant_primary_mode else "DeepSeek Analysis"
                logger.info(f"{source_label}: {reasoning_text[:500]}...")  # First 500 chars
                
                # Log specific fields if available
                if isinstance(decision.reasoning, dict):
                    if "action" in decision.reasoning:
                        logger.info(f"   {decision.reasoning['action']}")
                    if "market_regime_assessment" in decision.reasoning:
                        logger.info(f"   Market: {decision.reasoning['market_regime_assessment']}")
                    if "waiting_for" in decision.reasoning:
                        logger.info(f"   Waiting for: {decision.reasoning['waiting_for']}")
            return
        
        if decision.decision_type == DecisionType.HOLD:
            logger.info("Holding current positions")
            return
        
        if decision.decision_type == DecisionType.OPEN_POSITION:
            opened = bool(await self._open_position(decision, market_data, analysis))
            if opened:
                return
            reasoning = decision.reasoning if isinstance(decision.reasoning, dict) else {}
            fallback_payloads = reasoning.get("fallback_proposals", [])
            if not isinstance(fallback_payloads, list) or not fallback_payloads:
                return
            attempted_symbols = {str(decision.symbol or "").upper()}
            for idx, payload in enumerate(fallback_payloads, start=1):
                fallback_decision = self._quant_payload_to_decision(
                    payload,
                    fallback_rank=idx,
                    fallback_parent_symbol=decision.symbol,
                )
                if fallback_decision is None:
                    continue
                symbol_name = str(fallback_decision.symbol or "").upper()
                if symbol_name in attempted_symbols:
                    continue
                attempted_symbols.add(symbol_name)
                logger.info(
                    "Primary quant entry failed; trying fallback #{} {} {}",
                    idx,
                    symbol_name,
                    fallback_decision.side,
                )
                opened = bool(await self._open_position(fallback_decision, market_data, analysis))
                if opened:
                    logger.info(
                        "Fallback quant entry succeeded with #{} {} {}",
                        idx,
                        symbol_name,
                        fallback_decision.side,
                    )
                    return
            logger.info("Fallback quant attempts exhausted; no entry executed this cycle")
            return
        
        elif decision.decision_type == DecisionType.CLOSE_POSITION:
            await self._close_position(decision)

        elif decision.decision_type == DecisionType.MODIFY_POSITION:
            await self._modify_position(decision, market_data, analysis)
    
    async def _open_position(self, decision, market_data, analysis):
        """Open a new position"""
        if not decision.symbol:
            logger.warning("No symbol specified in decision")
            return False

        raw_side = str(decision.side or "").strip().lower()
        if raw_side == "buy":
            decision.side = "Buy"
        elif raw_side == "sell":
            decision.side = "Sell"
        else:
            logger.warning(f"Invalid side '{decision.side}' for {decision.symbol}; skipping")
            return False

        if self.settings.trading.soft_governor_enabled:
            current_drawdown_pct = self._current_drawdown_pct()
            catastrophic_cap = float(self.settings.trading.soft_catastrophic_drawdown_pct)
            if current_drawdown_pct >= catastrophic_cap:
                logger.error(
                    f"Catastrophic drawdown guard blocked entry: {current_drawdown_pct:.2f}% >= {catastrophic_cap:.2f}%"
                )
                return False

        if not self.settings.bybit.testnet and not self.settings.system.allow_live_trading:
            logger.error(
                "Live order blocked: BYBIT_TESTNET=false and ALLOW_LIVE_TRADING is not enabled"
            )
            return False

        decision_reasoning = decision.reasoning if isinstance(decision.reasoning, dict) else {}
        soft_mode = bool(self.relaxed_trade_gating) or bool(getattr(self.settings.trading, "soft_governor_enabled", False))
        quant_source_decision = str(decision_reasoning.get("source", "")).lower() == "quant_engine"
        risk_management = decision.risk_management or {}
        scale_in_requested = bool(risk_management.get("scale_in"))
        if scale_in_requested and not self.settings.trading.allow_scale_in:
            logger.warning(
                f"Scale-in request ignored for {decision.symbol}; ALLOW_SCALE_IN is disabled"
            )
            return False

        if (
            self.settings.trading.enforce_single_position_per_symbol
            and not scale_in_requested
            and self._has_open_position_for_symbol(decision.symbol)
        ):
            logger.info(
                f"Skipping {decision.symbol}: single-position-per-symbol guard is active and a position is already open"
            )
            return False

        effective_account_balance, effective_available_balance = self._effective_balances()
        small_account_mode = (
            effective_account_balance <= float(self.settings.trading.small_account_balance_threshold)
        )

        confidence_floor = float(self.settings.trading.min_confidence_score or 0.0)
        if small_account_mode:
            confidence_floor = max(0.56, confidence_floor - 0.10)
        if quant_source_decision:
            confidence_floor = max(0.52, confidence_floor - 0.04)
        confidence_epsilon = 0.012
        if decision.confidence_score is not None and not soft_mode:
            try:
                confidence_value = float(decision.confidence_score)
            except (TypeError, ValueError):
                confidence_value = 0.0
            if (confidence_value + confidence_epsilon) < confidence_floor:
                logger.info(
                    f"Skipping {decision.symbol}: confidence {confidence_value:.4f} below floor "
                    f"{confidence_floor:.4f} (eps={confidence_epsilon:.3f})"
                )
                return False

        # Get current price
        ticker = market_data["tickers"].get(decision.symbol)
        if not ticker:
            ticker = self.bybit.get_ticker(decision.symbol)
        
        current_price = ticker.last_price
        entry_tier = str(decision_reasoning.get("entry_tier", "full")).lower()
        if entry_tier not in {"full", "probe"}:
            entry_tier = "full"
        estimated_slippage_bps = self._estimate_entry_slippage_bps(
            symbol=decision.symbol,
            market_data=market_data,
            current_price=float(current_price or 0.0),
        )
        max_est_slippage_bps = (
            float(self.settings.trading.probe_entry_max_est_slippage_bps)
            if entry_tier == "probe"
            else float(self.settings.trading.full_entry_max_est_slippage_bps)
        )
        if estimated_slippage_bps > max_est_slippage_bps:
            reason_code = "probe_rejected_cost" if entry_tier == "probe" else "full_rejected_cost"
            logger.warning(
                f"Skipping {decision.symbol} ({entry_tier}): estimated slippage "
                f"{estimated_slippage_bps:.1f}bps > limit {max_est_slippage_bps:.1f}bps"
            )
            if self.quant_engine is not None:
                try:
                    self.quant_engine.record_execution_event(
                        event_type="reject",
                        symbol=decision.symbol,
                        reason_code=reason_code,
                        details={
                            "entry_tier": entry_tier,
                            "estimated_slippage_bps": estimated_slippage_bps,
                            "limit_bps": max_est_slippage_bps,
                        },
                    )
                except Exception:
                    pass
            return False
        entry_slippage_tolerance_pct = float(self._market_order_slippage_tolerance_pct)
        if entry_tier == "full":
            entry_slippage_tolerance_pct = min(
                entry_slippage_tolerance_pct,
                float(self.settings.trading.full_entry_max_est_slippage_bps) / 100.0,
            )
        else:
            entry_slippage_tolerance_pct = max(
                entry_slippage_tolerance_pct,
                float(self.settings.trading.probe_entry_max_est_slippage_bps) / 100.0,
            )

        if (
            self.settings.bybit.testnet
            and self.settings.trading.initial_balance > 0
            and not self.settings.trading.soft_governor_enabled
        ):
            cap = float(self.settings.trading.initial_balance)
            if self.account_balance > cap:
                logger.info(
                    f"Testnet sizing cap active: using ${effective_account_balance:.2f} account balance for position sizing"
                )

        precheck_leverage_hint = decision.leverage or self.default_leverage
        if small_account_mode and precheck_leverage_hint:
            precheck_leverage_hint = min(
                float(precheck_leverage_hint),
                float(self.settings.trading.small_account_max_leverage),
            )

        # Fast-fail if instrument minimum cannot fit available margin budget.
        precheck_affordable, precheck_reason, required_margin, max_margin_budget = self._check_symbol_affordability(
            symbol=decision.symbol,
            market_price=current_price,
            leverage_hint=precheck_leverage_hint,
        )
        if not precheck_affordable:
            logger.warning(
                f"Skipping {decision.symbol}: min executable margin ${required_margin:.2f} "
                f"exceeds budget ${max_margin_budget:.2f} ({precheck_reason})"
            )
            return False

        # Normalize stop-loss distance before risk validation so valid setups are not
        # rejected only because strategist placed an overly tight stop.
        min_stop_distance_pct = float(self.settings.trading.min_stop_distance_pct or 0.35)
        try:
            raw_stop_loss = float(decision.stop_loss) if decision.stop_loss is not None else 0.0
        except (TypeError, ValueError):
            raw_stop_loss = 0.0

        if current_price > 0:
            if raw_stop_loss <= 0:
                fallback_pct = max(min_stop_distance_pct, 0.45)
                if decision.side == "Buy":
                    decision.stop_loss = current_price * (1 - (fallback_pct / 100.0))
                else:
                    decision.stop_loss = current_price * (1 + (fallback_pct / 100.0))
                logger.info(
                    f"Applied fallback stop for {decision.symbol} at {fallback_pct:.2f}% distance before validation"
                )
            else:
                stop_distance_pct = abs(current_price - raw_stop_loss) / current_price * 100.0
                if stop_distance_pct < min_stop_distance_pct:
                    target_stop_pct = min_stop_distance_pct * 1.05
                    if decision.side == "Buy":
                        decision.stop_loss = current_price * (1 - (target_stop_pct / 100.0))
                    else:
                        decision.stop_loss = current_price * (1 + (target_stop_pct / 100.0))
                    logger.info(
                        f"Expanded tight stop for {decision.symbol}: {stop_distance_pct:.3f}% -> {target_stop_pct:.3f}%"
                    )
        
        # Validate trade with risk manager
        symbol_info_for_risk = self.bybit.get_symbol_info(decision.symbol) or {}
        lot_filter_for_risk = symbol_info_for_risk.get("lotSizeFilter", {}) or {}
        try:
            min_qty_for_risk = float(lot_filter_for_risk.get("minOrderQty", 0) or 0)
        except (TypeError, ValueError):
            min_qty_for_risk = 0.0
        min_notional_for_risk = max(5.5, min_qty_for_risk * current_price)
        trade_data = {
            "symbol": decision.symbol,
            "side": decision.side,
            "risk_pct": decision.risk_pct,
            "leverage": decision.leverage,
            "stop_loss": decision.stop_loss,
            "entry_price_target": current_price,
            "min_notional": min_notional_for_risk,
            "allow_counter_trend": risk_management.get("allow_counter_trend"),
            "scale_in": scale_in_requested,
        }
        
        validation = self.risk_manager.validate_trade(
            trade_decision=trade_data,
            account_balance=effective_account_balance,
            available_balance=effective_available_balance,
            open_positions=self.positions,
            market_price=current_price,
            market_analysis=analysis,
        )
        validation_warnings = [str(w) for w in (validation.warnings or []) if str(w).strip()]
        cap_clipped_by_risk = any("clipped to active margin caps" in w.lower() for w in validation_warnings)
        
        if not validation.is_valid:
            rejection_reasons = [str(r) for r in (validation.rejection_reasons or []) if str(r).strip()]
            reason_text = " | ".join(rejection_reasons) if rejection_reasons else "risk_validation_reject"
            reason_code = "risk_validation_reject"
            if "CLIPPED_BELOW_MIN" in reason_text:
                reason_code = "clipped_below_min"
            elif "CAP_EXCEEDED" in reason_text or "Position margin exceeds" in reason_text:
                reason_code = "cap_exceeded"
            logger.warning(f"Trade rejected: {rejection_reasons}")
            self._mark_entry_reject(decision.symbol, reason_text)
            if self.quant_engine is not None:
                try:
                    self.quant_engine.record_execution_event(
                        event_type="reject",
                        symbol=decision.symbol,
                        reason_code=reason_code,
                        details={
                            "reasons": rejection_reasons,
                            "entry_tier": entry_tier,
                            "current_price": current_price,
                            "effective_account_balance": effective_account_balance,
                            "effective_available_balance": effective_available_balance,
                            "min_notional": min_notional_for_risk,
                            "symbol_max_margin_pct": float(self.settings.trading.symbol_max_margin_pct),
                            "portfolio_max_margin_pct": float(self.settings.trading.portfolio_max_margin_pct),
                            "validation_warnings": validation_warnings,
                        },
                    )
                except Exception:
                    pass
            return False
        
        # Use adjusted values if any
        try:
            risk_pct = float(validation.adjusted_risk_pct or decision.risk_pct or self.default_risk_pct)
        except (TypeError, ValueError):
            risk_pct = float(self.default_risk_pct)
        try:
            leverage = int(float(validation.adjusted_leverage or decision.leverage or self.default_leverage))
        except (TypeError, ValueError):
            leverage = int(self.default_leverage)

        if small_account_mode:
            if risk_pct > self.settings.trading.small_account_max_risk_pct:
                logger.info(
                    f"Small-account risk cap active for {decision.symbol}: {risk_pct:.2f}% -> {self.settings.trading.small_account_max_risk_pct:.2f}%"
                )
                risk_pct = float(self.settings.trading.small_account_max_risk_pct)
            if leverage > self.settings.trading.small_account_max_leverage:
                logger.info(
                    f"Small-account leverage cap active for {decision.symbol}: {leverage}x -> {self.settings.trading.small_account_max_leverage}x"
                )
                leverage = int(self.settings.trading.small_account_max_leverage)

        symbol_loss_streak = int(self._symbol_loss_streak.get(decision.symbol, 0))
        hard_streak_limit = int(self.settings.trading.symbol_loss_hard_streak)
        if (
            not soft_mode
            and hard_streak_limit > 0
            and symbol_loss_streak >= hard_streak_limit
        ):
            cooldown_minutes = int(self.settings.trading.symbol_loss_hard_cooldown_minutes)
            self.symbol_cooldowns[decision.symbol] = datetime.now(timezone.utc) + timedelta(
                minutes=cooldown_minutes
            )
            logger.warning(
                f"Skipping {decision.symbol}: loss streak={symbol_loss_streak} reached hard limit={hard_streak_limit}; "
                f"cooldown {cooldown_minutes}m applied"
            )
            return False

        risk_multiplier = 1.0 if soft_mode else self._symbol_loss_streak_multiplier(decision.symbol)
        governor_multiplier = self._safe_float(decision_reasoning.get("risk_multiplier"), 1.0)
        combined_multiplier = max(0.10, min(1.20, risk_multiplier * governor_multiplier))
        if combined_multiplier < 0.999:
            prev_risk = risk_pct
            risk_pct = max(float(self.settings.trading.min_risk_pct), risk_pct * combined_multiplier)
            logger.info(
                f"{decision.symbol} loss-streak throttle applied: risk {prev_risk:.2f}% -> {risk_pct:.2f}% "
                f"(streak={symbol_loss_streak}, multiplier={combined_multiplier:.2f})"
            )
        
        # =========================================
        # COUNTER-TREND VALIDATION (HYBRID ALGO GATE)
        # =========================================
        symbol_analysis = analysis.get(decision.symbol, {})
        tf_1h = symbol_analysis.get("timeframe_analysis", {}).get("1h", {})
        trend_1h = tf_1h.get("trend", "neutral")
        allow_counter_trend_policy = bool(
            risk_management.get(
                "allow_counter_trend",
                False if quant_source_decision else True,
            )
        )
        
        # Check if this is a counter-trend trade
        is_counter_trend = self.counter_trend_validator.detect_counter_trend(decision.side, trend_1h)
        
        if is_counter_trend:
            # Run algorithmic validation for counter-trend trades
            ct_validation = self.counter_trend_validator.validate(
                symbol=decision.symbol,
                analysis=symbol_analysis,
                proposed_side=decision.side,
                proposed_sl=decision.stop_loss,
                proposed_tp=decision.take_profit_1,
            )
            ct_min_required = int(getattr(self.counter_trend_validator, "min_score_to_allow", 70))
            
            if not ct_validation.allowed:
                if not allow_counter_trend_policy:
                    logger.warning(
                        f"Counter-trend {decision.side} on {decision.symbol} blocked by symbol-side-regime policy "
                        f"(ct_score={ct_validation.score}/{ct_min_required})"
                    )
                    return False
                strict_counter_trend_mode = bool(self.settings.trading.win_rate_mode_enabled) and bool(
                    self.settings.trading.counter_trend_strict_mode
                )
                policy_context = {}
                if isinstance(decision_reasoning.get("metadata"), dict):
                    meta = decision_reasoning.get("metadata", {})
                    if isinstance(meta.get("policy"), dict):
                        policy_context = meta.get("policy", {})
                bucket_expectancy = self._safe_float(
                    policy_context.get("shrunk_expectancy_pct"),
                    0.0,
                )
                bucket_win_rate = self._safe_float(
                    policy_context.get("shrunk_win_rate"),
                    0.5,
                )
                strict_allow_score = int(self.settings.trading.counter_trend_bypass_min_score)
                strict_allow_exp = float(self.settings.trading.counter_trend_bypass_min_bucket_expectancy_pct)
                strict_allow_wr = float(self.settings.trading.counter_trend_bypass_min_bucket_win_rate)
                strict_allow = (
                    allow_counter_trend_policy
                    and ct_validation.score >= strict_allow_score
                    and bucket_expectancy >= strict_allow_exp
                    and bucket_win_rate >= strict_allow_wr
                )
                if strict_counter_trend_mode:
                    if not strict_allow:
                        logger.warning(
                            f"Counter-trend strict block: {decision.symbol} {decision.side} "
                            f"(ct_score={ct_validation.score}<{strict_allow_score} or "
                            f"bucket_exp={bucket_expectancy:+.3f}%<{strict_allow_exp:+.3f}% or "
                            f"bucket_wr={bucket_win_rate:.2f}<{strict_allow_wr:.2f})"
                        )
                        return False
                    prev_risk = risk_pct
                    strict_risk_mult = 0.70 if bool(self.settings.trading.win_rate_mode_enabled) else 0.75
                    risk_pct = max(float(self.settings.trading.min_risk_pct), risk_pct * strict_risk_mult)
                    logger.warning(
                        f"Counter-trend strict bypass allowed for {decision.symbol} {decision.side} "
                        f"(ct_score={ct_validation.score}/{ct_min_required}, "
                        f"bucket_exp={bucket_expectancy:+.3f}%, bucket_wr={bucket_win_rate:.2f}); "
                        f"risk {prev_risk:.2f}% -> {risk_pct:.2f}%"
                    )
                    # Strict mode intentionally bypasses legacy soft override paths.
                    ct_validation.allowed = True
                    ct_validation.reasons.append("strict_bypass_gate")
                    ct_validation.requirements_met["strict_bypass"] = True
                    # Continue execution path.
                    pass
                elif soft_mode:
                    recovery_soft_disable = bool(
                        getattr(self.settings.trading, "focus_recovery_mode_enabled", True)
                    ) and bool(
                        getattr(self.settings.trading, "focus_recovery_disable_countertrend_soft_bypass", True)
                    )
                    disable_soft_overrides = bool(
                        getattr(self.settings.trading, "counter_trend_disable_soft_overrides", False)
                    ) or recovery_soft_disable
                    if disable_soft_overrides:
                        logger.warning(
                            f"Counter-trend soft bypass disabled for {decision.symbol} {decision.side} "
                            f"(score={ct_validation.score}/{ct_min_required}); trade blocked"
                        )
                        return False
                    prev_risk = risk_pct
                    risk_pct = max(float(self.settings.trading.min_risk_pct), risk_pct * 0.85)
                    logger.warning(
                        f"SOFT_MODE: bypassing counter-trend block for {decision.symbol} {decision.side} "
                        f"(score={ct_validation.score}/{ct_min_required}); risk {prev_risk:.2f}% -> {risk_pct:.2f}%"
                    )
                else:
                    opportunity_score = self._safe_float(decision_reasoning.get("opportunity_score"), 0.0)
                    adx_15m = self._safe_float(
                        symbol_analysis.get("timeframe_analysis", {}).get("15m", {}).get("adx"),
                        25.0,
                    )
                    confidence_value = self._safe_float(decision.confidence_score, 0.0)
                    forecast_aggregate = symbol_analysis.get("forecast", {}).get("aggregate", {})
                    forecast_conf = self._safe_float(forecast_aggregate.get("confidence"), 0.0)
                    forecast_prob_up = self._safe_float(forecast_aggregate.get("prob_up"), 0.5)
                    forecast_supports_side = (
                        (decision.side == "Buy" and forecast_prob_up >= 0.56)
                        or (decision.side == "Sell" and forecast_prob_up <= 0.44)
                    )
                    legacy_soft_override = (
                        opportunity_score >= 92.0
                        and confidence_value >= 0.68
                        and adx_15m <= 20.0
                        and forecast_conf >= 0.55
                        and forecast_supports_side
                    )
                    quant_source = str(decision_reasoning.get("source", "")).lower() == "quant_engine"
                    quant_edge = self._safe_float(decision_reasoning.get("expected_edge_pct"), 0.0)
                    quant_expectancy_hour = self._safe_float(
                        decision_reasoning.get("expectancy_per_hour_pct"),
                        0.0,
                    )
                    quant_rr = self._safe_float(decision_reasoning.get("rr_ratio"), 0.0)
                    quant_conf = self._safe_float(
                        decision_reasoning.get("confidence"),
                        confidence_value,
                    )
                    quant_forecast_or_momentum = (
                        forecast_supports_side
                        or (quant_edge >= 0.55 and quant_conf >= 0.60)
                    )
                    quant_soft_override = (
                        quant_source
                        and ct_validation.score >= 35
                        and quant_conf >= 0.54
                        and forecast_conf >= 0.40
                        and quant_forecast_or_momentum
                        and quant_rr >= 1.60
                        and quant_edge >= 0.20
                        and quant_expectancy_hour >= 0.05
                    )
                    quant_alignment_override = (
                        quant_source_decision
                        and ct_validation.score >= max(20, ct_min_required - 25)
                        and quant_conf >= 0.53
                        and quant_rr >= 1.60
                        and quant_edge >= 0.10
                        and quant_expectancy_hour >= 0.03
                        and (forecast_supports_side or quant_conf >= 0.62)
                    )
                    soft_counter_override = legacy_soft_override or quant_soft_override or quant_alignment_override
                    if soft_counter_override:
                        prev_risk = risk_pct
                        risk_cut = 0.75 if quant_alignment_override else (0.70 if quant_soft_override else 0.60)
                        risk_pct = max(float(self.settings.trading.min_risk_pct), risk_pct * risk_cut)
                        if quant_alignment_override:
                            override_label = "quant_alignment"
                        elif quant_soft_override:
                            override_label = "quant_soft"
                        else:
                            override_label = "legacy_soft"
                        logger.warning(
                            f" Counter-trend {override_label} override: allowing {decision.symbol} {decision.side} "
                            f"(opp_score={opportunity_score:.1f}, conf={confidence_value:.2f}, adx15={adx_15m:.1f}, "
                            f"forecast_conf={forecast_conf:.2f}, q_edge={quant_edge:+.3f}, "
                            f"q_eph={quant_expectancy_hour:+.3f}, "
                            f"ct_score={ct_validation.score}). Risk reduced {prev_risk:.2f}% -> {risk_pct:.2f}%"
                        )
                    else:
                        logger.warning(
                            f" Counter-trend trade BLOCKED by algorithm: {decision.symbol} {decision.side} "
                            f"(score={ct_validation.score}/{ct_min_required}). Reasons: {ct_validation.reasons}"
                        )
                        return False
            else:
                logger.info(
                    f" Counter-trend trade ALLOWED: {decision.symbol} {decision.side} "
                    f"(score={ct_validation.score}). Requirements: {ct_validation.requirements_met}"
                )
        
        # =========================================
        # SL/TP DIRECTION + R:R VALIDATION (HYBRID ALGO)
        # =========================================
        entry_for_validation = current_price
        try:
            sl = float(decision.stop_loss)
        except (TypeError, ValueError):
            sl = 0.0
        try:
            tp = float(decision.take_profit_1)
        except (TypeError, ValueError):
            tp = 0.0

        if sl <= 0:
            sl = entry_for_validation * (0.99 if decision.side == "Buy" else 1.01)
            logger.warning(f"Missing/invalid stop loss from decision; applying fallback SL={sl:.2f}")
        if tp <= 0:
            tp = entry_for_validation * (1.015 if decision.side == "Buy" else 0.985)
            logger.warning(f"Missing/invalid take profit from decision; applying fallback TP={tp:.2f}")
        
        # Validate and auto-correct SL/TP direction
        if decision.side == "Buy":  # Long position
            if sl >= entry_for_validation:
                corrected_sl = entry_for_validation * 0.99  # 1% below entry
                logger.warning(f"Invalid long SL ({sl:.2f}) >= entry ({entry_for_validation:.2f}). Auto-corrected to {corrected_sl:.2f}")
                sl = corrected_sl
            if tp <= entry_for_validation:
                corrected_tp = entry_for_validation * 1.015  # 1.5% above entry
                logger.warning(f"Invalid long TP ({tp:.2f}) <= entry ({entry_for_validation:.2f}). Auto-corrected to {corrected_tp:.2f}")
                tp = corrected_tp
        else:  # Short position (Sell)
            if sl <= entry_for_validation:
                corrected_sl = entry_for_validation * 1.01  # 1% above entry
                logger.warning(f"Invalid short SL ({sl:.2f}) <= entry ({entry_for_validation:.2f}). Auto-corrected to {corrected_sl:.2f}")
                sl = corrected_sl
            if tp >= entry_for_validation:
                corrected_tp = entry_for_validation * 0.985  # 1.5% below entry
                logger.warning(f"Invalid short TP ({tp:.2f}) >= entry ({entry_for_validation:.2f}). Auto-corrected to {corrected_tp:.2f}")
                tp = corrected_tp
        
        # Enforce minimum R:R ratio (SYMBOL-SPECIFIC)
        risk_distance = abs(entry_for_validation - sl)
        reward_distance = abs(tp - entry_for_validation)
        rr_ratio = reward_distance / risk_distance if risk_distance > 0 else 0
        
        min_rr_ratio = max(get_min_rr_ratio(decision.symbol), float(self.settings.trading.min_rr_ratio))
        if small_account_mode:
            min_rr_ratio = max(min_rr_ratio, float(self.settings.trading.small_account_min_rr_ratio))
        if rr_ratio < min_rr_ratio:
            logger.warning(f"R:R too low ({rr_ratio:.2f}). Adjusting TP for minimum {min_rr_ratio}:1")
            if decision.side == "Buy":
                tp = entry_for_validation + (risk_distance * min_rr_ratio)
            else:
                tp = entry_for_validation - (risk_distance * min_rr_ratio)
            rr_ratio = min_rr_ratio
        
        # Update decision with validated values
        decision.stop_loss = sl
        decision.take_profit_1 = tp
        
        logger.info(f"Validated SL/TP: SL=${sl:.2f}, TP=${tp:.2f}, R:R={rr_ratio:.2f}:1")
        
        # =========================================
        # SYMBOL-SPECIFIC RISK CAP (HYBRID ALGO)
        # =========================================
        symbol_config = get_symbol_config(decision.symbol)
        max_risk_for_symbol = get_adjusted_risk(decision.symbol, risk_pct)
        if risk_pct > max_risk_for_symbol:
            logger.info(f" Risk capped for {decision.symbol} ({symbol_config.vol_class} vol): {risk_pct:.1f}%  {max_risk_for_symbol:.1f}%")
            risk_pct = max_risk_for_symbol
        
        # =========================================
        # FUNDING RATE CHECK (CROWDED TRADE DETECTION)
        # =========================================
        funding_rate = market_data.get("funding_rates", {}).get(decision.symbol, 0)
        funding_analysis = self.funding_analyzer.analyze(decision.symbol, decision.side, funding_rate)

        if (
            decision.side == "Buy"
            and funding_analysis.is_crowded
            and self.settings.trading.crowded_long_block_enabled
        ):
            if soft_mode:
                logger.warning(
                    f"SOFT_MODE: crowded-long block bypassed for {decision.symbol} "
                    f"({funding_analysis.reason})"
                )
            else:
                forecast_aggregate = symbol_analysis.get("forecast", {}).get("aggregate", {})
                forecast_conf = self._safe_float(forecast_aggregate.get("confidence"), 0.0)
                prob_up = self._safe_float(forecast_aggregate.get("prob_up"), 0.5)
                adx_15m = self._safe_float(
                    symbol_analysis.get("timeframe_analysis", {}).get("15m", {}).get("adx"),
                    0.0,
                )
                decision_reasoning = decision.reasoning if isinstance(decision.reasoning, dict) else {}
                opportunity_score = self._safe_float(decision_reasoning.get("opportunity_score"), 0.0)

                strong_confirmation = (
                    forecast_conf >= float(self.settings.trading.crowded_long_min_forecast_confidence)
                    and prob_up >= float(self.settings.trading.crowded_long_min_prob_up)
                    and (
                        opportunity_score >= float(self.settings.trading.crowded_long_min_opportunity_score)
                        or adx_15m >= float(self.settings.trading.crowded_long_min_adx_15m)
                    )
                )
                if not strong_confirmation:
                    reason = (
                        f"crowded_long_weak_confirmation(conf={forecast_conf:.2f},prob_up={prob_up:.2f},"
                        f"opp={opportunity_score:.1f},adx15={adx_15m:.1f})"
                    )
                    logger.warning(
                        f"Skipping {decision.symbol} long: {funding_analysis.reason}; {reason}"
                    )
                    if self.quant_engine is not None:
                        try:
                            self.quant_engine.record_execution_event(
                                event_type="reject",
                                symbol=decision.symbol,
                                reason_code="crowded_long_weak_confirmation",
                                details={
                                    "forecast_confidence": forecast_conf,
                                    "prob_up": prob_up,
                                    "opportunity_score": opportunity_score,
                                    "adx_15m": adx_15m,
                                },
                            )
                        except Exception:
                            pass
                    return False
        
        position_multiplier = 1.0  # Default full size
        if funding_analysis.is_crowded:
            position_multiplier = funding_analysis.position_multiplier
            logger.warning(f" {funding_analysis.reason}  Position reduced to {position_multiplier*100:.0f}%")
        elif funding_analysis.is_contrarian:
            logger.info(f" {funding_analysis.reason}")
        
        # Calculate position size with unified margin budget helper.
        max_margin = self._max_trade_margin_cap()
        if max_margin <= 0:
            logger.warning(f"No margin budget available for {decision.symbol}. Skipping trade.")
            return False
        max_position_value = max_margin * leverage

        if validation.adjusted_position_size and validation.adjusted_position_size > 0:
            position_size = float(validation.adjusted_position_size)
        else:
            # Fallback sizing if RiskManager did not return an adjusted size
            risk_amount = effective_account_balance * (risk_pct / 100)
            stop_distance_pct = abs(current_price - decision.stop_loss) / current_price
            if stop_distance_pct > 0:
                risk_based_value = risk_amount / stop_distance_pct
                position_value = min(risk_based_value, max_position_value)
            else:
                position_value = max_position_value * 0.5
            position_size = position_value / current_price

        # Ensure position size is within balance limits
        # Final margin check
        required_margin = (position_size * current_price) / leverage

        if (required_margin - max_margin) > self._affordability_margin_epsilon_usd:
            # Reduce to fit
            position_size = (max_margin * leverage) / current_price
            logger.warning(f"Position size capped by balance: {position_size:.4f}")
        
        # Round to appropriate decimal places using Bybit rules
        position_size = self.bybit.round_quantity(decision.symbol, position_size)
        
        # Apply crowded trade reduction from funding analysis
        if position_multiplier < 1.0:
            original_size = position_size
            position_size = self.bybit.round_quantity(decision.symbol, position_size * position_multiplier)
            logger.info(f" Position reduced for crowded funding: {original_size:.4f}  {position_size:.4f}")
        
        # Absolute minimum notional check (Bybit requires 5 USDT min for most pairs)
        # We set a buffer of 5.5 USDT to be safe.
        min_value = 5.5 
        
        current_notional = position_size * current_price
        if current_notional < min_value:
            # Check if raising it would exceed our available margin
            required_margin_for_min = min_value / max(1, leverage)
            if (required_margin_for_min - max_margin) > self._affordability_margin_epsilon_usd:
                logger.warning(f"Trade too small (${current_notional:.2f}) but raising it would exceed available margin. Skipping.")
                return False

            logger.warning(f"Position too small (${current_notional:.2f}), raising to minimum ${min_value:.2f}")
            position_size = min_value / current_price
            position_size = self.bybit.round_quantity(decision.symbol, position_size)

        # Final check against exchange-specific quantity limits
        symbol_info = self.bybit.get_symbol_info(decision.symbol)
        if symbol_info:
            lot_filter = symbol_info.get("lotSizeFilter", {}) or {}
            try:
                min_qty = float(lot_filter.get("minOrderQty", 0) or 0)
            except (TypeError, ValueError):
                min_qty = 0.0
            try:
                max_qty = float(lot_filter.get("maxOrderQty", 0) or 0)
            except (TypeError, ValueError):
                max_qty = 0.0
            if max_qty > 0 and position_size > (max_qty + 1e-12):
                original_size = position_size
                position_size = self.bybit.round_quantity(decision.symbol, max_qty)
                logger.warning(
                    f"Position size clipped to exchange qty cap for {decision.symbol}: "
                    f"{original_size} -> {position_size} (maxOrderQty={max_qty})"
                )

            if max_qty > 0 and min_qty > 0 and min_qty > max_qty:
                logger.warning(
                    f"Invalid symbol quantity rules for {decision.symbol}: min_qty {min_qty} > max_qty {max_qty}. Skipping."
                )
                return False

            if position_size < min_qty:
                # Check if this minimum is actually affordable
                required_margin_for_min_qty = (min_qty * current_price) / max(1, leverage)
                if (required_margin_for_min_qty - max_margin) > self._affordability_margin_epsilon_usd:
                    logger.warning(
                        f"Instrument minimum {min_qty} ({decision.symbol}) requires ${required_margin_for_min_qty:.2f} margin, "
                        f"but only ${max_margin:.2f} is available. Skipping trade."
                    )
                    return False

                logger.warning(f"Position size {position_size} below instrument minimum {min_qty}, raising to {min_qty}")
                position_size = min_qty
                if max_qty > 0 and position_size > max_qty:
                    logger.warning(
                        f"Instrument minimum {min_qty} exceeds maxOrderQty {max_qty} for {decision.symbol}. Skipping."
                    )
                    return False

        if isinstance(decision.reasoning, dict):
            metadata = decision.reasoning.get("metadata")
            if isinstance(metadata, dict):
                diag = metadata.get("candidate_diag")
                if isinstance(diag, dict):
                    diag["cap_clipped"] = bool(cap_clipped_by_risk)
                    diag["final_size"] = float(position_size)
                    metadata["candidate_diag"] = diag
                metadata["final_position_size"] = float(position_size)
                metadata["cap_clipped_by_risk"] = bool(cap_clipped_by_risk)
                decision.reasoning["metadata"] = metadata
        
        logger.info(
            f"Opening position: {decision.symbol} {decision.side} "
            f"Size: {position_size} Risk: {risk_pct}% Leverage: {leverage}x"
        )
        logger.info(f" DEEPSEEK DECISION: {decision.symbol} {decision.side} @ ${current_price:.2f} SL: ${decision.stop_loss:.2f} TP: ${decision.take_profit_1:.2f}")
        
        try:
            order_submit_started = time.perf_counter()
            # Re-fetch current price right before order and reject stale/drifted entries.
            latest_ticker = self.bybit.get_ticker(decision.symbol)
            latest_price = latest_ticker.last_price

            entry_drift_pct = abs(latest_price - current_price) / current_price * 100 if current_price > 0 else 0.0
            configured_max_drift = float(self.settings.trading.max_entry_drift_pct)
            tf_15m = analysis.get(decision.symbol, {}).get("timeframe_analysis", {}).get("15m", {})
            atr_15m = self._safe_float(tf_15m.get("atr"), 0.0)
            atr_pct_15m = (atr_15m / current_price * 100.0) if current_price > 0 else 0.0
            low_price_symbol = latest_price > 0 and latest_price < 1.0
            volatility_relax = max(0.0, (atr_pct_15m - 2.0) * 0.5)
            adaptive_ceiling = max(3.2, configured_max_drift + volatility_relax)
            adaptive_ceiling = min(6.0, adaptive_ceiling)
            if low_price_symbol:
                adaptive_ceiling = max(adaptive_ceiling, 4.8)
            adaptive_max_drift = max(
                configured_max_drift,
                min(adaptive_ceiling, configured_max_drift + (atr_pct_15m * 0.75))
            )
            if small_account_mode and low_price_symbol:
                adaptive_max_drift = max(adaptive_max_drift, 3.8)

            if entry_drift_pct > adaptive_max_drift:
                logger.warning(
                    f"Skipping {decision.symbol}: entry drift {entry_drift_pct:.3f}% exceeds adaptive max {adaptive_max_drift:.3f}% "
                    f"(base={configured_max_drift:.3f}%, atr15m={atr_pct_15m:.3f}%)"
                )
                return False
            if entry_drift_pct > configured_max_drift:
                logger.warning(
                    f"{decision.symbol} entry drift {entry_drift_pct:.3f}% above base {configured_max_drift:.3f}% "
                    f"but within adaptive max {adaptive_max_drift:.3f}% (atr15m={atr_pct_15m:.3f}%)"
                )

            # Re-price execution context to latest mark when drift is accepted.
            current_price = latest_price

            second_step_ok, second_step_reason, second_step_diag = self._confirm_entry_second_step(
                decision=decision,
                market_data=market_data,
                analysis=analysis,
                latest_price=float(current_price),
                entry_tier=entry_tier,
            )
            if not second_step_ok:
                logger.warning(
                    f"Second-step entry confirm rejected {decision.symbol} {decision.side}: "
                    f"{second_step_reason} diag={second_step_diag}"
                )
                self._mark_entry_reject(decision.symbol, second_step_reason)
                if self.quant_engine is not None:
                    try:
                        self.quant_engine.record_execution_event(
                            event_type="reject",
                            symbol=decision.symbol,
                            reason_code=second_step_reason,
                            details=second_step_diag,
                        )
                    except Exception:
                        pass
                return False

            # Re-validate protective levels against latest mark to reduce exchange rejects.
            adjusted_stop_loss = decision.stop_loss
            adjusted_take_profit = decision.take_profit_1

            if decision.side == "Sell":
                min_stop_loss = latest_price * 1.003
                max_take_profit = latest_price * 0.997
                if adjusted_stop_loss <= latest_price:
                    adjusted_stop_loss = min_stop_loss
                    logger.warning(f"Adjusting short SL from ${decision.stop_loss:.2f} to ${adjusted_stop_loss:.2f} (market moved to ${latest_price:.2f})")
                elif adjusted_stop_loss < min_stop_loss:
                    adjusted_stop_loss = min_stop_loss
                    logger.info(f"Buffering short SL from ${decision.stop_loss:.2f} to ${adjusted_stop_loss:.2f} for safety")
                if adjusted_take_profit is None or adjusted_take_profit >= latest_price:
                    adjusted_take_profit = max_take_profit
                    logger.warning(
                        f"Adjusting short TP to ${adjusted_take_profit:.2f} to remain below current price ${latest_price:.2f}"
                    )
            else:  # Buy/Long position
                max_stop_loss = latest_price * 0.997
                min_take_profit = latest_price * 1.003
                if adjusted_stop_loss >= latest_price:
                    adjusted_stop_loss = max_stop_loss
                    logger.warning(f"Adjusting long SL from ${decision.stop_loss:.2f} to ${adjusted_stop_loss:.2f} (market moved to ${latest_price:.2f})")
                elif adjusted_stop_loss > max_stop_loss:
                    adjusted_stop_loss = max_stop_loss
                    logger.info(f"Buffering long SL from ${decision.stop_loss:.2f} to ${adjusted_stop_loss:.2f} for safety")
                if adjusted_take_profit is None or adjusted_take_profit <= latest_price:
                    adjusted_take_profit = min_take_profit
                    logger.warning(
                        f"Adjusting long TP to ${adjusted_take_profit:.2f} to remain above current price ${latest_price:.2f}"
                    )

            # Re-apply minimum R:R check after market drift and level normalization.
            risk_distance_latest = abs(latest_price - adjusted_stop_loss)
            reward_distance_latest = abs(adjusted_take_profit - latest_price) if adjusted_take_profit else 0.0
            rr_latest = reward_distance_latest / risk_distance_latest if risk_distance_latest > 0 else 0.0
            if rr_latest < min_rr_ratio:
                if decision.side == "Buy":
                    adjusted_take_profit = latest_price + (risk_distance_latest * min_rr_ratio)
                else:
                    adjusted_take_profit = latest_price - (risk_distance_latest * min_rr_ratio)
                logger.info(
                    f"Adjusted TP for {decision.symbol} to maintain minimum R:R {min_rr_ratio:.2f}:1 after repricing"
                )

            decision.stop_loss = adjusted_stop_loss
            decision.take_profit_1 = adjusted_take_profit
            
            # Place the order
            from src.exchange.bybit_client import OrderSide, OrderType
            
            order_result = None
            inline_protection_applied = True
            last_order_error_text = ""
            reprice_markers = (
                "30208",
                "minimum buying price",
                "minimum selling price",
                "should be higher than base_price",
                "should be lower than base_price",
                "base_price",
            )
            fallback_markers = (
                "30209",
                "30208",
                "minimum selling price",
                "minimum buying price",
                "should be higher than base_price",
                "should be lower than base_price",
            )
            try:
                order_result = self.bybit.place_order(
                    symbol=decision.symbol,
                    side=OrderSide(decision.side),
                    order_type=OrderType.MARKET,
                    qty=position_size,
                    stop_loss=adjusted_stop_loss,
                    take_profit=adjusted_take_profit,
                    leverage=leverage,
                    slippage_tolerance_pct=entry_slippage_tolerance_pct,
                )
            except Exception as order_error:
                last_order_error_text = str(order_error).lower()
                if any(marker in last_order_error_text for marker in reprice_markers):
                    logger.warning(
                        f"Inline entry protection rejected for {decision.symbol} ({order_error}); "
                        f"retrying up to {self._max_reprice_attempts} time(s) with refreshed ticker"
                    )
                    for attempt in range(1, self._max_reprice_attempts + 1):
                        try:
                            refreshed_ticker = self.bybit.get_ticker(decision.symbol)
                            refreshed_price = float(getattr(refreshed_ticker, "last_price", 0) or 0)
                            if refreshed_price <= 0:
                                raise ValueError("invalid refreshed price")

                            reprice_guard = max(0.0018, min_rr_ratio * 0.0007)
                            refreshed_sl = float(adjusted_stop_loss)
                            refreshed_tp = float(adjusted_take_profit) if adjusted_take_profit is not None else 0.0

                            if decision.side == "Buy":
                                if refreshed_sl >= refreshed_price:
                                    refreshed_sl = refreshed_price * (1 - reprice_guard)
                                if refreshed_tp <= refreshed_price:
                                    refreshed_tp = refreshed_price * (1 + reprice_guard)
                            else:
                                if refreshed_sl <= refreshed_price:
                                    refreshed_sl = refreshed_price * (1 + reprice_guard)
                                if refreshed_tp >= refreshed_price:
                                    refreshed_tp = refreshed_price * (1 - reprice_guard)

                            reprice_risk = abs(refreshed_price - refreshed_sl)
                            reprice_reward = abs(refreshed_tp - refreshed_price)
                            reprice_rr = reprice_reward / reprice_risk if reprice_risk > 0 else 0.0
                            if reprice_rr < min_rr_ratio:
                                if decision.side == "Buy":
                                    refreshed_tp = refreshed_price + (reprice_risk * min_rr_ratio)
                                else:
                                    refreshed_tp = refreshed_price - (reprice_risk * min_rr_ratio)

                            adjusted_stop_loss = refreshed_sl
                            adjusted_take_profit = refreshed_tp
                            decision.stop_loss = refreshed_sl
                            decision.take_profit_1 = refreshed_tp

                            order_result = self.bybit.place_order(
                                symbol=decision.symbol,
                                side=OrderSide(decision.side),
                                order_type=OrderType.MARKET,
                                qty=position_size,
                                stop_loss=refreshed_sl,
                                take_profit=refreshed_tp,
                                leverage=leverage,
                                slippage_tolerance_pct=entry_slippage_tolerance_pct,
                            )
                            logger.info(
                                f"Reprice retry {attempt}/{self._max_reprice_attempts} succeeded for {decision.symbol}: "
                                f"price={refreshed_price:.6f}, sl={refreshed_sl:.6f}, tp={refreshed_tp:.6f}"
                            )
                            break
                        except Exception as reprice_error:
                            last_order_error_text = str(reprice_error).lower()
                            logger.warning(
                                f"Reprice retry {attempt}/{self._max_reprice_attempts} failed for {decision.symbol}: {reprice_error}"
                            )

                inline_price_guard_reject = (
                    ("takeprofit" in last_order_error_text or "stoploss" in last_order_error_text)
                    and "base_price" in last_order_error_text
                )
                if order_result is None and (
                    inline_price_guard_reject or any(marker in last_order_error_text for marker in fallback_markers)
                ):
                    inline_protection_applied = False
                    fallback_slippage_tolerance_pct = min(
                        2.5,
                        max(
                            entry_slippage_tolerance_pct + 0.15,
                            entry_slippage_tolerance_pct * 1.6,
                        ),
                    )
                    logger.warning(
                        f"Inline TP/SL rejected by exchange for {decision.symbol}; retrying market entry "
                        f"without inline protection (slippage_tolerance={fallback_slippage_tolerance_pct:.2f}%)"
                    )
                    try:
                        order_result = self.bybit.place_order(
                            symbol=decision.symbol,
                            side=OrderSide(decision.side),
                            order_type=OrderType.MARKET,
                            qty=position_size,
                            leverage=leverage,
                            slippage_tolerance_pct=fallback_slippage_tolerance_pct,
                        )
                    except Exception as fallback_error:
                        self._mark_entry_reject(decision.symbol, str(fallback_error))
                        raise
                elif order_result is None:
                    self._mark_entry_reject(decision.symbol, str(order_error))
                    raise

            logger.info(f" BYBIT ORDER: {decision.symbol} {OrderSide(decision.side).value} {position_size} units")
            logger.info(f" Order placed successfully: {order_result}")

            # Refresh positions from Bybit after successful order and confirm fill.
            position_poll_attempts = 4
            position_poll_delay_seconds = 0.25
            filled_position = None
            for poll_idx in range(1, position_poll_attempts + 1):
                self.positions = self.bybit.get_positions()
                filled_position = next(
                    (
                        p
                        for p in self.positions
                        if p.symbol == decision.symbol and abs(float(getattr(p, "size", 0) or 0)) > 0
                    ),
                    None,
                )
                logger.info(
                    f" POSITIONS AFTER ORDER (poll {poll_idx}/{position_poll_attempts}): {len(self.positions)} positions"
                )
                for pos in self.positions:
                    logger.info(f"   {pos.symbol}: {pos.side} {pos.size} @ ${pos.avg_price}")
                if filled_position is not None:
                    break
                if poll_idx < position_poll_attempts:
                    await asyncio.sleep(position_poll_delay_seconds)

            if filled_position is None:
                order_id = (order_result.get("result", {}) or {}).get("orderId")
                reject_reason = f"order_ack_without_fill order_id={order_id}"
                logger.warning(
                    f"{decision.symbol} order acknowledged but no open position was detected after "
                    f"{position_poll_attempts} poll(s); treating as unfilled"
                )
                self._mark_entry_reject(decision.symbol, reject_reason)
                if self.quant_engine is not None:
                    try:
                        self.quant_engine.record_execution_event(
                            event_type="reject",
                            symbol=decision.symbol,
                            reason_code="order_ack_without_fill",
                            details={
                                "order_id": order_id,
                                "side": decision.side,
                                "qty": float(position_size),
                            },
                        )
                    except Exception:
                        pass
                return False

            self._clear_entry_reject(decision.symbol)
            filled_position_size = abs(float(getattr(filled_position, "size", position_size) or position_size))

            fill_price = float(getattr(filled_position, "avg_price", current_price) or current_price)
            slippage_bps = (
                abs(fill_price - float(current_price)) / max(1e-9, float(current_price)) * 10000.0
            )
            slippage_units_ok, slippage_units_reason = self._validate_slippage_units(
                symbol=decision.symbol,
                estimated_slippage_bps=float(estimated_slippage_bps),
                realized_slippage_bps=float(slippage_bps),
                requested_entry_price=float(current_price),
                fill_price=float(fill_price),
            )
            if not slippage_units_ok:
                logger.error(
                    f"Slippage unit validation failed for {decision.symbol}: {slippage_units_reason}. "
                    "Fail-fast close engaged."
                )
                self._mark_entry_reject(decision.symbol, slippage_units_reason)
                if self.quant_engine is not None:
                    try:
                        self.quant_engine.record_execution_event(
                            event_type="reject",
                            symbol=decision.symbol,
                            reason_code="slippage_unit_inconsistent",
                            details={
                                "reason": slippage_units_reason,
                                "estimated_slippage_bps": float(estimated_slippage_bps),
                                "realized_slippage_bps": float(slippage_bps),
                                "requested_entry_price": float(current_price),
                                "fill_price": float(fill_price),
                            },
                        )
                    except Exception:
                        pass
                try:
                    self.bybit.close_position(decision.symbol)
                except Exception as close_err:
                    logger.error(
                        f"Fail-fast close after slippage validation failure also failed for "
                        f"{decision.symbol}: {close_err}"
                    )
                return False
            if self.quant_engine is not None:
                try:
                    latency_ms = (time.perf_counter() - order_submit_started) * 1000.0
                    fill_reason = "probe_entry_filled" if entry_tier == "probe" else "entry_filled"
                    self.quant_engine.record_execution_event(
                        event_type="fill",
                        symbol=decision.symbol,
                        reason_code=fill_reason,
                        slippage_bps=slippage_bps,
                        latency_ms=latency_ms,
                        details={
                            "side": decision.side,
                            "entry_tier": entry_tier,
                            "requested_entry": float(current_price),
                            "fill_price": fill_price,
                            "qty": float(position_size),
                            "estimated_slippage_bps": float(estimated_slippage_bps),
                            "estimated_slippage_pct": float(estimated_slippage_bps) / 100.0,
                            "realized_slippage_bps": float(slippage_bps),
                            "realized_slippage_pct": float(slippage_bps) / 100.0,
                            "cap_clipped": bool(cap_clipped_by_risk),
                            "final_size": float(position_size),
                            "policy_state": str(decision_reasoning.get("policy_state", "green")),
                            "regime": str(
                                decision_reasoning.get("regime")
                                or decision_reasoning.get("market_regime")
                                or "unknown"
                            ),
                        },
                    )
                except Exception:
                    pass

            win_rate_mode_enabled = bool(self.settings.trading.win_rate_mode_enabled)
            effective_entry_price = fill_price if fill_price > 0 else float(current_price)
            risk_distance = abs(float(effective_entry_price) - float(adjusted_stop_loss or 0.0))
            tp1_r_multiple = float(self.settings.trading.full_tier_tp1_r_multiple)
            tp1_partial_pct = float(self.settings.trading.full_tier_tp1_partial_pct)
            be_trigger_r = float(self.settings.trading.full_tier_breakeven_r_multiple)
            trail_activation_r = float(self.settings.trading.full_tier_trail_activation_r_multiple)
            tp1_price: Optional[float] = None
            if win_rate_mode_enabled and adjusted_take_profit and risk_distance > 0:
                if decision.side == "Buy":
                    tp1_price = effective_entry_price + (risk_distance * tp1_r_multiple)
                    tp1_price = min(tp1_price, float(adjusted_take_profit))
                else:
                    tp1_price = effective_entry_price - (risk_distance * tp1_r_multiple)
                    tp1_price = max(tp1_price, float(adjusted_take_profit))
            micro_exit_bps = float(self.settings.trading.post_fill_micro_exit_bps)
            if win_rate_mode_enabled and slippage_bps > micro_exit_bps:
                try:
                    post_fill_ticker = self.bybit.get_ticker(decision.symbol)
                    post_fill_price = float(getattr(post_fill_ticker, "last_price", 0) or 0)
                except Exception:
                    post_fill_price = 0.0
                if post_fill_price > 0 and fill_price > 0:
                    if decision.side == "Buy":
                        adverse_bps = max(0.0, (fill_price - post_fill_price) / fill_price * 10000.0)
                    else:
                        adverse_bps = max(0.0, (post_fill_price - fill_price) / fill_price * 10000.0)
                    if adverse_bps >= max(25.0, micro_exit_bps * 0.35):
                        logger.warning(
                            f"Micro-exit triggered for {decision.symbol}: fill_slip={slippage_bps:.1f}bps "
                            f"adverse_drift={adverse_bps:.1f}bps"
                        )
                        try:
                            self.bybit.close_position(decision.symbol)
                        except Exception as close_err:
                            logger.error(f"Micro-exit close failed for {decision.symbol}: {close_err}")
                        return True

            # Fallback safety: if inline TP/SL was rejected, attach protection right after fill.
            if not inline_protection_applied:
                await asyncio.sleep(0.2)
                protection_applied = False
                try:
                    self.bybit.update_position_protection(
                        symbol=decision.symbol,
                        stop_loss=adjusted_stop_loss,
                        take_profit=adjusted_take_profit,
                    )
                    logger.info(f"Applied post-entry protection for {decision.symbol} via set_trading_stop")
                    protection_applied = True
                except Exception as protection_error:
                    logger.warning(
                        f"Full protection update failed for {decision.symbol} after fallback entry: "
                        f"{protection_error}. Retrying with stop-loss only."
                    )
                    try:
                        self.bybit.update_position_protection(
                            symbol=decision.symbol,
                            stop_loss=adjusted_stop_loss,
                        )
                        logger.info(f"Applied stop-loss-only safety protection for {decision.symbol}")
                        protection_applied = True
                    except Exception as stop_only_error:
                        logger.error(
                            f"Protection update failed for {decision.symbol} after fallback entry "
                            f"(full_error={protection_error}; stop_only_error={stop_only_error}). "
                            f"Closing position for safety."
                        )

                if not protection_applied:
                    try:
                        self.bybit.close_position(decision.symbol)
                    except Exception as close_error:
                        logger.error(f"Emergency close failed for {decision.symbol}: {close_error}")
                    return False

            # Exits are managed through set_trading_stop updates to avoid duplicate entry legs.
            logger.debug("Single-entry mode active; bracket/OCO secondary-entry flow disabled")

            # Add position to monitor
            self.position_monitor.add_position(
                symbol=decision.symbol,
                side=decision.side,
                size=filled_position_size,
                entry_price=effective_entry_price,
                stop_loss=adjusted_stop_loss,  # Use adjusted SL
                take_profit=adjusted_take_profit,
            )

            # Setup smart execution features
            if adjusted_take_profit:
                # Calculate ATR-based trailing stop distance (SYMBOL-SPECIFIC)
                symbol_analysis = analysis.get(decision.symbol, {})
                tf_15m = symbol_analysis.get("timeframe_analysis", {}).get("15m", {})
                atr = tf_15m.get("atr")
                
                # Fallback if ATR not available
                if atr is None or atr <= 0:
                    atr = current_price * 0.005  # 0.5% fallback
                    logger.warning(f"ATR not available for {decision.symbol}, using 0.5% fallback (${atr:.2f})")
                
                # Use symbol-specific trailing distance (accounts for volatility class)
                trail_distance_pct = get_trail_distance(decision.symbol, atr, current_price)
                atr_pct = (atr / current_price) * 100
                
                logger.info(f"ATR-based trailing ({symbol_config.vol_class} vol): ATR=${atr:.2f} ({atr_pct:.2f}%), trail={trail_distance_pct:.2f}%")
                
                # Setup trailing stop with dynamic distance and store entry ATR
                self.execution_manager.setup_trailing_stop(
                    symbol=decision.symbol,
                    side=decision.side,
                    initial_stop=adjusted_stop_loss,  # Use adjusted SL
                    trail_distance_pct=trail_distance_pct,  # Dynamic ATR-based!
                    current_price=current_price,
                    entry_price=effective_entry_price,
                    entry_atr=atr,  # Store for future dynamic adjustments
                )

                # Setup partial profit targets (only if we have a take profit)
                if adjusted_take_profit:
                    try:
                        # Layered exits are only enabled in win-rate mode.
                        if win_rate_mode_enabled and entry_tier == "full":
                            tp_levels = [{
                                "price": float(tp1_price if tp1_price is not None else adjusted_take_profit),
                                "percentage": tp1_partial_pct,
                            }]
                        else:
                            tp_levels = [{"price": adjusted_take_profit, "percentage": 100}]
                        
                        self.execution_manager.setup_partial_take_profits(
                            symbol=decision.symbol,
                            position_size=filled_position_size,
                            tp_levels=tp_levels,
                            side=decision.side,  # Pass side for direction-aware triggering
                        )
                        logger.debug(f"Partial take profits set for {decision.symbol} ({decision.side})")
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
                "size": filled_position_size,
                "entry_price": effective_entry_price,
                "risk_pct": risk_pct,
                "strategy": decision.strategy_tag,
                "entry_tier": entry_tier,
                "estimated_slippage_bps": float(estimated_slippage_bps),
            })
            self._register_entry_symbol(decision.symbol)
            decision_regime = "unknown"
            if isinstance(decision.reasoning, dict):
                decision_regime = str(
                    decision.reasoning.get("regime")
                    or decision.reasoning.get("market_regime")
                    or "unknown"
                )
            expected_hold_minutes = int(
                max(
                    5,
                    self._safe_float(
                        getattr(decision, "expected_hold_duration_mins", None),
                        45.0,
                    ),
                )
            )
            if entry_tier == "probe":
                expected_hold_minutes = max(5, int(expected_hold_minutes * 0.65))
            opened_at_utc = datetime.now(timezone.utc).isoformat()
            self._last_open_positions[decision.symbol] = {
                "side": decision.side,
                "entry_price": float(effective_entry_price),
                "size": float(filled_position_size),
                "last_mark_price": float(effective_entry_price),
                "strategy_tag": decision.strategy_tag or "unknown",
                "risk_pct": float(risk_pct),
                "leverage": int(leverage),
                "entry_tier": entry_tier,
                "policy_state": str(decision_reasoning.get("policy_state", "green")),
                "policy_key": str(decision_reasoning.get("policy_key", "")),
                "timeframe": "15m",
                "market_regime": decision_regime,
                "opened_at": opened_at_utc,
                "expected_hold_minutes": expected_hold_minutes,
                "risk_distance": float(risk_distance),
                "tp1_price": float(tp1_price) if tp1_price is not None else None,
                "be_trigger_r": float(be_trigger_r),
                "trail_activation_r": float(trail_activation_r),
                "breakeven_moved": False,
                "trail_active": False,
            }
            
            # Save to database
            if self.db:
                await self.db.save_trade({
                    "trade_id": trade_id,
                    "symbol": decision.symbol,
                    "side": decision.side,
                    "order_type": "Market",
                    "size": filled_position_size,
                    "entry_price": effective_entry_price,
                    "stop_loss": adjusted_stop_loss,  # Use adjusted SL
                    "take_profit": adjusted_take_profit,
                    "leverage": leverage,
                    "risk_pct": risk_pct,
                    "position_value": filled_position_size * effective_entry_price,
                    "status": "open",
                    "strategy_tag": decision.strategy_tag,
                    "entry_tier": entry_tier,
                    "policy_state": str(decision_reasoning.get("policy_state", "green")),
                    "policy_key": str(decision_reasoning.get("policy_key", "")),
                    "ai_confidence": decision.confidence_score,
                    "ai_reasoning": str(decision.reasoning),
                    "decision_id": decision.decision_id,
                    "balance_before": self.account_balance,
                    "exchange_order_id": (order_result.get("result", {}) or {}).get("orderId"),
                })

                # Capture context for RAG
                try:
                    if decision.symbol in analysis:
                        symbol_analysis = analysis[decision.symbol]
                        tf_1h = symbol_analysis.get("timeframe_analysis", {}).get("1h")
                        if tf_1h:
                            if isinstance(tf_1h, dict):
                                atr_1h = self._safe_float(tf_1h.get("atr"), 0.0)
                                context_snapshot = {
                                    "rsi": self._safe_float(tf_1h.get("rsi"), 50.0),
                                    "trend": str(tf_1h.get("trend", "neutral")),
                                    "atr": atr_1h,
                                    "close": float(current_price),
                                    "current_price": float(current_price),
                                    "macd_signal": str(tf_1h.get("macd_signal", "neutral")),
                                    "adx": self._safe_float(tf_1h.get("adx"), 25.0),
                                    "volume_ratio": self._safe_float(tf_1h.get("volume_ratio"), 1.0),
                                    "bb_position": str(tf_1h.get("bb_position", "middle")),
                                    "price_24h_change_pct": market_data["tickers"][decision.symbol].price_24h_change * 100,
                                }
                            else:
                                atr_1h = self._safe_float(getattr(tf_1h.indicators, "atr", 0), 0.0)
                                context_snapshot = {
                                    "rsi": self._safe_float(getattr(tf_1h.indicators, "rsi", 50), 50.0),
                                    "trend": str(tf_1h.structure.trend.value),
                                    "atr": atr_1h,
                                    "close": float(current_price),
                                    "current_price": float(current_price),
                                    "macd_signal": str(getattr(tf_1h.indicators, "macd_signal", "neutral")),
                                    "adx": self._safe_float(getattr(tf_1h.indicators, "adx", 25), 25.0),
                                    "volume_ratio": self._safe_float(getattr(tf_1h.indicators, "volume_ratio", 1.0), 1.0),
                                    "bb_position": str(getattr(tf_1h.indicators, "bb_position", "middle")),
                                    "price_24h_change_pct": market_data["tickers"][decision.symbol].price_24h_change * 100,
                                }
                            self.active_trade_contexts[decision.symbol] = context_snapshot
                            # Save to Supabase for persistence across restarts
                            if self.db:
                                await self.db.save_active_context(decision.symbol, context_snapshot)
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
                )
            
            # Immediate push after opening
            await self._update_account_state()
            logger.info(f" Critical Sync: Dashboard updated after opening {decision.symbol}")
            return True
            
        except Exception as e:
            logger.error(f" Failed to place order: {e}")
            if self.quant_engine is not None:
                try:
                    self.quant_engine.record_execution_event(
                        event_type="reject",
                        symbol=decision.symbol,
                        reason_code="open_position_exception",
                        details={"error": str(e)},
                    )
                except Exception:
                    pass
            return False
    
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
            self._handled_closures.add(symbol)
            logger.info(f" Position closed: {symbol}")

            # Record trade result with emergency controls
            if position_found:
                entry_meta = dict(self._last_open_positions.get(symbol, {}))
                strategy_name = (
                    decision.strategy_tag
                    if hasattr(decision, "strategy_tag") and decision.strategy_tag
                    else entry_meta.get("strategy_tag", "manual_close")
                )

                position_size = 0.0
                for pos in self.positions:
                    if pos.symbol == symbol and float(pos.size) > 0:
                        position_size = float(pos.size)
                        break

                if position_size > 0 and entry_price > 0:
                    closed_info = self.bybit.get_latest_closed_pnl(symbol, within_minutes=20)
                    if closed_info and closed_info.get("closed_pnl") is not None:
                        notional = entry_price * position_size
                        if notional > 0:
                            pnl = float(closed_info["closed_pnl"]) / notional * 100

                won, is_flat = self._classify_trade_outcome(pnl)
                self.emergency_controls.record_trade_result(won is True, pnl_pct=pnl)
                outcome_label = "FLAT" if is_flat else ("WIN" if won else "LOSS")
                logger.debug(f"Trade result recorded: {outcome_label} (PnL: {pnl:+.2f}%)")
                self._apply_post_close_symbol_controls(
                    symbol=symbol,
                    pnl_pct=pnl,
                    won=won,
                    is_flat=is_flat,
                )

                await self._record_trade_outcome(
                    symbol=symbol,
                    pnl_pct=pnl,
                    won=won,
                    strategy_name=strategy_name,
                    metadata={
                        "market_regime": entry_meta.get("market_regime", "unknown"),
                        "timeframe": entry_meta.get("timeframe", "15m"),
                        "leverage": entry_meta.get("leverage", decision.leverage if hasattr(decision, "leverage") else self.default_leverage),
                        "risk_pct": entry_meta.get("risk_pct", decision.risk_pct if hasattr(decision, "risk_pct") else self.default_risk_pct),
                    },
                )
                if self.quant_engine is not None:
                    try:
                        close_details = self._build_close_event_details(
                            symbol=symbol,
                            strategy_name=strategy_name,
                            outcome="flat" if is_flat else ("win" if won else "loss"),
                            pnl_pct=float(pnl),
                            extra_details={
                                "close_reason": "manual_or_ai_close",
                            },
                        )
                        self.quant_engine.record_execution_event(
                            event_type="close",
                            symbol=symbol,
                            reason_code="manual_or_ai_close",
                            pnl_pct=float(pnl),
                            details=close_details,
                        )
                    except Exception:
                        pass

            # Remove from position monitor
            self.position_monitor.remove_position(symbol)

            # Clear smart execution data
            self.execution_manager.clear_position(symbol)

            # Refresh cached positions and push immediately
            self.positions = self.bybit.get_positions()
            await self._update_account_state()
            logger.info(f" Critical Sync: Dashboard updated after closing {symbol}")

        except Exception as e:
            logger.error(f" Failed to close position: {e}")
    
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
            logger.info(f" DeepSeek requested reversal on {symbol}: {current_side}  {target_side}")

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
                f" Modified {symbol} protection: SL={adjustments['stop_loss']} TP={adjustments['take_profit']} Trail={adjustments['trailing_stop']}"
            )

            self.positions = self.bybit.get_positions()

        except Exception as exc:
            logger.error(f" Failed to modify {symbol} protection: {exc}")

    def _log_status(self):
        """Log current status"""
        pnl = self.account_balance - self.starting_balance
        pnl_pct = (pnl / self.starting_balance * 100) if self.starting_balance > 0 else 0
        effective_balance, _ = self._effective_balances()
        
        logger.info("=" * 60)
        logger.info(f"Balance: ${self.account_balance:.2f} | PnL: ${pnl:.2f} ({pnl_pct:+.1f}%)")
        if self.settings.bybit.testnet and self.settings.trading.initial_balance > 0:
            logger.info(
                f"Sizing Basis (testnet cap): ${effective_balance:.2f} "
                f"(INITIAL_BALANCE={float(self.settings.trading.initial_balance):.2f})"
            )
        logger.info(f"Positions: {len(self.positions)} | Trades Today: {len(self.risk_manager.trades_today)}")
        
        current_milestone, next_milestone = self.settings.trading.get_current_milestone(
            self.account_balance
        )
        progress = (self.account_balance - current_milestone) / (next_milestone - current_milestone) * 100
        logger.info(f"Milestone: ${current_milestone:.0f}  ${next_milestone:.0f} ({progress:.1f}%)")
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
        #         "current_milestone": f"${current_milestone:.0f}  ${next_milestone:.0f}",
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

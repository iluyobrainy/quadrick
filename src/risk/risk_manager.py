"""
Risk Management Module - Validates trades and enforces safety constraints
"""
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
from loguru import logger


@dataclass
class RiskValidation:
    """Risk validation result"""
    is_valid: bool
    adjusted_risk_pct: Optional[float] = None
    adjusted_position_size: Optional[float] = None
    adjusted_leverage: Optional[int] = None
    rejection_reasons: List[str] = None
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.rejection_reasons is None:
            self.rejection_reasons = []
        if self.warnings is None:
            self.warnings = []


class RiskManager:
    """Risk management and safety enforcement"""
    
    def __init__(
        self,
        min_risk_pct: float = 1.0,
        max_risk_pct: float = 6.0,
        max_daily_drawdown_pct: float = 12.0,
        max_leverage: int = 10,
        max_concurrent_positions: int = 2,
        min_account_balance: float = 3.0,
        high_impact_news_blackout_mins: int = 15,
        small_account_balance_threshold: float = 150.0,
        small_account_max_risk_pct: float = 3.0,
        small_account_max_leverage: int = 5,
        min_stop_distance_pct: float = 0.35,
        max_stop_distance_pct: float = 8.0,
        symbol_max_margin_pct: float = 15.0,
        portfolio_max_margin_pct: float = 35.0,
        enforce_single_position_per_symbol: bool = True,
        allow_scale_in: bool = False,
    ):
        """Initialize risk manager"""
        self.min_risk_pct = min_risk_pct
        self.max_risk_pct = max_risk_pct
        self.max_daily_drawdown_pct = max_daily_drawdown_pct
        self.max_leverage = max_leverage
        self.max_concurrent_positions = max_concurrent_positions
        self.min_account_balance = min_account_balance
        self.high_impact_news_blackout_mins = high_impact_news_blackout_mins
        self.small_account_balance_threshold = small_account_balance_threshold
        self.small_account_max_risk_pct = small_account_max_risk_pct
        self.small_account_max_leverage = small_account_max_leverage
        self.min_stop_distance_pct = min_stop_distance_pct
        self.max_stop_distance_pct = max_stop_distance_pct
        self.symbol_max_margin_pct = symbol_max_margin_pct
        self.portfolio_max_margin_pct = portfolio_max_margin_pct
        self.enforce_single_position_per_symbol = enforce_single_position_per_symbol
        self.allow_scale_in = allow_scale_in
        
        # Track daily performance
        self.daily_start_balance: Optional[float] = None
        self.daily_low_balance: Optional[float] = None
        self.trades_today: List[Dict[str, Any]] = []
        self.last_reset_date: Optional[datetime] = None
        
        # Circuit breaker state
        self.trading_halted: bool = False
        self.halt_reason: Optional[str] = None
        self.halt_until: Optional[datetime] = None
        
        logger.info("Risk manager initialized")

    @staticmethod
    def _position_symbol(position: Any) -> str:
        if isinstance(position, dict):
            return str(position.get("symbol") or "")
        return str(getattr(position, "symbol", "") or "")

    @staticmethod
    def _position_size(position: Any) -> float:
        if isinstance(position, dict):
            raw = position.get("size", 0)
        else:
            raw = getattr(position, "size", 0)
        try:
            return float(raw or 0)
        except (TypeError, ValueError):
            return 0.0
    
    def validate_trade(
        self,
        trade_decision: Dict[str, Any],
        account_balance: float,
        open_positions: List[Dict[str, Any]],
        market_price: float,
        available_balance: float = None,
        economic_events: List[Dict[str, Any]] = None,
        market_analysis: Dict[str, Any] = None,
    ) -> RiskValidation:
        """
        Validate a trading decision against risk constraints
        
        Args:
            trade_decision: Trading decision from LLM
            account_balance: Current account balance (total equity)
            open_positions: List of open positions
            market_price: Current market price for the symbol
            available_balance: Usable funds (after other margins)
            economic_events: Upcoming economic events
            market_analysis: Full market context
        
        Returns:
            RiskValidation result
        """
        validation = RiskValidation(is_valid=True)
        
        # Use available balance for margin ceiling if provided
        liquid_balance = available_balance if available_balance is not None else account_balance
        
        # Reset daily tracking if new day
        self._check_daily_reset()
        
        # Check if trading is halted
        if self.trading_halted:
            if self.halt_until and datetime.now(timezone.utc) < self.halt_until:
                validation.is_valid = False
                validation.rejection_reasons.append(
                    f"Trading halted until {self.halt_until}: {self.halt_reason}"
                )
                return validation
            else:
                # Reset halt
                self.trading_halted = False
                self.halt_reason = None
                self.halt_until = None
        
        # 1. Check minimum balance
        if account_balance < self.min_account_balance:
            validation.is_valid = False
            validation.rejection_reasons.append(
                f"Account balance ${account_balance:.2f} below minimum ${self.min_account_balance}"
            )
            return validation
        
        symbol = str(trade_decision.get("symbol") or "")
        scale_in_requested = bool(trade_decision.get("scale_in"))

        active_positions = [p for p in open_positions if self._position_size(p) > 0]

        # 2. Check concurrent positions limit
        if len(active_positions) >= self.max_concurrent_positions:
            validation.is_valid = False
            validation.rejection_reasons.append(
                f"Already have {len(active_positions)} positions (max: {self.max_concurrent_positions})"
            )
            return validation

        # 2b. Default single-position-per-symbol safety.
        if self.enforce_single_position_per_symbol and symbol:
            same_symbol_open = [
                p for p in active_positions
                if self._position_symbol(p) == symbol and self._position_size(p) > 0
            ]
            scale_in_allowed = self.allow_scale_in and scale_in_requested
            if same_symbol_open and not scale_in_allowed:
                validation.is_valid = False
                validation.rejection_reasons.append(
                    f"Open position already exists on {symbol}; scale-in is disabled by policy"
                )
                return validation
        
        # 3. Validate risk percentage (adjust for small accounts)
        try:
            risk_pct = float(trade_decision.get("risk_pct", self.max_risk_pct))
        except (TypeError, ValueError):
            risk_pct = self.max_risk_pct

        if account_balance <= self.small_account_balance_threshold:
            max_risk_for_small = min(self.max_risk_pct, self.small_account_max_risk_pct)
            if risk_pct > max_risk_for_small:
                validation.adjusted_risk_pct = max_risk_for_small
                validation.warnings.append(
                    f"Risk capped at {max_risk_for_small}% for small-account mode"
                )
                risk_pct = max_risk_for_small

        if risk_pct < self.min_risk_pct or risk_pct > self.max_risk_pct:
            adjusted_risk = max(self.min_risk_pct, min(self.max_risk_pct, risk_pct))
            if adjusted_risk != risk_pct:
                validation.adjusted_risk_pct = adjusted_risk
                validation.warnings.append(
                    f"Risk adjusted from {risk_pct}% to {adjusted_risk}%"
                )
                risk_pct = adjusted_risk
        
        # 4. Check daily drawdown
        current_drawdown = self._calculate_daily_drawdown(account_balance)
        if current_drawdown >= self.max_daily_drawdown_pct:
            validation.is_valid = False
            validation.rejection_reasons.append(
                f"Daily drawdown {current_drawdown:.1f}% exceeds limit {self.max_daily_drawdown_pct}%"
            )
            self._trigger_circuit_breaker("Max daily drawdown reached")
            return validation
        
        # 5. Validate leverage
        try:
            leverage = int(float(trade_decision.get("leverage", self.max_leverage)))
        except (TypeError, ValueError):
            leverage = self.max_leverage

        if account_balance <= self.small_account_balance_threshold and leverage > self.small_account_max_leverage:
            validation.adjusted_leverage = self.small_account_max_leverage
            validation.warnings.append(
                f"Leverage reduced from {leverage}x to {self.small_account_max_leverage}x for small-account mode"
            )
            leverage = self.small_account_max_leverage

        if leverage > self.max_leverage:
            validation.adjusted_leverage = self.max_leverage
            validation.warnings.append(
                f"Leverage adjusted from {leverage}x to {self.max_leverage}x"
            )
            leverage = self.max_leverage
        
        # 6. Trend validation - prevent counter-trend trades in strong trends
        if market_analysis:
            symbol = trade_decision.get("symbol", "")
            side = trade_decision.get("side", "")

            # Get technical analysis for this symbol.
            # `market_analysis` may be the root dict keyed by symbol (main loop),
            # or wrapped under `technical_analysis` in other callers.
            symbol_analysis = {}
            if isinstance(market_analysis, dict):
                if symbol in market_analysis:
                    symbol_analysis = market_analysis.get(symbol, {})
                else:
                    symbol_analysis = market_analysis.get('technical_analysis', {}).get(symbol, {})

            tf_analysis = symbol_analysis.get('timeframe_analysis', symbol_analysis.get('timeframes', {}))

            # Check 1h timeframe for trend strength
            adx_1h = float(tf_analysis.get('1h', {}).get('adx', 0) or 0)
            trend_1h = tf_analysis.get('1h', {}).get('trend', '')

            allow_counter_trend = bool(trade_decision.get("allow_counter_trend"))

            if adx_1h > 25:
                is_counter_trend = (trend_1h == 'trending_down' and side == 'Buy') or (trend_1h == 'trending_up' and side == 'Sell')
                if is_counter_trend:
                    if allow_counter_trend:
                        validation.warnings.append(
                            f"Counter-trend trade approved by override (ADX {adx_1h:.1f}). Tight risk management required."
                        )
                    else:
                        validation.is_valid = False
                        validation.rejection_reasons.append(
                            f"COUNTER-TREND BLOCKED: ADX {adx_1h:.1f} indicates strong {trend_1h.replace('_', ' ')}. {side} trades forbidden unless explicitly allowed."
                        )
                        return validation

        # 7. Calculate position size
        stop_loss = trade_decision.get("stop_loss", 0)
        if stop_loss and market_price:
            position_size = self._calculate_position_size(
                account_balance,
                risk_pct,
                market_price,
                stop_loss,
                leverage,
            )
            desired_qty = max(0.0, float(position_size))
            desired_notional = desired_qty * float(market_price)
            desired_margin = desired_notional / max(1.0, float(leverage))

            existing_margin_used = 0.0
            for open_position in open_positions or []:
                try:
                    if isinstance(open_position, dict):
                        size_raw = open_position.get("size", 0)
                        price_raw = open_position.get("mark_price") or open_position.get("entry_price") or 0
                        lev_raw = open_position.get("leverage", 1)
                    else:
                        size_raw = getattr(open_position, "size", 0)
                        price_raw = getattr(open_position, "mark_price", None)
                        if not price_raw:
                            price_raw = getattr(open_position, "entry_price", 0)
                        lev_raw = getattr(open_position, "leverage", 1)
                    size_value = abs(float(size_raw or 0))
                    price_value = float(price_raw or 0)
                    lev_value = max(1.0, float(lev_raw or 1))
                    if size_value <= 0 or price_value <= 0:
                        continue
                    existing_margin_used += (size_value * price_value) / lev_value
                except (TypeError, ValueError):
                    continue

            symbol_cap_margin = account_balance * (float(self.symbol_max_margin_pct) / 100.0)
            portfolio_cap_margin = account_balance * (float(self.portfolio_max_margin_pct) / 100.0)
            remaining_portfolio_margin = max(0.0, portfolio_cap_margin - existing_margin_used)
            liquid_cap_margin = max(0.0, liquid_balance * 0.95)
            raw_effective_margin_cap = max(
                0.0,
                min(symbol_cap_margin, remaining_portfolio_margin, liquid_cap_margin),
            )
            margin_usd_epsilon = max(0.01, account_balance * 0.0001)
            effective_margin_cap = max(0.0, raw_effective_margin_cap - margin_usd_epsilon)

            min_notional_raw = trade_decision.get("min_notional", 5.5)
            try:
                min_notional = max(5.0, float(min_notional_raw or 5.5))
            except (TypeError, ValueError):
                min_notional = 5.5

            symbol_name = str(trade_decision.get("symbol") or "")

            if effective_margin_cap <= 0:
                validation.is_valid = False
                validation.rejection_reasons.append(
                    "CAP_EXCEEDED: no remaining margin budget after portfolio and liquidity caps"
                )
                logger.warning(
                    "CAP_REJECT symbol={} equity=${:.2f} cap_type=MARGIN cap_value=${:.2f} "
                    "desired_notional=${:.2f} desired_margin=${:.2f} min_notional=${:.2f} "
                    "clipped_notional=${:.2f} clipped_margin=${:.2f} reason=CAP_EXCEEDED",
                    symbol_name,
                    account_balance,
                    symbol_cap_margin,
                    desired_notional,
                    desired_margin,
                    min_notional,
                    0.0,
                    0.0,
                )
                return validation

            cap_qty = (effective_margin_cap * float(leverage)) / float(market_price)
            final_qty = min(desired_qty, cap_qty)
            clipped_notional = final_qty * float(market_price)
            clipped_margin = clipped_notional / max(1.0, float(leverage))

            if final_qty < desired_qty:
                position_size = final_qty
                validation.warnings.append(
                    "Position size clipped to active margin caps "
                    f"(symbol={self.symbol_max_margin_pct:.1f}%, portfolio={self.portfolio_max_margin_pct:.1f}%)"
                )
                logger.info(
                    "CAP_CLIP symbol={} equity=${:.2f} cap_type=MARGIN cap_value=${:.2f} "
                    "desired_notional=${:.2f} desired_margin=${:.2f} min_notional=${:.2f} "
                    "clipped_notional=${:.2f} clipped_margin=${:.2f} reason=CAP_EXCEEDED",
                    symbol_name,
                    account_balance,
                    symbol_cap_margin,
                    desired_notional,
                    desired_margin,
                    min_notional,
                    clipped_notional,
                    clipped_margin,
                )
            else:
                position_size = desired_qty

            if (clipped_notional + margin_usd_epsilon) < min_notional:
                validation.is_valid = False
                validation.rejection_reasons.append(
                    f"CLIPPED_BELOW_MIN: ${clipped_notional:.2f} < min_notional ${min_notional:.2f}"
                )
                logger.warning(
                    "CAP_REJECT symbol={} equity=${:.2f} cap_type=MARGIN cap_value=${:.2f} "
                    "desired_notional=${:.2f} desired_margin=${:.2f} min_notional=${:.2f} "
                    "clipped_notional=${:.2f} clipped_margin=${:.2f} reason=CLIPPED_BELOW_MIN",
                    symbol_name,
                    account_balance,
                    symbol_cap_margin,
                    desired_notional,
                    desired_margin,
                    min_notional,
                    clipped_notional,
                    clipped_margin,
                )
                return validation

            validation.adjusted_position_size = position_size
        
        # 7. Check economic events blackout
        if economic_events and self._check_news_blackout(economic_events):
            validation.is_valid = False
            validation.rejection_reasons.append(
                "High-impact economic event within blackout window"
            )
            return validation
        
        # 8. Validate stop loss placement
        if stop_loss and market_price:
            stop_distance_pct = abs(market_price - stop_loss) / market_price * 100

            # Hard-reject tight stops to avoid noise-driven churn.
            if stop_distance_pct < self.min_stop_distance_pct:
                validation.is_valid = False
                validation.rejection_reasons.append(
                    f"Stop loss too tight at {stop_distance_pct:.2f}% (min {self.min_stop_distance_pct:.2f}%)"
                )
                return validation

            # Warn if stop is too wide
            if stop_distance_pct > self.max_stop_distance_pct:
                validation.warnings.append(
                    f"Stop loss very wide at {stop_distance_pct:.2f}%"
                )
        
        # 9. Check correlation risk
        if open_positions:
            correlation_warning = self._check_correlation_risk(
                trade_decision.get("symbol"),
                open_positions
            )
            if correlation_warning:
                validation.warnings.append(correlation_warning)

        # 10. Portfolio-level risk assessment
        if trade_decision.get("symbol"):
            # Estimate position value
            entry_price = trade_decision.get("entry_price_target", market_price or 1)
            position_size = getattr(validation, 'adjusted_position_size', trade_decision.get("position_size", 0))
            position_value = abs(position_size * entry_price)

            logger.info(f"About to check portfolio risk: balance=${account_balance:.2f}, position_value=${position_value:.2f}, symbol={trade_decision.get('symbol')}")

            portfolio_assessment = self.check_portfolio_risk(
                account_balance=account_balance,
                open_positions=open_positions,
                new_position_value=position_value,
                new_symbol=trade_decision.get("symbol"),
                leverage=leverage
            )

            # Add portfolio warnings
            validation.warnings.extend(portfolio_assessment.get("warnings", []))

            # Block if portfolio restrictions
            if not portfolio_assessment.get("can_open", True):
                validation.is_valid = False
                validation.rejection_reasons.extend(portfolio_assessment.get("restrictions", []))

            # Add portfolio metrics to validation
            validation.portfolio_metrics = portfolio_assessment.get("portfolio_metrics", {})
        
        # 11. Milestone-based risk adjustment
        suggested_risk = self._get_suggested_risk(account_balance, risk_pct)
        if abs(suggested_risk - risk_pct) > 5:
            validation.warnings.append(
                f"Consider {suggested_risk}% risk for current milestone"
            )
        
        # Log validation result
        if validation.is_valid:
            logger.info(
                f"Trade validated: {trade_decision.get('symbol')} "
                f"Risk: {risk_pct}% Leverage: {leverage}x"
            )
            if validation.warnings:
                logger.warning(f"Warnings: {', '.join(validation.warnings)}")
        else:
            logger.warning(
                f"Trade rejected: {', '.join(validation.rejection_reasons)}"
            )
        
        return validation
    
    def update_trade_result(
        self,
        trade_id: str,
        pnl: float,
        account_balance: float,
    ):
        """Update risk manager with trade result"""
        self.trades_today.append({
            "id": trade_id,
            "pnl": pnl,
            "balance_after": account_balance,
            "timestamp": datetime.now(timezone.utc),
        })
        
        # Update daily low
        if self.daily_low_balance is None or account_balance < self.daily_low_balance:
            self.daily_low_balance = account_balance
        
        # Check if drawdown limit hit
        drawdown = self._calculate_daily_drawdown(account_balance)
        if drawdown >= self.max_daily_drawdown_pct:
            self._trigger_circuit_breaker("Max daily drawdown reached after trade")
    
    def _calculate_position_size(
        self,
        account_balance: float,
        risk_pct: float,
        entry_price: float,
        stop_loss: float,
        leverage: int,
    ) -> float:
        """Calculate position size based on risk parameters"""
        # Risk amount in USD
        risk_amount = account_balance * (risk_pct / 100)
        
        # Stop distance
        stop_distance = abs(entry_price - stop_loss)
        stop_distance_pct = stop_distance / entry_price
        
        # Position value (before leverage)
        position_value = risk_amount / stop_distance_pct
        
        # Position size in units
        position_size = position_value / entry_price
        
        # Adjust for leverage (leverage affects margin requirement, not position size calculation)
        # The position size stays the same, but margin required = position_value / leverage
        
        return position_size
    
    def calculate_kelly_criterion(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
    ) -> float:
        """
        Calculate Kelly Criterion for optimal position sizing
        
        Args:
            win_rate: Historical win rate (0-1)
            avg_win: Average win amount
            avg_loss: Average loss amount (positive number)
        
        Returns:
            Optimal risk percentage (0-100)
        """
        if avg_loss == 0 or win_rate == 0:
            return 15.0  # Default
        
        # Kelly Formula: f = (bp - q) / b
        # where: b = avg_win/avg_loss, p = win_rate, q = 1 - win_rate
        b = avg_win / avg_loss
        p = win_rate
        q = 1 - p
        
        kelly = (b * p - q) / b
        
        # Convert to percentage
        kelly_pct = kelly * 100
        
        # Use fractional Kelly (0.25 to 0.5) for safety
        fractional_kelly = kelly_pct * 0.25  # 25% of full Kelly
        
        # Clamp to reasonable range
        optimal_risk = max(10, min(30, fractional_kelly))
        
        logger.debug(f"Kelly Criterion: {kelly_pct:.1f}%, Fractional: {fractional_kelly:.1f}%, Using: {optimal_risk:.1f}%")
        
        return optimal_risk
    
    def adjust_risk_for_volatility(
        self,
        base_risk_pct: float,
        current_atr: float,
        avg_atr: float,
    ) -> float:
        """
        Adjust risk based on current volatility
        
        Args:
            base_risk_pct: Base risk percentage
            current_atr: Current ATR
            avg_atr: Average ATR
        
        Returns:
            Adjusted risk percentage
        """
        # Volatility ratio
        vol_ratio = current_atr / avg_atr if avg_atr > 0 else 1.0
        
        # Reduce risk in high volatility
        if vol_ratio > 1.5:  # 50% above average
            adjusted = base_risk_pct * 0.7  # Reduce by 30%
            logger.debug(f"High volatility ({vol_ratio:.2f}x) - reducing risk: {base_risk_pct}% -> {adjusted:.1f}%")
            return adjusted
        
        # Increase risk slightly in low volatility
        elif vol_ratio < 0.7:  # 30% below average
            adjusted = base_risk_pct * 1.15  # Increase by 15%
            adjusted = min(adjusted, self.max_risk_pct)  # Cap at max
            logger.debug(f"Low volatility ({vol_ratio:.2f}x) - increasing risk: {base_risk_pct}% -> {adjusted:.1f}%")
            return adjusted
        
        return base_risk_pct
    
    def _calculate_daily_drawdown(self, current_balance: float) -> float:
        """Calculate current daily drawdown percentage"""
        if self.daily_start_balance is None:
            return 0.0
        
        if current_balance >= self.daily_start_balance:
            return 0.0
        
        drawdown = (self.daily_start_balance - current_balance) / self.daily_start_balance * 100
        return drawdown
    
    def _check_daily_reset(self):
        """Reset daily tracking if new day"""
        today = datetime.now(timezone.utc).date()
        
        if self.last_reset_date is None or self.last_reset_date != today:
            self.daily_start_balance = None
            self.daily_low_balance = None
            self.trades_today = []
            self.last_reset_date = today
            logger.info("Daily risk tracking reset")
    
    def _check_news_blackout(self, economic_events: List[Dict[str, Any]]) -> bool:
        """Check if we're in economic news blackout period"""
        now = datetime.now(timezone.utc)
        
        for event in economic_events:
            if event.get("impact") != "high":
                continue
            
            event_time = event.get("time_utc")
            if isinstance(event_time, str):
                event_time = datetime.fromisoformat(event_time.replace("Z", "+00:00"))
            
            # Check if event is within blackout window
            time_until = (event_time - now).total_seconds() / 60
            time_since = (now - event_time).total_seconds() / 60
            
            if -self.high_impact_news_blackout_mins <= time_until <= self.high_impact_news_blackout_mins:
                return True
            if 0 <= time_since <= self.high_impact_news_blackout_mins:
                return True
        
        return False
    
    def _check_correlation_risk(
        self,
        symbol: str,
        open_positions: List[Dict[str, Any]]
    ) -> Optional[str]:
        """Check for correlated positions"""
        if not symbol or not open_positions:
            return None

        # Expanded correlation matrix
        correlations = {
            "BTCUSDT": ["ETHUSDT", "SOLUSDT", "AVAXUSDT", "LINKUSDT"],
            "ETHUSDT": ["BTCUSDT", "SOLUSDT", "MATICUSDT", "AVAXUSDT"],
            "SOLUSDT": ["BTCUSDT", "ETHUSDT", "AVAXUSDT", "MATICUSDT"],
            "ARBUSDT": ["OPUSDT", "MATICUSDT"],
            "OPUSDT": ["ARBUSDT", "MATICUSDT"],
            "DOGEUSDT": ["1000PEPEUSDT"],
            "1000PEPEUSDT": ["DOGEUSDT"],
        }

        correlated_symbols = correlations.get(symbol, [])
        correlated_positions = []

        for position in open_positions:
            position_symbol = self._position_symbol(position)
            if position_symbol in correlated_symbols:
                correlated_positions.append(position_symbol)

        if len(correlated_positions) >= 2:
            return f"High correlation risk: {len(correlated_positions)} correlated positions"
        elif len(correlated_positions) == 1:
            return f"Moderate correlation with {correlated_positions[0]}"

        return None

    def check_portfolio_risk(
        self,
        account_balance: float,
        open_positions: List[Dict[str, Any]],
        new_position_value: float,
        new_symbol: str,
        leverage: int = 1,
    ) -> Dict[str, Any]:
        """
        Portfolio-level risk assessment

        Args:
            account_balance: Current account balance
            open_positions: List of open positions
            new_position_value: Value of potential new position
            new_symbol: Symbol of potential new position

        Returns:
            Risk assessment dict
        """
        assessment = {
            "can_open": True,
            "warnings": [],
            "restrictions": [],
            "portfolio_metrics": {}
        }

        def _position_notional(pos: Any) -> float:
            if isinstance(pos, dict):
                size_raw = pos.get("size", 0)
                price_raw = pos.get("avg_price", pos.get("entry_price", 0))
            else:
                size_raw = getattr(pos, "size", 0)
                price_raw = getattr(pos, "avg_price", getattr(pos, "entry_price", 0))
            return abs(
                float(size_raw or 0)
                * float(price_raw or 0)
            )

        def _position_leverage(pos: Any) -> float:
            raw = (pos.get("leverage", 1) if isinstance(pos, dict) else getattr(pos, "leverage", 1)) or 1
            try:
                value = float(raw)
                return max(1.0, value)
            except (TypeError, ValueError):
                return 1.0

        active_positions = [p for p in open_positions if _position_notional(p) > 0]

        # Portfolio exposure metrics: notional and margin usage.
        total_position_value = sum(_position_notional(pos) for pos in active_positions)
        total_margin_used = sum(_position_notional(pos) / _position_leverage(pos) for pos in active_positions)

        # Margin load is a more accurate risk proxy than raw notional for leveraged accounts.
        portfolio_margin_pct = (total_margin_used / account_balance) * 100 if account_balance > 0 else 0
        effective_leverage = (total_position_value / account_balance) if account_balance > 0 else 0

        assessment["portfolio_metrics"] = {
            "total_positions": len(active_positions),
            "total_position_value": round(total_position_value, 2),
            "total_margin_used": round(total_margin_used, 2),
            "portfolio_margin_pct": round(portfolio_margin_pct, 1),
            "account_balance": round(account_balance, 2),
            "effective_leverage": round(effective_leverage, 1),
        }

        # Position count limits
        if len(active_positions) >= 2:
            assessment["warnings"].append(f"Portfolio has {len(active_positions)} positions (recommended max: 2)")

        # Portfolio concentration limits (margin-based)
        max_portfolio_risk = float(self.portfolio_max_margin_pct)

        new_margin_used = new_position_value / max(1, leverage)
        new_risk_pct = (new_margin_used / account_balance) * 100 if account_balance > 0 else 100
        total_risk_after = portfolio_margin_pct + new_risk_pct

        logger.info(
            f"Portfolio risk check: balance=${account_balance:.2f}, max_margin={max_portfolio_risk}%, "
            f"current_margin={portfolio_margin_pct:.1f}%, new_margin={new_risk_pct:.1f}%, total_after={total_risk_after:.1f}%"
        )

        margin_pct_epsilon = 0.05  # 0.05 percentage points tolerance for float/rounding noise.
        if total_risk_after - max_portfolio_risk > margin_pct_epsilon:
            assessment["restrictions"].append(
                f"Portfolio margin would exceed {max_portfolio_risk}% (would be {total_risk_after:.1f}%)"
            )
            assessment["can_open"] = False
        elif abs(total_risk_after - max_portfolio_risk) <= margin_pct_epsilon:
            logger.debug(
                "Portfolio margin near limit accepted: "
                f"total_after={total_risk_after:.6f}% max={max_portfolio_risk:.6f}% eps={margin_pct_epsilon:.6f}%"
            )

        # Single position limits (based on margin used, not notional value)
        margin_used_by_position = new_position_value / leverage
        max_single_margin = account_balance * (float(self.symbol_max_margin_pct) / 100.0)

        margin_usd_epsilon = max(0.01, account_balance * 0.0001)  # 1 cent minimum, 1bp balance-scaled.
        if margin_used_by_position - max_single_margin > margin_usd_epsilon:
            max_pct = float(self.symbol_max_margin_pct)
            assessment["restrictions"].append(f"Position margin exceeds {max_pct}% of account (${margin_used_by_position:.2f} > ${max_single_margin:.2f})")
            assessment["can_open"] = False
        elif abs(margin_used_by_position - max_single_margin) <= margin_usd_epsilon:
            logger.debug(
                "Single-position margin near limit accepted: "
                f"used=${margin_used_by_position:.6f} cap=${max_single_margin:.6f} eps=${margin_usd_epsilon:.6f}"
            )

        return assessment
    
    def _get_suggested_risk(self, balance: float, current_risk: float) -> float:
        """Get suggested risk percentage based on balance/milestone"""
        if balance <= self.small_account_balance_threshold:
            return min(3.0, current_risk)
        elif balance < 500:
            return min(4.0, max(2.0, current_risk))
        else:
            return min(5.0, max(2.5, current_risk))
    
    def _trigger_circuit_breaker(self, reason: str):
        """Trigger trading halt circuit breaker"""
        self.trading_halted = True
        self.halt_reason = reason
        self.halt_until = datetime.now(timezone.utc) + timedelta(hours=1)
        
        logger.critical(f"CIRCUIT BREAKER TRIGGERED: {reason}")
        logger.critical(f"Trading halted until {self.halt_until}")
    
    def get_risk_metrics(self) -> Dict[str, Any]:
        """Get current risk metrics"""
        return {
            "trading_enabled": not self.trading_halted,
            "halt_reason": self.halt_reason,
            "daily_trades": len(self.trades_today),
            "daily_pnl": sum(t["pnl"] for t in self.trades_today),
            "daily_drawdown": self._calculate_daily_drawdown(
                self.trades_today[-1]["balance_after"] if self.trades_today else 0
            ),
            "last_trade": self.trades_today[-1] if self.trades_today else None,
        }
    
    def emergency_stop(self, reason: str = "Manual emergency stop"):
        """Emergency stop - halt all trading"""
        self.trading_halted = True
        self.halt_reason = reason
        self.halt_until = datetime.now(timezone.utc) + timedelta(hours=24)
        
        logger.critical(f"EMERGENCY STOP: {reason}")
        logger.critical("Trading halted for 24 hours")

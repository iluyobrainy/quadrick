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
        min_risk_pct: float = 10.0,
        max_risk_pct: float = 30.0,
        max_daily_drawdown_pct: float = 30.0,
        max_leverage: int = 50,
        max_concurrent_positions: int = 3,
        min_account_balance: float = 3.0,
        high_impact_news_blackout_mins: int = 15,
    ):
        """Initialize risk manager"""
        self.min_risk_pct = min_risk_pct
        self.max_risk_pct = max_risk_pct
        self.max_daily_drawdown_pct = max_daily_drawdown_pct
        self.max_leverage = max_leverage
        self.max_concurrent_positions = max_concurrent_positions
        self.min_account_balance = min_account_balance
        self.high_impact_news_blackout_mins = high_impact_news_blackout_mins
        
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
        
        # 2. Check concurrent positions limit
        if len(open_positions) >= self.max_concurrent_positions:
            validation.is_valid = False
            validation.rejection_reasons.append(
                f"Already have {len(open_positions)} positions (max: {self.max_concurrent_positions})"
            )
            return validation
        
        # 3. Validate risk percentage (adjust for small accounts)
        risk_pct = trade_decision.get("risk_pct", 15)

        # For very small accounts, reduce risk to prevent oversized positions
        # unless it's so small that we need some risk just to meet exchange minimums
        if account_balance < 20:
            max_risk_for_small = 15.0  # Increased from 8.0 to allow 0.001 BTC trades
            if risk_pct > max_risk_for_small:
                validation.adjusted_risk_pct = max_risk_for_small
                validation.warnings.append(
                    f"Risk capped at {max_risk_for_small}% for small account safety"
                )
                risk_pct = max_risk_for_small
        elif account_balance < 50:
            max_risk_for_small = 10.0  # Max 10% risk for accounts < $50
            if risk_pct > max_risk_for_small:
                validation.adjusted_risk_pct = max_risk_for_small
                validation.warnings.append(
                    f"Risk reduced from {risk_pct}% to {max_risk_for_small}% for small account safety"
                )
                risk_pct = max_risk_for_small

        if risk_pct < self.min_risk_pct or risk_pct > self.max_risk_pct:
            # Adjust to valid range (allow lower risk for very small accounts)
            if risk_pct < self.min_risk_pct and account_balance < 20:
                adjusted_risk = risk_pct  # honor conservative sizing for tiny balances
            else:
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
        leverage = trade_decision.get("leverage", 10)

        # Reduce leverage for tiny accounts to make margin requirements smaller
        if account_balance < 10:
            max_leverage_for_tiny = 10  # Max 10x for accounts under $10
            if leverage > max_leverage_for_tiny:
                validation.adjusted_leverage = max_leverage_for_tiny
                validation.warnings.append(
                    f"Leverage reduced from {leverage}x to {max_leverage_for_tiny}x for tiny account"
                )
                leverage = max_leverage_for_tiny

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

            # Get technical analysis for this symbol
            symbol_analysis = market_analysis.get('technical_analysis', {}).get(symbol, {})
            tf_analysis = symbol_analysis.get('timeframe_analysis', {})

            # Check 1h timeframe for trend strength
            adx_1h = tf_analysis.get('1h', {}).get('adx', 0)
            trend_1h = tf_analysis.get('1h', {}).get('trend', '')

            allow_counter_trend = bool(trade_decision.get("allow_counter_trend"))

            if adx_1h > 25:
                is_counter_trend = (trend_1h == 'trending_down' and side == 'Buy') or (trend_1h == 'trending_up' and side == 'Sell')
                if is_counter_trend:
                    if allow_counter_trend:
                        validation.warnings.append(
                            f"⚠️ Counter-trend trade approved by override (ADX {adx_1h:.1f}). Tight risk management required."
                        )
                    else:
                        validation.is_valid = False
                        validation.rejection_reasons.append(
                            f"🚫 COUNTER-TREND BLOCKED: ADX {adx_1h:.1f} indicates strong {trend_1h.replace('_', ' ')}. {side} trades forbidden unless explicitly allowed."
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
            # Dynamic margin caps for small accounts
            if account_balance < 20:
                max_margin_pct = 0.7  # up to 70% of balance for tiny accounts
                max_exposure_pct = 400  # up to 400% notional exposure
            elif account_balance < 50:
                max_margin_pct = 0.5
                max_exposure_pct = 300
            else:
                max_margin_pct = 0.25
                max_exposure_pct = 200

            # Cap by margin - CRITICAL: Respect LIQUID balance
            margin_used = (position_size * market_price) / max(1, leverage)
            # The maximum we can commit is based on total balance constraint, 
            # BUT it's capped by the physical liquid cash available (95% to allow for fees).
            max_margin = min(account_balance * max_margin_pct, liquid_balance * 0.95)
            
            if margin_used > max_margin:
                position_size = (max_margin * leverage) / market_price
                validation.warnings.append(
                    f"Position size reduced to fit margin cap ({max_margin_pct*100:.0f}%)"
                )

            # Cap by exposure
            position_value = position_size * market_price
            max_exposure_value = account_balance * (max_exposure_pct / 100)
            if position_value > max_exposure_value:
                position_size = max_exposure_value / market_price
                validation.warnings.append(
                    f"Position size reduced to fit exposure cap ({max_exposure_pct}%)"
                )

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
            
            # Warning if stop is too tight
            if stop_distance_pct < 0.5:
                validation.warnings.append(
                    f"Stop loss very tight at {stop_distance_pct:.2f}%"
                )
            
            # Warning if stop is too wide
            elif stop_distance_pct > 10:
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
            logger.debug(f"High volatility ({vol_ratio:.2f}x) - reducing risk: {base_risk_pct}% → {adjusted:.1f}%")
            return adjusted
        
        # Increase risk slightly in low volatility
        elif vol_ratio < 0.7:  # 30% below average
            adjusted = base_risk_pct * 1.15  # Increase by 15%
            adjusted = min(adjusted, self.max_risk_pct)  # Cap at max
            logger.debug(f"Low volatility ({vol_ratio:.2f}x) - increasing risk: {base_risk_pct}% → {adjusted:.1f}%")
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
            if position.symbol in correlated_symbols:
                correlated_positions.append(position.symbol)

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

        # Calculate portfolio metrics
        total_position_value = sum(
            abs(getattr(pos, 'size', 0) * getattr(pos, 'avg_price', getattr(pos, 'entry_price', 0)))
            for pos in open_positions
        )

        # For leveraged trading, risk is total exposure / account balance
        # Account balance is the margin/capital at risk
        total_exposure = total_position_value
        portfolio_risk_pct = (total_exposure / account_balance) * 100 if account_balance > 0 else 0

        assessment["portfolio_metrics"] = {
            "total_positions": len(open_positions),
            "total_position_value": round(total_position_value, 2),
            "portfolio_leverage": round(portfolio_risk_pct, 1),
            "account_balance": round(account_balance, 2),
            "effective_leverage": round(total_exposure / account_balance, 1) if account_balance > 0 else 0
        }

        # Position count limits
        if len(open_positions) >= 3:
            assessment["warnings"].append(f"Portfolio has {len(open_positions)} positions (recommended max: 3)")

        # Portfolio concentration limits (adjusted for leveraged trading)
        # For leveraged trading, allow much higher exposure ratios
        if account_balance < 10:
            max_portfolio_risk = 5000  # Allow up to 50x effective leverage for micro accounts
        elif account_balance < 20:
            max_portfolio_risk = 2000  # Allow up to 20x effective leverage for ultra-small accounts
        elif account_balance < 50:
            max_portfolio_risk = 400  # Allow up to 4x leverage for small accounts
        else:
            max_portfolio_risk = 300  # Standard 3x leverage limit

        new_risk_pct = (new_position_value / account_balance) * 100 if account_balance > 0 else 100
        total_risk_after = portfolio_risk_pct + new_risk_pct

        logger.info(f"Portfolio risk check: balance=${account_balance:.2f}, max_risk={max_portfolio_risk}%, current_risk={portfolio_risk_pct:.1f}%, new_risk={new_risk_pct:.1f}%, total_after={total_risk_after:.1f}%")

        if total_risk_after > max_portfolio_risk:
            assessment["restrictions"].append(f"Portfolio risk would exceed {max_portfolio_risk}% (would be {total_risk_after:.1f}%)")
            assessment["can_open"] = False

        # Single position size limits (based on margin used, not notional value)
        # For leveraged trading, check margin usage per position
        margin_used_by_position = new_position_value / leverage

        if account_balance < 10:
            max_single_margin = account_balance * 0.90  # Allow up to 90% margin for micro accounts
        elif account_balance < 20:
            max_single_margin = account_balance * 0.80  # Allow up to 80% margin for very small accounts
        elif account_balance < 50:
            max_single_margin = account_balance * 0.60  # Allow up to 60% margin for small accounts
        elif account_balance < 100:
            max_single_margin = account_balance * 0.35   # Moderate accounts
        else:
            max_single_margin = account_balance * 0.40   # Larger accounts

        if margin_used_by_position > max_single_margin:
            if account_balance < 10:
                max_pct = 90
            elif account_balance < 20:
                max_pct = 80
            elif account_balance < 50:
                max_pct = 60
            elif account_balance < 100:
                max_pct = 35
            else:
                max_pct = 40
            assessment["restrictions"].append(f"Position margin exceeds {max_pct}% of account (${margin_used_by_position:.2f} > ${max_single_margin:.2f})")
            assessment["can_open"] = False

        return assessment
    
    def _get_suggested_risk(self, balance: float, current_risk: float) -> float:
        """Get suggested risk percentage based on balance/milestone"""
        if balance < 50:
            # Conservative phase
            return min(15, current_risk)
        elif balance < 200:
            # Moderate phase
            return min(20, max(15, current_risk))
        elif balance < 500:
            # Moderate aggressive
            return min(25, max(18, current_risk))
        else:
            # Maximum aggressive
            return min(30, max(22, current_risk))
    
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

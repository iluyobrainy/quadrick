import asyncio
from typing import Dict, Any, Optional
from datetime import datetime
from loguru import logger
from src.agents.analyst import AnalystAgent, MarketRegime
from src.agents.strategist import StrategistAgent, TradePlan
from src.strategies import TrendFollowingStrategy, MeanReversionStrategy, BreakoutStrategy, ScalpingMomentumStrategy, BaseStrategy
from src.risk.risk_manager import RiskManager

# Regime-to-Strategy mapping with confidence multipliers
REGIME_STRATEGY_MAP = {
    "bull_trend": [("TrendFollowing", 1.0), ("Breakout", 0.9)],
    "bear_trend": [("TrendFollowing", 1.0), ("Breakout", 0.9)],
    "range_chop": [("MeanReversion", 1.0), ("ScalpingMomentum", 0.85)],
    "volatile": [("Breakout", 1.0), ("ScalpingMomentum", 0.9)],
    "low_volatility": [("Breakout", 1.0)],  # Squeeze breakout opportunity
}

class TradingCouncil:
    """
    The Council: Orchestrates the Multi-Agent Workflow.
    Replaces the monolithic DeepSeekClient.get_trading_decision call.
    """
    
    def __init__(self, llm_client, risk_manager: RiskManager, db_client=None, bridge=None):
        self.llm = llm_client
        self.risk_manager = risk_manager
        self.db = db_client  # Supabase client for RAG
        self.bridge = bridge # Dashboard bridge
        
        # Initialize Agents
        self.analyst = AnalystAgent(llm_client)
        self.strategist = StrategistAgent(llm_client)
        
        # Initialize Strategy Library (expanded with new strategies)
        self.strategies = {
            "TrendFollowing": TrendFollowingStrategy(),
            "MeanReversion": MeanReversionStrategy(),
            "Breakout": BreakoutStrategy(),
            "ScalpingMomentum": ScalpingMomentumStrategy(),
        }

    @staticmethod
    def _as_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _normalize_trend(value: Any) -> str:
        trend = str(value or "neutral").lower()
        mapping = {
            "uptrend": "uptrend",
            "bullish": "uptrend",
            "trending_up": "uptrend",
            "up": "uptrend",
            "downtrend": "downtrend",
            "bearish": "downtrend",
            "trending_down": "downtrend",
            "down": "downtrend",
            "neutral": "neutral",
            "range_chop": "neutral",
            "ranging": "neutral",
            "sideways": "neutral",
        }
        return mapping.get(trend, "neutral")

    def _build_fallback_plan_from_opportunity(
        self,
        symbol: str,
        market_data: Dict[str, Any],
        account_balance: float,
    ) -> Optional[TradePlan]:
        score = self._as_float(market_data.get("opportunity_score"), 0.0)
        direction = str(market_data.get("opportunity_direction") or "neutral").lower()
        if score < 72 or direction not in {"long", "short"}:
            return None

        current_price = self._as_float(market_data.get("current_price"), 0.0)
        if current_price <= 0:
            return None

        tf_15m = market_data.get("timeframe_analysis", {}).get("15m", {})
        atr_15m = self._as_float(tf_15m.get("atr"), current_price * 0.005)
        if atr_15m <= 0:
            atr_15m = current_price * 0.005

        action = "buy" if direction == "long" else "sell"
        suggested_sl = self._as_float(market_data.get("opportunity_suggested_sl"), 0.0)
        suggested_tp = self._as_float(market_data.get("opportunity_suggested_tp"), 0.0)

        if action == "buy":
            stop_loss = suggested_sl if 0 < suggested_sl < current_price else current_price - (atr_15m * 1.25)
            take_profit = suggested_tp if suggested_tp > current_price else current_price + (atr_15m * 2.0)
        else:
            stop_loss = suggested_sl if suggested_sl > current_price else current_price + (atr_15m * 1.25)
            take_profit = suggested_tp if 0 < suggested_tp < current_price else current_price - (atr_15m * 2.0)

        risk_pct = 1.2 if account_balance <= 150 else 1.8
        confidence = max(0.60, min(0.78, 0.60 + ((score - 72) * 0.006)))
        return TradePlan(
            symbol=symbol,
            action=action,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_pct=risk_pct,
            reasoning=(
                f"Deterministic fallback from OpportunityScorer (score={score:.1f}, direction={direction}). "
                "LLM returned WAIT, so execution used ATR-anchored fallback levels."
            ),
            confidence=confidence,
        )

    async def make_decision(self, symbol: str, market_data: Dict[str, Any], account_balance: float, autonomous_mode: bool = True) -> Dict[str, Any]:
        """
        Execute the Council Workflow:
        Analyst -> [Select Strategy] -> Strategist -> Risk Officer -> Decision
        
        If autonomous_mode is True, we skip the hardcoded strategies and let the 
        Strategist design the plan based on professional knowledge and RAG.
        """
        logger.info(f"🏛️ Council convening for {symbol} (Autonomous: {autonomous_mode})...")
        
        # 0. Fetch RAG Memory (Historical Context)
        similar_trades = []
        if self.db:
            try:
                similar_trades = await self.db.find_similar_trades(symbol, market_data, limit=5)
                if similar_trades:
                    wins = sum(1 for t in similar_trades if t.get("win"))
                    total = len(similar_trades)
                    avg_pnl = sum(float(t.get("pnl_pct", 0)) for t in similar_trades) / total
                    logger.info(f"📚 RAG: Found {total} similar setups from memory. History in this context: {wins}W/{total-wins}L (Avg PnL: {avg_pnl:+.2f}%)")
                    
                    # Log specific examples for transparency
                    for i, t in enumerate(similar_trades[:2]):
                        res = "WON" if t.get("win") else "LOST"
                        logger.debug(f"   Memory #{i+1}: {t.get('symbol')} {res} ({t.get('pnl_pct'):+.1f}%) using {t.get('strategy')}")
                
                # Add to market_data for agents to use
                market_data["similar_trades"] = similar_trades
            except Exception as e:
                logger.warning(f"Failed to fetch RAG memories: {e}")
        
        # 1. The Analyst: Determine Regime
        try:
            regime, a_prompt, a_response = await self.analyst.analyze_market(symbol, market_data)
            logger.info(f"🧐 Analyst: {regime.regime} (Conf: {regime.confidence:.2f}) -> Rec: {regime.recommended_strategy}")
            
            # Push Analyst Insights
            if hasattr(self, 'bridge') and self.bridge:
                self.bridge.update_ai_insights({
                    "symbol": symbol,
                    "agent": "Analyst",
                    "prompt": a_prompt,
                    "response": a_response,
                    "timestamp": datetime.now().isoformat()
                })
        except Exception as e:
            logger.error(f"Analyst failed for {symbol}: {e}")
            return self._create_wait_decision(symbol, f"Analyst error: {e}")
        
        if autonomous_mode:
            # In autonomous mode, analyst guidance is advisory.
            # Only block if regime confidence is very weak.
            if regime.confidence < 0.55:
                return self._create_wait_decision(
                    symbol,
                    f"Analyst confidence too low for autonomous plan ({regime.regime}, {regime.confidence:.2f})"
                )
            if regime.recommended_strategy == "Wait":
                logger.info(
                    f"Analyst suggested WAIT for {symbol}, but autonomous planning remains enabled "
                    f"(regime={regime.regime}, conf={regime.confidence:.2f})"
                )
        else:
            if regime.recommended_strategy == "Wait" or regime.confidence < 0.72:
                return self._create_wait_decision(symbol, f"Analyst suggests waiting ({regime.regime})")

        # 3. Create Trade Plan
        try:
            if autonomous_mode:
                # --- AUTONOMOUS MODE: AI Designs the Strategy ---
                plan, s_prompt, s_response = await self.strategist.create_autonomous_plan(symbol, regime, market_data, account_balance)
                strategy_name = "AutonomousAI"
            else:
                # --- LEGACY MODE: Uses Hardcoded Heuristics ---
                strategy_name = regime.recommended_strategy
                strategy = self.strategies.get(strategy_name)
                
                if not strategy:
                    logger.warning(f"Strategy {strategy_name} not found, defaulting to Wait")
                    return self._create_wait_decision(symbol, f"Strategy {strategy_name} not implemented")
                
                plan, s_prompt, s_response = await self.strategist.create_trade_plan(symbol, regime, strategy, market_data, account_balance)
            
            logger.info(f"♟️ Strategist: {plan.action.upper()} {symbol} @ {plan.entry_price}")
            
            # Push Strategist Insights
            if hasattr(self, 'bridge') and self.bridge:
                self.bridge.update_ai_insights({
                    "symbol": symbol,
                    "agent": "Strategist",
                    "prompt": s_prompt,
                    "response": s_response,
                    "timestamp": datetime.now().isoformat()
                })
        except Exception as e:
            logger.error(f"Strategist failed for {symbol}: {e}")
            return self._create_wait_decision(symbol, f"Strategist error: {e}")

        if plan.action == "wait" and autonomous_mode:
            tf_15m = market_data.get("timeframe_analysis", {}).get("15m", {})
            try:
                volume_ratio_15m = float(tf_15m.get("volume_ratio", 1.0) or 1.0)
            except (TypeError, ValueError):
                volume_ratio_15m = 1.0

            if regime.confidence >= 0.58 and volume_ratio_15m >= 1.0:
                logger.info(
                    f"{symbol}: strategist returned WAIT under actionable conditions "
                    f"(regime_conf={regime.confidence:.2f}, vol={volume_ratio_15m:.2f}x); retrying in decisive mode"
                )
                try:
                    retry_plan, retry_prompt, retry_response = await self.strategist.create_autonomous_plan(
                        symbol=symbol,
                        regime=regime,
                        market_data=market_data,
                        account_balance=account_balance,
                        force_action=True,
                    )
                    if retry_plan.action != "wait":
                        plan = retry_plan
                        s_prompt = retry_prompt
                        s_response = retry_response
                        logger.info(f"{symbol}: decisive retry produced {plan.action.upper()} plan")
                except Exception as retry_error:
                    logger.warning(f"Decisive retry failed for {symbol}: {retry_error}")

        if plan.action == "wait":
            fallback_plan = self._build_fallback_plan_from_opportunity(
                symbol=symbol,
                market_data=market_data,
                account_balance=account_balance,
            )
            if fallback_plan:
                plan = fallback_plan
                strategy_name = "OpportunityFallback"
                logger.info(
                    f"{symbol}: replacing WAIT with deterministic fallback plan "
                    f"(opportunity_score={self._as_float(market_data.get('opportunity_score'), 0.0):.1f})"
                )
            else:
                return self._create_wait_decision(symbol, f"Strategist decided to wait: {plan.reasoning}")
        
        # 3.5 Multi-Timeframe Confirmation (NEW)
        htf_soft_override = False
        htf_aligned = self._check_htf_alignment(symbol, market_data, plan.action)
        if not htf_aligned:
            score = self._as_float(market_data.get("opportunity_score"), 0.0)
            tf_15m = market_data.get("timeframe_analysis", {}).get("15m", {})
            adx_15m = self._as_float(tf_15m.get("adx"), 25.0)
            regime_name = str(regime.regime or "").lower()
            soft_override = (
                score >= 86
                and plan.confidence >= 0.66
                and (
                    (regime_name in {"volatile", "range_chop"} and adx_15m <= 18)
                    or adx_15m <= 16
                )
            )
            if soft_override:
                logger.warning(
                    f"⚠️ {symbol}: HTF soft-override enabled for {plan.action.upper()} "
                    f"(score={score:.1f}, conf={plan.confidence:.2f}, regime={regime_name}, adx15={adx_15m:.1f})"
                )
                htf_aligned = True
                htf_soft_override = True
            else:
                logger.warning(
                    f"⚠️ {symbol}: HTF not aligned for {plan.action.upper()} "
                    f"(score={score:.1f}, conf={plan.confidence:.2f}) - skipping trade"
                )
                return self._create_wait_decision(symbol, f"HTF not aligned for {plan.action} direction")
        
        logger.info(f"✅ {symbol}: HTF alignment confirmed for {plan.action.upper()}")
        
        # 4. Validate SL/TP
        current_price = market_data.get("current_price", plan.entry_price)
        
        # Absolute validation
        if plan.stop_loss <= 0 or plan.take_profit <= 0:
            logger.warning(f"Invalid SL/TP from Strategist: SL={plan.stop_loss}, TP={plan.take_profit}")
            return self._create_wait_decision(symbol, "Invalid stop loss or take profit")

        # Range validation (check if targets are within 50% of price - very generous buffer)
        # This catches "offsets" like 0.002 when price is 1.0+
        if current_price > 0:
            sl_dist_pct = abs(plan.stop_loss - current_price) / current_price
            tp_dist_pct = abs(plan.take_profit - current_price) / current_price
            
            if sl_dist_pct > 0.5 or tp_dist_pct > 2.0: # Very loose but catches hallucinations
                logger.warning(f"SL/TP too far from price: SL={plan.stop_loss} ({sl_dist_pct:.1%}), TP={plan.take_profit} ({tp_dist_pct:.1%}) @ Price={current_price}")
                return self._create_wait_decision(symbol, "Targets too far from current market price")
            
            if plan.stop_loss < (current_price * 0.1) or plan.take_profit < (current_price * 0.1):
                logger.warning(f"SL/TP suspiciously low: SL={plan.stop_loss}, TP={plan.take_profit} @ Price={current_price}")
                return self._create_wait_decision(symbol, "Targets are suspiciously low compared to market price")

        # 5. Calculate leverage (simple logic for now)
        leverage = min(10, max(5, int(plan.confidence * 15)))  # 5-10x based on confidence

        # 6. Final Decision Output (Match DeepSeekClient parser format)
        return {
            "decision_type": "open_position",
            "trade": {  # Wrapper expected by _parse_response
                "symbol": symbol,
                "side": "Buy" if plan.action == "buy" else "Sell",
                "order_type": "Market",
                "risk_pct": plan.risk_pct,
                "leverage": leverage,
                "entry_price_target": plan.entry_price,
                "stop_loss": plan.stop_loss,
                "take_profit_1": plan.take_profit,
                "take_profit_2": None,
                "strategy_tag": strategy_name,
                "confidence_score": plan.confidence,
            },
            "reasoning": {
                "analyst": regime.reasoning,
                "strategist": plan.reasoning,
                "regime": regime.regime,
                "htf_aligned": True,
                "htf_soft_override": htf_soft_override,
                "opportunity_score": self._as_float(market_data.get("opportunity_score"), 0.0),
                "opportunity_direction": str(market_data.get("opportunity_direction") or "neutral"),
            }
        }

    def _create_wait_decision(self, symbol: str, reason: str) -> Dict[str, Any]:
        return {
            "decision_type": "wait",
            "symbol": symbol,
            "reasoning": {"reason": reason}
        }
    
    def _check_htf_alignment(self, symbol: str, data: Dict[str, Any], direction: str) -> bool:
        """
        Multi-Timeframe Confirmation: Ensure 1H and 4H trends align with trade direction.
        
        For LONG trades: 1H should be bullish/uptrend, 4H should be bullish/uptrend/neutral
        For SHORT trades: 1H should be bearish/downtrend, 4H should be bearish/downtrend/neutral
        
        This prevents taking trades against strong higher timeframe trends.
        """
        tf = data.get("timeframe_analysis", {})
        
        # Get 1H and 4H trends
        h1_data = tf.get("1h", {})
        h4_data = tf.get("4h", {})
        
        h1_trend = self._normalize_trend(h1_data.get("trend", "neutral"))
        h4_trend = self._normalize_trend(h4_data.get("trend", "neutral"))
        adx_1h = self._as_float(h1_data.get("adx"), 0.0)
        
        # Normalize trend values (handle different naming conventions)
        bullish_trends = ["uptrend"]
        bearish_trends = ["downtrend"]
        neutral_trends = ["neutral"]
        low_trend_strength = adx_1h > 0 and adx_1h < 18

        if direction.lower() == "buy":
            # For longs: prefer strict alignment; allow relaxed alignment only in weak HTF trend.
            h1_aligned = h1_trend.lower() in bullish_trends
            h4_ok = h4_trend.lower() in bullish_trends or h4_trend.lower() in neutral_trends

            if h1_aligned and h4_ok:
                return True
            if h1_aligned and low_trend_strength:
                logger.info(
                    f"{symbol}: HTF relaxed long alignment accepted (h1={h1_trend}, h4={h4_trend}, adx1h={adx_1h:.1f})"
                )
                return True
            
            # Log why alignment failed
            if not h1_aligned:
                logger.debug(f"{symbol}: 1H trend '{h1_trend}' not aligned for LONG")
            if not h4_ok:
                logger.debug(f"{symbol}: 4H trend '{h4_trend}' not supportive for LONG")
                
        elif direction.lower() == "sell":
            # For shorts: prefer strict alignment; allow relaxed alignment only in weak HTF trend.
            h1_aligned = h1_trend.lower() in bearish_trends
            h4_ok = h4_trend.lower() in bearish_trends or h4_trend.lower() in neutral_trends

            if h1_aligned and h4_ok:
                return True
            if h1_aligned and low_trend_strength:
                logger.info(
                    f"{symbol}: HTF relaxed short alignment accepted (h1={h1_trend}, h4={h4_trend}, adx1h={adx_1h:.1f})"
                )
                return True
            
            # Log why alignment failed
            if not h1_aligned:
                logger.debug(f"{symbol}: 1H trend '{h1_trend}' not aligned for SHORT")
            if not h4_ok:
                logger.debug(f"{symbol}: 4H trend '{h4_trend}' not supportive for SHORT")
        
        return False

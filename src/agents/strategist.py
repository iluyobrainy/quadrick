from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from loguru import logger
import json
from .analyst import MarketRegime
from src.strategies.base_strategy import BaseStrategy, StrategySignal

@dataclass
class TradePlan:
    symbol: str
    action: str # "buy", "sell", "wait"
    entry_price: float
    stop_loss: float
    take_profit: float
    risk_pct: float
    reasoning: str
    confidence: float

class StrategistAgent:
    """
    Agent 2: The Strategist
    Role: Executes the chosen strategy with precision.
    Input: MarketRegime, Strategy Logic (Code), Market Data
    Output: TradePlan (Entry, SL, TP)
    """
    
    def __init__(self, llm_client):
        self.llm = llm_client

    async def create_trade_plan(self, 
                              symbol: str, 
                              regime: MarketRegime, 
                              strategy: BaseStrategy, 
                              market_data: Dict[str, Any],
                              account_balance: float) -> Tuple[TradePlan, str, str]:
        """
        Generate a specific trade plan based on the strategy and regime.
        """
        
        # 1. Run the Code-Based Strategy Logic first (The "Guardrails")
        # This gives us a raw signal based on math/indicators
        raw_signal = strategy.analyze(market_data)
        
        if raw_signal.signal_type == "neutral":
            return TradePlan(symbol, "wait", 0, 0, 0, 0, f"Strategy {strategy.name} returned neutral", 0.0), "N/A (Strategy neutral)", "N/A"
        
        # 2. Validate entry using strategy's guardrails
        if not strategy.validate_entry(raw_signal, market_data):
            return TradePlan(symbol, "wait", 0, 0, 0, 0, f"Strategy {strategy.name} validation failed", 0.0), "N/A (Strategy validation failed)", "N/A"

        # 2. Use LLM to refine the execution (The "Art")
        # The LLM sees the code's signal and the regime, and optimizes entry/exit
        prompt = self._build_strategist_prompt(symbol, regime, strategy, raw_signal, market_data, account_balance)
        
        try:
            response = await self.llm.get_completion(
                prompt=prompt,
                system_prompt="You are an elite Execution Trader. Your job is to refine a trade signal into a precise plan. Optimize Entry, SL, and TP.",
                json_mode=True
            )
            
            data = json.loads(response)
            
            plan = TradePlan(
                symbol=symbol,
                action=data.get("action", "wait").lower(),
                entry_price=data.get("entry_price", 0.0),
                stop_loss=data.get("stop_loss", 0.0),
                take_profit=data.get("take_profit", 0.0),
                risk_pct=data.get("risk_pct", 1.0),
                reasoning=data.get("reasoning", ""),
                confidence=data.get("confidence", 0.0)
            )
            return plan, prompt, response
            
        except Exception as e:
            logger.error(f"Strategist Agent failed for {symbol}: {e}")
            return TradePlan(symbol, "wait", 0, 0, 0, 0, f"Error: {e}", 0.0), prompt, str(e)

    async def create_autonomous_plan(self, 
                                     symbol: str, 
                                     regime: MarketRegime, 
                                     market_data: Dict[str, Any],
                                     account_balance: float,
                                     force_action: bool = False) -> Tuple[TradePlan, str, str]:
        """
        AI-Only Mode: Designs a trade plan without any code-based heuristics.
        Uses professional Knowledge Base + RAG memories + Real-time Indicators.
        """
        from .knowledge_base import get_knowledge_summary
        
        # 1. Gather context
        knowledge = get_knowledge_summary()
        similar_trades = market_data.get("similar_trades", [])
        
        # 2. Build the autonomous prompt
        prompt = self._build_autonomous_prompt(
            symbol,
            regime,
            market_data,
            knowledge,
            similar_trades,
            account_balance,
            force_action=force_action,
        )
        
        try:
            response = await self.llm.get_completion(
                prompt=prompt,
                system_prompt="You are an Autonomous AI Quant Trader. You don't use simple rules; you understand market dynamics, liquidity, and probability. Create a high-conviction trade plan.",
                json_mode=True
            )
            
            data = json.loads(response)
            
            plan = TradePlan(
                symbol=symbol,
                action=data.get("action", "wait").lower(),
                entry_price=data.get("entry_price", 0.0),
                stop_loss=data.get("stop_loss", 0.0),
                take_profit=data.get("take_profit", 0.0),
                risk_pct=data.get("risk_pct", 1.0),
                reasoning=data.get("reasoning", ""),
                confidence=data.get("confidence", 0.0)
            )
            return plan, prompt, response
            
        except Exception as e:
            logger.error(f"Autonomous Strategist failed for {symbol}: {e}")
            return TradePlan(symbol, "wait", 0, 0, 0, 0, f"Error: {e}", 0.0), prompt, str(e)

    def _build_autonomous_prompt(
        self,
        symbol: str,
        regime: MarketRegime,
        data: Dict[str, Any],
        knowledge: str,
        similar_trades: list,
        account_balance: float,
        force_action: bool = False,
    ) -> str:
        current_price = data.get("current_price", 0)
        tf_analysis = data.get("timeframe_analysis", {})
        
        # Format similar trades for the prompt
        rag_context = "No similar historical trades found."
        if similar_trades:
            rag_context = "\n".join([
                f"- Past Trade: {t.get('symbol')} | Result: {'WIN' if t.get('win') else 'LOSS'} | PnL: {t.get('pnl_pct')}% | Strategy used: {t.get('strategy')}"
                for t in similar_trades
            ])

        force_directive = ""
        if force_action:
            force_directive = (
                "\nDECISIVE MODE:\n"
                "- Prefer an actionable trade plan if the setup is coherent.\n"
                "- Do not default to WAIT because of minor uncertainty.\n"
                "- Use conservative risk if conviction is moderate.\n"
                "- WAIT is forbidden unless market data is clearly missing or structurally contradictory."
            )

        return f"""
AUTHENTIC AUTONOMOUS TRADING PLAN: {symbol}
CURRENT PRICE: {current_price}
ACCOUNT BALANCE: ${account_balance:.2f}

{knowledge}

### RAG MEMORY (Similar Past Setups):
{rag_context}

### REAL-TIME MARKET CONTEXT:
MARKET REGIME: {regime.regime} (Confidence: {regime.confidence})
ANALYST INSIGHTS: {regime.reasoning}

HTF ANALYSIS (1H/4H):
- 4H Trend: {tf_analysis.get('4h', {}).get('trend', 'neutral')} | RSI: {tf_analysis.get('4h', {}).get('rsi', 50)}
- 1H Trend: {tf_analysis.get('1h', {}).get('trend', 'neutral')} | RSI: {tf_analysis.get('1h', {}).get('rsi', 50)}

LTF TRIGGER (15M/5M):
- 15m Trend: {tf_analysis.get('15m', {}).get('trend', 'neutral')} | RSI: {tf_analysis.get('15m', {}).get('rsi', 50)}
- 15m Volume Ratio: {tf_analysis.get('15m', {}).get('volume_ratio', 1.0)}x average
- Key Levels: {data.get('key_levels', 'N/A')}
{force_directive}

TASK:
1. FORECAST: Analyze the 1-minute chart. Predict the Price Action for the next 2-5 minutes (Next 3 candles).
2. DIRECTION: Determine whether the impulse is UP (Long) or DOWN (Short). Evaluate both directions equally without bias.
3. TRADE PLAN:
   - SL: The price level that INVALIDATES your 2-minute forecast.
   - TP: The peak of the impulse.
   - RISK: High conviction = High leverage.

OUTPUT JSON:
{{
    "action": "buy" | "sell" | "wait",
    "entry_price": {current_price},
    "stop_loss": float (Invalidation Level),
    "take_profit": float (Impulse Target),
    "risk_pct": float (suggest 1.0 to 15.0),
    "confidence": 0.0 to 1.0,
    "reasoning": "Forecast: Price will reject off {{level}} and drop to..."
}}
"""

    def _build_strategist_prompt(self, symbol: str, regime: MarketRegime, strategy: BaseStrategy, signal: StrategySignal, data: Dict[str, Any], account_balance: float) -> str:
        current_price = data.get("current_price", 0)
        
        return f"""
REFINED STRATEGY: {strategy.name}
SYMBOL: {symbol}
CURRENT PRICE: {current_price}
ACCOUNT BALANCE: ${account_balance:.2f}

MARKET REGIME: {regime.regime} (Confidence: {regime.confidence})
ANALYST REASONING: {regime.reasoning}

CODE-BASED HEURISTIC SIGNAL:
- Type: {signal.signal_type} ({signal.direction})
- Confidence: {signal.confidence}
- Suggested SL: {signal.stop_loss}
- Suggested TP: {signal.take_profit}
- Reasoning: {signal.reasoning}

TASK:
1. FORECAST: Predict the exact price action for the next 2-5 minutes (next 3-5 candles).
2. DETERMINE TREND: Is the *immediate* impulse UP (Long) or DOWN (Short)? Analyze objectively without directional bias.
3. DESIGN SNIPER PLAN:
   - If Forecast is Neutral/Weak -> WAIT.
   - If Forecast is Strong -> ENTER.
   - STOP LOSS: Must be the specific INVALIDATION LEVEL of your forecast (Where is the thesis wrong?).
   - TAKE PROFIT: maximise profit based on the impulse velocity.

OUTPUT JSON:
{{
    "action": "buy" | "sell" | "wait",
    "entry_price": {current_price},
    "stop_loss": float (Absolute Prediction Invalidation Level),
    "take_profit": float (Impulse Max Target),
    "risk_pct": float (suggest 1.0 to 15.0),
    "confidence": 0.0 to 1.0,
    "reasoning": "Forecast: Price will sweep {{level}} then reject..."
}}
"""

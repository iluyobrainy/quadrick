import asyncio
import os
from dotenv import load_dotenv
from src.agents.council import TradingCouncil
from src.llm.deepseek_client import DeepSeekClient
from src.risk.risk_manager import RiskManager
from config.settings import settings

# Load env
load_dotenv()

async def run_verification():
    print("🧪 Initializing Council Verification...")
    
    # 1. Setup Components
    llm = DeepSeekClient(
        api_key=settings.llm.deepseek_api_key,
        model=settings.llm.deepseek_model
    )
    
    risk = RiskManager() # Defaults are fine
    
    council = TradingCouncil(llm, risk)
    
    # 2. Mock Market Data (Strong Uptrend Scenario)
    print("\n📊 Simulating Market Data (Bullish Scenario)...")
    market_data = {
        "current_price": 50000.0,
        "timeframe_analysis": {
            "1h": {
                "trend": "uptrend",
                "rsi": 65,
                "macd_signal": "bullish",
                "adx": 30,
                "atr": 500.0
            },
            "15m": {
                "trend": "uptrend",
                "rsi": 55,
                "bb_position": "middle"
            }
        }
    }
    
    # 3. Run Council
    print("🤔 Asking Council for decision on BTCUSDT...")
    decision = await council.make_decision("BTCUSDT", market_data, account_balance=1000.0)
    
    # 4. Output Results
    print("\n🤖 COUNCIL DECISION OUTPUT:")
    print(f"Type: {decision.get('decision_type')}")
    print(f"Strategy: {decision.get('strategy_tag')}")
    print(f"Confidence: {decision.get('confidence_score')}")
    
    reasoning = decision.get("reasoning", {})
    print(f"\n🧐 Analyst View: {reasoning.get('analyst')}")
    print(f"♟️ Strategist Plan: {reasoning.get('strategist')}")
    
    if decision.get('decision_type') == 'open_position':
        print(f"\n✅ Trade Plan: {decision.get('side')} @ Market")
        print(f"   SL: {decision.get('stop_loss')}")
        print(f"   TP: {decision.get('take_profit_1')}")
        print(f"   Risk: {decision.get('risk_pct')}%")

if __name__ == "__main__":
    asyncio.run(run_verification())

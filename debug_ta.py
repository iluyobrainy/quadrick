#!/usr/bin/env python3
"""
DEBUG SCRIPT: Check Technical Analysis Data Flow
"""

import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import QuadrickTradingBot

async def debug_ta_data():
    print("🔍 DEBUGGING TECHNICAL ANALYSIS DATA STRUCTURE")
    print("=" * 60)

    try:
        bot = QuadrickTradingBot()
        print("✅ Bot initialized")

        # Get market data and analysis
        market_data = await bot._fetch_market_data()
        analysis = await bot._analyze_markets(market_data)

        # Prepare LLM context (same as production)
        context = bot._prepare_llm_context(market_data, analysis)

        # Check technical analysis structure
        ta = context.get('technical_analysis', {})
        print(f"\n📊 Technical Analysis Keys: {list(ta.keys())[:3]}...")

        if ta:
            first_symbol = list(ta.keys())[0]
            symbol_data = ta[first_symbol]
            print(f"\n🔍 {first_symbol} structure:")
            print(f"  Keys: {list(symbol_data.keys())}")

            # Check if there's timeframe_analysis
            if 'timeframe_analysis' in symbol_data:
                tf_analysis = symbol_data['timeframe_analysis']
                print(f"  timeframe_analysis keys: {list(tf_analysis.keys())}")

                # Check timeframe data
                for tf in ['1h', '15m']:
                    if tf in tf_analysis:
                        tf_data = tf_analysis[tf]
                        print(f"  {tf} keys: {list(tf_data.keys()) if isinstance(tf_data, dict) else type(tf_data)}")

                        # Check key_levels specifically
                        if 'key_levels' in tf_data:
                            kl = tf_data['key_levels']
                            print(f"    key_levels: {list(kl.keys()) if isinstance(kl, dict) else type(kl)}")
                            if isinstance(kl, dict):
                                pivot = kl.get('pivot_point', 'missing')
                                support = kl.get('immediate_support', 'missing')
                                print(f"    Sample values: pivot={pivot}, support={support}")
                        else:
                            print(f"    ❌ key_levels MISSING from {tf}")

                        # Check ATR
                        if 'atr' in tf_data:
                            print(f"    ATR: {tf_data['atr']}")
                        else:
                            print(f"    ❌ ATR MISSING from {tf}")
                    else:
                        print(f"  ❌ {tf} timeframe MISSING from timeframe_analysis")
            else:
                print("  ❌ timeframe_analysis MISSING from symbol data")

        # Test what HybridDeepSeekClient sees
        from src.llm.deepseek_client import HybridDeepSeekClient
        hybrid = HybridDeepSeekClient()
        formatted_ta = hybrid._format_technical_analysis(context)

        print("\n🤖 Formatted TA for DeepSeek (first 500 chars):")
        print(formatted_ta[:500] + "...")

        if 'Key Levels:' in formatted_ta:
            print("✅ Key Levels found in formatted output")
        else:
            print("❌ Key Levels MISSING from formatted output")

        if 'ATR:' in formatted_ta:
            print("✅ ATR found in formatted output")
        else:
            print("❌ ATR MISSING from formatted output")

        print("\n🎯 Data structure analysis complete")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_ta_data())

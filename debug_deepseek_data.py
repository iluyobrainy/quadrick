#!/usr/bin/env python3
"""
Debug script to show what data DeepSeek receives
"""
import asyncio
import sys
sys.path.insert(0, '.')

async def debug_deepseek_data():
    from main import QuadrickTradingBot

    bot = QuadrickTradingBot()

    # Get all the data that would be sent to DeepSeek
    market_data = await bot._fetch_market_data()
    analysis = await bot._analyze_markets(market_data)

    # Use the same call as main.py
    positions_data = []  # Simplified for debug
    position_monitor_summary = {"total_positions": 0, "positions": []}
    ta_summary = analysis.get("technical_analysis", {})
    milestone_progress = {"current": 0, "next": 50, "progress": 0}
    performance_feedback = {"total_trades": 0, "message": "No trades yet"}

    context = bot.deepseek.prepare_market_context(
        account_balance=bot.account_balance,
        positions=positions_data,
        position_monitor=position_monitor_summary,
        market_data=market_data,
        technical_analysis=ta_summary,
        funding_rates=market_data.get("funding_rates", {}),
        top_movers=market_data.get("top_movers", {"gainers": [], "losers": []}),
        milestone_progress=milestone_progress,
        recent_trades=[],
        performance_feedback=performance_feedback,
        order_flow_data=None,
        sentiment_data=None,
        portfolio_metrics=None,
        strategy_insights=None,
        onchain_data=None
    )

    print('=== DEEPSEEK DATA DEBUG ===')

    # Account state
    account = context.get('account_state', {})
    print(f'Balance: ${account.get("available_balance", 0):.2f}')
    print(f'Positions: {len(account.get("open_positions", []))}')

    # Technical analysis
    ta = context.get('technical_analysis', {})
    print(f'Symbols analyzed: {len(ta)}')

    # Check first symbol
    if ta:
        symbol = list(ta.keys())[0]
        data = ta[symbol]
        print(f'First symbol: {symbol}')
        print(f'Current price: ${data.get("current_price", 0):.2f}')

        tf_analysis = data.get('timeframe_analysis', {})
        if '1h' in tf_analysis:
            h1 = tf_analysis['1h']
            print(f'1h RSI: {h1.get("rsi", 0):.1f}')
            print(f'1h Trend: {h1.get("trend", "unknown")}')
            print(f'1h ATR: ${h1.get("atr", 0):.2f}')

            # Key levels
            kl = h1.get('key_levels', {})
            if kl:
                print(f'Support: ${kl.get("immediate_support", 0):.2f}')
                print(f'Resistance: ${kl.get("immediate_resistance", 0):.2f}')

    # Market regime
    market = context.get('market_overview', {})
    print(f'Market regime: {market.get("market_regime", "unknown")}')

    # Check what DeepSeek would decide
    print('\n=== TESTING DEEPSEEK DECISION ===')
    decision = await bot.deepseek.get_trading_decision(context)

    print(f'Decision: {decision.decision_type}')
    if decision.symbol:
        print(f'Symbol: {decision.symbol}')
        print(f'Side: {decision.side}')
        print(f'Stop Loss: ${decision.stop_loss}')
        print(f'Take Profit: ${decision.take_profit_1}')
        print(f'Strategy: {decision.strategy_tag}')

if __name__ == "__main__":
    asyncio.run(debug_deepseek_data())

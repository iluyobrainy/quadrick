#!/usr/bin/env python3
"""
Debug balance API call
"""
import sys
sys.path.insert(0, '.')

from main import QuadrickTradingBot

bot = QuadrickTradingBot()

try:
    response = bot.bybit.http.get_wallet_balance(accountType='UNIFIED')
    print('Raw API Response:')
    print(f'RetCode: {response.get("retCode")}')
    print(f'RetMsg: {response.get("retMsg")}')

    if response.get('retCode') == 0:
        print('Success - checking coin data...')
        coin_list = response.get('result', {}).get('list', [])
        if coin_list:
            coins = coin_list[0].get('coin', [])
            usdt_found = False
            for coin in coins:
                if coin.get('coin') == 'USDT':
                    print('USDT found:')
                    for key, value in coin.items():
                        print(f'  {key}: {value}')
                    usdt_found = True
                    break
            if not usdt_found:
                print('USDT not found in coins')
                print(f'Available coins: {[c.get("coin") for c in coins]}')
        else:
            print('No coin list')
    else:
        print(f'API Error: {response}')

except Exception as e:
    print(f'Exception: {e}')
    import traceback
    traceback.print_exc()

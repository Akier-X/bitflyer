#!/usr/bin/env python3
"""システムテスト - 取引が動作するか確認"""

import asyncio
import os
import sys

# .envファイルを読み込み
from pathlib import Path
env_path = Path('/home/user/bitflyer/.env')
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key] = value
                print(f"  Loaded: {key}={'*' * 8 if 'KEY' in key or 'SECRET' in key or 'TOKEN' in key else value}")

sys.path.insert(0, '/home/user/bitflyer')

from config.settings import get_config, reload_config
from src.api.bitflyer_client import BitFlyerClient, MockBitFlyerClient

async def test_api():
    print("\n" + "=" * 60)
    print("🧪 SYSTEM TEST - 世界最強AIトレーダー")
    print("=" * 60)

    # 設定再読み込み
    config = reload_config()

    print(f"\n📋 Configuration:")
    print(f"  Paper Trading: {config.trading.paper_trading}")
    print(f"  API Key Set: {bool(config.bitflyer.api_key)}")
    print(f"  API Secret Set: {bool(config.bitflyer.api_secret)}")

    # クライアント作成
    if config.trading.paper_trading:
        print("\n⚠️ PAPER TRADING MODE - Using MockClient")
        client = MockBitFlyerClient(initial_balance=5000, product_code="XRP_JPY")
    else:
        print("\n✅ LIVE TRADING MODE - Using Real API")
        client = BitFlyerClient(
            api_key=config.bitflyer.api_key,
            api_secret=config.bitflyer.api_secret,
            product_code="XRP_JPY"
        )

    # 残高テスト
    print("\n📊 Testing Balance API...")
    try:
        balances = await client.get_balance()
        if balances:
            print("  ✅ Balance API works!")
            for b in balances:
                currency = b.get('currency_code', 'N/A')
                amount = b.get('amount', 0)
                available = b.get('available', 0)
                if amount > 0 or currency == 'JPY':
                    print(f"    {currency}: {amount:,.4f} (available: {available:,.4f})")
        else:
            print("  ❌ No balance data")
    except Exception as e:
        print(f"  ❌ Balance failed: {e}")

    # 価格テスト
    print("\n📈 Testing Ticker API...")
    pairs = ["XRP_JPY", "XLM_JPY", "MONA_JPY"]
    for pair in pairs:
        try:
            if config.trading.paper_trading:
                test_client = MockBitFlyerClient(product_code=pair)
            else:
                test_client = BitFlyerClient(
                    api_key=config.bitflyer.api_key,
                    api_secret=config.bitflyer.api_secret,
                    product_code=pair
                )
            ticker = await test_client.get_ticker()
            if ticker:
                print(f"  ✅ {pair}: ¥{ticker.ltp:,.0f}")
            else:
                print(f"  ❌ {pair}: No data")
            await asyncio.sleep(0.3)
        except Exception as e:
            print(f"  ❌ {pair}: {e}")

    print("\n" + "=" * 60)
    print("✅ TEST COMPLETE")
    print("=" * 60)

    if not config.trading.paper_trading:
        print("\n🚀 Live trading is enabled. To start:")
        print("   cd /home/user/bitflyer")
        print("   python -m src.bot.aggressive_trader")
    else:
        print("\n⚠️ Paper trading mode. Set PAPER_TRADING=false for live trading.")

if __name__ == "__main__":
    asyncio.run(test_api())

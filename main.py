#!/usr/bin/env python3
"""
Ultimate AI Trading System - Main Entry Point
==============================================
世界最強AIトレーディングシステム

Usage:
    python main.py                  # Start trading bot
    python main.py --paper          # Paper trading mode
    python main.py --status         # Show status
    python main.py --health         # Health check endpoint
"""

import asyncio
import argparse
import sys
import os
from pathlib import Path

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Fix Windows encoding issues
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        # Python < 3.7
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from loguru import logger

# Configure logging
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",
)
logger.add(
    "logs/trading_{time:YYYY-MM-DD}.log",
    rotation="1 day",
    retention="7 days",
    level="DEBUG",
)


def print_banner(multi_asset: bool = False, aggressive: bool = False):
    """バナーを表示"""
    if aggressive:
        banner = """
+===============================================================+
|                                                               |
|     AGGRESSIVE SMALL ACCOUNT TRADER                           |
|     5000 yen -> 15000+ yen in 1 Month                         |
|                                                               |
|     Strategy:                                                 |
|     +-- XRP/JPY, MONA/JPY, XLM/JPY (Low Cost)                |
|     +-- High-Frequency Scalping (0.5s interval)               |
|     +-- Momentum + Trend Following                            |
|     +-- Auto Take-Profit (0.8%) / Stop-Loss (0.5%)           |
|     +-- Compound Gains for Maximum Growth                     |
|                                                               |
|     Target: 3x Return (200% profit)                           |
|                                                               |
+===============================================================+
"""
    elif multi_asset:
        banner = """
+===============================================================+
|                                                               |
|     ULTIMATE MULTI-ASSET AI TRADING SYSTEM                    |
|     World's Strongest Multi-Asset AI Trader                   |
|                                                               |
|     Supported Pairs:                                          |
|     +-- BTC/JPY, ETH/JPY, XRP/JPY, MONA/JPY, XLM/JPY         |
|     +-- FX_BTC/JPY (Lightning), ETH/BTC, BCH/BTC             |
|                                                               |
|     Features:                                                 |
|     +-- All Pairs Simultaneous Monitoring                     |
|     +-- Cross-Pair Arbitrage Detection                        |
|     +-- Dynamic Portfolio Allocation                          |
|     +-- High-Frequency Trading (0.5s interval)                |
|     +-- SOTA AI + Multi-Agent System                          |
|                                                               |
+===============================================================+
"""
    else:
        banner = """
+===============================================================+
|                                                               |
|     ULTIMATE AI TRADING SYSTEM                                |
|     World's Strongest AI Trading System                       |
|                                                               |
|     Components:                                               |
|     +-- PatchTST / Mamba / iTransformer                       |
|     +-- PPO / SAC / C51 / QR-DQN                              |
|     +-- DreamerV3 World Model                                 |
|     +-- Vector DB Pattern Matching                            |
|     +-- Multi-Agent System                                    |
|     +-- 500+ Dimension Feature Engineering                    |
|                                                               |
+===============================================================+
"""
    print(banner)


async def run_trading_bot(paper_trading: bool = True):
    """取引Botを実行"""
    from config.settings import get_config, reload_config
    from src.bot.live_trader import LiveTrader

    # Force paper trading if specified
    if paper_trading:
        os.environ["PAPER_TRADING"] = "true"
        reload_config()

    config = get_config()
    trader = LiveTrader(config)

    try:
        await trader.start()
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
        await trader.stop()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        await trader.stop()
        raise


def run_health_server(port: int = 8080):
    """ヘルスチェックサーバーを実行"""
    from aiohttp import web
    from config.settings import get_config

    config = get_config()

    async def health_handler(request):
        """ヘルスチェックエンドポイント"""
        return web.json_response({
            "status": "healthy",
            "service": "ultimate-ai-trader",
            "config": config.to_dict(),
        })

    async def status_handler(request):
        """ステータスエンドポイント"""
        return web.json_response({
            "status": "running",
            "paper_trading": config.trading.paper_trading,
            "product": config.bitflyer.product_code,
        })

    app = web.Application()
    app.router.add_get("/health", health_handler)
    app.router.add_get("/", health_handler)
    app.router.add_get("/status", status_handler)

    logger.info(f"Starting health server on port {port}")
    web.run_app(app, host="0.0.0.0", port=port)


async def run_aggressive_trader(paper_trading: bool = True, port: int = 8080, capital: float = 5000):
    """攻撃的スモールアカウントトレーダーを実行"""
    from aiohttp import web
    from config.settings import get_config, reload_config
    from src.bot.aggressive_trader import AggressiveTrader

    if paper_trading:
        os.environ["PAPER_TRADING"] = "true"
        reload_config()

    config = get_config()
    trader = AggressiveTrader(config, initial_capital=capital)

    # Health check handlers
    async def health_handler(request):
        return web.json_response({
            "status": "healthy",
            "mode": "aggressive",
            "capital": trader.current_capital,
            "pnl": trader.total_pnl,
        })

    async def status_handler(request):
        return web.json_response(trader.get_status())

    # Setup web app
    app = web.Application()
    app.router.add_get("/health", health_handler)
    app.router.add_get("/", health_handler)
    app.router.add_get("/status", status_handler)

    # Start web server in background
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", port)
    await site.start()
    logger.info(f"Health server started on port {port}")

    # Run trader
    try:
        await trader.start()
    except KeyboardInterrupt:
        await trader.stop()
    finally:
        await runner.cleanup()


async def run_multi_asset_trader(paper_trading: bool = True, port: int = 8080):
    """マルチアセットトレーダーを実行"""
    from aiohttp import web
    from config.settings import get_config, reload_config
    from src.bot.multi_asset_trader import MultiAssetTrader

    if paper_trading:
        os.environ["PAPER_TRADING"] = "true"
        reload_config()

    config = get_config()
    trader = MultiAssetTrader(config)

    # Health check handlers
    async def health_handler(request):
        return web.json_response({
            "status": "healthy",
            "mode": "multi-asset",
            "trading": trader.running,
            "pairs": trader.active_pairs,
        })

    async def status_handler(request):
        return web.json_response(trader.get_status())

    # Setup web app
    app = web.Application()
    app.router.add_get("/health", health_handler)
    app.router.add_get("/", health_handler)
    app.router.add_get("/status", status_handler)

    # Start web server in background
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", port)
    await site.start()
    logger.info(f"Health server started on port {port}")

    # Run trader
    try:
        await trader.start()
    except KeyboardInterrupt:
        await trader.stop()
    finally:
        await runner.cleanup()


async def run_with_health_check(paper_trading: bool = True, port: int = 8080):
    """Botとヘルスチェックを同時実行"""
    from aiohttp import web
    from config.settings import get_config, reload_config
    from src.bot.live_trader import LiveTrader

    if paper_trading:
        os.environ["PAPER_TRADING"] = "true"
        reload_config()

    config = get_config()
    trader = LiveTrader(config)

    # Health check handlers
    async def health_handler(request):
        return web.json_response({
            "status": "healthy",
            "trading": trader.running,
            "stats": trader.stats.to_dict() if trader.stats else {},
        })

    async def status_handler(request):
        return web.json_response(trader.get_status())

    # Setup web app
    app = web.Application()
    app.router.add_get("/health", health_handler)
    app.router.add_get("/", health_handler)
    app.router.add_get("/status", status_handler)

    # Start web server in background
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", port)
    await site.start()
    logger.info(f"Health server started on port {port}")

    # Run trader
    try:
        await trader.start()
    except KeyboardInterrupt:
        await trader.stop()
    finally:
        await runner.cleanup()


def show_status():
    """現在のステータスを表示"""
    from config.settings import get_config

    config = get_config()
    print("\n📊 Current Configuration:")
    print("-" * 40)

    import json
    print(json.dumps(config.to_dict(), indent=2))

    print("\n🔧 Environment Variables Required:")
    print("-" * 40)
    print("""
For Live Trading:
  BITFLYER_API_KEY       - bitFlyer API Key
  BITFLYER_API_SECRET    - bitFlyer API Secret

For LINE Notifications:
  LINE_NOTIFY_TOKEN      - LINE Notify Token
                          (Get from https://notify-bot.line.me/)

For Paper Trading (default):
  PAPER_TRADING=true     - Enable paper trading mode
""")


def generate_env_file():
    """環境変数テンプレートを生成"""
    from config.settings import generate_env_template

    generate_env_template(".env.template")
    print("✅ .env.template generated")
    print("Copy to .env and fill in your credentials:")
    print("  cp .env.template .env")


def main():
    """メインエントリーポイント"""
    parser = argparse.ArgumentParser(
        description="Ultimate AI Trading System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    Start trading (paper mode by default)
  python main.py --live             Start live trading
  python main.py --paper --port 8080  Paper trading with health endpoint
  python main.py --status           Show configuration status
  python main.py --gen-env          Generate .env template
        """,
    )

    parser.add_argument(
        "--paper",
        action="store_true",
        default=True,
        help="Paper trading mode (default)",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Live trading mode (requires API keys)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Health check server port (default: 8080)",
    )
    parser.add_argument(
        "--health-only",
        action="store_true",
        help="Run health server only (for testing)",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show current status and configuration",
    )
    parser.add_argument(
        "--gen-env",
        action="store_true",
        help="Generate .env.template file",
    )
    parser.add_argument(
        "--multi",
        action="store_true",
        help="Multi-asset trading mode (all pairs)",
    )
    parser.add_argument(
        "--aggressive",
        action="store_true",
        help="Aggressive small account mode (5000yen -> 15000yen)",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=5000,
        help="Initial capital for aggressive mode (default: 5000)",
    )
    parser.add_argument(
        "--pairs",
        type=str,
        default="",
        help="Comma-separated trading pairs (e.g., BTC_JPY,ETH_JPY)",
    )

    args = parser.parse_args()

    # Set trading pairs from command line
    if args.pairs:
        os.environ["TRADING_PAIRS"] = args.pairs

    # Create logs directory
    os.makedirs("logs", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    if args.gen_env:
        generate_env_file()
        return

    if args.status:
        show_status()
        return

    print_banner(multi_asset=args.multi, aggressive=args.aggressive)

    if args.health_only:
        run_health_server(args.port)
        return

    paper_trading = not args.live

    if paper_trading:
        logger.info("🧪 Starting in PAPER TRADING mode")
    else:
        logger.info("💰 Starting in LIVE TRADING mode")
        logger.warning("⚠️ Real money will be used!")

    if args.aggressive:
        logger.info(f"🔥 AGGRESSIVE MODE: ¥{args.capital:,.0f} -> ¥{args.capital * 3:,.0f}")
        logger.info("🎯 Target: 3x return in 1 month")
        try:
            asyncio.run(run_aggressive_trader(
                paper_trading=paper_trading,
                port=args.port,
                capital=args.capital,
            ))
        except KeyboardInterrupt:
            logger.info("Shutdown requested")
        except Exception as e:
            logger.error(f"Fatal error: {e}")
            sys.exit(1)
    elif args.multi:
        logger.info("🌐 MULTI-ASSET MODE: Trading all pairs")
        try:
            asyncio.run(run_multi_asset_trader(
                paper_trading=paper_trading,
                port=args.port,
            ))
        except KeyboardInterrupt:
            logger.info("Shutdown requested")
        except Exception as e:
            logger.error(f"Fatal error: {e}")
            sys.exit(1)
    else:
        try:
            asyncio.run(run_with_health_check(
                paper_trading=paper_trading,
                port=args.port,
            ))
        except KeyboardInterrupt:
            logger.info("Shutdown requested")
        except Exception as e:
            logger.error(f"Fatal error: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()

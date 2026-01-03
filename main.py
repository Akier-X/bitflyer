#!/usr/bin/env python3
"""
World's Strongest AI Trader
===========================
bitFlyer用 高頻度AIトレーディングシステム
月利30%以上を目指す世界最強トレーダー

Usage:
    python main.py [--dry-run] [--config CONFIG_PATH]

Environment Variables:
    BITFLYER_API_KEY: bitFlyer APIキー
    BITFLYER_API_SECRET: bitFlyer APIシークレット
"""

import os
import sys
import argparse
import signal
import time
from datetime import datetime
from loguru import logger

# ログ設定
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
    level="INFO",
)
logger.add(
    "logs/trader_{time:YYYY-MM-DD}.log",
    rotation="1 day",
    retention="30 days",
    level="DEBUG",
)


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description="World's Strongest AI Trader for bitFlyer"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run mode (no actual trades)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/settings.yaml",
        help="Config file path",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.getenv("BITFLYER_API_KEY", ""),
        help="bitFlyer API key",
    )
    parser.add_argument(
        "--api-secret",
        type=str,
        default=os.getenv("BITFLYER_API_SECRET", ""),
        help="bitFlyer API secret",
    )

    args = parser.parse_args()

    # APIキーチェック
    if not args.api_key or not args.api_secret:
        logger.error("API credentials not provided!")
        logger.info("Set BITFLYER_API_KEY and BITFLYER_API_SECRET environment variables")
        logger.info("Or use --api-key and --api-secret arguments")
        sys.exit(1)

    # トレーダー初期化
    from src.bot.trader import AITrader

    trader = AITrader(
        api_key=args.api_key,
        api_secret=args.api_secret,
        config_path=args.config,
    )

    # シグナルハンドラ設定
    def signal_handler(sig, frame):
        logger.info("Shutdown signal received...")
        trader.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # スタートバナー
    print_banner()

    if args.dry_run:
        logger.info("🔍 DRY RUN MODE - No actual trades will be executed")

    # トレーディング開始
    logger.info("🚀 Starting World's Strongest AI Trader...")
    trader.start()

    # メインループ
    try:
        while True:
            # ステータス表示（1分ごと）
            status = trader.get_status()
            logger.info(
                f"Status: Trades={status['trade_count']} "
                f"PnL={status['daily_pnl']:+,.0f} JPY "
                f"Positions={status['positions']['total_positions']}"
            )
            time.sleep(60)

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received...")
        trader.stop()


def print_banner():
    """バナー表示"""
    banner = """
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║   🏆 WORLD'S STRONGEST AI TRADER 🏆                                  ║
║                                                                      ║
║   ██╗    ██╗ ██████╗ ██████╗ ██╗     ██████╗ ███████╗                ║
║   ██║    ██║██╔═══██╗██╔══██╗██║     ██╔══██╗██╔════╝                ║
║   ██║ █╗ ██║██║   ██║██████╔╝██║     ██║  ██║███████╗                ║
║   ██║███╗██║██║   ██║██╔══██╗██║     ██║  ██║╚════██║                ║
║   ╚███╔███╔╝╚██████╔╝██║  ██║███████╗██████╔╝███████║                ║
║    ╚══╝╚══╝  ╚═════╝ ╚═╝  ╚═╝╚══════╝╚═════╝ ╚══════╝                ║
║                                                                      ║
║   🎯 TARGET: 30%+ MONTHLY RETURNS                                    ║
║   ⚡ HIGH-FREQUENCY TRADING SYSTEM                                   ║
║   🤖 6 AI-POWERED STRATEGIES                                         ║
║   🛡️  ADVANCED RISK MANAGEMENT                                       ║
║                                                                      ║
║   Powered by: Machine Learning, Ensemble Strategies, Smart Execution║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
    """
    print(banner)


if __name__ == "__main__":
    main()

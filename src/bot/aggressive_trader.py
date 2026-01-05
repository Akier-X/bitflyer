#!/usr/bin/env python3
"""
================================================================================
    ULTIMATE AI TRADER v6.0 - 完全版
================================================================================
    すべての問題を解決した永久実行可能な最強トレーディングシステム

    解決済み問題:
    - 売却時の小数点精度問題（安全な整数単位で売却）
    - API残高同期問題（毎回APIから取得）
    - 連続エラー問題（クールダウン機能）
    - API制限問題（適切な待機時間）
================================================================================
"""

import asyncio
import math
import sys
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque

import numpy as np
from loguru import logger

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
env_path = PROJECT_ROOT / '.env'
if env_path.exists():
    load_dotenv(env_path)

from config.settings import get_config
from src.api.bitflyer_client import BitFlyerClient, OrderSide, OrderType


# =============================================================================
# 設定
# =============================================================================

TRADING_FEE = 0.0015        # 0.15%
TAKE_PROFIT = 0.015         # 1.5% で利確（手数料0.3%考慮後 実質1.2%）
STOP_LOSS = 0.01            # 1.0% で損切り
TRADE_COOLDOWN = 10         # 取引後10秒は再取引しない
FAIL_COOLDOWN = 120         # 失敗後120秒は再試行しない
MIN_CASH_RESERVE = 100      # 最低100円は残す

# API制限対策
BALANCE_REFRESH = 60        # 残高は60秒ごとに更新
PRICE_DELAY = 1.5           # 価格取得間隔: 1.5秒
LOOP_DELAY = 5              # メインループ: 5秒

# テクニカル設定
RSI_PERIOD = 14
RSI_BUY = 30
RSI_SELL = 70
PRICE_HISTORY = 100


# =============================================================================
# ペア設定（売却時は整数部分のみ使用して安全に）
# =============================================================================

@dataclass
class PairConfig:
    code: str
    min_size: float         # 最小注文サイズ
    buy_decimals: int       # 買い注文の小数点桁数
    sell_decimals: int      # 売り注文の小数点桁数（より少なく）


# 売却時は整数または少ない小数点で安全に
PAIRS = {
    "XLM_JPY": PairConfig("XLM_JPY", 1.0, 6, 0),    # 売却: 整数のみ
    "XRP_JPY": PairConfig("XRP_JPY", 1.0, 6, 0),    # 売却: 整数のみ
    "MONA_JPY": PairConfig("MONA_JPY", 1.0, 6, 0),  # 売却: 整数のみ
    "ETH_JPY": PairConfig("ETH_JPY", 0.01, 7, 2),   # 売却: 0.01単位
    "BTC_JPY": PairConfig("BTC_JPY", 0.001, 8, 3),  # 売却: 0.001単位
}


# =============================================================================
# 価格トラッカー
# =============================================================================

@dataclass
class PriceTracker:
    prices: deque = field(default_factory=lambda: deque(maxlen=PRICE_HISTORY))
    entry_price: float = 0.0
    entry_time: Optional[datetime] = None

    def add(self, price: float):
        self.prices.append(price)

    def set_entry(self, price: float):
        self.entry_price = price
        self.entry_time = datetime.now()

    def clear(self):
        self.entry_price = 0.0
        self.entry_time = None

    def pnl(self, current: float) -> float:
        if self.entry_price > 0:
            return (current - self.entry_price) / self.entry_price
        return 0.0

    def rsi(self) -> Optional[float]:
        if len(self.prices) < RSI_PERIOD + 1:
            return None
        prices = list(self.prices)[-RSI_PERIOD-1:]
        deltas = np.diff(prices)
        gains = np.mean(np.clip(deltas, 0, None))
        losses = np.mean(np.clip(-deltas, 0, None))
        if losses < 1e-10:
            return 100.0
        return 100 - (100 / (1 + gains / losses))

    def momentum(self, periods: int = 5) -> Optional[float]:
        if len(self.prices) < periods + 1:
            return None
        return (self.prices[-1] - self.prices[-periods-1]) / self.prices[-periods-1]


# =============================================================================
# トレーダー
# =============================================================================

class Trader:
    def __init__(self):
        self.config = get_config()
        self.clients: Dict[str, BitFlyerClient] = {}
        self.trackers: Dict[str, PriceTracker] = {}
        self.last_trade: Dict[str, datetime] = {}
        self.failed: Dict[str, datetime] = {}

        # 残高キャッシュ
        self._balances: Dict[str, float] = {}
        self._balance_time: Optional[datetime] = None

        # 統計
        self.trades = 0
        self.wins = 0
        self.pnl = 0.0
        self.start_value = 0.0
        self.start_time: Optional[datetime] = None

    # -------------------------------------------------------------------------
    # API
    # -------------------------------------------------------------------------

    async def get_balances(self, force: bool = False) -> Dict[str, float]:
        """残高取得（キャッシュ付き）"""
        if not force and self._balance_time:
            if (datetime.now() - self._balance_time).seconds < BALANCE_REFRESH:
                return self._balances

        try:
            client = next(iter(self.clients.values()))
            data = await asyncio.wait_for(client.get_balance(), timeout=15)
            if data:
                self._balances = {
                    b["currency_code"]: float(b.get("available", 0))
                    for b in data if float(b.get("available", 0)) > 0
                }
                self._balance_time = datetime.now()
        except Exception as e:
            logger.debug(f"残高取得エラー: {e}")
        return self._balances

    async def get_price(self, pair: str) -> Optional[float]:
        """価格取得"""
        client = self.clients.get(pair)
        if not client:
            return None
        try:
            ticker = await asyncio.wait_for(client.get_ticker(), timeout=10)
            return ticker.ltp if ticker and ticker.ltp > 0 else None
        except:
            return None

    def can_trade(self, pair: str) -> bool:
        """取引可能チェック"""
        now = datetime.now()

        # 最近の取引チェック
        if pair in self.last_trade:
            if (now - self.last_trade[pair]).seconds < TRADE_COOLDOWN:
                return False

        # 失敗クールダウンチェック
        if pair in self.failed:
            if (now - self.failed[pair]).seconds < FAIL_COOLDOWN:
                return False
            del self.failed[pair]

        return True

    # -------------------------------------------------------------------------
    # 売買
    # -------------------------------------------------------------------------

    async def buy(self, pair: str, reason: str) -> bool:
        """購入"""
        cfg = PAIRS.get(pair)
        client = self.clients.get(pair)
        if not cfg or not client or not self.can_trade(pair):
            return False

        # 残高確認
        balances = await self.get_balances(force=True)
        jpy = balances.get("JPY", 0) - MIN_CASH_RESERVE
        if jpy < 100:
            return False

        price = await self.get_price(pair)
        if not price:
            return False

        # サイズ計算
        budget = jpy * 0.4  # 40%を使用
        size = budget / price
        size = round(size, cfg.buy_decimals)
        if size < cfg.min_size:
            size = cfg.min_size

        cost = size * price * (1 + TRADING_FEE)
        if cost > jpy:
            return False

        logger.info(f"  📤 BUY: {pair} {size} @ ¥{price:,.0f}")

        try:
            order_id = await asyncio.wait_for(
                client.send_order(OrderSide.BUY, size, OrderType.MARKET),
                timeout=20
            )
            if order_id:
                self.last_trade[pair] = datetime.now()
                self.trades += 1
                self.trackers[pair].set_entry(price)
                self._balances = {}  # キャッシュクリア
                logger.info(f"  ✅ BUY成功: {pair} {size} @ ¥{price:,.0f} [{reason}]")
                return True
            else:
                self.failed[pair] = datetime.now()
                logger.warning(f"  ⚠️ BUY失敗: {pair} - {FAIL_COOLDOWN}秒待機")
        except Exception as e:
            self.failed[pair] = datetime.now()
            logger.error(f"  ❌ BUY失敗: {pair} - {e}")
        return False

    async def sell(self, pair: str, reason: str) -> bool:
        """売却"""
        cfg = PAIRS.get(pair)
        client = self.clients.get(pair)
        tracker = self.trackers.get(pair)
        if not cfg or not client or not tracker or not self.can_trade(pair):
            return False

        # 未決済注文キャンセル
        try:
            await client.cancel_all_orders(pair)
            await asyncio.sleep(0.5)
        except:
            pass

        # 残高確認
        balances = await self.get_balances(force=True)
        currency = pair.replace("_JPY", "")
        holding = balances.get(currency, 0)

        # 売却サイズ（整数部分または安全な桁数で切り捨て）
        multiplier = 10 ** cfg.sell_decimals
        size = math.floor(holding * multiplier) / multiplier

        if size < cfg.min_size:
            logger.debug(f"  {pair}: 売却可能量不足 ({holding:.8f} → {size})")
            return False

        price = await self.get_price(pair)
        if not price:
            return False

        logger.info(f"  📤 SELL: {pair} {size} (保有: {holding:.6f}) @ ¥{price:,.0f}")

        try:
            order_id = await asyncio.wait_for(
                client.send_order(OrderSide.SELL, size, OrderType.MARKET),
                timeout=20
            )
            if order_id:
                self.last_trade[pair] = datetime.now()
                self.trades += 1

                # 損益計算
                entry = tracker.entry_price if tracker.entry_price > 0 else price
                gross = (price - entry) * size
                fee = price * size * TRADING_FEE * 2
                net = gross - fee

                self.pnl += net
                if net > 0:
                    self.wins += 1

                tracker.clear()
                self._balances = {}

                emoji = "💰" if net >= 0 else "📉"
                pct = (price - entry) / entry * 100 if entry > 0 else 0
                logger.info(f"  {emoji} SELL成功: {pair} {size} → ¥{net:+,.0f} ({pct:+.1f}%) [{reason}]")
                return True
            else:
                self.failed[pair] = datetime.now()
                logger.warning(f"  ⚠️ SELL失敗: {pair} - {FAIL_COOLDOWN}秒待機")
        except Exception as e:
            self.failed[pair] = datetime.now()
            logger.error(f"  ❌ SELL失敗: {pair} - {e}")
        return False

    # -------------------------------------------------------------------------
    # シグナル
    # -------------------------------------------------------------------------

    def should_buy(self, pair: str) -> Tuple[bool, str]:
        """買いシグナル"""
        tracker = self.trackers.get(pair)
        if not tracker or len(tracker.prices) < 20:
            return False, ""

        rsi = tracker.rsi()
        mom = tracker.momentum()

        if rsi is not None and rsi < RSI_BUY:
            if mom is not None and mom > 0:
                return True, f"RSI={rsi:.0f},MOM↑"
        return False, ""

    def should_sell(self, pair: str, price: float) -> Tuple[bool, str]:
        """売りシグナル"""
        tracker = self.trackers.get(pair)
        if not tracker:
            return False, ""

        pnl = tracker.pnl(price)

        # 利確
        if pnl >= TAKE_PROFIT:
            return True, f"利確+{pnl*100:.1f}%"

        # 損切り
        if pnl <= -STOP_LOSS:
            return True, f"損切{pnl*100:.1f}%"

        # テクニカル売り
        rsi = tracker.rsi()
        if rsi is not None and rsi > RSI_SELL and pnl > 0:
            return True, f"RSI={rsi:.0f}"

        return False, ""

    # -------------------------------------------------------------------------
    # 初期化
    # -------------------------------------------------------------------------

    async def init(self) -> bool:
        """初期化"""
        api_key = self.config.bitflyer.api_key
        api_secret = self.config.bitflyer.api_secret

        if not api_key or not api_secret:
            logger.error("APIキーが設定されていません")
            return False

        logger.info("=" * 60)
        logger.info("  🚀 ULTIMATE AI TRADER v6.0 - 完全版")
        logger.info("=" * 60)
        logger.info(f"  利確: {TAKE_PROFIT*100:.1f}% | 損切: {STOP_LOSS*100:.1f}%")
        logger.info("=" * 60)

        # クライアント初期化
        for pair, cfg in PAIRS.items():
            try:
                client = BitFlyerClient(api_key, api_secret, pair)
                ticker = await asyncio.wait_for(client.get_ticker(), timeout=10)
                if ticker and ticker.ltp > 0:
                    self.clients[pair] = client
                    self.trackers[pair] = PriceTracker()
                    self.trackers[pair].add(ticker.ltp)
                    min_jpy = cfg.min_size * ticker.ltp
                    logger.info(f"  ✓ {pair}: ¥{ticker.ltp:,.0f} (最小¥{min_jpy:,.0f})")
            except Exception as e:
                logger.debug(f"  ✗ {pair}: {e}")

        if not self.clients:
            logger.error("接続可能なペアがありません")
            return False

        # 未決済注文をキャンセル
        logger.info("-" * 60)
        for pair, client in self.clients.items():
            try:
                await client.cancel_all_orders(pair)
            except:
                pass
        await asyncio.sleep(1)

        # 残高表示
        balances = await self.get_balances(force=True)
        jpy = balances.get("JPY", 0)
        logger.info(f"  💴 現金: ¥{jpy:,.0f}")

        total = jpy
        for pair in self.clients:
            currency = pair.replace("_JPY", "")
            amount = balances.get(currency, 0)
            if amount > 0:
                price = await self.get_price(pair) or 0
                value = amount * price
                total += value
                cfg = PAIRS.get(pair)

                # 売却可能量を計算
                multiplier = 10 ** cfg.sell_decimals
                sellable = math.floor(amount * multiplier) / multiplier
                can_sell = sellable >= cfg.min_size

                status = f"売却可({sellable})" if can_sell else "少額"
                logger.info(f"  💎 {currency}: {amount:.6f} (¥{value:,.0f}) [{status}]")

                if can_sell:
                    self.trackers[pair].set_entry(price)

        self.start_value = total
        logger.info("-" * 60)
        logger.info(f"  📊 総資産: ¥{total:,.0f}")
        logger.info("=" * 60)
        return True

    # -------------------------------------------------------------------------
    # メインループ
    # -------------------------------------------------------------------------

    async def run(self):
        """メインループ"""
        if not await self.init():
            return

        self.start_time = datetime.now()
        logger.info("")
        logger.info("  🎯 トレード開始!")
        logger.info("  Ctrl+C で停止")
        logger.info("")

        tick = 0
        while True:
            try:
                tick += 1

                # 残高取得
                balances = await self.get_balances()
                jpy = balances.get("JPY", 0)

                # 各ペア処理
                for pair, cfg in PAIRS.items():
                    if pair not in self.clients:
                        continue

                    tracker = self.trackers[pair]
                    currency = pair.replace("_JPY", "")
                    holding = balances.get(currency, 0)

                    # 価格取得
                    price = await self.get_price(pair)
                    if not price:
                        await asyncio.sleep(PRICE_DELAY)
                        continue
                    tracker.add(price)

                    # 売却可能量を計算
                    multiplier = 10 ** cfg.sell_decimals
                    sellable = math.floor(holding * multiplier) / multiplier
                    has_position = sellable >= cfg.min_size

                    # === ポジションあり: 売りを検討 ===
                    if has_position:
                        should, reason = self.should_sell(pair, price)
                        if should:
                            await self.sell(pair, reason)

                    # === ポジションなし: 買いを検討 ===
                    else:
                        if tracker.entry_price > 0:
                            tracker.clear()

                        min_cost = cfg.min_size * price * 1.01
                        if jpy - MIN_CASH_RESERVE < min_cost:
                            continue

                        should, reason = self.should_buy(pair)
                        if should:
                            await self.buy(pair, reason)

                    await asyncio.sleep(PRICE_DELAY)

                # ステータス表示（5分ごと）
                if tick % 60 == 0:
                    self.log_status()

                await asyncio.sleep(LOOP_DELAY)

            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"エラー: {e}")
                await asyncio.sleep(10)

        logger.info("")
        logger.info("  🛑 停止")
        self.log_status()

    def log_status(self):
        """ステータス表示"""
        elapsed = datetime.now() - self.start_time if self.start_time else timedelta(0)
        win_rate = (self.wins / self.trades * 100) if self.trades > 0 else 0

        logger.info("")
        logger.info("=" * 50)
        logger.info(f"  ⏱️ 経過: {elapsed.seconds // 3600}h {(elapsed.seconds % 3600) // 60}m")
        logger.info(f"  📊 取引: {self.trades}回 | 勝率: {win_rate:.0f}%")
        logger.info(f"  💰 損益: ¥{self.pnl:+,.0f}")
        logger.info("=" * 50)


# =============================================================================
# メイン
# =============================================================================

async def main():
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>",
        level="INFO",
    )

    trader = Trader()
    await trader.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass

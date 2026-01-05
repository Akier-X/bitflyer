#!/usr/bin/env python3
"""
================================================================================
    ULTIMATE AI TRADER v7.0 - 積極取引版
================================================================================
    リアルタイム損益表示 + 積極的な取引ロジック
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
# 設定（より積極的に）
# =============================================================================

TRADING_FEE = 0.0015
TAKE_PROFIT = 0.008          # 0.8%で利確（手数料後0.5%利益）
STOP_LOSS = 0.005            # 0.5%で損切り
TRADE_COOLDOWN = 5           # 5秒クールダウン
FAIL_COOLDOWN = 30           # 失敗後30秒待機
MIN_CASH_RESERVE = 50

# API設定
BALANCE_REFRESH = 30
PRICE_DELAY = 0.8
LOOP_DELAY = 3
STATUS_INTERVAL = 30         # 30秒ごとにステータス表示

# テクニカル（緩い条件）
RSI_PERIOD = 7
RSI_BUY = 45                 # 45以下で買い（緩く）
RSI_SELL = 55                # 55以上で売り（緩く）


# =============================================================================
# ペア設定
# =============================================================================

@dataclass
class PairConfig:
    code: str
    min_size: float
    buy_decimals: int
    sell_decimals: int


PAIRS = {
    "XLM_JPY": PairConfig("XLM_JPY", 1.0, 6, 0),
    "XRP_JPY": PairConfig("XRP_JPY", 1.0, 6, 0),
    "MONA_JPY": PairConfig("MONA_JPY", 1.0, 6, 0),
    "ETH_JPY": PairConfig("ETH_JPY", 0.01, 7, 2),
    "BTC_JPY": PairConfig("BTC_JPY", 0.001, 8, 3),
}


# =============================================================================
# 価格トラッカー
# =============================================================================

@dataclass
class PriceTracker:
    prices: deque = field(default_factory=lambda: deque(maxlen=50))
    entry_price: float = 0.0
    entry_time: Optional[datetime] = None
    highest: float = 0.0
    lowest: float = float('inf')

    def add(self, price: float):
        self.prices.append(price)
        if self.entry_price > 0:
            self.highest = max(self.highest, price)
            self.lowest = min(self.lowest, price)

    def set_entry(self, price: float):
        self.entry_price = price
        self.entry_time = datetime.now()
        self.highest = price
        self.lowest = price

    def clear(self):
        self.entry_price = 0.0
        self.entry_time = None
        self.highest = 0.0
        self.lowest = float('inf')

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

    def trend(self) -> Optional[float]:
        """短期トレンド（%）"""
        if len(self.prices) < 3:
            return None
        return (self.prices[-1] - self.prices[-3]) / self.prices[-3] * 100


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

        self._balances: Dict[str, float] = {}
        self._balance_time: Optional[datetime] = None

        self.trades = 0
        self.wins = 0
        self.pnl = 0.0
        self.start_value = 0.0
        self.start_time: Optional[datetime] = None

    async def get_balances(self, force: bool = False) -> Dict[str, float]:
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
        except:
            pass
        return self._balances

    async def get_price(self, pair: str) -> Optional[float]:
        client = self.clients.get(pair)
        if not client:
            return None
        try:
            ticker = await asyncio.wait_for(client.get_ticker(), timeout=10)
            return ticker.ltp if ticker and ticker.ltp > 0 else None
        except:
            return None

    def can_trade(self, pair: str) -> bool:
        now = datetime.now()
        if pair in self.last_trade:
            if (now - self.last_trade[pair]).seconds < TRADE_COOLDOWN:
                return False
        if pair in self.failed:
            if (now - self.failed[pair]).seconds < FAIL_COOLDOWN:
                return False
            del self.failed[pair]
        return True

    async def buy(self, pair: str, reason: str) -> bool:
        cfg = PAIRS.get(pair)
        client = self.clients.get(pair)
        if not cfg or not client or not self.can_trade(pair):
            return False

        balances = await self.get_balances(force=True)
        jpy = balances.get("JPY", 0) - MIN_CASH_RESERVE
        if jpy < 50:
            return False

        price = await self.get_price(pair)
        if not price:
            return False

        budget = jpy * 0.5
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
                self._balances = {}
                logger.info(f"  ✅ BUY成功: {pair} {size} @ ¥{price:,.0f} [{reason}]")
                return True
            else:
                self.failed[pair] = datetime.now()
        except Exception as e:
            self.failed[pair] = datetime.now()
            logger.error(f"  ❌ BUY失敗: {e}")
        return False

    async def sell(self, pair: str, reason: str) -> bool:
        cfg = PAIRS.get(pair)
        client = self.clients.get(pair)
        tracker = self.trackers.get(pair)
        if not cfg or not client or not tracker or not self.can_trade(pair):
            return False

        try:
            await client.cancel_all_orders(pair)
            await asyncio.sleep(0.3)
        except:
            pass

        balances = await self.get_balances(force=True)
        currency = pair.replace("_JPY", "")
        holding = balances.get(currency, 0)

        multiplier = 10 ** cfg.sell_decimals
        size = math.floor(holding * multiplier) / multiplier

        if size < cfg.min_size:
            return False

        price = await self.get_price(pair)
        if not price:
            return False

        logger.info(f"  📤 SELL: {pair} {size} @ ¥{price:,.0f}")

        try:
            order_id = await asyncio.wait_for(
                client.send_order(OrderSide.SELL, size, OrderType.MARKET),
                timeout=20
            )
            if order_id:
                self.last_trade[pair] = datetime.now()
                self.trades += 1

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
                logger.info(f"  {emoji} SELL成功: {size} → ¥{net:+,.0f} ({pct:+.2f}%) [{reason}]")
                return True
            else:
                self.failed[pair] = datetime.now()
        except Exception as e:
            self.failed[pair] = datetime.now()
            logger.error(f"  ❌ SELL失敗: {e}")
        return False

    def should_sell(self, pair: str, price: float) -> Tuple[bool, str]:
        tracker = self.trackers.get(pair)
        if not tracker or tracker.entry_price <= 0:
            return False, ""

        pnl = tracker.pnl(price)

        # 利確
        if pnl >= TAKE_PROFIT:
            return True, f"利確+{pnl*100:.2f}%"

        # 損切り
        if pnl <= -STOP_LOSS:
            return True, f"損切{pnl*100:.2f}%"

        # 高値から0.3%下落で利確（トレーリング）
        if tracker.highest > 0 and pnl > 0.003:
            drop = (tracker.highest - price) / tracker.highest
            if drop > 0.003:
                return True, f"トレール+{pnl*100:.2f}%"

        # RSI売りシグナル + 利益あり
        rsi = tracker.rsi()
        if rsi and rsi > RSI_SELL and pnl > 0.002:
            return True, f"RSI={rsi:.0f}"

        return False, ""

    def should_buy(self, pair: str) -> Tuple[bool, str]:
        tracker = self.trackers.get(pair)
        if not tracker or len(tracker.prices) < 10:
            return False, ""

        rsi = tracker.rsi()
        trend = tracker.trend()

        # RSI低め + 上昇トレンド
        if rsi and rsi < RSI_BUY and trend and trend > 0:
            return True, f"RSI={rsi:.0f},↑{trend:.2f}%"

        # 強い上昇トレンド
        if trend and trend > 0.3:
            return True, f"急騰+{trend:.2f}%"

        return False, ""

    async def init(self) -> bool:
        api_key = self.config.bitflyer.api_key
        api_secret = self.config.bitflyer.api_secret

        if not api_key or not api_secret:
            logger.error("APIキーが設定されていません")
            return False

        logger.info("=" * 60)
        logger.info("  🚀 ULTIMATE AI TRADER v7.0 - 積極取引版")
        logger.info("=" * 60)
        logger.info(f"  利確: {TAKE_PROFIT*100:.1f}% | 損切: {STOP_LOSS*100:.1f}%")
        logger.info(f"  RSI: 買い<{RSI_BUY} 売り>{RSI_SELL}")
        logger.info("=" * 60)

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
            except:
                pass

        if not self.clients:
            logger.error("接続可能なペアがありません")
            return False

        logger.info("-" * 60)
        for pair, client in self.clients.items():
            try:
                await client.cancel_all_orders(pair)
            except:
                pass
        await asyncio.sleep(0.5)

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
                multiplier = 10 ** cfg.sell_decimals
                sellable = math.floor(amount * multiplier) / multiplier
                can_sell = sellable >= cfg.min_size

                if can_sell:
                    self.trackers[pair].set_entry(price)
                    logger.info(f"  💎 {currency}: {amount:.4f} (¥{value:,.0f}) [売却可]")
                else:
                    logger.info(f"  💎 {currency}: {amount:.6f} (¥{value:,.0f}) [少額]")

        self.start_value = total
        logger.info("-" * 60)
        logger.info(f"  📊 総資産: ¥{total:,.0f}")
        logger.info("=" * 60)
        return True

    def show_realtime_status(self):
        """リアルタイムステータス表示"""
        lines = []
        total_unrealized = 0.0

        for pair in self.clients:
            tracker = self.trackers.get(pair)
            if not tracker or len(tracker.prices) == 0:
                continue

            price = tracker.prices[-1]
            currency = pair.replace("_JPY", "")
            holding = self._balances.get(currency, 0)
            cfg = PAIRS.get(pair)
            multiplier = 10 ** cfg.sell_decimals
            sellable = math.floor(holding * multiplier) / multiplier

            if sellable >= cfg.min_size and tracker.entry_price > 0:
                pnl_pct = tracker.pnl(price) * 100
                pnl_jpy = (price - tracker.entry_price) * sellable
                total_unrealized += pnl_jpy
                rsi = tracker.rsi()
                rsi_str = f"RSI:{rsi:.0f}" if rsi else "RSI:--"

                if pnl_pct >= 0:
                    lines.append(f"  📈 {currency}: ¥{price:,.0f} ({pnl_pct:+.2f}% ¥{pnl_jpy:+,.0f}) {rsi_str}")
                else:
                    lines.append(f"  📉 {currency}: ¥{price:,.0f} ({pnl_pct:+.2f}% ¥{pnl_jpy:+,.0f}) {rsi_str}")

        if lines:
            logger.info("-" * 50)
            for line in lines:
                logger.info(line)
            logger.info(f"  💰 含み損益: ¥{total_unrealized:+,.0f} | 確定損益: ¥{self.pnl:+,.0f}")
            logger.info(f"  📊 取引: {self.trades}回 | 勝率: {(self.wins/self.trades*100) if self.trades > 0 else 0:.0f}%")

    async def run(self):
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

                balances = await self.get_balances()
                jpy = balances.get("JPY", 0)

                for pair, cfg in PAIRS.items():
                    if pair not in self.clients:
                        continue

                    tracker = self.trackers[pair]
                    currency = pair.replace("_JPY", "")
                    holding = balances.get(currency, 0)

                    price = await self.get_price(pair)
                    if not price:
                        await asyncio.sleep(PRICE_DELAY)
                        continue
                    tracker.add(price)

                    multiplier = 10 ** cfg.sell_decimals
                    sellable = math.floor(holding * multiplier) / multiplier
                    has_position = sellable >= cfg.min_size

                    if has_position:
                        should, reason = self.should_sell(pair, price)
                        if should:
                            await self.sell(pair, reason)
                    else:
                        if tracker.entry_price > 0:
                            tracker.clear()

                        min_cost = cfg.min_size * price * 1.01
                        if jpy - MIN_CASH_RESERVE >= min_cost:
                            should, reason = self.should_buy(pair)
                            if should:
                                await self.buy(pair, reason)

                    await asyncio.sleep(PRICE_DELAY)

                # リアルタイムステータス表示
                if tick % STATUS_INTERVAL == 0:
                    self.show_realtime_status()

                await asyncio.sleep(LOOP_DELAY)

            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"エラー: {e}")
                await asyncio.sleep(5)

        logger.info("")
        logger.info("  🛑 停止")
        self.show_realtime_status()


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

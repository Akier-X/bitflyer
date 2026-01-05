#!/usr/bin/env python3
"""
================================================================================
    🏆 ULTIMATE AI TRADER v9.2 - 高利益アグレッシブ版 (BULLETPROOF)
================================================================================
    - 積極的な利確（0.6%）・損切り（0.4%）
    - 急騰時即利確機能
    - 反発検知・モメンタム売買
    - RSI安定性維持（期間14）
    - 🛡️ BULLETPROOF ORDER SYSTEM (Insufficient funds 完全防止)
================================================================================
"""

import asyncio
import math
import sys
import os
from datetime import datetime, timedelta
from typing import Dict, Optional, List
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
# 設定 - 高利益アグレッシブ版
# =============================================================================

TRADING_FEE = 0.0015
TAKE_PROFIT = 0.006          # 0.6%で利確（手数料0.3%差引後0.3%利益）
STOP_LOSS = 0.004            # 0.4%で損切り（素早く撤退）
MIN_HOLD_TIME = 45           # 最低45秒（急騰対応可能）
TRADE_COOLDOWN = 30          # 30秒クールダウン
FAIL_COOLDOWN = 60           # 失敗後1分待機

# API設定
PRICE_DELAY = 1.5
LOOP_DELAY = 3
STATUS_INTERVAL = 20

# テクニカル
RSI_PERIOD = 14              # 標準RSI期間（安定性維持）
RSI_BUY = 42                 # RSI42以下で買い（機会増加）
RSI_SELL = 58                # RSI58以上で売り検討

# =============================================================================
# 🛡️ BULLETPROOF ORDER SYSTEM - 絶対安全設定
# =============================================================================

SAFETY_MARGIN_BUY = 0.05     # 購入時5%の余裕
SAFETY_MARGIN_SELL = 0.02    # 売却時2%の余裕
JPY_RESERVE = 100            # 常に¥100を残す
MAX_BUDGET_RATIO = 0.90      # 最大投資比率90%
MAX_ORDER_RETRIES = 3        # 最大リトライ回数
RETRY_SIZE_REDUCTION = 0.90  # リトライ時のサイズ縮小率
BALANCE_CACHE_TTL = 5        # 残高キャッシュ有効期限（5秒）


# =============================================================================
# ペア設定
# =============================================================================

@dataclass
class PairConfig:
    code: str
    min_size: float
    decimals: int
    priority: int


PAIRS = {
    "MONA_JPY": PairConfig("MONA_JPY", 1.0, 0, 1),
    "XLM_JPY": PairConfig("XLM_JPY", 1.0, 0, 2),
    "XRP_JPY": PairConfig("XRP_JPY", 1.0, 0, 3),
    "ETH_JPY": PairConfig("ETH_JPY", 0.01, 2, 4),
    "BTC_JPY": PairConfig("BTC_JPY", 0.001, 3, 5),
}


# =============================================================================
# ポジション管理
# =============================================================================

@dataclass
class Position:
    pair: str
    size: float
    entry_price: float
    entry_time: datetime
    highest: float = 0.0

    def pnl_pct(self, current: float) -> float:
        return (current - self.entry_price) / self.entry_price

    def hold_seconds(self) -> int:
        return int((datetime.now() - self.entry_time).total_seconds())


# =============================================================================
# 価格分析
# =============================================================================

class PriceAnalyzer:
    def __init__(self):
        self.prices: deque = deque(maxlen=200)

    def add(self, price: float):
        self.prices.append(price)

    def rsi(self) -> Optional[float]:
        if len(self.prices) < RSI_PERIOD + 1:
            return None
        prices = list(self.prices)[-RSI_PERIOD-1:]
        deltas = np.diff(prices)
        gains = deltas.copy()
        losses = deltas.copy()
        gains[gains < 0] = 0
        losses[losses > 0] = 0
        losses = abs(losses)

        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)

        if avg_loss < 1e-10:
            return 100.0
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def trend_pct(self) -> Optional[float]:
        if len(self.prices) < 10:
            return None
        return (self.prices[-1] - self.prices[-10]) / self.prices[-10] * 100

    def current(self) -> Optional[float]:
        return self.prices[-1] if self.prices else None


# =============================================================================
# トレーダー
# =============================================================================

class Trader:
    def __init__(self):
        self.config = get_config()
        self.clients: Dict[str, BitFlyerClient] = {}
        self.analyzers: Dict[str, PriceAnalyzer] = {}
        self.positions: Dict[str, Position] = {}
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
        """🛡️ BULLETPROOF残高取得（5秒キャッシュ）"""
        if not force and self._balance_time:
            if (datetime.now() - self._balance_time).seconds < BALANCE_CACHE_TTL:
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

    async def get_fresh_balance(self, currency: str = "JPY") -> float:
        """🛡️ 絶対最新の残高を取得（キャッシュ無視）"""
        try:
            client = next(iter(self.clients.values()))
            data = await asyncio.wait_for(client.get_balance(), timeout=10)
            if data:
                for b in data:
                    if b.get("currency_code") == currency:
                        return float(b.get("available", 0))
        except:
            pass
        return 0.0

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

    async def execute_buy(self, pair: str, reason: str) -> bool:
        """🛡️ BULLETPROOF購入 - Insufficient fundsエラー完全防止"""
        cfg = PAIRS.get(pair)
        client = self.clients.get(pair)
        if not cfg or not client or not self.can_trade(pair):
            return False

        currency = pair.replace("_JPY", "")

        # === STEP 1: 初回残高チェック ===
        balances = await self.get_balances(force=True)
        jpy = balances.get("JPY", 0)
        if jpy < JPY_RESERVE + 50:
            return False

        price = await self.get_price(pair)
        if not price:
            return False

        # === STEP 2: 安全サイズ計算（5%マージン込み） ===
        available_jpy = jpy - JPY_RESERVE
        budget = available_jpy * 0.5 * (1 - SAFETY_MARGIN_BUY)  # 50%の投資で5%マージン
        size = budget / price
        size = math.floor(size * (10 ** cfg.decimals)) / (10 ** cfg.decimals)

        if size < cfg.min_size:
            min_cost = cfg.min_size * price * (1 + SAFETY_MARGIN_BUY)
            if min_cost <= available_jpy:
                size = cfg.min_size
            else:
                return False

        # === STEP 3: リトライループ（最大3回） ===
        for attempt in range(MAX_ORDER_RETRIES):
            fresh_jpy = await self.get_fresh_balance("JPY")
            final_cost = size * price * (1 + SAFETY_MARGIN_BUY)
            available = fresh_jpy - JPY_RESERVE

            if final_cost > available:
                if attempt < MAX_ORDER_RETRIES - 1:
                    size = size * RETRY_SIZE_REDUCTION
                    size = math.floor(size * (10 ** cfg.decimals)) / (10 ** cfg.decimals)
                    if size < cfg.min_size:
                        return False
                    continue
                else:
                    return False

            if attempt == 0:
                logger.info(f"")
                logger.info(f"  🛒 【購入】{currency}")
                logger.info(f"     数量: {size} @ ¥{price:,.0f}")
                logger.info(f"     安全コスト: ¥{final_cost:,.0f} (5%マージン込み)")
                logger.info(f"     理由: {reason}")

            try:
                order_id = await asyncio.wait_for(
                    client.send_order(OrderSide.BUY, size, OrderType.MARKET),
                    timeout=20
                )
                if order_id:
                    self.last_trade[pair] = datetime.now()
                    self.trades += 1
                    self.positions[pair] = Position(
                        pair=pair, size=size, entry_price=price,
                        entry_time=datetime.now(), highest=price
                    )
                    self._balances = {}
                    logger.info(f"  ✅ 購入成功！ 投資額: ¥{size * price:,.0f}")
                    return True
                else:
                    if attempt < MAX_ORDER_RETRIES - 1:
                        size = size * RETRY_SIZE_REDUCTION
                        size = math.floor(size * (10 ** cfg.decimals)) / (10 ** cfg.decimals)
                        if size < cfg.min_size:
                            self.failed[pair] = datetime.now()
                            return False
                        logger.info(f"  🔄 {currency}: リトライ ({attempt+1}/{MAX_ORDER_RETRIES})")
                        await asyncio.sleep(1)
                    else:
                        self.failed[pair] = datetime.now()
                        return False
            except Exception as e:
                if attempt < MAX_ORDER_RETRIES - 1:
                    size = size * RETRY_SIZE_REDUCTION
                    size = math.floor(size * (10 ** cfg.decimals)) / (10 ** cfg.decimals)
                    if size >= cfg.min_size:
                        await asyncio.sleep(1)
                        continue
                self.failed[pair] = datetime.now()
                return False

        return False

    async def execute_sell(self, pair: str, reason: str) -> bool:
        """🛡️ BULLETPROOF売却 - Insufficient fundsエラー完全防止"""
        cfg = PAIRS.get(pair)
        client = self.clients.get(pair)
        pos = self.positions.get(pair)
        if not cfg or not client or not self.can_trade(pair):
            return False

        currency = pair.replace("_JPY", "")

        # キャンセル（静かに）
        try:
            await client.cancel_all_orders(pair)
            await asyncio.sleep(0.3)
        except:
            pass

        # === STEP 1: 最新残高を取得 ===
        holding = await self.get_fresh_balance(currency)
        if holding <= 0:
            if pair in self.positions:
                del self.positions[pair]
            return False

        # === STEP 2: 安全サイズ計算（2%マージン込み） ===
        safe_holding = holding * (1 - SAFETY_MARGIN_SELL)
        multiplier = 10 ** cfg.decimals
        size = math.floor(safe_holding * multiplier) / multiplier

        if size < cfg.min_size * 1.01:
            if pair in self.positions:
                del self.positions[pair]
            return False

        price = await self.get_price(pair)
        if not price:
            return False

        logger.info(f"")
        logger.info(f"  💰 【売却】{currency}")
        logger.info(f"     保有量: {holding:.8f}")
        logger.info(f"     売却サイズ: {size} (2%マージン適用)")
        logger.info(f"     理由: {reason}")

        # === STEP 3: リトライループ（最大3回） ===
        for attempt in range(MAX_ORDER_RETRIES):
            fresh_holding = await self.get_fresh_balance(currency)

            if size > fresh_holding:
                safe_holding = fresh_holding * (1 - SAFETY_MARGIN_SELL)
                size = math.floor(safe_holding * multiplier) / multiplier
                if size < cfg.min_size:
                    if attempt < MAX_ORDER_RETRIES - 1:
                        await asyncio.sleep(1)
                        continue
                    else:
                        if pair in self.positions:
                            del self.positions[pair]
                        return False

            try:
                order_id = await asyncio.wait_for(
                    client.send_order(OrderSide.SELL, size, OrderType.MARKET),
                    timeout=20
                )
                if order_id:
                    self.last_trade[pair] = datetime.now()
                    self.trades += 1

                    entry = pos.entry_price if pos else price
                    gross = (price - entry) * size
                    fee = price * size * TRADING_FEE * 2
                    net = gross - fee
                    pct = (price - entry) / entry * 100 if entry > 0 else 0

                    self.pnl += net
                    if net > 0:
                        self.wins += 1
                        emoji = "💰"
                    else:
                        emoji = "📉"

                    if pair in self.positions:
                        del self.positions[pair]
                    self._balances = {}

                    logger.info(f"  {emoji} 売却完了！")
                    logger.info(f"     損益: ¥{net:+,.0f} ({pct:+.2f}%)")
                    logger.info(f"     累計: ¥{self.pnl:+,.0f}")
                    return True
                else:
                    if attempt < MAX_ORDER_RETRIES - 1:
                        size = size * RETRY_SIZE_REDUCTION
                        size = math.floor(size * multiplier) / multiplier
                        if size < cfg.min_size:
                            self.failed[pair] = datetime.now()
                            return False
                        logger.info(f"  🔄 {currency}: リトライ ({attempt+1}/{MAX_ORDER_RETRIES})")
                        await asyncio.sleep(1)
                    else:
                        self.failed[pair] = datetime.now()
                        return False
            except:
                if attempt < MAX_ORDER_RETRIES - 1:
                    size = size * RETRY_SIZE_REDUCTION
                    size = math.floor(size * multiplier) / multiplier
                    if size >= cfg.min_size:
                        await asyncio.sleep(1)
                        continue
                self.failed[pair] = datetime.now()
                return False

        return False

    def should_sell(self, pair: str, price: float) -> tuple:
        pos = self.positions.get(pair)
        analyzer = self.analyzers.get(pair)
        if not pos or not analyzer:
            return False, ""

        pnl = pos.pnl_pct(price)
        hold_time = pos.hold_seconds()

        # 高値更新
        if price > pos.highest:
            pos.highest = price

        # 緊急損切り（いつでも）
        if pnl <= -0.015:  # -1.5%
            return True, f"緊急損切り {pnl*100:.2f}%"

        # 最低保有時間チェック（ただし大きな利益は即確定）
        if hold_time < MIN_HOLD_TIME:
            # 急騰時は即利確
            if pnl >= 0.008:  # +0.8%以上
                return True, f"急騰利確 +{pnl*100:.2f}%"
            return False, ""

        # 利確
        if pnl >= TAKE_PROFIT:
            return True, f"利確 +{pnl*100:.2f}%"

        # 損切り
        if pnl <= -STOP_LOSS:
            return True, f"損切り {pnl*100:.2f}%"

        # トレーリング（高値から0.3%下落、かつ利益あり）
        if pos.highest > 0 and pnl > 0.003:
            drop = (pos.highest - price) / pos.highest
            if drop > 0.003:
                return True, f"トレール +{pnl*100:.2f}%"

        # RSI売りシグナル + 少しでも利益あり
        rsi = analyzer.rsi()
        if rsi and rsi > RSI_SELL and pnl > 0.002 and hold_time > 60:
            return True, f"RSI={rsi:.0f} +{pnl*100:.2f}%"

        return False, ""

    def should_buy(self, pair: str) -> tuple:
        analyzer = self.analyzers.get(pair)
        if not analyzer or len(analyzer.prices) < 15:
            return False, ""

        rsi = analyzer.rsi()
        trend = analyzer.trend_pct()

        # RSI低め（買いチャンス）
        if rsi and rsi < RSI_BUY:
            if trend is None or trend > -0.3:  # 急落中でなければOK
                return True, f"RSI={rsi:.0f} 買い時"

        # 上昇トレンド（モメンタム）
        if trend and trend > 0.3:
            if rsi is None or rsi < 65:  # 過熱してなければOK
                return True, f"上昇 +{trend:.2f}%"

        # 反発シグナル（下落後の上昇開始）
        if len(analyzer.prices) >= 5:
            recent = list(analyzer.prices)[-5:]
            if recent[-1] > recent[-2] > recent[-3] and recent[-3] < recent[-4]:
                return True, f"反発検知"

        return False, ""

    async def init(self) -> bool:
        api_key = self.config.bitflyer.api_key
        api_secret = self.config.bitflyer.api_secret

        if not api_key or not api_secret:
            logger.error("❌ APIキー未設定")
            return False

        logger.info("")
        logger.info("╔══════════════════════════════════════════════════════════╗")
        logger.info("║  🏆 ULTIMATE AI TRADER v9.1 - 高利益アグレッシブ版      ║")
        logger.info("╠══════════════════════════════════════════════════════════╣")
        logger.info(f"║  利確: +{TAKE_PROFIT*100:.1f}% | 損切: -{STOP_LOSS*100:.1f}% | 急騰: 即利確       ║")
        logger.info("╚══════════════════════════════════════════════════════════╝")

        for pair, cfg in sorted(PAIRS.items(), key=lambda x: x[1].priority):
            try:
                client = BitFlyerClient(api_key, api_secret, pair)
                ticker = await asyncio.wait_for(client.get_ticker(), timeout=10)
                if ticker and ticker.ltp > 0:
                    self.clients[pair] = client
                    self.analyzers[pair] = PriceAnalyzer()
                    self.analyzers[pair].add(ticker.ltp)
                    min_jpy = cfg.min_size * ticker.ltp
                    logger.info(f"  ✓ {pair}: ¥{ticker.ltp:,.0f} (最小: ¥{min_jpy:,.0f})")
            except:
                pass

        if not self.clients:
            return False

        # キャンセル
        for pair in self.clients:
            try:
                await self.clients[pair].cancel_all_orders(pair)
            except:
                pass
        await asyncio.sleep(0.5)

        logger.info("")
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
                multiplier = 10 ** cfg.decimals
                sellable = math.floor(amount * 0.95 * multiplier) / multiplier

                if sellable >= cfg.min_size:
                    self.positions[pair] = Position(
                        pair=pair, size=sellable, entry_price=price,
                        entry_time=datetime.now(), highest=price
                    )
                    logger.info(f"  💎 {currency}: {sellable} (¥{value:,.0f})")

        self.start_value = total
        logger.info(f"  📊 総資産: ¥{total:,.0f}")
        logger.info("")
        return True

    def show_status(self):
        logger.info("")
        logger.info("  ══════════════════════════════════════════════════")
        logger.info("  📊 【戦況】")

        unrealized = 0.0
        for pair, pos in self.positions.items():
            analyzer = self.analyzers.get(pair)
            if not analyzer:
                continue
            price = analyzer.current()
            if not price:
                continue

            currency = pair.replace("_JPY", "")
            pnl_pct = pos.pnl_pct(price) * 100
            pnl_jpy = (price - pos.entry_price) * pos.size
            unrealized += pnl_jpy
            hold = pos.hold_seconds()
            rsi = analyzer.rsi()

            emoji = "📈" if pnl_pct >= 0 else "📉"
            rsi_str = f"RSI:{rsi:.0f}" if rsi else ""
            logger.info(f"  {emoji} {currency}: {pnl_pct:+.2f}% (¥{pnl_jpy:+,.0f}) {hold}秒 {rsi_str}")

        logger.info("  ──────────────────────────────────────────────────")

        win_rate = (self.wins / self.trades * 100) if self.trades > 0 else 0
        logger.info(f"  💰 含み: ¥{unrealized:+,.0f} | 確定: ¥{self.pnl:+,.0f}")
        logger.info(f"  📊 取引: {self.trades}回 (勝率: {win_rate:.0f}%)")

        jpy = self._balances.get("JPY", 0)
        if jpy >= 5000:
            logger.info(f"  🎉 ETH取引可能！")
        logger.info("  ══════════════════════════════════════════════════")

    async def run(self):
        if not await self.init():
            return

        self.start_time = datetime.now()
        logger.info("  🚀 開始！ Ctrl+C で停止")
        logger.info("")

        tick = 0
        while True:
            try:
                tick += 1

                balances = await self.get_balances()
                jpy = balances.get("JPY", 0)

                for pair, cfg in sorted(PAIRS.items(), key=lambda x: x[1].priority):
                    if pair not in self.clients:
                        continue

                    analyzer = self.analyzers[pair]
                    currency = pair.replace("_JPY", "")

                    price = await self.get_price(pair)
                    if not price:
                        await asyncio.sleep(PRICE_DELAY)
                        continue
                    analyzer.add(price)

                    # ポジション更新
                    if pair in self.positions:
                        pos = self.positions[pair]
                        if price > pos.highest:
                            pos.highest = price

                    has_position = pair in self.positions

                    if has_position:
                        should, reason = self.should_sell(pair, price)
                        if should:
                            await self.execute_sell(pair, reason)
                    else:
                        min_cost = cfg.min_size * price * 1.01
                        if jpy - 50 >= min_cost:
                            should, reason = self.should_buy(pair)
                            if should:
                                await self.execute_buy(pair, reason)

                    await asyncio.sleep(PRICE_DELAY)

                if tick % STATUS_INTERVAL == 0:
                    self.show_status()

                await asyncio.sleep(LOOP_DELAY)

            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"  エラー: {e}")
                await asyncio.sleep(10)

        logger.info("")
        logger.info("  🛑 停止")
        self.show_status()


async def main():
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | {message}",
        level="INFO",
        filter=lambda r: "Cancel all failed" not in r["message"]
                     and "Request error: 200" not in r["message"]
    )

    import logging
    logging.getLogger("src.api.bitflyer_client").setLevel(logging.CRITICAL)

    trader = Trader()
    await trader.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass

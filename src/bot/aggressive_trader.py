#!/usr/bin/env python3
"""
================================================================================
    ULTIMATE AI TRADER v4.3 - 現在保有最適化版
================================================================================
    現在の保有コインを活用して利益を最大化

    実際の保有状況 (総資産: ¥6,000):
    - 現金: ¥920
    - XLM: 84.89個 (¥3,144) → メイン利確対象
    - XRP: 5.57個 (¥1,888) → 利確対象
    - ETH: 微量 (売却不可)

    戦略:
    - XLM/XRPポジションの利確を狙う
    - DOGE/MONA/FLRで小額トレード
    - 資金が増えたらETH/BTCへ
================================================================================
"""

import asyncio
import sys
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from collections import deque
import traceback

import numpy as np
from loguru import logger

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
env_path = PROJECT_ROOT / '.env'
if env_path.exists():
    load_dotenv(env_path)

from config.settings import Config, get_config
from src.api.bitflyer_client import BitFlyerClient, OrderSide, OrderType


# =============================================================================
# 設定
# =============================================================================

TRADING_FEE_RATE = 0.0015   # 0.15%
TAKE_PROFIT_PCT = 1.0       # 1.0%で利確
STOP_LOSS_PCT = 0.8         # 0.8%で損切り
MIN_TRADE_INTERVAL = 3      # 3秒

# テクニカル指標
RSI_PERIOD = 7
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
EMA_SHORT = 3
EMA_LONG = 10


# =============================================================================
# ペア設定
# =============================================================================

@dataclass
class PairConfig:
    code: str
    min_size: float
    decimals: int
    priority: int


# 保有コイン優先 (Lightning対応ペアのみ)
PAIRS = {
    "XLM_JPY": PairConfig("XLM_JPY", 1.0, 6, 1),      # 保有中: 84.89 XLM ¥3,126
    "XRP_JPY": PairConfig("XRP_JPY", 1.0, 6, 2),      # 保有中: 5.57 XRP ¥1,872
    "MONA_JPY": PairConfig("MONA_JPY", 1.0, 6, 3),    # ¥14/個 - 低価格
    "ETH_JPY": PairConfig("ETH_JPY", 0.01, 7, 4),
    "BTC_JPY": PairConfig("BTC_JPY", 0.001, 8, 5),
}


# =============================================================================
# テクニカル分析
# =============================================================================

class Analyzer:
    def __init__(self):
        self.prices: Dict[str, deque] = {}

    def add(self, pair: str, price: float):
        if pair not in self.prices:
            self.prices[pair] = deque(maxlen=100)
        self.prices[pair].append(price)

    def get_signal(self, pair: str) -> Tuple[int, float, str]:
        """0=SELL, 1=HOLD, 2=BUY"""
        prices = list(self.prices.get(pair, []))
        if len(prices) < 15:
            return 1, 0.0, "waiting"

        # RSI
        deltas = np.diff(prices[-RSI_PERIOD-1:])
        gains = np.mean(np.where(deltas > 0, deltas, 0))
        losses = np.mean(np.where(deltas < 0, -deltas, 0))
        rsi = 100 - (100 / (1 + gains / max(losses, 0.0001)))

        # EMA
        def ema(data, period):
            m = 2 / (period + 1)
            e = data[0]
            for p in data[1:]:
                e = p * m + e * (1 - m)
            return e

        ema_s = ema(prices[-EMA_SHORT:], EMA_SHORT)
        ema_l = ema(prices[-EMA_LONG:], EMA_LONG)

        # モメンタム
        mom = (prices[-1] - prices[-5]) / prices[-5] if len(prices) >= 5 else 0

        # シグナル判定
        signals = []
        reasons = []

        if rsi < RSI_OVERSOLD:
            signals.append(2)
            reasons.append(f"RSI={rsi:.0f}")
        elif rsi > RSI_OVERBOUGHT:
            signals.append(0)
            reasons.append(f"RSI={rsi:.0f}")

        if ema_s > ema_l * 1.002:
            signals.append(2)
            reasons.append("EMA↑")
        elif ema_s < ema_l * 0.998:
            signals.append(0)
            reasons.append("EMA↓")

        if mom > 0.003:
            signals.append(2)
            reasons.append(f"MOM+{mom*100:.1f}%")
        elif mom < -0.003:
            signals.append(0)
            reasons.append(f"MOM{mom*100:.1f}%")

        if not signals:
            return 1, 0.0, "neutral"

        buy = signals.count(2)
        sell = signals.count(0)

        if buy > sell:
            return 2, buy / len(signals), ",".join(reasons)
        elif sell > buy:
            return 0, sell / len(signals), ",".join(reasons)
        return 1, 0.0, "mixed"


# =============================================================================
# ポジション
# =============================================================================

@dataclass
class Position:
    pair: str
    size: float = 0.0
    entry_price: float = 0.0
    current_price: float = 0.0

    @property
    def value(self) -> float:
        return self.size * self.current_price

    @property
    def pnl_pct(self) -> float:
        if self.entry_price > 0:
            return (self.current_price / self.entry_price - 1) * 100
        return 0.0


# =============================================================================
# トレーダー
# =============================================================================

class Trader:
    def __init__(self):
        self.config = get_config()
        self.running = False
        self.start_time = None

        self.cash = 0.0
        self.initial_value = 0.0
        self.clients: Dict[str, BitFlyerClient] = {}
        self.positions: Dict[str, Position] = {}
        self.analyzer = Analyzer()

        self.trades = 0
        self.wins = 0
        self.pnl = 0.0
        self.last_trade: Dict[str, datetime] = {}

    async def init(self) -> bool:
        """初期化"""
        api_key = self.config.bitflyer.api_key
        api_secret = self.config.bitflyer.api_secret

        if not api_key or not api_secret:
            logger.error("APIキーが設定されていません")
            return False

        logger.info("=" * 60)
        logger.info("  ULTIMATE AI TRADER v4.3 - 現在保有最適化版")
        logger.info("=" * 60)
        logger.info(f"  利確: {TAKE_PROFIT_PCT}% | 損切り: {STOP_LOSS_PCT}%")
        logger.info("=" * 60)

        # クライアント初期化
        for pair, cfg in PAIRS.items():
            try:
                client = BitFlyerClient(
                    api_key=api_key,
                    api_secret=api_secret,
                    product_code=pair,
                )
                ticker = await asyncio.wait_for(client.get_ticker(), timeout=10)
                if ticker and ticker.ltp > 0:
                    self.clients[pair] = client
                    self.positions[pair] = Position(pair=pair, current_price=ticker.ltp)
                    min_cost = cfg.min_size * ticker.ltp
                    logger.info(f"  {pair}: ¥{ticker.ltp:,.0f} (最小: ¥{min_cost:,.0f})")
            except Exception as e:
                logger.warning(f"  {pair}: 接続失敗")

        if not self.clients:
            logger.error("接続できるペアがありません")
            return False

        # 残高取得
        balances = await self._get_balance()
        self.cash = balances.get("JPY", 0)
        logger.info(f"  現金: ¥{self.cash:,.0f}")

        # 保有コイン
        for currency, amount in balances.items():
            if currency == "JPY":
                continue
            pair = f"{currency}_JPY"
            if pair in self.positions and amount > 0:
                pos = self.positions[pair]
                pos.size = amount
                pos.entry_price = pos.current_price  # 現在価格をエントリー価格とする
                cfg = PAIRS.get(pair)
                can_sell = cfg and amount >= cfg.min_size
                status = "売却可" if can_sell else "売却不可"
                logger.info(f"  {pair}: {amount} (¥{pos.value:,.0f}) [{status}]")

        # 初期ポートフォリオ
        self.initial_value = self._portfolio()
        logger.info(f"  総資産: ¥{self.initial_value:,.0f}")

        return True

    async def _get_balance(self) -> Dict[str, float]:
        """残高取得"""
        result = {}
        try:
            client = list(self.clients.values())[0]
            balances = await asyncio.wait_for(client.get_balance(), timeout=10)
            if balances:
                for b in balances:
                    currency = b.get("currency_code", "")
                    amount = float(b.get("available", 0))
                    if amount > 0:
                        result[currency] = amount
        except Exception as e:
            logger.error(f"残高取得エラー: {e}")
        return result

    async def _get_price(self, pair: str) -> Optional[float]:
        """価格取得"""
        client = self.clients.get(pair)
        if not client:
            return None
        try:
            ticker = await asyncio.wait_for(client.get_ticker(), timeout=10)
            if ticker and ticker.ltp > 0:
                self.analyzer.add(pair, ticker.ltp)
                self.positions[pair].current_price = ticker.ltp
                return ticker.ltp
        except:
            pass
        return None

    def _portfolio(self) -> float:
        """ポートフォリオ価値"""
        total = self.cash
        for pos in self.positions.values():
            total += pos.value
        return total

    def _can_trade(self, pair: str) -> bool:
        """取引可能か"""
        last = self.last_trade.get(pair)
        if last and (datetime.now() - last).seconds < MIN_TRADE_INTERVAL:
            return False
        return True

    async def _buy(self, pair: str, reason: str) -> bool:
        """購入"""
        if not self._can_trade(pair):
            return False

        cfg = PAIRS.get(pair)
        pos = self.positions.get(pair)
        client = self.clients.get(pair)
        if not cfg or not pos or not client:
            return False

        price = pos.current_price
        min_cost = cfg.min_size * price

        # 資金チェック
        if self.cash < min_cost * 1.01:  # 手数料込み
            return False

        # サイズ計算 (資金の40%)
        budget = min(self.cash * 0.4, self.cash - 100)  # 100円は残す
        if budget < min_cost:
            budget = min_cost

        size = budget / price
        size = max(cfg.min_size, size)
        size = round(size, cfg.decimals)

        cost = size * price
        if cost > self.cash:
            return False

        try:
            order_id = await asyncio.wait_for(
                client.send_order(side=OrderSide.BUY, size=size, order_type=OrderType.MARKET),
                timeout=15
            )
            if order_id:
                self.last_trade[pair] = datetime.now()
                self.trades += 1
                self.cash -= cost * (1 + TRADING_FEE_RATE)
                pos.size += size
                pos.entry_price = price
                logger.info(f"  ✓ BUY {pair}: {size} @ ¥{price:,.0f} [{reason}]")
                return True
        except Exception as e:
            logger.error(f"購入エラー: {e}")
        return False

    async def _sell(self, pair: str, reason: str) -> bool:
        """売却"""
        if not self._can_trade(pair):
            return False

        cfg = PAIRS.get(pair)
        pos = self.positions.get(pair)
        client = self.clients.get(pair)
        if not cfg or not pos or not client:
            return False

        if pos.size < cfg.min_size:
            return False

        size = round(pos.size, cfg.decimals)
        if size < cfg.min_size:
            return False

        price = pos.current_price
        proceeds = size * price * (1 - TRADING_FEE_RATE)

        try:
            order_id = await asyncio.wait_for(
                client.send_order(side=OrderSide.SELL, size=size, order_type=OrderType.MARKET),
                timeout=15
            )
            if order_id:
                self.last_trade[pair] = datetime.now()
                self.trades += 1

                profit = proceeds - (pos.size * pos.entry_price)
                self.pnl += profit
                if profit > 0:
                    self.wins += 1

                self.cash += proceeds
                pos.size = 0
                pos.entry_price = 0

                emoji = "💰" if profit >= 0 else "📉"
                logger.info(f"  {emoji} SELL {pair}: ¥{profit:+,.0f} ({pos.pnl_pct:+.1f}%) [{reason}]")
                return True
        except Exception as e:
            logger.error(f"売却エラー: {e}")
        return False

    def _log_status(self):
        """ステータス表示"""
        portfolio = self._portfolio()
        roi = (portfolio / self.initial_value - 1) * 100 if self.initial_value > 0 else 0
        win_rate = (self.wins / self.trades * 100) if self.trades > 0 else 0

        logger.info("")
        logger.info("=" * 50)
        logger.info(f"  💼 Portfolio: ¥{portfolio:,.0f} ({roi:+.2f}%)")
        logger.info(f"  💴 Cash: ¥{self.cash:,.0f}")
        logger.info(f"  📊 Trades: {self.trades} | Win: {win_rate:.0f}%")
        logger.info("-" * 50)
        for pair, pos in self.positions.items():
            if pos.size > 0:
                logger.info(f"  {pair}: {pos.size:.4f} @ ¥{pos.current_price:,.0f} ({pos.pnl_pct:+.2f}%)")
        logger.info("=" * 50)

    async def run(self):
        """メインループ"""
        self.running = True
        self.start_time = datetime.now()

        if not await self.init():
            return

        logger.info("")
        logger.info("  🚀 Trading started!")
        logger.info("")

        tick = 0
        while self.running:
            try:
                tick += 1

                # 価格更新
                for pair in self.clients:
                    await self._get_price(pair)
                    await asyncio.sleep(0.3)

                # 各ペアの取引判断
                for pair in self.clients:
                    pos = self.positions.get(pair)
                    cfg = PAIRS.get(pair)
                    if not pos or not cfg:
                        continue

                    has_position = pos.size >= cfg.min_size

                    # === 保有中の場合 ===
                    if has_position:
                        # 利確
                        if pos.pnl_pct >= TAKE_PROFIT_PCT:
                            logger.info(f"  📈 利確シグナル: {pair} ({pos.pnl_pct:+.2f}%)")
                            await self._sell(pair, "TAKE_PROFIT")
                            continue

                        # 損切り
                        if pos.pnl_pct <= -STOP_LOSS_PCT:
                            logger.info(f"  📉 損切りシグナル: {pair} ({pos.pnl_pct:+.2f}%)")
                            await self._sell(pair, "STOP_LOSS")
                            continue

                        # テクニカル売りシグナル
                        action, conf, reason = self.analyzer.get_signal(pair)
                        if action == 0 and conf >= 0.5:
                            await self._sell(pair, reason)

                    # === 保有なしの場合 ===
                    else:
                        min_cost = cfg.min_size * pos.current_price
                        if self.cash < min_cost * 1.01:
                            continue  # 資金不足

                        action, conf, reason = self.analyzer.get_signal(pair)
                        if action == 2 and conf >= 0.5:
                            await self._buy(pair, reason)

                # ステータス表示 (60秒ごと)
                if tick % 60 == 0:
                    self._log_status()

                await asyncio.sleep(1)

            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"エラー: {e}")
                await asyncio.sleep(5)

        logger.info("")
        logger.info("  🛑 Stopped")
        self._log_status()


async def main():
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>",
        level="INFO",
    )

    trader = Trader()
    try:
        await trader.run()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    asyncio.run(main())

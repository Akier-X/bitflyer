#!/usr/bin/env python3
"""
================================================================================
    ULTIMATE AI TRADER v4.2 - 世界最強・短期高利益システム
================================================================================
    短期間で最大利益を狙う超積極的トレードシステム

    設定:
    - 利確: 1.5% (大きな利益を狙う)
    - 損切り: 0.8% (損失を限定)
    - 取引間隔: 2秒 (高頻度)
    - シグナル閾値: 30% (積極的)

    テクニカル分析 (短期最適化):
    - RSI(7): 25/75 (極端な水準のみ)
    - EMA(3/10): 超短期クロス
    - ボリンジャー(10, 1.5σ): 狭いバンド
================================================================================
"""

import asyncio
import sys
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from collections import deque
import traceback

import numpy as np
from loguru import logger

# パス設定 (Windows/Linux両対応)
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 環境変数読み込み
from dotenv import load_dotenv
env_path = PROJECT_ROOT / '.env'
if env_path.exists():
    load_dotenv(env_path)

from config.settings import Config, get_config
from src.api.bitflyer_client import BitFlyerClient, OrderSide, OrderType


# =============================================================================
# 定数
# =============================================================================

TRADING_FEE_RATE = 0.0015   # 0.15%
TAKE_PROFIT_RATE = 0.015    # 1.5% (より大きな利益を狙う)
STOP_LOSS_RATE = 0.008      # 0.8% (損失を抑える)
MIN_TRADE_INTERVAL = 2      # 2秒 (より高速)

# テクニカル指標 (短期トレード最適化)
RSI_PERIOD = 7              # 短期RSI
RSI_OVERSOLD = 25           # より極端な売られすぎ
RSI_OVERBOUGHT = 75         # より極端な買われすぎ
EMA_SHORT = 3               # 超短期EMA
EMA_LONG = 10               # 短期EMA
BOLLINGER_PERIOD = 10       # 短期ボリンジャー
BOLLINGER_STD = 1.5         # 狭いバンド（より多くのシグナル）

# API設定
API_TIMEOUT = 15.0
MAX_RETRIES = 3
RETRY_DELAY = 2.0


# =============================================================================
# 取引ペア設定
# =============================================================================

@dataclass
class TradingPair:
    code: str
    min_size: float
    size_decimals: int
    priority: int
    current_price: float = 0.0


# 残高¥920で取引可能なペアを優先
TRADING_PAIRS = {
    "XRP_JPY": TradingPair("XRP_JPY", 1.0, 6, 1),      # ¥337 × 1 = ¥337 ✓
    "MONA_JPY": TradingPair("MONA_JPY", 1.0, 6, 2),    # ¥14 × 1 = ¥14 ✓
    "ETH_JPY": TradingPair("ETH_JPY", 0.01, 7, 3),     # ¥500,890 × 0.01 = ¥5,008 ✗
    "BTC_JPY": TradingPair("BTC_JPY", 0.001, 8, 4),    # ¥14,561,322 × 0.001 = ¥14,561 ✗
}


# =============================================================================
# テクニカル分析
# =============================================================================

class TechnicalAnalyzer:
    def __init__(self, max_history: int = 200):
        self.prices: Dict[str, deque] = {}
        self.max_history = max_history

    def add_price(self, pair: str, price: float) -> None:
        if pair not in self.prices:
            self.prices[pair] = deque(maxlen=self.max_history)
        self.prices[pair].append(price)

    def get_prices(self, pair: str) -> List[float]:
        return list(self.prices.get(pair, []))

    def calculate_rsi(self, pair: str) -> Optional[float]:
        prices = self.get_prices(pair)
        if len(prices) < RSI_PERIOD + 1:
            return None
        deltas = np.diff(prices[-RSI_PERIOD-1:])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def calculate_ema(self, pair: str, period: int) -> Optional[float]:
        prices = self.get_prices(pair)
        if len(prices) < period:
            return None
        multiplier = 2 / (period + 1)
        ema = prices[-period]
        for price in prices[-period+1:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
        return ema

    def calculate_bollinger(self, pair: str) -> Optional[Tuple[float, float, float]]:
        prices = self.get_prices(pair)
        if len(prices) < BOLLINGER_PERIOD:
            return None
        arr = np.array(prices[-BOLLINGER_PERIOD:])
        middle = np.mean(arr)
        std = np.std(arr)
        return middle - std * BOLLINGER_STD, middle, middle + std * BOLLINGER_STD

    def calculate_momentum(self, pair: str, period: int = 10) -> Optional[float]:
        prices = self.get_prices(pair)
        if len(prices) < period + 1:
            return None
        return (prices[-1] - prices[-period-1]) / prices[-period-1]

    def get_signal(self, pair: str) -> Tuple[int, float, str]:
        """シグナル生成: 0=SELL, 1=HOLD, 2=BUY"""
        prices = self.get_prices(pair)
        if len(prices) < 20:
            return 1, 0.0, "waiting_data"

        signals = []
        reasons = []

        # RSI
        rsi = self.calculate_rsi(pair)
        if rsi is not None:
            if rsi < RSI_OVERSOLD:
                signals.append(2)
                reasons.append(f"RSI={rsi:.0f}")
            elif rsi > RSI_OVERBOUGHT:
                signals.append(0)
                reasons.append(f"RSI={rsi:.0f}")
            else:
                signals.append(1)

        # EMA
        ema_s = self.calculate_ema(pair, EMA_SHORT)
        ema_l = self.calculate_ema(pair, EMA_LONG)
        if ema_s and ema_l:
            if ema_s > ema_l * 1.001:
                signals.append(2)
                reasons.append("EMA_UP")
            elif ema_s < ema_l * 0.999:
                signals.append(0)
                reasons.append("EMA_DOWN")
            else:
                signals.append(1)

        # Bollinger
        bb = self.calculate_bollinger(pair)
        if bb:
            lower, middle, upper = bb
            if prices[-1] < lower:
                signals.append(2)
                reasons.append("BB_LOW")
            elif prices[-1] > upper:
                signals.append(0)
                reasons.append("BB_HIGH")
            else:
                signals.append(1)

        # Momentum
        mom = self.calculate_momentum(pair)
        if mom is not None:
            if mom > 0.002:
                signals.append(2)
                reasons.append(f"MOM={mom*100:.2f}%")
            elif mom < -0.002:
                signals.append(0)
                reasons.append(f"MOM={mom*100:.2f}%")
            else:
                signals.append(1)

        if not signals:
            return 1, 0.0, "no_signal"

        buy_count = signals.count(2)
        sell_count = signals.count(0)
        total = len(signals)

        if buy_count > sell_count and buy_count >= 2:
            return 2, buy_count / total, ",".join(reasons)
        elif sell_count > buy_count and sell_count >= 2:
            return 0, sell_count / total, ",".join(reasons)
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
    def pnl(self) -> float:
        if self.size > 0 and self.entry_price > 0:
            return (self.current_price - self.entry_price) * self.size
        return 0.0

    @property
    def pnl_pct(self) -> float:
        if self.entry_price > 0:
            return (self.current_price / self.entry_price - 1) * 100
        return 0.0


# =============================================================================
# メイントレーダー
# =============================================================================

class UltimateTrader:
    """本番専用AIトレーダー v4.1"""

    def __init__(self, config: Config = None):
        self.config = config or get_config()
        self.running = False
        self.start_time: Optional[datetime] = None

        self.initial_capital = 0.0
        self.current_capital = 0.0
        self.clients: Dict[str, BitFlyerClient] = {}
        self.active_pairs: List[str] = []
        self.analyzer = TechnicalAnalyzer()
        self.positions: Dict[str, Position] = {}

        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0
        self.max_drawdown = 0.0
        self.peak_capital = 0.0
        self.last_trade_time: Dict[str, datetime] = {}

        logger.info("=" * 70)
        logger.info("    ULTIMATE AI TRADER v4.2 - 世界最強・短期高利益")
        logger.info("=" * 70)
        logger.info(f"    利確: {TAKE_PROFIT_RATE*100:.1f}% | 損切り: {STOP_LOSS_RATE*100:.1f}%")
        logger.info(f"    取引間隔: {MIN_TRADE_INTERVAL}秒 | RSI: {RSI_OVERSOLD}/{RSI_OVERBOUGHT}")
        logger.info("=" * 70)

    async def _api_call_with_retry(self, coro, retries: int = MAX_RETRIES):
        """API呼び出し（リトライ付き）"""
        for attempt in range(retries):
            try:
                result = await asyncio.wait_for(coro, timeout=API_TIMEOUT)
                return result
            except asyncio.TimeoutError:
                logger.warning(f"API timeout (attempt {attempt + 1}/{retries})")
            except Exception as e:
                logger.warning(f"API error (attempt {attempt + 1}/{retries}): {e}")

            if attempt < retries - 1:
                await asyncio.sleep(RETRY_DELAY * (attempt + 1))

        return None

    async def _init_clients(self) -> bool:
        """本番APIクライアント初期化"""
        api_key = self.config.bitflyer.api_key
        api_secret = self.config.bitflyer.api_secret

        if not api_key or not api_secret:
            logger.error("API credentials not found in .env!")
            logger.error("Please set BITFLYER_API_KEY and BITFLYER_API_SECRET")
            return False

        logger.info(f"API Key: {'*'*16}{api_key[-4:]}")
        logger.info("Connecting to bitFlyer API...")

        sorted_pairs = sorted(TRADING_PAIRS.values(), key=lambda x: x.priority)

        for pair_info in sorted_pairs:
            pair = pair_info.code

            client = BitFlyerClient(
                api_key=api_key,
                api_secret=api_secret,
                product_code=pair,
            )

            # 接続テスト（リトライ付き）
            ticker = await self._api_call_with_retry(client.get_ticker())

            if ticker and ticker.ltp > 0:
                self.clients[pair] = client
                self.active_pairs.append(pair)
                self.positions[pair] = Position(pair=pair)
                pair_info.current_price = ticker.ltp
                logger.info(f"  ✓ {pair}: ¥{ticker.ltp:,.0f}")
            else:
                logger.warning(f"  ✗ {pair}: 接続失敗")

        if not self.active_pairs:
            logger.error("")
            logger.error("=" * 50)
            logger.error("  bitFlyer APIに接続できません")
            logger.error("")
            logger.error("  確認事項:")
            logger.error("  1. インターネット接続")
            logger.error("  2. APIキーが正しいか")
            logger.error("  3. API権限（取引権限が必要）")
            logger.error("=" * 50)
            return False

        return True

    async def _get_balance(self) -> Tuple[float, Dict[str, float]]:
        """残高取得"""
        jpy = 0.0
        holdings: Dict[str, float] = {}

        if not self.active_pairs:
            return jpy, holdings

        client = self.clients.get(self.active_pairs[0])
        if not client:
            return jpy, holdings

        balances = await self._api_call_with_retry(client.get_balance())

        if balances and isinstance(balances, list):
            for b in balances:
                currency = b.get('currency_code', '')
                available = float(b.get('available', 0))
                if currency == 'JPY':
                    jpy = available
                elif available > 0:
                    holdings[currency] = available

        return jpy, holdings

    async def _get_price(self, pair: str) -> Optional[float]:
        """価格取得"""
        client = self.clients.get(pair)
        if not client:
            return None

        ticker = await self._api_call_with_retry(client.get_ticker())

        if ticker and ticker.ltp > 0:
            price = ticker.ltp
            self.analyzer.add_price(pair, price)

            if pair in self.positions:
                self.positions[pair].current_price = price
            if pair in TRADING_PAIRS:
                TRADING_PAIRS[pair].current_price = price

            return price

        return None

    def _calculate_size(self, pair: str, price: float) -> float:
        """サイズ計算"""
        pair_info = TRADING_PAIRS.get(pair)
        if not pair_info:
            return 0.0

        # 最小取引金額チェック
        min_cost = pair_info.min_size * price
        if self.current_capital < min_cost:
            logger.debug(f"{pair}: 資金不足 (必要: ¥{min_cost:,.0f}, 残高: ¥{self.current_capital:,.0f})")
            return 0.0

        # 資金の50%を使用（より積極的）
        available = self.current_capital * 0.50

        if available < min_cost:
            available = min_cost  # 最小取引額を使用

        size = available / price
        size = max(pair_info.min_size, size)

        # 正しい小数点桁数に丸める
        size = round(size, pair_info.size_decimals)

        # 最終確認
        if size < pair_info.min_size:
            return 0.0

        return size

    def _can_trade(self, pair: str) -> bool:
        """取引可能判定"""
        last = self.last_trade_time.get(pair)
        if last:
            if (datetime.now() - last).total_seconds() < MIN_TRADE_INTERVAL:
                return False
        return True

    async def _execute_trade(self, pair: str, action: int, reason: str) -> bool:
        """取引実行"""
        if not self._can_trade(pair):
            return False

        client = self.clients.get(pair)
        pos = self.positions.get(pair)
        if not client or not pos:
            return False

        price = pos.current_price
        if price <= 0:
            return False

        # BUY
        if action == 2:
            size = self._calculate_size(pair, price)
            if size <= 0:
                return False

            cost = size * price
            fee = cost * TRADING_FEE_RATE

            if cost + fee > self.current_capital:
                return False

            order_id = await self._api_call_with_retry(
                client.send_order(
                    side=OrderSide.BUY,
                    size=size,
                    order_type=OrderType.MARKET,
                )
            )

            if order_id:
                self.last_trade_time[pair] = datetime.now()
                self.total_trades += 1
                self.current_capital -= (cost + fee)
                pos.size += size
                if pos.entry_price == 0:
                    pos.entry_price = price
                else:
                    total = pos.size
                    pos.entry_price = (pos.entry_price * (total - size) + price * size) / total

                logger.info(f"  ✓ BUY {pair}: {size} @ ¥{price:,.0f} (¥{cost:,.0f}) [{reason}]")
                return True

        # SELL
        elif action == 0:
            if pos.size <= 0:
                return False

            size = pos.size
            proceeds = size * price
            fee = proceeds * TRADING_FEE_RATE
            net = proceeds - fee

            order_id = await self._api_call_with_retry(
                client.send_order(
                    side=OrderSide.SELL,
                    size=size,
                    order_type=OrderType.MARKET,
                )
            )

            if order_id:
                self.last_trade_time[pair] = datetime.now()
                self.total_trades += 1

                entry_cost = pos.size * pos.entry_price
                pnl = net - entry_cost
                pnl_pct = (net / entry_cost - 1) * 100 if entry_cost > 0 else 0

                self.total_pnl += pnl
                if pnl > 0:
                    self.winning_trades += 1

                self.current_capital += net
                pos.size = 0
                pos.entry_price = 0

                emoji = "  " if pnl >= 0 else "  "
                logger.info(f"{emoji} SELL {pair}: ¥{pnl:+,.0f} ({pnl_pct:+.2f}%) [{reason}]")
                return True

        return False

    def _get_portfolio(self) -> float:
        """ポートフォリオ価値"""
        value = self.current_capital
        for pos in self.positions.values():
            if pos.size > 0:
                value += pos.value
        return value

    def _log_status(self) -> None:
        """ステータス表示"""
        portfolio = self._get_portfolio()
        roi = ((portfolio / self.initial_capital) - 1) * 100 if self.initial_capital > 0 else 0
        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
        elapsed = datetime.now() - self.start_time if self.start_time else timedelta(0)

        logger.info("")
        logger.info("=" * 60)
        logger.info(f"    PORTFOLIO: ¥{portfolio:,.0f} ({roi:+.2f}%)")
        logger.info(f"    Cash: ¥{self.current_capital:,.0f}")
        logger.info(f"    Trades: {self.total_trades} | Win: {win_rate:.0f}% | PnL: ¥{self.total_pnl:,.0f}")
        logger.info(f"    Runtime: {elapsed}")
        logger.info("-" * 60)

        for pair, pos in self.positions.items():
            if pos.size > 0:
                logger.info(f"    {pair}: {pos.size} @ ¥{pos.current_price:,.0f} ({pos.pnl_pct:+.2f}%)")

        logger.info("=" * 60)

    async def run(self) -> None:
        """メインループ"""
        self.running = True
        self.start_time = datetime.now()

        logger.info("")
        logger.info("Initializing...")

        if not await self._init_clients():
            return

        logger.info(f"Active: {self.active_pairs}")

        # 残高取得
        jpy, holdings = await self._get_balance()

        if jpy <= 0:
            logger.error("残高を取得できません。API権限を確認してください。")
            return

        self.initial_capital = jpy
        self.current_capital = jpy
        self.peak_capital = jpy

        logger.info(f"Balance: ¥{self.initial_capital:,.0f}")

        # 既存ポジション
        for currency, amount in holdings.items():
            pair = f"{currency}_JPY"
            if pair in self.positions:
                self.positions[pair].size = amount
                price = await self._get_price(pair)
                if price:
                    self.positions[pair].entry_price = price
                logger.info(f"Position: {pair} = {amount}")

        logger.info("")
        logger.info("  Trading started! (Press Ctrl+C to stop)")
        logger.info("")

        tick = 0
        while self.running:
            try:
                tick += 1

                # 価格取得
                for pair in self.active_pairs:
                    await self._get_price(pair)
                    await asyncio.sleep(0.3)

                # 取引判断
                for pair in self.active_pairs:
                    pos = self.positions.get(pair)
                    if not pos:
                        continue

                    has_pos = pos.size > 0

                    # 利確
                    if has_pos and pos.pnl_pct >= TAKE_PROFIT_RATE * 100:
                        await self._execute_trade(pair, 0, "TAKE_PROFIT")
                        continue

                    # 損切り
                    if has_pos and pos.pnl_pct <= -STOP_LOSS_RATE * 100:
                        await self._execute_trade(pair, 0, "STOP_LOSS")
                        continue

                    # シグナル
                    action, conf, reason = self.analyzer.get_signal(pair)

                    # より積極的なトレード（信頼度30%以上で実行）
                    if action == 2 and not has_pos and conf >= 0.3:
                        await self._execute_trade(pair, 2, reason)
                    elif action == 0 and has_pos and conf >= 0.3:
                        await self._execute_trade(pair, 0, reason)

                # ドローダウン
                portfolio = self._get_portfolio()
                if portfolio > self.peak_capital:
                    self.peak_capital = portfolio
                if self.peak_capital > 0:
                    dd = (self.peak_capital - portfolio) / self.peak_capital
                    if dd > self.max_drawdown:
                        self.max_drawdown = dd

                # ステータス (60秒ごと)
                if tick % 60 == 0:
                    self._log_status()

                await asyncio.sleep(1)

            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Error: {e}")
                await asyncio.sleep(5)

        await self.stop()

    async def stop(self) -> None:
        self.running = False
        logger.info("")
        logger.info("  Trader stopped")
        self._log_status()


# =============================================================================
# エントリーポイント
# =============================================================================

async def main():
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>",
        level="INFO",
    )

    config = get_config()
    trader = UltimateTrader(config)

    try:
        await trader.run()
    except KeyboardInterrupt:
        await trader.stop()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopped by user")

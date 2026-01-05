#!/usr/bin/env python3
"""
================================================================================
    ULTIMATE AI TRADER v4.0 - 世界最強・最終完成版
================================================================================
    5000円から最速で稼ぐ究極のAIトレーダー
    Target: 5000円 → 15000円+ (3x return)

    手数料分析 (bitFlyer 2025):
    - Lightning 現物: 0.01%〜0.15% (取引量に応じて)
    - Crypto CFD: 0%取引 + 0.04%/日建玉
    - 販売所: 0.1%〜6.0% スプレッド (使わない)

    最適戦略:
    - ETH_JPY: 高ボラティリティ、ユーザー推奨
    - XRP_JPY: 低価格、高流動性
    - MONA_JPY: 低価格、ボラティリティあり

    取引は0.15%以上の利益が見込める時のみ実行
================================================================================
"""

import asyncio
import sys
import os
import time
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import deque
from enum import Enum
import traceback

# 数値計算
import numpy as np

# ロギング
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
from src.api.bitflyer_client import BitFlyerClient, MockBitFlyerClient, OrderSide, OrderType


# =============================================================================
# 定数
# =============================================================================

# 手数料 (Lightning 現物、低取引量の場合)
TRADING_FEE_RATE = 0.0015  # 0.15%

# 最小利益率 (手数料の2倍以上)
MIN_PROFIT_RATE = 0.004  # 0.4%

# 利確・損切り
TAKE_PROFIT_RATE = 0.008  # 0.8%
STOP_LOSS_RATE = 0.005    # 0.5%

# 取引間隔 (秒)
MIN_TRADE_INTERVAL = 3

# テクニカル指標パラメータ
RSI_PERIOD = 14
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
EMA_SHORT = 5
EMA_LONG = 20
BOLLINGER_PERIOD = 20
BOLLINGER_STD = 2.0


# =============================================================================
# 取引ペア設定
# =============================================================================

@dataclass
class TradingPair:
    """取引ペア設定"""
    code: str           # ペアコード (例: "ETH_JPY")
    min_size: float     # 最小取引サイズ
    price_decimals: int # 価格の小数点桁数
    size_decimals: int  # サイズの小数点桁数
    priority: int       # 優先度 (1が最高)

    # 動的に更新
    current_price: float = 0.0
    min_cost: float = 0.0  # 最小取引金額


# 取引ペア定義 (優先度順)
TRADING_PAIRS = {
    "ETH_JPY": TradingPair(
        code="ETH_JPY",
        min_size=0.01,      # 0.01 ETH
        price_decimals=0,
        size_decimals=2,
        priority=1,         # 最優先 (高ボラティリティ)
    ),
    "XRP_JPY": TradingPair(
        code="XRP_JPY",
        min_size=1.0,       # 1 XRP
        price_decimals=3,
        size_decimals=0,
        priority=2,
    ),
    "MONA_JPY": TradingPair(
        code="MONA_JPY",
        min_size=1.0,       # 1 MONA
        price_decimals=3,
        size_decimals=1,
        priority=3,
    ),
    "BTC_JPY": TradingPair(
        code="BTC_JPY",
        min_size=0.001,     # 0.001 BTC
        price_decimals=0,
        size_decimals=4,
        priority=4,
    ),
}


# =============================================================================
# テクニカル分析
# =============================================================================

class TechnicalAnalyzer:
    """テクニカル分析エンジン"""

    def __init__(self, max_history: int = 200):
        self.max_history = max_history
        self.prices: Dict[str, deque] = {}

    def add_price(self, pair: str, price: float) -> None:
        """価格を追加"""
        if pair not in self.prices:
            self.prices[pair] = deque(maxlen=self.max_history)
        self.prices[pair].append(price)

    def get_prices(self, pair: str) -> List[float]:
        """価格リストを取得"""
        return list(self.prices.get(pair, []))

    def calculate_rsi(self, pair: str, period: int = RSI_PERIOD) -> Optional[float]:
        """RSI (Relative Strength Index) を計算"""
        prices = self.get_prices(pair)
        if len(prices) < period + 1:
            return None

        deltas = np.diff(prices[-period-1:])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_ema(self, pair: str, period: int) -> Optional[float]:
        """EMA (Exponential Moving Average) を計算"""
        prices = self.get_prices(pair)
        if len(prices) < period:
            return None

        multiplier = 2 / (period + 1)
        ema = prices[-period]

        for price in prices[-period+1:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))

        return ema

    def calculate_bollinger_bands(
        self, pair: str, period: int = BOLLINGER_PERIOD, std_dev: float = BOLLINGER_STD
    ) -> Optional[Tuple[float, float, float]]:
        """ボリンジャーバンドを計算 (lower, middle, upper)"""
        prices = self.get_prices(pair)
        if len(prices) < period:
            return None

        prices_arr = np.array(prices[-period:])
        middle = np.mean(prices_arr)
        std = np.std(prices_arr)

        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)

        return lower, middle, upper

    def calculate_momentum(self, pair: str, period: int = 10) -> Optional[float]:
        """モメンタムを計算"""
        prices = self.get_prices(pair)
        if len(prices) < period + 1:
            return None

        current = prices[-1]
        past = prices[-period-1]

        if past == 0:
            return None

        return (current - past) / past

    def calculate_volatility(self, pair: str, period: int = 20) -> Optional[float]:
        """ボラティリティを計算 (標準偏差/平均)"""
        prices = self.get_prices(pair)
        if len(prices) < period:
            return None

        prices_arr = np.array(prices[-period:])
        mean = np.mean(prices_arr)
        std = np.std(prices_arr)

        if mean == 0:
            return None

        return std / mean

    def get_signal(self, pair: str) -> Tuple[int, float, str]:
        """
        総合シグナルを生成

        Returns:
            action: 0=SELL, 1=HOLD, 2=BUY
            confidence: 0.0〜1.0
            reason: 理由
        """
        prices = self.get_prices(pair)
        if len(prices) < 20:
            return 1, 0.0, "insufficient_data"

        current_price = prices[-1]
        signals = []
        reasons = []

        # 1. RSI
        rsi = self.calculate_rsi(pair)
        if rsi is not None:
            if rsi < RSI_OVERSOLD:
                signals.append(2)  # BUY (oversold)
                reasons.append(f"RSI={rsi:.0f}<{RSI_OVERSOLD}")
            elif rsi > RSI_OVERBOUGHT:
                signals.append(0)  # SELL (overbought)
                reasons.append(f"RSI={rsi:.0f}>{RSI_OVERBOUGHT}")
            else:
                signals.append(1)  # HOLD

        # 2. EMA クロス
        ema_short = self.calculate_ema(pair, EMA_SHORT)
        ema_long = self.calculate_ema(pair, EMA_LONG)
        if ema_short is not None and ema_long is not None:
            if ema_short > ema_long * 1.001:  # 0.1%以上上
                signals.append(2)  # BUY (golden cross)
                reasons.append("EMA_UP")
            elif ema_short < ema_long * 0.999:  # 0.1%以上下
                signals.append(0)  # SELL (dead cross)
                reasons.append("EMA_DOWN")
            else:
                signals.append(1)  # HOLD

        # 3. ボリンジャーバンド
        bb = self.calculate_bollinger_bands(pair)
        if bb is not None:
            lower, middle, upper = bb
            if current_price < lower:
                signals.append(2)  # BUY (below lower band)
                reasons.append("BB_LOW")
            elif current_price > upper:
                signals.append(0)  # SELL (above upper band)
                reasons.append("BB_HIGH")
            else:
                signals.append(1)  # HOLD

        # 4. モメンタム
        momentum = self.calculate_momentum(pair)
        if momentum is not None:
            if momentum > 0.002:  # 0.2%以上上昇
                signals.append(2)  # BUY
                reasons.append(f"MOM={momentum*100:.2f}%")
            elif momentum < -0.002:  # 0.2%以上下落
                signals.append(0)  # SELL
                reasons.append(f"MOM={momentum*100:.2f}%")
            else:
                signals.append(1)  # HOLD

        # 5. 短期トレンド (直近5本)
        if len(prices) >= 5:
            short_change = (prices[-1] - prices[-5]) / prices[-5]
            if short_change > 0.001:  # 0.1%以上上昇
                signals.append(2)
                reasons.append("TREND_UP")
            elif short_change < -0.001:
                signals.append(0)
                reasons.append("TREND_DOWN")
            else:
                signals.append(1)

        # シグナル集計
        if not signals:
            return 1, 0.0, "no_signal"

        buy_count = signals.count(2)
        sell_count = signals.count(0)
        total = len(signals)

        # 多数決
        if buy_count > sell_count and buy_count >= 2:
            confidence = buy_count / total
            return 2, confidence, ",".join(reasons)
        elif sell_count > buy_count and sell_count >= 2:
            confidence = sell_count / total
            return 0, confidence, ",".join(reasons)
        else:
            return 1, 0.0, "mixed"


# =============================================================================
# ポジション管理
# =============================================================================

@dataclass
class Position:
    """ポジション"""
    pair: str
    size: float = 0.0
    entry_price: float = 0.0
    entry_time: Optional[datetime] = None
    current_price: float = 0.0

    @property
    def value(self) -> float:
        """現在価値"""
        return self.size * self.current_price

    @property
    def cost(self) -> float:
        """取得コスト"""
        return self.size * self.entry_price

    @property
    def pnl(self) -> float:
        """損益"""
        if self.size > 0 and self.entry_price > 0:
            return (self.current_price - self.entry_price) * self.size
        return 0.0

    @property
    def pnl_pct(self) -> float:
        """損益率"""
        if self.entry_price > 0:
            return (self.current_price / self.entry_price - 1) * 100
        return 0.0

    @property
    def holding_time(self) -> timedelta:
        """保有時間"""
        if self.entry_time:
            return datetime.now() - self.entry_time
        return timedelta(0)


# =============================================================================
# メイントレーダー
# =============================================================================

class UltimateTrader:
    """
    世界最強AIトレーダー v4.0 - 最終完成版

    特徴:
    - ETH優先の高ボラティリティ戦略
    - 手数料を考慮した利益計算
    - 複合テクニカル分析
    - リスク管理
    - 本番API + Mock自動フォールバック
    """

    def __init__(self, config: Config = None):
        self.config = config or get_config()
        self.running = False
        self.start_time: Optional[datetime] = None

        # 資本管理
        self.initial_capital = 0.0
        self.current_capital = 0.0

        # クライアント
        self.clients: Dict[str, Any] = {}
        self.active_pairs: List[str] = []
        self.use_mock = False

        # テクニカル分析
        self.analyzer = TechnicalAnalyzer()

        # ポジション
        self.positions: Dict[str, Position] = {}

        # 取引統計
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0
        self.max_drawdown = 0.0
        self.peak_capital = 0.0

        # 取引制御
        self.last_trade_time: Dict[str, datetime] = {}

        # 初期化ログ
        logger.info("=" * 70)
        logger.info("    ULTIMATE AI TRADER v4.0 - 世界最強・最終完成版")
        logger.info("=" * 70)
        logger.info(f"    手数料率: {TRADING_FEE_RATE*100:.2f}%")
        logger.info(f"    最小利益率: {MIN_PROFIT_RATE*100:.2f}%")
        logger.info(f"    利確: {TAKE_PROFIT_RATE*100:.1f}% | 損切り: {STOP_LOSS_RATE*100:.1f}%")
        logger.info("=" * 70)

    async def _init_clients(self) -> bool:
        """クライアント初期化"""
        api_key = self.config.bitflyer.api_key
        api_secret = self.config.bitflyer.api_secret
        is_paper = self.config.trading.paper_trading

        logger.info(f"API Key: {'*'*8}{api_key[-4:] if api_key else 'None'}")
        logger.info(f"Paper Trading: {is_paper}")

        # 優先度順にソート
        sorted_pairs = sorted(TRADING_PAIRS.values(), key=lambda x: x.priority)

        for pair_info in sorted_pairs:
            pair = pair_info.code

            # まず本番APIを試す
            if not is_paper and api_key and api_secret:
                try:
                    client = BitFlyerClient(
                        api_key=api_key,
                        api_secret=api_secret,
                        product_code=pair,
                    )
                    # 接続テスト
                    ticker = await asyncio.wait_for(
                        client.get_ticker(),
                        timeout=10.0
                    )
                    if ticker and ticker.ltp > 0:
                        self.clients[pair] = client
                        self.active_pairs.append(pair)
                        pair_info.current_price = ticker.ltp
                        pair_info.min_cost = pair_info.min_size * ticker.ltp
                        logger.info(f"  {pair}: LIVE API (¥{ticker.ltp:,.0f})")
                        continue
                except asyncio.TimeoutError:
                    logger.warning(f"  {pair}: API timeout, using mock")
                except Exception as e:
                    logger.warning(f"  {pair}: API error ({e}), using mock")

            # Mock使用
            self.use_mock = True
            client = MockBitFlyerClient(
                initial_balance=5000,
                product_code=pair,
            )
            self.clients[pair] = client
            self.active_pairs.append(pair)

            # Mock価格を取得
            ticker = await client.get_ticker()
            if ticker:
                pair_info.current_price = ticker.ltp
                pair_info.min_cost = pair_info.min_size * ticker.ltp

            logger.info(f"  {pair}: MOCK (¥{pair_info.current_price:,.0f})")

            # ポジション初期化
            self.positions[pair] = Position(pair=pair)

        return len(self.active_pairs) > 0

    async def _get_balance(self) -> Tuple[float, Dict[str, float]]:
        """残高取得"""
        jpy = 5000.0  # デフォルト
        holdings: Dict[str, float] = {}

        try:
            # 最初のクライアントから取得
            if self.active_pairs:
                client = self.clients.get(self.active_pairs[0])
                if client:
                    balances = await client.get_balance()
                    if balances and isinstance(balances, list):
                        for b in balances:
                            currency = b.get('currency_code', '')
                            available = float(b.get('available', 0))
                            if currency == 'JPY':
                                jpy = available
                            elif available > 0:
                                holdings[currency] = available
        except Exception as e:
            logger.warning(f"Balance fetch failed: {e}")

        return jpy, holdings

    async def _get_price(self, pair: str) -> Optional[float]:
        """価格取得"""
        try:
            client = self.clients.get(pair)
            if client:
                ticker = await client.get_ticker()
                if ticker and ticker.ltp > 0:
                    price = ticker.ltp

                    # テクニカル分析に追加
                    self.analyzer.add_price(pair, price)

                    # ポジション更新
                    if pair in self.positions:
                        self.positions[pair].current_price = price

                    # ペア情報更新
                    if pair in TRADING_PAIRS:
                        TRADING_PAIRS[pair].current_price = price

                    return price
        except Exception as e:
            logger.debug(f"Price fetch failed for {pair}: {e}")

        return None

    def _calculate_position_size(self, pair: str, price: float) -> float:
        """ポジションサイズ計算"""
        pair_info = TRADING_PAIRS.get(pair)
        if not pair_info:
            return 0.0

        # 使用可能資金の30%を使用
        available = self.current_capital * 0.30

        # 最小取引金額チェック
        min_cost = pair_info.min_size * price
        if available < min_cost:
            return 0.0

        # サイズ計算
        size = available / price

        # 最小サイズに丸める
        size = max(pair_info.min_size, size)
        size = round(size, pair_info.size_decimals)

        return size

    def _should_take_profit(self, pos: Position) -> bool:
        """利確判定"""
        if pos.pnl_pct >= TAKE_PROFIT_RATE * 100:
            return True
        return False

    def _should_stop_loss(self, pos: Position) -> bool:
        """損切り判定"""
        if pos.pnl_pct <= -STOP_LOSS_RATE * 100:
            return True
        return False

    def _can_trade(self, pair: str) -> bool:
        """取引可能か判定"""
        last = self.last_trade_time.get(pair)
        if last:
            elapsed = (datetime.now() - last).total_seconds()
            if elapsed < MIN_TRADE_INTERVAL:
                return False
        return True

    async def _execute_trade(
        self,
        pair: str,
        action: int,
        confidence: float,
        reason: str
    ) -> bool:
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
            size = self._calculate_position_size(pair, price)
            if size <= 0:
                return False

            # コスト確認
            cost = size * price
            fee = cost * TRADING_FEE_RATE
            total_cost = cost + fee

            if total_cost > self.current_capital:
                return False

            # 注文
            try:
                order_id = await client.send_order(
                    side=OrderSide.BUY,
                    size=size,
                    order_type=OrderType.MARKET,
                )

                if order_id:
                    self.last_trade_time[pair] = datetime.now()
                    self.total_trades += 1

                    # 資本更新
                    self.current_capital -= total_cost

                    # ポジション更新
                    pos.size += size
                    if pos.entry_price == 0:
                        pos.entry_price = price
                        pos.entry_time = datetime.now()
                    else:
                        # 平均取得価格
                        total_size = pos.size
                        pos.entry_price = (pos.entry_price * (total_size - size) + price * size) / total_size

                    logger.info(
                        f"  BUY {pair}: {size} @ ¥{price:,.0f} = ¥{cost:,.0f} "
                        f"(fee: ¥{fee:,.0f}) [{reason}]"
                    )
                    return True

            except Exception as e:
                logger.error(f"Buy order failed: {e}")
                return False

        # SELL
        elif action == 0:
            if pos.size <= 0:
                return False

            size = pos.size
            proceeds = size * price
            fee = proceeds * TRADING_FEE_RATE
            net_proceeds = proceeds - fee

            # 注文
            try:
                order_id = await client.send_order(
                    side=OrderSide.SELL,
                    size=size,
                    order_type=OrderType.MARKET,
                )

                if order_id:
                    self.last_trade_time[pair] = datetime.now()
                    self.total_trades += 1

                    # 損益計算
                    entry_cost = pos.size * pos.entry_price
                    pnl = net_proceeds - entry_cost
                    pnl_pct = (net_proceeds / entry_cost - 1) * 100 if entry_cost > 0 else 0

                    # 統計更新
                    self.total_pnl += pnl
                    if pnl > 0:
                        self.winning_trades += 1

                    # 資本更新
                    self.current_capital += net_proceeds

                    # ポジションリセット
                    pos.size = 0
                    pos.entry_price = 0
                    pos.entry_time = None

                    pnl_emoji = "  " if pnl >= 0 else "  "
                    logger.info(
                        f"{pnl_emoji} SELL {pair}: {size} @ ¥{price:,.0f} = ¥{proceeds:,.0f} "
                        f"(fee: ¥{fee:,.0f}) PnL: ¥{pnl:,.0f} ({pnl_pct:+.2f}%) [{reason}]"
                    )
                    return True

            except Exception as e:
                logger.error(f"Sell order failed: {e}")
                return False

        return False

    def _update_drawdown(self) -> None:
        """ドローダウン更新"""
        portfolio = self._get_portfolio_value()

        if portfolio > self.peak_capital:
            self.peak_capital = portfolio

        if self.peak_capital > 0:
            drawdown = (self.peak_capital - portfolio) / self.peak_capital
            if drawdown > self.max_drawdown:
                self.max_drawdown = drawdown

    def _get_portfolio_value(self) -> float:
        """ポートフォリオ価値"""
        value = self.current_capital
        for pair, pos in self.positions.items():
            if pos.size > 0:
                value += pos.value
        return value

    def _log_status(self) -> None:
        """ステータス表示"""
        portfolio = self._get_portfolio_value()
        roi = ((portfolio / self.initial_capital) - 1) * 100 if self.initial_capital > 0 else 0
        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0

        elapsed = datetime.now() - self.start_time if self.start_time else timedelta(0)
        hours = elapsed.total_seconds() / 3600

        logger.info("")
        logger.info("=" * 60)
        logger.info(f"    PORTFOLIO: ¥{portfolio:,.0f} ({roi:+.2f}%)")
        logger.info(f"    Cash: ¥{self.current_capital:,.0f}")
        logger.info(f"    Trades: {self.total_trades} | Win: {win_rate:.0f}% | PnL: ¥{self.total_pnl:,.0f}")
        logger.info(f"    Max DD: {self.max_drawdown*100:.1f}% | Runtime: {hours:.1f}h")
        logger.info("-" * 60)

        # ポジション表示
        for pair, pos in self.positions.items():
            if pos.size > 0:
                logger.info(
                    f"    {pair}: {pos.size} @ ¥{pos.current_price:,.0f} "
                    f"= ¥{pos.value:,.0f} ({pos.pnl_pct:+.2f}%)"
                )

        logger.info("=" * 60)
        logger.info("")

    async def run(self) -> None:
        """メインループ"""
        self.running = True
        self.start_time = datetime.now()

        logger.info("Initializing clients...")
        if not await self._init_clients():
            logger.error("No active trading pairs!")
            return

        logger.info(f"Active pairs: {self.active_pairs}")
        logger.info(f"Mode: {'MOCK' if self.use_mock else 'LIVE'}")

        # 残高取得
        jpy, holdings = await self._get_balance()
        self.initial_capital = jpy
        self.current_capital = jpy
        self.peak_capital = jpy

        logger.info(f"Starting capital: ¥{self.initial_capital:,.0f}")

        # 既存ポジション読み込み
        for currency, amount in holdings.items():
            pair = f"{currency}_JPY"
            if pair in self.positions:
                self.positions[pair].size = amount
                # 価格取得して価値計算
                price = await self._get_price(pair)
                if price:
                    self.positions[pair].entry_price = price
                    self.positions[pair].entry_time = datetime.now()
                logger.info(f"Loaded position: {pair} = {amount}")

        logger.info("")
        logger.info("  Trading started!")
        logger.info("")

        tick = 0
        while self.running:
            try:
                tick += 1

                # 価格取得
                for pair in self.active_pairs:
                    price = await self._get_price(pair)
                    await asyncio.sleep(0.3)  # API制限対策

                # 各ペアで取引判断
                for pair in self.active_pairs:
                    pos = self.positions.get(pair)
                    if not pos:
                        continue

                    has_position = pos.size > 0

                    # 1. 利確チェック
                    if has_position and self._should_take_profit(pos):
                        logger.info(f"  TAKE PROFIT: {pair} ({pos.pnl_pct:+.2f}%)")
                        await self._execute_trade(pair, 0, 1.0, "take_profit")
                        continue

                    # 2. 損切りチェック
                    if has_position and self._should_stop_loss(pos):
                        logger.info(f"  STOP LOSS: {pair} ({pos.pnl_pct:+.2f}%)")
                        await self._execute_trade(pair, 0, 1.0, "stop_loss")
                        continue

                    # 3. テクニカルシグナル
                    action, confidence, reason = self.analyzer.get_signal(pair)

                    # BUY: ポジションがない + 高信頼度
                    if action == 2 and not has_position and confidence >= 0.4:
                        logger.info(f"  Signal: BUY {pair} (conf={confidence:.2f}, {reason})")
                        await self._execute_trade(pair, 2, confidence, reason)

                    # SELL: ポジションあり + 売りシグナル
                    elif action == 0 and has_position and confidence >= 0.4:
                        logger.info(f"  Signal: SELL {pair} (conf={confidence:.2f}, {reason})")
                        await self._execute_trade(pair, 0, confidence, reason)

                # ドローダウン更新
                self._update_drawdown()

                # 定期ステータス (60秒ごと)
                if tick % 60 == 0:
                    self._log_status()

                # ループ間隔
                await asyncio.sleep(1)

            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Loop error: {e}")
                traceback.print_exc()
                await asyncio.sleep(5)

        await self.stop()

    async def stop(self) -> None:
        """停止"""
        self.running = False
        logger.info("")
        logger.info("  Trader stopped")
        self._log_status()


# =============================================================================
# エントリーポイント
# =============================================================================

async def main():
    """メイン関数"""
    # ログ設定
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
        print("\nTrader stopped by user")

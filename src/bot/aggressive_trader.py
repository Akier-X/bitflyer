"""
Aggressive Small Account Trader
================================
5000円から最速で資産を増やす超攻撃的トレーダー

Target: 5000円 → 15000円+ in 1 month (3x return)
Strategy: High-frequency scalping on volatile altcoins

Key Features:
- 低コスト通貨に集中（XRP, MONA, XLM）
- 高頻度スキャルピング
- モメンタム追従
- 複利運用
- リスク管理付き攻撃的取引
"""

import asyncio
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
from enum import Enum
import traceback
import numpy as np
from loguru import logger

sys.path.insert(0, '/home/user/bitflyer')

from config.settings import Config, get_config
from src.api.bitflyer_client import BitFlyerClient, MockBitFlyerClient, OrderSide, OrderType
from src.notifications.line_messaging import LINEMessagingAPI, TradeNotification


@dataclass
class PriceHistory:
    """価格履歴"""
    prices: deque = field(default_factory=lambda: deque(maxlen=100))
    volumes: deque = field(default_factory=lambda: deque(maxlen=100))
    timestamps: deque = field(default_factory=lambda: deque(maxlen=100))

    def add(self, price: float, volume: float):
        self.prices.append(price)
        self.volumes.append(volume)
        self.timestamps.append(datetime.now())

    @property
    def sma_5(self) -> float:
        if len(self.prices) < 5:
            return 0
        return np.mean(list(self.prices)[-5:])

    @property
    def sma_20(self) -> float:
        if len(self.prices) < 20:
            return 0
        return np.mean(list(self.prices)[-20:])

    @property
    def momentum(self) -> float:
        """短期モメンタム（-1 to 1）"""
        if len(self.prices) < 5:
            return 0
        recent = list(self.prices)[-5:]
        return (recent[-1] - recent[0]) / recent[0] if recent[0] > 0 else 0

    @property
    def volatility(self) -> float:
        """ボラティリティ"""
        if len(self.prices) < 10:
            return 0
        return np.std(list(self.prices)[-10:]) / np.mean(list(self.prices)[-10:])

    @property
    def trend(self) -> int:
        """トレンド方向 (-1, 0, 1)"""
        if self.sma_5 == 0 or self.sma_20 == 0:
            return 0
        if self.sma_5 > self.sma_20 * 1.001:
            return 1  # Uptrend
        elif self.sma_5 < self.sma_20 * 0.999:
            return -1  # Downtrend
        return 0


@dataclass
class AssetPosition:
    """ポジション管理"""
    pair: str
    size: float = 0.0
    entry_price: float = 0.0
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0
    entry_time: Optional[datetime] = None

    def update(self, current_price: float):
        self.current_price = current_price
        if self.size != 0 and self.entry_price > 0:
            if self.size > 0:  # Long
                self.unrealized_pnl = (current_price - self.entry_price) * self.size
                self.unrealized_pnl_pct = (current_price / self.entry_price - 1) * 100
            else:  # Short (FX only)
                self.unrealized_pnl = (self.entry_price - current_price) * abs(self.size)
                self.unrealized_pnl_pct = (self.entry_price / current_price - 1) * 100


class AggressiveTrader:
    """
    超攻撃的スモールアカウントトレーダー

    5000円スタートで最速利益追求
    """

    # 5000円で取引可能なペア（最小注文額順）
    SMALL_ACCOUNT_PAIRS = [
        ("XRP_JPY", 1.0, 100),      # 1 XRP ≈ 100円
        ("MONA_JPY", 1.0, 50),      # 1 MONA ≈ 50円
        ("XLM_JPY", 1.0, 50),       # 1 XLM ≈ 50円
        ("ETH_JPY", 0.01, 5000),    # 0.01 ETH ≈ 5000円
    ]

    # 利確・損切りライン
    TAKE_PROFIT_PCT = 0.8    # 0.8%で利確
    STOP_LOSS_PCT = -0.5     # -0.5%で損切り
    TRAILING_STOP_PCT = 0.3  # 0.3%トレーリングストップ

    # 取引頻度
    MIN_INTERVAL_SECONDS = 3  # 最小3秒間隔
    MAX_TRADES_PER_HOUR = 200  # 1時間200回まで

    def __init__(self, config: Config = None, initial_capital: float = None):
        self.config = config or get_config()
        # 初期資本は後でAPIから取得
        self.initial_capital = initial_capital or 0
        self.current_capital = self.initial_capital

        # API Client for balance check
        self.balance_client = None
        if not self.config.trading.paper_trading:
            self.balance_client = BitFlyerClient(
                api_key=self.config.bitflyer.api_key,
                api_secret=self.config.bitflyer.api_secret,
                product_code="BTC_JPY",
            )

        # アクティブペア（初期化後に更新される）
        self.active_pairs = self._select_pairs_for_capital(initial_capital or 5000)

        # クライアント
        self.clients: Dict[str, any] = {}
        self._init_clients()

        # 通知
        self.notifier = self._init_notifier()

        # 価格履歴
        self.price_history: Dict[str, PriceHistory] = {
            pair: PriceHistory() for pair, _, _ in self.SMALL_ACCOUNT_PAIRS
        }

        # ポジション
        self.positions: Dict[str, AssetPosition] = {
            pair: AssetPosition(pair=pair) for pair, _, _ in self.SMALL_ACCOUNT_PAIRS
        }

        # 統計
        self.running = False
        self.start_time = datetime.now()
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0
        self.max_capital = initial_capital
        self.trades_this_hour = 0
        self.last_hour_reset = datetime.now()
        self.last_trade_time: Dict[str, datetime] = {}

        # 取引履歴
        self.trade_history: deque = deque(maxlen=500)

        logger.info(f"AggressiveTrader initialized")
        logger.info(f"  Initial Capital: ¥{initial_capital:,.0f}")
        logger.info(f"  Active Pairs: {self.active_pairs}")
        logger.info(f"  Target: ¥{initial_capital * 3:,.0f} (3x)")

    def _select_pairs_for_capital(self, capital: float) -> List[str]:
        """資本に応じたペアを選択"""
        pairs = []
        for pair, min_size, approx_cost in self.SMALL_ACCOUNT_PAIRS:
            if approx_cost <= capital * 0.9:  # 資本の90%以下
                pairs.append(pair)
        return pairs if pairs else ["XRP_JPY", "MONA_JPY"]  # フォールバック

    def _init_clients(self):
        """クライアント初期化"""
        for pair, _, _ in self.SMALL_ACCOUNT_PAIRS:
            if self.config.trading.paper_trading:
                self.clients[pair] = MockBitFlyerClient(
                    initial_balance=self.initial_capital,
                    product_code=pair,
                )
            else:
                self.clients[pair] = BitFlyerClient(
                    api_key=self.config.bitflyer.api_key,
                    api_secret=self.config.bitflyer.api_secret,
                    product_code=pair,
                )

    def _init_notifier(self) -> Optional[LINEMessagingAPI]:
        """通知システム初期化"""
        if self.config.line.is_configured and self.config.line.use_messaging_api:
            return LINEMessagingAPI(
                channel_access_token=self.config.line.channel_access_token,
                user_id=self.config.line.user_id,
            )
        return None

    async def _fetch_prices(self) -> Dict[str, Dict]:
        """全ペアの価格を取得"""
        tasks = []
        pairs = []
        for pair in self.active_pairs:
            if pair in self.clients:
                tasks.append(self.clients[pair].get_ticker())
                pairs.append(pair)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        prices = {}
        for pair, result in zip(pairs, results):
            if not isinstance(result, Exception) and result:
                prices[pair] = {
                    'price': result.ltp,
                    'bid': result.best_bid,
                    'ask': result.best_ask,
                    'spread': result.spread,
                    'volume': result.volume,
                }
                # 価格履歴更新
                self.price_history[pair].add(result.ltp, result.volume)
                # ポジション更新
                self.positions[pair].update(result.ltp)

        return prices

    def _generate_signal(self, pair: str, prices: Dict) -> Tuple[int, float, str]:
        """
        シグナル生成（超攻撃的）

        Returns: (action, confidence, reason)
            action: 0=SELL, 1=HOLD, 2=BUY
        """
        history = self.price_history.get(pair)
        if not history or len(history.prices) < 10:
            return 1, 0.0, "insufficient_data"

        price_data = prices.get(pair, {})
        if not price_data:
            return 1, 0.0, "no_price"

        current_price = price_data['price']
        spread_pct = price_data['spread'] / current_price if current_price > 0 else 1

        # スプレッドが広すぎる場合はスキップ
        if spread_pct > 0.005:  # 0.5%以上
            return 1, 0.0, "wide_spread"

        momentum = history.momentum
        volatility = history.volatility
        trend = history.trend

        action = 1  # HOLD
        confidence = 0.0
        reasons = []

        # === 攻撃的シグナル生成 ===

        # 1. モメンタムシグナル
        if momentum > 0.002:  # 0.2%上昇
            action = 2  # BUY
            confidence += 0.3
            reasons.append("momentum_up")
        elif momentum < -0.002:
            action = 0  # SELL
            confidence += 0.3
            reasons.append("momentum_down")

        # 2. トレンドフォロー
        if trend == 1 and action == 2:
            confidence += 0.25
            reasons.append("uptrend")
        elif trend == -1 and action == 0:
            confidence += 0.25
            reasons.append("downtrend")

        # 3. ボラティリティボーナス
        if volatility > 0.005:  # 高ボラ
            confidence += 0.2
            reasons.append("high_volatility")

        # 4. ゴールデンクロス/デッドクロス
        sma_5 = history.sma_5
        sma_20 = history.sma_20
        if sma_5 > 0 and sma_20 > 0:
            if sma_5 > sma_20 * 1.002 and action != 0:
                action = 2
                confidence += 0.15
                reasons.append("golden_cross")
            elif sma_5 < sma_20 * 0.998 and action != 2:
                action = 0
                confidence += 0.15
                reasons.append("dead_cross")

        # 5. 価格の急変検出
        if len(history.prices) >= 3:
            last_3 = list(history.prices)[-3:]
            quick_change = (last_3[-1] - last_3[0]) / last_3[0]
            if quick_change > 0.003:  # 0.3%急騰
                action = 2
                confidence += 0.2
                reasons.append("quick_pump")
            elif quick_change < -0.003:
                action = 0
                confidence += 0.2
                reasons.append("quick_dump")

        # ポジションチェック
        position = self.positions.get(pair)
        if position and position.size != 0:
            # 利確チェック
            if position.unrealized_pnl_pct >= self.TAKE_PROFIT_PCT:
                action = 0 if position.size > 0 else 2
                confidence = 0.95
                reasons = ["take_profit"]
            # 損切りチェック
            elif position.unrealized_pnl_pct <= self.STOP_LOSS_PCT:
                action = 0 if position.size > 0 else 2
                confidence = 0.9
                reasons = ["stop_loss"]

        return action, min(confidence, 1.0), ",".join(reasons)

    def _calculate_order_size(self, pair: str, price: float) -> float:
        """注文サイズを計算（全力投資）"""
        # ペアの最小サイズを取得
        min_size = 1.0
        for p, ms, _ in self.SMALL_ACCOUNT_PAIRS:
            if p == pair:
                min_size = ms
                break

        # 資本の80%を使用
        available = self.current_capital * 0.8
        max_size = available / price if price > 0 else 0

        # 最小サイズ以上で最大サイズ以下
        size = max(min_size, min(max_size, min_size * 10))

        return round(size, 2)

    async def _execute_trade(
        self,
        pair: str,
        action: int,
        confidence: float,
        reason: str,
        price: float,
    ) -> bool:
        """取引実行"""
        # 取引間隔チェック
        last_trade = self.last_trade_time.get(pair)
        if last_trade:
            elapsed = (datetime.now() - last_trade).total_seconds()
            if elapsed < self.MIN_INTERVAL_SECONDS:
                return False

        # 1時間あたりの取引数チェック
        if (datetime.now() - self.last_hour_reset).seconds >= 3600:
            self.trades_this_hour = 0
            self.last_hour_reset = datetime.now()

        if self.trades_this_hour >= self.MAX_TRADES_PER_HOUR:
            return False

        side = OrderSide.BUY if action == 2 else OrderSide.SELL
        size = self._calculate_order_size(pair, price)

        client = self.clients.get(pair)
        if not client:
            return False

        try:
            order_id = await client.send_order(
                side=side,
                size=size,
                order_type=OrderType.MARKET,
            )

            if order_id:
                self.last_trade_time[pair] = datetime.now()
                self.trades_this_hour += 1
                self.total_trades += 1

                # ポジション更新
                position = self.positions[pair]
                if side == OrderSide.BUY:
                    position.size += size
                    position.entry_price = price
                    position.entry_time = datetime.now()
                else:
                    # 決済時のPnL計算
                    if position.size > 0:
                        pnl = (price - position.entry_price) * min(size, position.size)
                        self.total_pnl += pnl
                        self.current_capital += pnl
                        if pnl > 0:
                            self.winning_trades += 1
                    position.size -= size
                    if position.size <= 0:
                        position.size = 0
                        position.entry_price = 0

                # 最大資本更新
                if self.current_capital > self.max_capital:
                    self.max_capital = self.current_capital

                logger.info(
                    f"✅ [{pair}] {side.value} {size} @ ¥{price:,.1f} "
                    f"(conf={confidence:.2f}, {reason})"
                )

                # 通知
                if self.notifier and self.total_trades % 10 == 0:  # 10回ごと
                    win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
                    self.notifier.send_text(
                        f"📈 Trade #{self.total_trades}\n"
                        f"{pair}: {side.value} {size}\n"
                        f"Price: ¥{price:,.0f}\n"
                        f"Capital: ¥{self.current_capital:,.0f}\n"
                        f"PnL: ¥{self.total_pnl:,.0f}\n"
                        f"Win Rate: {win_rate:.1f}%"
                    )

                return True

        except Exception as e:
            logger.error(f"Trade failed: {e}")

        return False

    async def _trading_loop(self):
        """メイン取引ループ"""
        logger.info("🚀 Aggressive Trading Loop Started")

        tick = 0
        rate_limit_backoff = 0  # レート制限バックオフ（秒）
        poll_interval = 2.0  # 基本ポーリング間隔（秒）

        while self.running:
            try:
                tick += 1

                # レート制限バックオフ中は待機
                if rate_limit_backoff > 0:
                    logger.warning(f"⏳ Rate limit backoff: {rate_limit_backoff}s")
                    await asyncio.sleep(rate_limit_backoff)
                    rate_limit_backoff = max(0, rate_limit_backoff - 5)

                # 価格取得（順次取得でレート制限回避）
                prices = {}
                for pair in self.active_pairs:
                    try:
                        if pair in self.clients:
                            ticker = await self.clients[pair].get_ticker()
                            if ticker:
                                prices[pair] = {
                                    'price': ticker.ltp,
                                    'bid': ticker.best_bid,
                                    'ask': ticker.best_ask,
                                    'spread': ticker.spread,
                                    'volume': ticker.volume,
                                }
                                self.price_history[pair].add(ticker.ltp, ticker.volume)
                                self.positions[pair].update(ticker.ltp)
                        await asyncio.sleep(0.3)  # API間隔
                    except Exception as e:
                        if "429" in str(e) or "rate" in str(e).lower():
                            rate_limit_backoff = min(rate_limit_backoff + 10, 60)
                            logger.warning(f"Rate limit hit, backing off {rate_limit_backoff}s")
                            break

                if not prices:
                    await asyncio.sleep(poll_interval)
                    continue

                # 各ペアでシグナル生成・取引
                for pair in self.active_pairs:
                    if pair not in prices:
                        continue

                    action, confidence, reason = self._generate_signal(pair, prices)

                    # 信頼度閾値（攻撃的: 0.35）
                    if action != 1 and confidence >= 0.35:
                        price = prices[pair]['price']
                        await self._execute_trade(pair, action, confidence, reason, price)

                # 定期ステータス（30秒ごと = 15 tick）
                if tick % 15 == 0:
                    self._log_status()

                # 目標達成チェック
                if self.current_capital >= self.initial_capital * 3:
                    logger.info(f"🎉 TARGET ACHIEVED! Capital: ¥{self.current_capital:,.0f}")
                    if self.notifier:
                        self.notifier.send_text(
                            f"🎉 目標達成！\n"
                            f"資本: ¥{self.current_capital:,.0f}\n"
                            f"利益: ¥{self.total_pnl:,.0f}\n"
                            f"取引数: {self.total_trades}"
                        )

                # ループ間隔（2秒 - レート制限対応）
                await asyncio.sleep(poll_interval)

            except Exception as e:
                if "429" in str(e):
                    rate_limit_backoff = min(rate_limit_backoff + 15, 60)
                    logger.warning(f"Rate limit in loop, backing off {rate_limit_backoff}s")
                else:
                    logger.error(f"Loop error: {e}")
                await asyncio.sleep(3)

    def _log_status(self):
        """ステータスログ"""
        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
        roi = ((self.current_capital / self.initial_capital) - 1) * 100

        logger.info(
            f"📊 Capital: ¥{self.current_capital:,.0f} ({roi:+.1f}%) | "
            f"Trades: {self.total_trades} | "
            f"Win: {win_rate:.0f}% | "
            f"PnL: ¥{self.total_pnl:,.0f}"
        )

        for pair in self.active_pairs[:3]:
            pos = self.positions.get(pair)
            if pos and pos.size > 0:
                logger.info(
                    f"  {pair}: {pos.size:.2f} @ ¥{pos.entry_price:,.0f} "
                    f"({pos.unrealized_pnl_pct:+.2f}%)"
                )

    async def _fetch_balance_from_api(self) -> float:
        """APIから残高を取得"""
        if self.balance_client:
            try:
                balances = await self.balance_client.get_balance()
                if balances and not isinstance(balances, dict):
                    for balance in balances:
                        if balance.get('currency_code') == 'JPY':
                            jpy_balance = float(balance.get('available', 0))
                            logger.info(f"💰 API Balance: ¥{jpy_balance:,.0f}")
                            return jpy_balance
                elif isinstance(balances, dict) and 'error' not in balances:
                    # Single balance response
                    for balance in balances if isinstance(balances, list) else [balances]:
                        if balance.get('currency_code') == 'JPY':
                            return float(balance.get('available', 0))
            except Exception as e:
                logger.warning(f"Failed to fetch balance from API: {e}")
        return 0

    async def start(self):
        """トレーダー開始"""
        if self.running:
            return

        self.running = True
        self.start_time = datetime.now()

        # APIから残高を取得（指定がない場合）
        if self.initial_capital == 0 and not self.config.trading.paper_trading:
            logger.info("📡 Fetching balance from bitFlyer API...")
            api_balance = await self._fetch_balance_from_api()
            if api_balance > 0:
                self.initial_capital = api_balance
                self.current_capital = api_balance
                self.max_capital = api_balance
                # アクティブペアを再選択
                self.active_pairs = self._select_pairs_for_capital(api_balance)
                logger.info(f"✅ Balance fetched: ¥{api_balance:,.0f}")
            else:
                logger.warning("⚠️ Could not fetch balance, using default 5000")
                self.initial_capital = 5000
                self.current_capital = 5000
                self.max_capital = 5000

        # Paper mode default
        if self.initial_capital == 0:
            self.initial_capital = 5000
            self.current_capital = 5000
            self.max_capital = 5000

        logger.info("=" * 60)
        logger.info("  AGGRESSIVE SMALL ACCOUNT TRADER")
        logger.info("  超攻撃的スモールアカウントトレーダー")
        logger.info("=" * 60)
        logger.info(f"  Initial: ¥{self.initial_capital:,.0f}")
        logger.info(f"  Target:  ¥{self.initial_capital * 3:,.0f} (3x in 1 month)")
        logger.info(f"  Pairs:   {', '.join(self.active_pairs)}")
        logger.info(f"  Mode:    {'Paper' if self.config.trading.paper_trading else 'LIVE'}")
        logger.info("=" * 60)

        # 開始通知
        if self.notifier:
            self.notifier.send_text(
                f"🚀 Aggressive Trader Started\n\n"
                f"💰 Initial: ¥{self.initial_capital:,.0f}\n"
                f"🎯 Target: ¥{self.initial_capital * 3:,.0f}\n"
                f"📊 Pairs: {', '.join(self.active_pairs)}\n"
                f"⚡ Mode: {'Paper' if self.config.trading.paper_trading else 'LIVE'}"
            )

        await self._trading_loop()

    async def stop(self):
        """トレーダー停止"""
        logger.info("Stopping Aggressive Trader...")
        self.running = False

        # 全ポジション決済
        for pair, position in self.positions.items():
            if position.size > 0:
                try:
                    client = self.clients.get(pair)
                    if client:
                        await client.send_order(
                            side=OrderSide.SELL,
                            size=position.size,
                            order_type=OrderType.MARKET,
                        )
                except Exception:
                    pass

        # 最終レポート
        running_hours = (datetime.now() - self.start_time).total_seconds() / 3600
        roi = ((self.current_capital / self.initial_capital) - 1) * 100
        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0

        logger.info("=" * 60)
        logger.info("  FINAL REPORT")
        logger.info("=" * 60)
        logger.info(f"  Runtime: {running_hours:.1f} hours")
        logger.info(f"  Capital: ¥{self.initial_capital:,.0f} → ¥{self.current_capital:,.0f}")
        logger.info(f"  ROI: {roi:+.1f}%")
        logger.info(f"  Trades: {self.total_trades}")
        logger.info(f"  Win Rate: {win_rate:.1f}%")
        logger.info(f"  Total PnL: ¥{self.total_pnl:,.0f}")
        logger.info("=" * 60)

        if self.notifier:
            self.notifier.send_text(
                f"🛑 Trader Stopped\n\n"
                f"📊 Final Report:\n"
                f"💰 ¥{self.initial_capital:,.0f} → ¥{self.current_capital:,.0f}\n"
                f"📈 ROI: {roi:+.1f}%\n"
                f"🔄 Trades: {self.total_trades}\n"
                f"✅ Win Rate: {win_rate:.1f}%"
            )

    def get_status(self) -> Dict:
        """ステータス取得"""
        return {
            'running': self.running,
            'capital': self.current_capital,
            'initial': self.initial_capital,
            'pnl': self.total_pnl,
            'trades': self.total_trades,
            'win_rate': (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0,
            'positions': {
                pair: {
                    'size': pos.size,
                    'entry': pos.entry_price,
                    'pnl_pct': pos.unrealized_pnl_pct,
                }
                for pair, pos in self.positions.items() if pos.size > 0
            },
        }


async def run_aggressive_trader(initial_capital: float = 5000):
    """攻撃的トレーダー実行"""
    config = get_config()
    trader = AggressiveTrader(config, initial_capital)

    try:
        await trader.start()
    except KeyboardInterrupt:
        await trader.stop()


if __name__ == "__main__":
    asyncio.run(run_aggressive_trader(5000))

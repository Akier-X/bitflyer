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
import json
import os
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

    # 利確・損切りライン（超攻撃的設定）
    TAKE_PROFIT_PCT = 0.5    # 0.5%で即利確（高速回転）
    STOP_LOSS_PCT = -0.3     # -0.3%で即損切り（損失最小化）
    TRAILING_STOP_PCT = 0.2  # 0.2%トレーリングストップ

    # 取引頻度（超高速設定）
    MIN_INTERVAL_SECONDS = 0.5  # 500ms間隔（ミリ秒単位の高速取引）
    MAX_TRADES_PER_HOUR = 1000  # 1時間1000回まで（無制限に近い）

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

        # 状態保存ファイル
        self.state_file = "logs/trader_state.json"
        self.last_state_save = datetime.now()
        self.state_save_interval = 60  # 60秒ごとに保存

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

        # 現金不足チェック - 買いは禁止、売りのみ許可
        min_size = 1.0
        for p, ms, _ in self.SMALL_ACCOUNT_PAIRS:
            if p == pair:
                min_size = ms
                break
        min_order_cost = min_size * current_price
        can_buy = self.current_capital >= min_order_cost

        momentum = history.momentum
        volatility = history.volatility
        trend = history.trend

        action = 1  # HOLD
        confidence = 0.0
        reasons = []

        # ポジション取得（早期に取得）
        position = self.positions.get(pair)

        # === 超攻撃的シグナル生成（世界最強設定） ===

        # 1. モメンタムシグナル（超高感度）
        if momentum > 0.001:  # 0.1%上昇で即反応
            action = 2  # BUY
            confidence += 0.4
            reasons.append("momentum_up")
        elif momentum < -0.001:  # 0.1%下落で即反応
            action = 0  # SELL
            confidence += 0.4
            reasons.append("momentum_down")

        # 2. トレンドフォロー（強化）
        if trend == 1:
            if action == 2:
                confidence += 0.3
            elif action == 1:  # HOLDでもトレンド中は買い
                action = 2
                confidence += 0.25
            reasons.append("uptrend")
        elif trend == -1:
            if action == 0:
                confidence += 0.3
            reasons.append("downtrend")

        # 3. ボラティリティボーナス（低閾値）
        if volatility > 0.003:  # 低ボラでも反応
            confidence += 0.25
            reasons.append("volatility_opportunity")

        # 4. ゴールデンクロス/デッドクロス（高感度）
        sma_5 = history.sma_5
        sma_20 = history.sma_20
        if sma_5 > 0 and sma_20 > 0:
            if sma_5 > sma_20 * 1.001 and action != 0:  # 0.1%差で反応
                action = 2
                confidence += 0.2
                reasons.append("golden_cross")
            elif sma_5 < sma_20 * 0.999 and action != 2:
                action = 0
                confidence += 0.2
                reasons.append("dead_cross")

        # 5. 価格の急変検出（超高感度）
        if len(history.prices) >= 3:
            last_3 = list(history.prices)[-3:]
            quick_change = (last_3[-1] - last_3[0]) / last_3[0]
            if quick_change > 0.002:  # 0.2%急騰で即買い
                action = 2
                confidence += 0.3
                reasons.append("quick_pump")
            elif quick_change < -0.002:  # 0.2%急落で即売り
                action = 0
                confidence += 0.3
                reasons.append("quick_dump")

        # 6. スキャルピングボーナス（ポジションなしで買いやすく）
        if action == 1 and (not position or position.size == 0):
            # ポジションがなく、わずかでも上昇傾向なら買い
            if momentum > 0.0005:
                action = 2
                confidence += 0.2
                reasons.append("scalp_entry")

        # ポジションチェック（利確・損切り）
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

        # 現金不足時はBUYをブロック
        if action == 2 and not can_buy:
            # ポジションがあれば売りを検討
            if position and position.size > 0:
                action = 0  # SELL
                reasons = ["low_cash_sell"]
            else:
                action = 1  # HOLD
                reasons = ["insufficient_cash"]
                confidence = 0

        return action, min(confidence, 1.0), ",".join(reasons)

    def _calculate_order_size(self, pair: str, price: float, is_buy: bool) -> float:
        """注文サイズを計算"""
        # ペアの最小サイズを取得
        min_size = 1.0
        for p, ms, _ in self.SMALL_ACCOUNT_PAIRS:
            if p == pair:
                min_size = ms
                break

        if is_buy:
            # 最小注文金額を計算
            min_order_cost = min_size * price

            # 現金が最小注文金額未満なら買えない
            if self.current_capital < min_order_cost:
                return 0  # 資金不足

            # 買い: 資本の50%を使用（攻撃的分散投資）
            # 全額投入でリターン最大化
            available = self.current_capital * 0.5

            # 最低でも最小注文金額は確保
            available = max(available, min_order_cost)

            # 全資金を使用可能（制限なし）
            available = min(available, self.current_capital)

            max_size = available / price if price > 0 else 0

            # 最小サイズ以上かチェック
            if max_size < min_size:
                return 0  # 資金不足

            # サイズ制限なし - 最大限の利益追求
            size = max_size
        else:
            # 売り: 保有ポジションのみ
            position = self.positions.get(pair)
            if not position or position.size <= 0:
                return 0  # ポジションなし
            size = position.size

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
        is_buy = (action == 2)
        size = self._calculate_order_size(pair, price, is_buy)

        # サイズが0なら取引しない
        if size <= 0:
            return False

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
                cost = price * size

                if side == OrderSide.BUY:
                    # 買い: 資本から購入コストを引く
                    self.current_capital -= cost
                    position.size += size
                    # 平均取得単価を計算
                    if position.entry_price > 0:
                        total_cost = (position.entry_price * (position.size - size)) + cost
                        position.entry_price = total_cost / position.size
                    else:
                        position.entry_price = price
                    position.entry_time = datetime.now()
                    logger.info(f"  💰 Available: ¥{self.current_capital:,.0f}")
                else:
                    # 売り: 売却収入を資本に加算
                    self.current_capital += cost
                    # 決済PnL計算
                    if position.size > 0 and position.entry_price > 0:
                        pnl = (price - position.entry_price) * min(size, position.size)
                        self.total_pnl += pnl
                        if pnl > 0:
                            self.winning_trades += 1
                        logger.info(f"  💵 PnL: ¥{pnl:,.0f}")
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

                # 取引後に状態を保存
                self._save_state()

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
        """メイン取引ループ（超高速モード）"""
        logger.info("🚀 ULTRA AGGRESSIVE Trading Loop Started - 世界最強モード")

        tick = 0
        rate_limit_backoff = 0  # レート制限バックオフ（秒）
        poll_interval = 1.0  # 高速ポーリング（1秒間隔）

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
                        await asyncio.sleep(0.15)  # 超高速API間隔（150ms）
                    except Exception as e:
                        if "429" in str(e) or "rate" in str(e).lower():
                            rate_limit_backoff = min(rate_limit_backoff + 5, 30)
                            logger.warning(f"Rate limit hit, backing off {rate_limit_backoff}s")
                            break

                if not prices:
                    await asyncio.sleep(poll_interval)
                    continue

                # 各ペアでシグナル生成・取引（超攻撃的）
                for pair in self.active_pairs:
                    if pair not in prices:
                        continue

                    action, confidence, reason = self._generate_signal(pair, prices)

                    # 信頼度閾値（超攻撃的: 0.25）- わずかなチャンスも逃さない
                    if action != 1 and confidence >= 0.25:
                        price = prices[pair]['price']
                        await self._execute_trade(pair, action, confidence, reason, price)

                # 定期ステータス（30秒ごと = 30 tick @ 1秒間隔）
                if tick % 30 == 0:
                    self._log_status()

                # 定期残高更新（3分ごと = 180 tick @ 1秒間隔）
                if tick % 180 == 0 and not self.config.trading.paper_trading:
                    try:
                        api_balance, _ = await self._fetch_balance_from_api()
                        if api_balance > 0:
                            self.current_capital = api_balance
                            logger.info(f"💰 Balance refreshed: ¥{api_balance:,.0f}")
                    except Exception:
                        pass

                # 定期状態保存（60秒ごと）
                if (datetime.now() - self.last_state_save).seconds >= self.state_save_interval:
                    self._save_state()
                    self.last_state_save = datetime.now()

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

    def _calculate_current_portfolio_value(self) -> float:
        """現在のポートフォリオ価値を計算（現金 + ポジションの時価）"""
        total = self.current_capital

        for pair, pos in self.positions.items():
            if pos.size > 0 and pos.current_price > 0:
                total += pos.size * pos.current_price

        return total

    def _log_status(self):
        """ステータスログ"""
        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0

        # 全ポートフォリオ価値を計算
        portfolio_value = self._calculate_current_portfolio_value()
        roi = ((portfolio_value / self.initial_capital) - 1) * 100

        # ポジション価値
        position_value = portfolio_value - self.current_capital

        logger.info(
            f"📊 Portfolio: ¥{portfolio_value:,.0f} ({roi:+.1f}%) | "
            f"Cash: ¥{self.current_capital:,.0f} | "
            f"Crypto: ¥{position_value:,.0f}"
        )
        logger.info(
            f"   Trades: {self.total_trades} | "
            f"Win: {win_rate:.0f}% | "
            f"PnL: ¥{self.total_pnl:,.0f}"
        )

        for pair in self.active_pairs[:4]:
            pos = self.positions.get(pair)
            if pos and pos.size > 0:
                value = pos.size * pos.current_price
                logger.info(
                    f"  {pair}: {pos.size:.2f} @ ¥{pos.current_price:,.0f} "
                    f"= ¥{value:,.0f} ({pos.unrealized_pnl_pct:+.2f}%)"
                )

    async def _fetch_balance_from_api(self) -> Tuple[float, Dict[str, float]]:
        """APIから残高と保有コインを取得"""
        jpy_balance = 0
        crypto_holdings = {}

        if self.balance_client:
            try:
                balances = await self.balance_client.get_balance()
                if balances and not isinstance(balances, dict):
                    for balance in balances:
                        currency = balance.get('currency_code', '')
                        available = float(balance.get('available', 0))

                        if currency == 'JPY':
                            jpy_balance = available
                            logger.info(f"💰 JPY Balance: ¥{jpy_balance:,.0f}")
                        elif available > 0:
                            crypto_holdings[currency] = available
                            logger.info(f"🪙 {currency}: {available:.4f}")

                elif isinstance(balances, dict) and 'error' not in balances:
                    for balance in balances if isinstance(balances, list) else [balances]:
                        currency = balance.get('currency_code', '')
                        available = float(balance.get('available', 0))
                        if currency == 'JPY':
                            jpy_balance = available
                        elif available > 0:
                            crypto_holdings[currency] = available

            except Exception as e:
                logger.warning(f"Failed to fetch balance from API: {e}")

        return jpy_balance, crypto_holdings

    async def _calculate_total_portfolio_value(self, jpy_balance: float, holdings: Dict[str, float]) -> float:
        """全資産の合計価値（JPY換算）を計算"""
        total_value = jpy_balance

        currency_to_pair = {
            'XRP': 'XRP_JPY',
            'MONA': 'MONA_JPY',
            'XLM': 'XLM_JPY',
            'ETH': 'ETH_JPY',
            'BTC': 'BTC_JPY',
        }

        for currency, amount in holdings.items():
            pair = currency_to_pair.get(currency)
            if pair and pair in self.clients:
                try:
                    ticker = await self.clients[pair].get_ticker()
                    if ticker:
                        value = amount * ticker.ltp
                        total_value += value
                        logger.info(f"  {currency}: {amount:.4f} × ¥{ticker.ltp:,.0f} = ¥{value:,.0f}")
                except Exception:
                    pass

        return total_value

    def _save_state(self):
        """状態をファイルに保存（再起動後も継続可能）"""
        try:
            state = {
                'saved_at': datetime.now().isoformat(),
                'initial_capital': self.initial_capital,
                'current_capital': self.current_capital,
                'max_capital': self.max_capital,
                'total_trades': self.total_trades,
                'winning_trades': self.winning_trades,
                'total_pnl': self.total_pnl,
                'start_time': self.start_time.isoformat() if self.start_time else None,
                'positions': {
                    pair: {
                        'size': pos.size,
                        'entry_price': pos.entry_price,
                        'entry_time': pos.entry_time.isoformat() if pos.entry_time else None,
                    }
                    for pair, pos in self.positions.items() if pos.size > 0
                },
            }

            os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)

            logger.debug(f"💾 State saved: {len(state['positions'])} positions")
        except Exception as e:
            logger.warning(f"Failed to save state: {e}")

    def _load_state(self) -> bool:
        """保存された状態を読み込み"""
        try:
            if not os.path.exists(self.state_file):
                logger.info("📂 No saved state found, starting fresh")
                return False

            with open(self.state_file, 'r') as f:
                state = json.load(f)

            # 状態が古すぎる場合はスキップ（24時間以上前）
            saved_at = datetime.fromisoformat(state['saved_at'])
            age_hours = (datetime.now() - saved_at).total_seconds() / 3600
            if age_hours > 24:
                logger.warning(f"⚠️ Saved state is {age_hours:.1f} hours old, ignoring")
                return False

            # 統計情報を復元
            self.total_trades = state.get('total_trades', 0)
            self.winning_trades = state.get('winning_trades', 0)
            self.total_pnl = state.get('total_pnl', 0.0)

            # ポジションを復元
            saved_positions = state.get('positions', {})
            for pair, pos_data in saved_positions.items():
                if pair in self.positions:
                    self.positions[pair].size = pos_data.get('size', 0)
                    self.positions[pair].entry_price = pos_data.get('entry_price', 0)
                    entry_time = pos_data.get('entry_time')
                    if entry_time:
                        self.positions[pair].entry_time = datetime.fromisoformat(entry_time)

            logger.info(f"📥 State loaded from {saved_at.strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info(f"   Trades: {self.total_trades}, PnL: ¥{self.total_pnl:,.0f}")
            logger.info(f"   Positions: {len(saved_positions)}")

            return True
        except Exception as e:
            logger.warning(f"Failed to load state: {e}")
            return False

    async def _init_positions_from_holdings(self, holdings: Dict[str, float]):
        """既存の保有コインからポジションを初期化"""
        # 通貨コードとペアのマッピング
        currency_to_pair = {
            'XRP': 'XRP_JPY',
            'MONA': 'MONA_JPY',
            'XLM': 'XLM_JPY',
            'ETH': 'ETH_JPY',
            'BTC': 'BTC_JPY',
        }

        for currency, amount in holdings.items():
            pair = currency_to_pair.get(currency)
            if pair and pair in self.positions:
                # 現在価格を取得して平均取得単価として使用
                try:
                    if pair in self.clients:
                        ticker = await self.clients[pair].get_ticker()
                        if ticker:
                            self.positions[pair].size = amount
                            self.positions[pair].entry_price = ticker.ltp
                            self.positions[pair].current_price = ticker.ltp
                            logger.info(f"📦 Position loaded: {pair} = {amount:.2f} @ ¥{ticker.ltp:,.0f}")
                except Exception as e:
                    logger.warning(f"Failed to get price for {pair}: {e}")

    async def start(self):
        """トレーダー開始"""
        if self.running:
            return

        self.running = True
        self.start_time = datetime.now()

        # 保存された状態を読み込み
        state_loaded = self._load_state()
        if state_loaded:
            logger.info("📥 Resuming from saved state...")

        # APIから残高と保有コインを取得
        if not self.config.trading.paper_trading:
            logger.info("📡 Fetching balance and holdings from bitFlyer API...")
            api_balance, holdings = await self._fetch_balance_from_api()

            # 保有コインをポジションに反映
            if holdings:
                await self._init_positions_from_holdings(holdings)
                logger.info(f"📦 Loaded {len(holdings)} existing positions")

            if self.initial_capital == 0:
                if api_balance > 0 or holdings:
                    # 全資産の合計価値を計算（JPY + 保有コインのJPY換算額）
                    logger.info("📊 Calculating total portfolio value...")
                    total_portfolio = await self._calculate_total_portfolio_value(api_balance, holdings)

                    self.initial_capital = total_portfolio
                    self.current_capital = api_balance  # 取引可能な現金のみ
                    self.max_capital = total_portfolio

                    # 保有コインの価値を記録
                    crypto_value = total_portfolio - api_balance

                    logger.info("=" * 50)
                    logger.info("💼 PORTFOLIO SUMMARY")
                    logger.info("=" * 50)
                    logger.info(f"  💴 JPY Balance:    ¥{api_balance:,.0f}")
                    logger.info(f"  🪙 Crypto Value:   ¥{crypto_value:,.0f}")
                    logger.info(f"  💰 TOTAL VALUE:    ¥{total_portfolio:,.0f}")
                    logger.info("=" * 50)

                    # アクティブペアを全ポートフォリオ価値で選択
                    self.active_pairs = self._select_pairs_for_capital(total_portfolio)
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

        # 停止時に状態を保存
        self._save_state()
        logger.info("💾 State saved for resume on restart")

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
        portfolio_value = self._calculate_current_portfolio_value()
        position_value = portfolio_value - self.current_capital

        return {
            'running': self.running,
            'portfolio_value': portfolio_value,
            'cash': self.current_capital,
            'crypto_value': position_value,
            'initial': self.initial_capital,
            'pnl': self.total_pnl,
            'roi_pct': ((portfolio_value / self.initial_capital) - 1) * 100 if self.initial_capital > 0 else 0,
            'trades': self.total_trades,
            'win_rate': (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0,
            'positions': {
                pair: {
                    'size': pos.size,
                    'entry': pos.entry_price,
                    'current': pos.current_price,
                    'value': pos.size * pos.current_price,
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

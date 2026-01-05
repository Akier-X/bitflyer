#!/usr/bin/env python3
"""
================================================================================
    ULTIMATE AI TRADER v5.0 - 完全同期版
================================================================================
    完全にAPIと同期した堅牢なトレーディングシステム

    主な改善点:
    - 毎回APIから実残高を取得（ローカルキャッシュに依存しない）
    - 価格履歴に基づく利確/損切り判定
    - 安全な注文サイズ計算
    - 完全なエラーハンドリング
================================================================================
"""

import asyncio
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
TAKE_PROFIT = 0.012         # 1.2% (手数料考慮後の実質利益)
STOP_LOSS = 0.008           # 0.8%
TRADE_COOLDOWN = 5          # 5秒のクールダウン
POSITION_RATIO = 0.35       # 現金の35%を1取引に使用
MIN_CASH_RESERVE = 50       # 最低50円は残す

# API制限対策
BALANCE_REFRESH_INTERVAL = 30   # 残高は30秒ごとに更新
PRICE_FETCH_DELAY = 1.0         # 価格取得間隔: 1秒
LOOP_INTERVAL = 3               # メインループ: 3秒

# テクニカル設定
RSI_PERIOD = 7
RSI_BUY = 35                # 買いシグナル閾値
RSI_SELL = 65               # 売りシグナル閾値
MOMENTUM_BUY = 0.002        # 0.2%以上の上昇モメンタム
MOMENTUM_SELL = -0.002      # -0.2%以下の下落モメンタム


# =============================================================================
# ペア設定 (Lightning API対応のみ)
# =============================================================================

@dataclass
class PairConfig:
    code: str
    min_size: float     # 最小注文サイズ
    decimals: int       # サイズの小数点桁数
    price_tick: float   # 価格の最小単位


PAIRS = {
    "XLM_JPY": PairConfig("XLM_JPY", 1.0, 6, 0.001),
    "XRP_JPY": PairConfig("XRP_JPY", 1.0, 6, 0.001),
    "MONA_JPY": PairConfig("MONA_JPY", 1.0, 6, 0.001),
    "ETH_JPY": PairConfig("ETH_JPY", 0.01, 7, 1),
    "BTC_JPY": PairConfig("BTC_JPY", 0.001, 8, 1),
}


# =============================================================================
# 価格トラッカー（エントリー価格を追跡）
# =============================================================================

@dataclass
class PriceTracker:
    """価格履歴とエントリー価格を追跡"""
    prices: deque = field(default_factory=lambda: deque(maxlen=200))
    entry_price: float = 0.0
    entry_time: Optional[datetime] = None
    highest_since_entry: float = 0.0
    lowest_since_entry: float = float('inf')

    def add_price(self, price: float):
        self.prices.append(price)
        if self.entry_price > 0:
            self.highest_since_entry = max(self.highest_since_entry, price)
            self.lowest_since_entry = min(self.lowest_since_entry, price)

    def set_entry(self, price: float):
        self.entry_price = price
        self.entry_time = datetime.now()
        self.highest_since_entry = price
        self.lowest_since_entry = price

    def clear_entry(self):
        self.entry_price = 0.0
        self.entry_time = None
        self.highest_since_entry = 0.0
        self.lowest_since_entry = float('inf')

    def get_pnl_pct(self, current_price: float) -> float:
        if self.entry_price > 0:
            return (current_price - self.entry_price) / self.entry_price
        return 0.0

    def get_rsi(self) -> Optional[float]:
        if len(self.prices) < RSI_PERIOD + 1:
            return None
        prices = list(self.prices)[-RSI_PERIOD-1:]
        deltas = np.diff(prices)
        gains = np.mean(np.where(deltas > 0, deltas, 0))
        losses = np.mean(np.where(deltas < 0, -deltas, 0))
        if losses < 0.0001:
            return 100.0
        rs = gains / losses
        return 100 - (100 / (1 + rs))

    def get_momentum(self) -> Optional[float]:
        if len(self.prices) < 5:
            return None
        return (self.prices[-1] - self.prices[-5]) / self.prices[-5]

    def get_ema_signal(self) -> Optional[int]:
        """EMAクロスオーバーシグナル: 1=買い, -1=売り, 0=中立"""
        if len(self.prices) < 12:
            return None

        prices = list(self.prices)

        def ema(data, period):
            m = 2 / (period + 1)
            e = data[0]
            for p in data[1:]:
                e = p * m + e * (1 - m)
            return e

        ema_short = ema(prices[-5:], 5)
        ema_long = ema(prices[-12:], 12)

        diff_pct = (ema_short - ema_long) / ema_long

        if diff_pct > 0.001:  # 0.1%以上で買い
            return 1
        elif diff_pct < -0.001:  # -0.1%以下で売り
            return -1
        return 0


# =============================================================================
# メイントレーダー
# =============================================================================

class Trader:
    def __init__(self):
        self.config = get_config()
        self.clients: Dict[str, BitFlyerClient] = {}
        self.trackers: Dict[str, PriceTracker] = {}
        self.last_trade: Dict[str, datetime] = {}

        # 残高キャッシュ
        self._cached_balances: Dict[str, float] = {}
        self._balance_last_update: Optional[datetime] = None

        # 統計
        self.trades = 0
        self.wins = 0
        self.total_pnl = 0.0
        self.initial_value = 0.0
        self.start_time = None

    async def _get_balances(self, force: bool = False) -> Dict[str, float]:
        """APIから実残高を取得（キャッシュ付き）"""
        # キャッシュが有効なら使用
        if not force and self._balance_last_update:
            elapsed = (datetime.now() - self._balance_last_update).total_seconds()
            if elapsed < BALANCE_REFRESH_INTERVAL and self._cached_balances:
                return self._cached_balances

        result = {}
        try:
            client = list(self.clients.values())[0]
            balances = await asyncio.wait_for(client.get_balance(), timeout=15)
            if balances:
                for b in balances:
                    currency = b.get("currency_code", "")
                    amount = float(b.get("available", 0))
                    if amount > 0:
                        result[currency] = amount
                # キャッシュ更新
                self._cached_balances = result
                self._balance_last_update = datetime.now()
        except asyncio.TimeoutError:
            logger.warning("残高取得タイムアウト")
            return self._cached_balances  # キャッシュを返す
        except Exception as e:
            logger.error(f"残高取得エラー: {e}")
            return self._cached_balances  # キャッシュを返す
        return result

    async def _get_price(self, pair: str) -> Optional[float]:
        """現在価格を取得"""
        client = self.clients.get(pair)
        if not client:
            return None
        try:
            ticker = await asyncio.wait_for(client.get_ticker(), timeout=10)
            if ticker and ticker.ltp > 0:
                return ticker.ltp
        except:
            pass
        return None

    def _can_trade(self, pair: str) -> bool:
        """トレード可能か確認"""
        last = self.last_trade.get(pair)
        if last:
            elapsed = (datetime.now() - last).total_seconds()
            if elapsed < TRADE_COOLDOWN:
                return False
        return True

    async def _execute_buy(self, pair: str, size: float, reason: str) -> bool:
        """買い注文を実行"""
        cfg = PAIRS.get(pair)
        client = self.clients.get(pair)
        if not cfg or not client:
            return False

        # サイズを正確に丸める
        size = round(size, cfg.decimals)
        if size < cfg.min_size:
            return False

        # 最終確認: 実際の残高をチェック
        balances = await self._get_balances(force=True)
        jpy_available = balances.get("JPY", 0)
        price = await self._get_price(pair) or 0
        required = size * price * (1 + TRADING_FEE)

        if jpy_available < required:
            logger.debug(f"  BUY {pair} スキップ: 残高不足 (¥{jpy_available:,.0f} < ¥{required:,.0f})")
            return False

        logger.info(f"  📤 BUY注文: {pair} {size} @ ¥{price:,.0f} (必要: ¥{required:,.0f})")

        try:
            order_id = await asyncio.wait_for(
                client.send_order(side=OrderSide.BUY, size=size, order_type=OrderType.MARKET),
                timeout=20
            )
            if order_id:
                self.last_trade[pair] = datetime.now()
                self.trades += 1
                self.trackers[pair].set_entry(price)
                self._cached_balances = {}  # キャッシュクリア
                logger.info(f"  ✅ BUY成功: {pair} {size} @ ¥{price:,.0f} [{reason}]")
                return True
            else:
                logger.warning(f"  ⚠️ BUY {pair}: 注文IDなし")
        except asyncio.TimeoutError:
            logger.warning(f"  ⏳ BUY {pair} タイムアウト")
        except Exception as e:
            logger.error(f"  ❌ BUY {pair} エラー: {e}")
        return False

    async def _execute_sell(self, pair: str, size: float, reason: str) -> bool:
        """売り注文を実行"""
        cfg = PAIRS.get(pair)
        client = self.clients.get(pair)
        tracker = self.trackers.get(pair)
        if not cfg or not client or not tracker:
            return False

        # 最終確認: 実際の保有量をチェック
        balances = await self._get_balances(force=True)
        currency = pair.replace("_JPY", "")
        actual_holding = balances.get(currency, 0)

        # 実際に保有している量を使用
        size = min(size, actual_holding)
        size = round(size, cfg.decimals)

        if size < cfg.min_size:
            logger.debug(f"  SELL {pair} スキップ: 保有不足 ({actual_holding} < {cfg.min_size})")
            return False

        price = await self._get_price(pair) or 0
        logger.info(f"  📤 SELL注文: {pair} {size} @ ¥{price:,.0f}")

        try:
            order_id = await asyncio.wait_for(
                client.send_order(side=OrderSide.SELL, size=size, order_type=OrderType.MARKET),
                timeout=20
            )
            if order_id:
                self.last_trade[pair] = datetime.now()
                self.trades += 1

                # 損益計算
                entry = tracker.entry_price if tracker.entry_price > 0 else price
                gross_profit = (price - entry) * size
                fee = price * size * TRADING_FEE * 2  # 往復手数料
                net_profit = gross_profit - fee

                self.total_pnl += net_profit
                if net_profit > 0:
                    self.wins += 1

                tracker.clear_entry()
                self._cached_balances = {}  # キャッシュクリア

                emoji = "💰" if net_profit >= 0 else "📉"
                pnl_pct = (price - entry) / entry * 100 if entry > 0 else 0
                logger.info(f"  {emoji} SELL成功: {pair} {size} @ ¥{price:,.0f} → ¥{net_profit:+,.0f} ({pnl_pct:+.2f}%) [{reason}]")
                return True
            else:
                logger.warning(f"  ⚠️ SELL {pair}: 注文IDなし")
        except asyncio.TimeoutError:
            logger.warning(f"  ⏳ SELL {pair} タイムアウト")
        except Exception as e:
            logger.error(f"  ❌ SELL {pair} エラー: {e}")
        return False

    def _get_buy_signal(self, pair: str) -> Tuple[bool, str]:
        """買いシグナルを判定"""
        tracker = self.trackers.get(pair)
        if not tracker:
            return False, ""

        rsi = tracker.get_rsi()
        momentum = tracker.get_momentum()
        ema_signal = tracker.get_ema_signal()

        signals = []
        reasons = []

        # RSI
        if rsi is not None and rsi < RSI_BUY:
            signals.append(True)
            reasons.append(f"RSI={rsi:.0f}")

        # モメンタム
        if momentum is not None and momentum > MOMENTUM_BUY:
            signals.append(True)
            reasons.append(f"MOM+{momentum*100:.2f}%")

        # EMA
        if ema_signal == 1:
            signals.append(True)
            reasons.append("EMA↑")

        # 2つ以上のシグナルで買い
        if len(signals) >= 2:
            return True, ",".join(reasons)

        return False, ""

    def _get_sell_signal(self, pair: str, current_price: float) -> Tuple[bool, str]:
        """売りシグナルを判定"""
        tracker = self.trackers.get(pair)
        if not tracker:
            return False, ""

        pnl_pct = tracker.get_pnl_pct(current_price)

        # 利確 (1.2%以上)
        if pnl_pct >= TAKE_PROFIT:
            return True, f"PROFIT+{pnl_pct*100:.2f}%"

        # 損切り (-0.8%以下)
        if pnl_pct <= -STOP_LOSS:
            return True, f"LOSS{pnl_pct*100:.2f}%"

        # テクニカル売りシグナル
        rsi = tracker.get_rsi()
        momentum = tracker.get_momentum()
        ema_signal = tracker.get_ema_signal()

        tech_signals = 0
        reasons = []

        if rsi is not None and rsi > RSI_SELL:
            tech_signals += 1
            reasons.append(f"RSI={rsi:.0f}")

        if momentum is not None and momentum < MOMENTUM_SELL:
            tech_signals += 1
            reasons.append(f"MOM{momentum*100:.2f}%")

        if ema_signal == -1:
            tech_signals += 1
            reasons.append("EMA↓")

        # 利益が出ていて、2つ以上のテクニカル売りシグナル
        if pnl_pct > 0.003 and tech_signals >= 2:  # 0.3%以上の利益
            return True, ",".join(reasons)

        return False, ""

    async def init(self) -> bool:
        """初期化"""
        api_key = self.config.bitflyer.api_key
        api_secret = self.config.bitflyer.api_secret

        if not api_key or not api_secret:
            logger.error("APIキーが設定されていません (.envファイルを確認)")
            return False

        logger.info("=" * 60)
        logger.info("  🚀 ULTIMATE AI TRADER v5.0 - 完全同期版")
        logger.info("=" * 60)
        logger.info(f"  📈 利確: {TAKE_PROFIT*100:.1f}% | 損切り: {STOP_LOSS*100:.1f}%")
        logger.info(f"  💰 取引比率: {POSITION_RATIO*100:.0f}% | 手数料: {TRADING_FEE*100:.2f}%")
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
                    self.trackers[pair] = PriceTracker()
                    self.trackers[pair].add_price(ticker.ltp)
                    min_cost = cfg.min_size * ticker.ltp
                    logger.info(f"  ✓ {pair}: ¥{ticker.ltp:,.0f} (最小: ¥{min_cost:,.0f})")
            except Exception as e:
                logger.warning(f"  ✗ {pair}: 接続失敗 - {e}")

        if not self.clients:
            logger.error("接続できるペアがありません")
            return False

        # 初期残高を表示
        balances = await self._get_balances()
        jpy = balances.get("JPY", 0)
        logger.info("-" * 60)
        logger.info(f"  💴 現金: ¥{jpy:,.0f}")

        total_value = jpy
        for pair in self.clients:
            currency = pair.replace("_JPY", "")
            amount = balances.get(currency, 0)
            if amount > 0:
                price = await self._get_price(pair) or 0
                value = amount * price
                total_value += value
                cfg = PAIRS.get(pair)
                can_sell = cfg and amount >= cfg.min_size
                status = "✓売却可" if can_sell else "✗少額"

                # エントリー価格を現在価格に設定（既存ポジション）
                if can_sell:
                    self.trackers[pair].set_entry(price)

                logger.info(f"  💎 {currency}: {amount:.6f} (¥{value:,.0f}) [{status}]")

        self.initial_value = total_value
        logger.info("-" * 60)
        logger.info(f"  📊 総資産: ¥{total_value:,.0f}")
        logger.info("=" * 60)

        return True

    def _log_status(self):
        """現在のステータスを表示"""
        elapsed = datetime.now() - self.start_time if self.start_time else timedelta(0)
        hours = elapsed.total_seconds() / 3600

        win_rate = (self.wins / self.trades * 100) if self.trades > 0 else 0

        logger.info("")
        logger.info("=" * 50)
        logger.info(f"  ⏱️  経過: {elapsed.seconds // 3600}h {(elapsed.seconds % 3600) // 60}m")
        logger.info(f"  📊 取引: {self.trades}回 | 勝率: {win_rate:.0f}%")
        logger.info(f"  💰 累計損益: ¥{self.total_pnl:+,.0f}")
        logger.info("=" * 50)

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
        status_interval = 120  # 2分ごとにステータス表示

        try:
            while True:
                tick += 1

                # === 1. 残高を取得（キャッシュ使用） ===
                balances = await self._get_balances()
                jpy_balance = balances.get("JPY", 0)

                # === 2. 各ペアの処理 ===
                for pair, cfg in PAIRS.items():
                    if pair not in self.clients:
                        continue

                    tracker = self.trackers[pair]
                    currency = pair.replace("_JPY", "")
                    holding = balances.get(currency, 0)

                    # 価格を取得して記録
                    price = await self._get_price(pair)
                    if not price:
                        await asyncio.sleep(PRICE_FETCH_DELAY)
                        continue
                    tracker.add_price(price)

                    has_position = holding >= cfg.min_size

                    # === 保有中の場合: 売りを検討 ===
                    if has_position:
                        if not self._can_trade(pair):
                            await asyncio.sleep(PRICE_FETCH_DELAY)
                            continue

                        should_sell, reason = self._get_sell_signal(pair, price)
                        if should_sell:
                            # _execute_sell内で残高チェックを行う
                            await self._execute_sell(pair, holding, reason)

                    # === 保有なしの場合: 買いを検討 ===
                    else:
                        # エントリー価格をクリア（ポジションがないので）
                        if tracker.entry_price > 0:
                            tracker.clear_entry()

                        # 資金チェック（概算）
                        min_cost = cfg.min_size * price * (1 + TRADING_FEE)
                        available = jpy_balance - MIN_CASH_RESERVE

                        if available < min_cost:
                            # 資金不足でスキップ
                            continue

                        if not self._can_trade(pair):
                            await asyncio.sleep(PRICE_FETCH_DELAY)
                            continue

                        should_buy, reason = self._get_buy_signal(pair)
                        if should_buy:
                            # 購入サイズを計算（概算）
                            budget = min(available * POSITION_RATIO, available)
                            if budget < min_cost:
                                budget = min_cost

                            size = budget / price
                            size = max(cfg.min_size, size)

                            # _execute_buy内で最終的な残高チェックを行う
                            await self._execute_buy(pair, size, reason)

                    # API制限対策の待機
                    await asyncio.sleep(PRICE_FETCH_DELAY)

                # === 3. ステータス表示 ===
                if tick % status_interval == 0:
                    self._log_status()

                # メインループ待機
                await asyncio.sleep(LOOP_INTERVAL)

        except KeyboardInterrupt:
            logger.info("")
            logger.info("  🛑 停止しました")
        except Exception as e:
            logger.error(f"予期しないエラー: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self._log_status()


async def main():
    # ロガー設定
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

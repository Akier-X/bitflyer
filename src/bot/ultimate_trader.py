#!/usr/bin/env python3
"""
================================================================================
    🏆 ULTIMATE AI TRADER v10.0 - 世界最強・絶対的AIトレーダー
================================================================================
    全機能統合版:
    - MACD / ボリンジャーバンド / RSI / 複数時間足
    - LSTM機械学習予測
    - WebSocketリアルタイム価格
    - LINE通知
    - パターン認識
    - 適応型パラメータ
================================================================================
"""

import asyncio
import math
import sys
import os
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Tuple
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
from src.analysis.indicators import TechnicalIndicators
from src.analysis.ml_predictor import EnsemblePredictor, Prediction
from src.notifications.line_notify import SmartNotifier, TradeNotification, DailyReport

# WebSocket (オプション)
try:
    from src.api.websocket_client import MultiPairWebSocket
    WS_AVAILABLE = True
except ImportError:
    WS_AVAILABLE = False


# =============================================================================
# 設定 - 世界最強パラメータ
# =============================================================================

TRADING_FEE = 0.0015

# 動的調整される基本パラメータ
BASE_TAKE_PROFIT = 0.006     # 0.6%
BASE_STOP_LOSS = 0.004       # 0.4%
MIN_HOLD_TIME = 30           # 30秒
TRADE_COOLDOWN = 20          # 20秒
FAIL_COOLDOWN = 60           # 60秒

# API設定
PRICE_DELAY = 1.0
LOOP_DELAY = 2
STATUS_INTERVAL = 15

# AI信頼度閾値
AI_CONFIDENCE_THRESHOLD = 0.4
ML_WEIGHT = 0.3              # ML予測の重み
TECH_WEIGHT = 0.7            # テクニカル指標の重み


# =============================================================================
# ペア設定
# =============================================================================

@dataclass
class PairConfig:
    code: str
    min_size: float
    decimals: int
    priority: int
    volatility_mult: float = 1.0  # ボラティリティ倍率


PAIRS = {
    "MONA_JPY": PairConfig("MONA_JPY", 1.0, 0, 1, 1.2),
    "XLM_JPY": PairConfig("XLM_JPY", 1.0, 0, 2, 1.0),
    "XRP_JPY": PairConfig("XRP_JPY", 1.0, 0, 3, 0.9),
    "ETH_JPY": PairConfig("ETH_JPY", 0.01, 2, 4, 1.1),
    "BTC_JPY": PairConfig("BTC_JPY", 0.001, 3, 5, 0.8),
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
    lowest: float = float('inf')
    ai_confidence: float = 0.0
    entry_reason: str = ""

    def pnl_pct(self, current: float) -> float:
        return (current - self.entry_price) / self.entry_price

    def hold_seconds(self) -> int:
        return int((datetime.now() - self.entry_time).total_seconds())


# =============================================================================
# 取引履歴
# =============================================================================

@dataclass
class TradeRecord:
    pair: str
    action: str
    size: float
    price: float
    pnl: float
    pnl_pct: float
    reason: str
    timestamp: datetime


# =============================================================================
# メイントレーダー
# =============================================================================

class UltimateTrader:
    """世界最強AIトレーダー"""

    def __init__(self):
        self.config = get_config()

        # API/データ
        self.clients: Dict[str, BitFlyerClient] = {}
        self.indicators: Dict[str, TechnicalIndicators] = {}
        self.predictors: Dict[str, EnsemblePredictor] = {}
        self.positions: Dict[str, Position] = {}

        # WebSocket
        self.ws: Optional[MultiPairWebSocket] = None
        self.use_websocket = WS_AVAILABLE

        # 通知
        self.notifier = SmartNotifier()

        # 状態管理
        self.last_trade: Dict[str, datetime] = {}
        self.failed: Dict[str, datetime] = {}
        self._balances: Dict[str, float] = {}
        self._balance_time: Optional[datetime] = None

        # 統計
        self.trades = 0
        self.wins = 0
        self.pnl = 0.0
        self.start_value = 0.0
        self.start_time: Optional[datetime] = None
        self.trade_history: List[TradeRecord] = []

        # 適応パラメータ
        self.volatility: Dict[str, float] = {}
        self.win_streak = 0
        self.loss_streak = 0

    # =========================================================================
    # データ取得
    # =========================================================================

    async def get_balances(self, force: bool = False) -> Dict[str, float]:
        if not force and self._balance_time:
            if (datetime.now() - self._balance_time).seconds < 20:
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
        # WebSocket優先
        if self.ws and self.ws.is_connected:
            price = self.ws.get_price(pair)
            if price and price > 0:
                return price

        # REST APIフォールバック
        client = self.clients.get(pair)
        if not client:
            return None
        try:
            ticker = await asyncio.wait_for(client.get_ticker(), timeout=10)
            return ticker.ltp if ticker and ticker.ltp > 0 else None
        except:
            return None

    # =========================================================================
    # 動的パラメータ
    # =========================================================================

    def get_dynamic_params(self, pair: str) -> Tuple[float, float]:
        """ボラティリティに応じた動的利確/損切り"""
        cfg = PAIRS.get(pair)
        vol = self.volatility.get(pair, 1.0)

        # ボラティリティ調整
        vol_factor = max(0.5, min(2.0, vol))

        take_profit = BASE_TAKE_PROFIT * vol_factor * (cfg.volatility_mult if cfg else 1.0)
        stop_loss = BASE_STOP_LOSS * vol_factor * (cfg.volatility_mult if cfg else 1.0)

        # 連勝/連敗調整
        if self.win_streak >= 3:
            take_profit *= 1.2  # 調子がいい時は欲張る
        if self.loss_streak >= 2:
            stop_loss *= 0.8  # 損失時は早く切る
            take_profit *= 0.9

        return take_profit, stop_loss

    # =========================================================================
    # AI分析
    # =========================================================================

    def analyze(self, pair: str, price: float) -> Tuple[str, float, str]:
        """
        AI統合分析
        Returns: (シグナル, 信頼度, 理由)
        """
        indicators = self.indicators.get(pair)
        predictor = self.predictors.get(pair)

        if not indicators:
            return "NEUTRAL", 0.0, "データなし"

        # 価格追加
        indicators.add_price(price)
        if predictor:
            predictor.add_price(price)

        # テクニカル分析
        tech_signal, tech_conf, tech_reason = indicators.composite_signal()

        # ML予測
        ml_signal = "NEUTRAL"
        ml_conf = 0.0
        ml_reason = ""

        if predictor:
            prediction = predictor.predict()
            if prediction:
                ml_signal = prediction.direction
                ml_conf = prediction.confidence
                ml_reason = f"AI予測:{prediction.predicted_change:+.2f}%"

        # 統合
        if tech_signal == ml_signal and tech_signal != "NEUTRAL":
            # 一致 = 高信頼度
            combined_conf = tech_conf * TECH_WEIGHT + ml_conf * ML_WEIGHT + 0.2
            combined_reason = f"{tech_reason} {ml_reason}"
            return tech_signal, min(combined_conf, 1.0), combined_reason

        elif tech_conf > ml_conf * 1.5:
            # テクニカル優勢
            return tech_signal, tech_conf * 0.8, tech_reason

        elif ml_conf > tech_conf * 1.5 and ml_conf > 0.5:
            # ML優勢
            return ml_signal, ml_conf * 0.7, ml_reason

        else:
            # 不一致 = 慎重に
            return "NEUTRAL", 0.0, "シグナル不一致"

    # =========================================================================
    # 売買判断
    # =========================================================================

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

    def should_buy(self, pair: str, price: float) -> Tuple[bool, float, str]:
        """買い判断"""
        if not self.can_trade(pair):
            return False, 0.0, ""

        signal, confidence, reason = self.analyze(pair, price)

        if signal == "BUY" and confidence >= AI_CONFIDENCE_THRESHOLD:
            return True, confidence, reason

        return False, 0.0, ""

    def should_sell(self, pair: str, price: float) -> Tuple[bool, str]:
        """売り判断"""
        pos = self.positions.get(pair)
        if not pos:
            return False, ""

        pnl = pos.pnl_pct(price)
        hold_time = pos.hold_seconds()
        take_profit, stop_loss = self.get_dynamic_params(pair)

        # 高値/安値更新
        if price > pos.highest:
            pos.highest = price
        if price < pos.lowest:
            pos.lowest = price

        # === 損切り判断 ===

        # 緊急損切り (-2%)
        if pnl <= -0.02:
            return True, f"緊急損切り {pnl*100:.2f}%"

        # 通常損切り
        if hold_time >= MIN_HOLD_TIME and pnl <= -stop_loss:
            return True, f"損切り {pnl*100:.2f}%"

        # === 利確判断 ===

        # 急騰利確 (+1%以上、即時)
        if pnl >= 0.01:
            return True, f"急騰利確 +{pnl*100:.2f}%"

        # 通常利確
        if hold_time >= MIN_HOLD_TIME and pnl >= take_profit:
            return True, f"利確 +{pnl*100:.2f}%"

        # トレーリングストップ
        if pos.highest > 0 and pnl > 0.003:
            drop = (pos.highest - price) / pos.highest
            if drop > 0.003:
                return True, f"トレール +{pnl*100:.2f}%"

        # AI売りシグナル
        if hold_time >= MIN_HOLD_TIME:
            signal, confidence, reason = self.analyze(pair, price)
            if signal == "SELL" and confidence > 0.5 and pnl > 0.002:
                return True, f"AI売り {reason}"

        return False, ""

    # =========================================================================
    # 注文実行
    # =========================================================================

    async def execute_buy(self, pair: str, confidence: float, reason: str) -> bool:
        cfg = PAIRS.get(pair)
        client = self.clients.get(pair)
        if not cfg or not client:
            return False

        balances = await self.get_balances(force=True)
        jpy = balances.get("JPY", 0) - 50
        if jpy < 30:
            return False

        price = await self.get_price(pair)
        if not price:
            return False

        # 信頼度に応じた投資額
        budget_ratio = 0.3 + (confidence * 0.4)  # 30%~70%
        budget = jpy * budget_ratio
        size = budget / price
        size = round(size, cfg.decimals)
        if size < cfg.min_size:
            size = cfg.min_size

        cost = size * price * 1.002
        if cost > jpy:
            return False

        currency = pair.replace("_JPY", "")
        logger.info(f"")
        logger.info(f"  🛒 【購入】{currency}")
        logger.info(f"     数量: {size} @ ¥{price:,.0f}")
        logger.info(f"     AI信頼度: {confidence*100:.0f}%")
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
                    entry_time=datetime.now(), highest=price, lowest=price,
                    ai_confidence=confidence, entry_reason=reason
                )
                self._balances = {}

                logger.info(f"  ✅ 購入成功！ 投資額: ¥{cost:,.0f}")

                # LINE通知
                await self.notifier.on_trade(TradeNotification(
                    action="BUY", pair=pair, size=size, price=price, reason=reason
                ))

                return True
            else:
                self.failed[pair] = datetime.now()
        except Exception as e:
            self.failed[pair] = datetime.now()
            logger.debug(f"  購入エラー: {e}")
        return False

    async def execute_sell(self, pair: str, reason: str) -> bool:
        cfg = PAIRS.get(pair)
        client = self.clients.get(pair)
        pos = self.positions.get(pair)
        if not cfg or not client:
            return False

        # 未決済キャンセル
        try:
            await client.cancel_all_orders(pair)
            await asyncio.sleep(0.3)
        except:
            pass

        balances = await self.get_balances(force=True)
        currency = pair.replace("_JPY", "")
        holding = balances.get(currency, 0)

        # 安全な売却サイズ
        size = holding * 0.95
        multiplier = 10 ** cfg.decimals
        size = math.floor(size * multiplier) / multiplier

        if size < cfg.min_size:
            return False

        price = await self.get_price(pair)
        if not price:
            return False

        logger.info(f"")
        logger.info(f"  💰 【売却】{currency}")
        logger.info(f"     数量: {size} @ ¥{price:,.0f}")
        logger.info(f"     理由: {reason}")

        try:
            order_id = await asyncio.wait_for(
                client.send_order(OrderSide.SELL, size, OrderType.MARKET),
                timeout=20
            )
            if order_id:
                self.last_trade[pair] = datetime.now()
                self.trades += 1

                # 損益計算
                entry = pos.entry_price if pos else price
                gross = (price - entry) * size
                fee = price * size * TRADING_FEE * 2
                net = gross - fee
                pct = (price - entry) / entry * 100 if entry > 0 else 0

                self.pnl += net

                # 勝敗記録
                if net > 0:
                    self.wins += 1
                    self.win_streak += 1
                    self.loss_streak = 0
                    emoji = "💰"
                else:
                    self.loss_streak += 1
                    self.win_streak = 0
                    emoji = "📉"

                # 履歴追加
                self.trade_history.append(TradeRecord(
                    pair=pair, action="SELL", size=size, price=price,
                    pnl=net, pnl_pct=pct, reason=reason, timestamp=datetime.now()
                ))

                if pair in self.positions:
                    del self.positions[pair]
                self._balances = {}

                logger.info(f"  {emoji} 売却完了！")
                logger.info(f"     損益: ¥{net:+,.0f} ({pct:+.2f}%)")
                logger.info(f"     累計: ¥{self.pnl:+,.0f}")

                # LINE通知
                await self.notifier.on_trade(TradeNotification(
                    action="SELL", pair=pair, size=size, price=price,
                    pnl=net, pnl_pct=pct, reason=reason
                ))

                # マイルストーンチェック
                await self._check_milestones()

                return True
            else:
                self.failed[pair] = datetime.now()
        except Exception as e:
            self.failed[pair] = datetime.now()
            logger.debug(f"  売却エラー: {e}")
        return False

    async def _check_milestones(self):
        """マイルストーンチェック"""
        balances = await self.get_balances(force=True)
        total = balances.get("JPY", 0)

        for pair in self.clients:
            currency = pair.replace("_JPY", "")
            amount = balances.get(currency, 0)
            if amount > 0:
                price = await self.get_price(pair)
                if price:
                    total += amount * price

        milestones = [
            (5000, "¥5,000達成！ETH取引可能に！"),
            (7500, "¥7,500達成！順調に成長中！"),
            (10000, "¥10,000達成！2倍到達！"),
            (15000, "¥15,000達成！目標達成！"),
            (20000, "¥20,000達成！4倍到達！"),
        ]

        for threshold, message in milestones:
            if self.start_value < threshold <= total:
                await self.notifier.notifier.notify_milestone(message, total, self.pnl)

    # =========================================================================
    # 初期化
    # =========================================================================

    async def init(self) -> bool:
        api_key = self.config.bitflyer.api_key
        api_secret = self.config.bitflyer.api_secret

        if not api_key or not api_secret:
            logger.error("❌ APIキー未設定")
            return False

        logger.info("")
        logger.info("╔══════════════════════════════════════════════════════════════╗")
        logger.info("║  🏆 ULTIMATE AI TRADER v10.0 - 世界最強・絶対的AIトレーダー ║")
        logger.info("╠══════════════════════════════════════════════════════════════╣")
        logger.info("║  MACD | ボリンジャー | RSI | 複数時間足 | LSTM | LINE通知   ║")
        logger.info("╚══════════════════════════════════════════════════════════════╝")

        active_pairs = []

        for pair, cfg in sorted(PAIRS.items(), key=lambda x: x[1].priority):
            try:
                client = BitFlyerClient(api_key, api_secret, pair)
                ticker = await asyncio.wait_for(client.get_ticker(), timeout=10)
                if ticker and ticker.ltp > 0:
                    self.clients[pair] = client
                    self.indicators[pair] = TechnicalIndicators()
                    self.predictors[pair] = EnsemblePredictor()
                    self.volatility[pair] = 1.0
                    active_pairs.append(pair)

                    min_jpy = cfg.min_size * ticker.ltp
                    logger.info(f"  ✓ {pair}: ¥{ticker.ltp:,.0f} (最小: ¥{min_jpy:,.0f})")
            except:
                pass

        if not self.clients:
            return False

        # WebSocket開始
        if self.use_websocket and WS_AVAILABLE:
            try:
                self.ws = MultiPairWebSocket(active_pairs)
                if await self.ws.start():
                    logger.info("  ⚡ WebSocket: 有効")
                else:
                    self.ws = None
                    logger.info("  ⚡ WebSocket: 無効 (REST API使用)")
            except:
                self.ws = None

        # 未決済キャンセル
        for pair in self.clients:
            try:
                await self.clients[pair].cancel_all_orders(pair)
            except:
                pass
        await asyncio.sleep(0.5)

        # 残高確認
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
                        entry_time=datetime.now(), highest=price, lowest=price
                    )
                    logger.info(f"  💎 {currency}: {sellable} (¥{value:,.0f})")

        self.start_value = total
        logger.info(f"  📊 総資産: ¥{total:,.0f}")
        logger.info("")

        # LINE通知
        await self.notifier.start(total, active_pairs)

        return True

    # =========================================================================
    # ステータス表示
    # =========================================================================

    def show_status(self):
        logger.info("")
        logger.info("  ══════════════════════════════════════════════════════════════")
        logger.info("  📊 【AI戦況】")

        unrealized = 0.0

        for pair, pos in self.positions.items():
            indicators = self.indicators.get(pair)
            if not indicators:
                continue

            price = indicators.current_price()
            if not price:
                continue

            currency = pair.replace("_JPY", "")
            pnl_pct = pos.pnl_pct(price) * 100
            pnl_jpy = (price - pos.entry_price) * pos.size
            unrealized += pnl_jpy
            hold = pos.hold_seconds()

            # AI分析
            signal, confidence, _ = self.analyze(pair, price)
            ai_str = f"AI:{signal[0]}{confidence*100:.0f}%"

            emoji = "📈" if pnl_pct >= 0 else "📉"
            logger.info(f"  {emoji} {currency}: {pnl_pct:+.2f}% (¥{pnl_jpy:+,.0f}) {hold}秒 {ai_str}")

        logger.info("  ──────────────────────────────────────────────────────────────")

        win_rate = (self.wins / self.trades * 100) if self.trades > 0 else 0
        logger.info(f"  💰 含み: ¥{unrealized:+,.0f} | 確定: ¥{self.pnl:+,.0f}")
        logger.info(f"  📊 取引: {self.trades}回 (勝率: {win_rate:.0f}%)")

        # 連勝/連敗表示
        if self.win_streak >= 2:
            logger.info(f"  🔥 {self.win_streak}連勝中！")
        if self.loss_streak >= 2:
            logger.info(f"  ⚠️ {self.loss_streak}連敗中...")

        jpy = self._balances.get("JPY", 0)
        if jpy >= 5000:
            logger.info(f"  🎉 ETH取引可能！")

        logger.info("  ══════════════════════════════════════════════════════════════")

    # =========================================================================
    # メインループ
    # =========================================================================

    async def run(self):
        if not await self.init():
            return

        self.start_time = datetime.now()
        logger.info("  🚀 世界最強AIトレーダー起動！ Ctrl+C で停止")
        logger.info("")

        tick = 0
        last_daily_report = datetime.now().date()

        while True:
            try:
                tick += 1

                balances = await self.get_balances()
                jpy = balances.get("JPY", 0)

                for pair, cfg in sorted(PAIRS.items(), key=lambda x: x[1].priority):
                    if pair not in self.clients:
                        continue

                    price = await self.get_price(pair)
                    if not price:
                        await asyncio.sleep(PRICE_DELAY)
                        continue

                    # ボラティリティ更新
                    indicators = self.indicators[pair]
                    vol = indicators.volatility()
                    if vol:
                        self.volatility[pair] = vol

                    has_position = pair in self.positions

                    if has_position:
                        should, reason = self.should_sell(pair, price)
                        if should:
                            await self.execute_sell(pair, reason)
                    else:
                        min_cost = cfg.min_size * price * 1.01
                        if jpy - 50 >= min_cost:
                            should, confidence, reason = self.should_buy(pair, price)
                            if should:
                                await self.execute_buy(pair, confidence, reason)

                    await asyncio.sleep(PRICE_DELAY)

                # ステータス表示
                if tick % STATUS_INTERVAL == 0:
                    self.show_status()

                # 日次レポート
                today = datetime.now().date()
                if today != last_daily_report and datetime.now().hour >= 0:
                    await self._send_daily_report()
                    last_daily_report = today

                await asyncio.sleep(LOOP_DELAY)

            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"  エラー: {e}")
                await asyncio.sleep(10)

        # 終了処理
        logger.info("")
        logger.info("  🛑 停止")
        self.show_status()

        await self.notifier.stop(self.pnl, self.trades)

        if self.ws:
            await self.ws.stop()

    async def _send_daily_report(self):
        """日次レポート送信"""
        today_trades = [t for t in self.trade_history
                        if t.timestamp.date() == datetime.now().date()]

        if not today_trades:
            return

        wins = sum(1 for t in today_trades if t.pnl > 0)
        losses = len(today_trades) - wins
        total_pnl = sum(t.pnl for t in today_trades)

        best = max((t.pnl for t in today_trades), default=None)
        worst = min((t.pnl for t in today_trades), default=None)

        balances = await self.get_balances(force=True)
        end_balance = balances.get("JPY", 0)
        for pair in self.clients:
            currency = pair.replace("_JPY", "")
            amount = balances.get(currency, 0)
            price = await self.get_price(pair)
            if amount > 0 and price:
                end_balance += amount * price

        report = DailyReport(
            total_trades=len(today_trades),
            wins=wins,
            losses=losses,
            total_pnl=total_pnl,
            best_trade=best,
            worst_trade=worst,
            start_balance=self.start_value,
            end_balance=end_balance
        )

        await self.notifier.daily_report(report)


# =============================================================================
# エントリーポイント
# =============================================================================

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
    logging.getLogger("websockets").setLevel(logging.WARNING)

    trader = UltimateTrader()
    await trader.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass

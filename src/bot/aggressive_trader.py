#!/usr/bin/env python3
"""
================================================================================
    🏆 ULTIMATE AI TRADER v8.0 - 世界最強システム
================================================================================
    - リアルタイム取引状況・AI確率表示
    - LINE通知対応
    - エラーメッセージ抑制
    - 高速利益追求
================================================================================
"""

import asyncio
import math
import sys
import os
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple, List
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
# 設定 - 高速利益追求
# =============================================================================

TRADING_FEE = 0.0015
TAKE_PROFIT = 0.006          # 0.6%で利確（素早く利確）
STOP_LOSS = 0.004            # 0.4%で損切り（損失最小化）
TRADE_COOLDOWN = 3           # 3秒クールダウン（高速取引）
FAIL_COOLDOWN = 20           # 失敗後20秒
MIN_CASH_RESERVE = 30        # 最小リザーブを下げて積極投資

# API設定
BALANCE_REFRESH = 20
PRICE_DELAY = 0.5
LOOP_DELAY = 2
STATUS_INTERVAL = 20         # 20秒ごとにステータス

# テクニカル設定
RSI_PERIOD = 5
RSI_BUY = 50
RSI_SELL = 50

# LINE通知設定（.envにLINE_TOKENを設定）
LINE_TOKEN = os.getenv("LINE_TOKEN", "")
LINE_API = "https://notify-api.line.me/api/notify"


# =============================================================================
# ペア設定
# =============================================================================

@dataclass
class PairConfig:
    code: str
    min_size: float
    buy_decimals: int
    sell_decimals: int
    priority: int  # 低いほど優先


PAIRS = {
    "MONA_JPY": PairConfig("MONA_JPY", 1.0, 6, 0, 1),    # 最優先（低価格）
    "XLM_JPY": PairConfig("XLM_JPY", 1.0, 6, 0, 2),
    "XRP_JPY": PairConfig("XRP_JPY", 1.0, 6, 0, 3),
    "ETH_JPY": PairConfig("ETH_JPY", 0.01, 7, 2, 4),     # 資金増えたら
    "BTC_JPY": PairConfig("BTC_JPY", 0.001, 8, 3, 5),    # 最終目標
}


# =============================================================================
# AI分析エンジン
# =============================================================================

@dataclass
class AIAnalysis:
    """AI分析結果"""
    signal: str = "HOLD"      # BUY, SELL, HOLD
    confidence: float = 0.0   # 0-100%
    reasons: List[str] = field(default_factory=list)
    price_prediction: str = "横ばい"


@dataclass
class PriceTracker:
    prices: deque = field(default_factory=lambda: deque(maxlen=100))
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

    def trend(self, periods: int = 3) -> Optional[float]:
        if len(self.prices) < periods + 1:
            return None
        return (self.prices[-1] - self.prices[-periods-1]) / self.prices[-periods-1] * 100

    def volatility(self) -> Optional[float]:
        if len(self.prices) < 10:
            return None
        prices = list(self.prices)[-10:]
        return np.std(prices) / np.mean(prices) * 100

    def analyze(self, has_position: bool) -> AIAnalysis:
        """AI分析を実行"""
        analysis = AIAnalysis()

        if len(self.prices) < 10:
            analysis.signal = "WAIT"
            analysis.confidence = 0
            analysis.reasons = ["データ収集中"]
            return analysis

        rsi = self.rsi()
        trend = self.trend()
        vol = self.volatility()

        score = 0
        reasons = []

        # RSI分析
        if rsi:
            if rsi < 30:
                score += 30
                reasons.append(f"RSI低({rsi:.0f})=買いチャンス")
            elif rsi < 45:
                score += 15
                reasons.append(f"RSI低め({rsi:.0f})")
            elif rsi > 70:
                score -= 30
                reasons.append(f"RSI高({rsi:.0f})=売りシグナル")
            elif rsi > 55:
                score -= 15
                reasons.append(f"RSI高め({rsi:.0f})")

        # トレンド分析
        if trend:
            if trend > 0.5:
                score += 25
                reasons.append(f"上昇トレンド(+{trend:.2f}%)")
                analysis.price_prediction = "上昇"
            elif trend > 0.1:
                score += 10
                reasons.append(f"やや上昇(+{trend:.2f}%)")
            elif trend < -0.5:
                score -= 25
                reasons.append(f"下降トレンド({trend:.2f}%)")
                analysis.price_prediction = "下落"
            elif trend < -0.1:
                score -= 10
                reasons.append(f"やや下降({trend:.2f}%)")

        # ボラティリティ
        if vol:
            if vol > 1.0:
                score += 10
                reasons.append(f"高ボラ({vol:.2f}%)=チャンス")
            elif vol < 0.2:
                reasons.append(f"低ボラ({vol:.2f}%)")

        # ポジションがある場合の損益チェック
        if has_position and self.entry_price > 0:
            current_pnl = self.pnl(self.prices[-1]) * 100
            if current_pnl >= TAKE_PROFIT * 100:
                analysis.signal = "SELL"
                analysis.confidence = 90
                analysis.reasons = [f"利確シグナル(+{current_pnl:.2f}%)"]
                return analysis
            elif current_pnl <= -STOP_LOSS * 100:
                analysis.signal = "SELL"
                analysis.confidence = 85
                analysis.reasons = [f"損切りシグナル({current_pnl:.2f}%)"]
                return analysis

        # シグナル決定
        if has_position:
            if score < -20:
                analysis.signal = "SELL"
                analysis.confidence = min(90, 50 + abs(score))
            else:
                analysis.signal = "HOLD"
                analysis.confidence = 50
        else:
            if score > 20:
                analysis.signal = "BUY"
                analysis.confidence = min(90, 50 + score)
            else:
                analysis.signal = "WAIT"
                analysis.confidence = 50

        analysis.reasons = reasons if reasons else ["市場安定"]
        return analysis


# =============================================================================
# LINE通知
# =============================================================================

async def send_line_notify(message: str):
    """LINE通知を送信"""
    if not LINE_TOKEN:
        return
    try:
        async with aiohttp.ClientSession() as session:
            headers = {"Authorization": f"Bearer {LINE_TOKEN}"}
            data = {"message": message}
            await session.post(LINE_API, headers=headers, data=data)
    except:
        pass


# =============================================================================
# メイントレーダー
# =============================================================================

class UltimateTrader:
    def __init__(self):
        self.config = get_config()
        self.clients: Dict[str, BitFlyerClient] = {}
        self.trackers: Dict[str, PriceTracker] = {}
        self.last_trade: Dict[str, datetime] = {}
        self.failed: Dict[str, datetime] = {}

        self._balances: Dict[str, float] = {}
        self._balance_time: Optional[datetime] = None

        # 統計
        self.trades = 0
        self.wins = 0
        self.losses = 0
        self.pnl = 0.0
        self.start_value = 0.0
        self.start_time: Optional[datetime] = None
        self.trade_history: List[dict] = []

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

    async def cancel_orders_silent(self, pair: str):
        """エラー出力なしで注文キャンセル"""
        try:
            client = self.clients.get(pair)
            if client:
                # 直接APIを呼び出してエラーを無視
                await asyncio.wait_for(client.cancel_all_orders(pair), timeout=5)
        except:
            pass  # 完全に無視

    async def execute_buy(self, pair: str, analysis: AIAnalysis) -> bool:
        cfg = PAIRS.get(pair)
        client = self.clients.get(pair)
        if not cfg or not client or not self.can_trade(pair):
            return False

        balances = await self.get_balances(force=True)
        jpy = balances.get("JPY", 0) - MIN_CASH_RESERVE
        if jpy < 30:
            return False

        price = await self.get_price(pair)
        if not price:
            return False

        budget = jpy * 0.6  # 60%投入
        size = budget / price
        size = round(size, cfg.buy_decimals)
        if size < cfg.min_size:
            size = cfg.min_size

        cost = size * price * (1 + TRADING_FEE)
        if cost > jpy:
            return False

        currency = pair.replace("_JPY", "")
        logger.info("")
        logger.info(f"  🤖 AI判定: {currency} 【{analysis.signal}】 確信度:{analysis.confidence:.0f}%")
        for reason in analysis.reasons:
            logger.info(f"     └─ {reason}")
        logger.info(f"  📤 買い注文: {size} {currency} @ ¥{price:,.0f}")

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

                logger.info(f"  ✅ 購入成功！ {size} {currency} @ ¥{price:,.0f}")
                logger.info(f"     投資額: ¥{cost:,.0f}")

                # LINE通知
                await send_line_notify(f"\n🛒 購入完了\n{currency}: {size}個\n価格: ¥{price:,.0f}\n投資: ¥{cost:,.0f}")
                return True
            else:
                self.failed[pair] = datetime.now()
        except Exception as e:
            self.failed[pair] = datetime.now()
        return False

    async def execute_sell(self, pair: str, analysis: AIAnalysis) -> bool:
        cfg = PAIRS.get(pair)
        client = self.clients.get(pair)
        tracker = self.trackers.get(pair)
        if not cfg or not client or not tracker or not self.can_trade(pair):
            return False

        await self.cancel_orders_silent(pair)
        await asyncio.sleep(0.2)

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

        logger.info("")
        logger.info(f"  🤖 AI判定: {currency} 【{analysis.signal}】 確信度:{analysis.confidence:.0f}%")
        for reason in analysis.reasons:
            logger.info(f"     └─ {reason}")
        logger.info(f"  📤 売り注文: {size} {currency} @ ¥{price:,.0f}")

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
                pct = (price - entry) / entry * 100 if entry > 0 else 0

                self.pnl += net
                if net > 0:
                    self.wins += 1
                    emoji = "💰"
                    result = "利益確定"
                else:
                    self.losses += 1
                    emoji = "📉"
                    result = "損切り"

                tracker.clear()
                self._balances = {}

                self.trade_history.append({
                    "time": datetime.now(),
                    "pair": currency,
                    "pnl": net,
                    "pct": pct
                })

                logger.info(f"  {emoji} 売却完了！")
                logger.info(f"     売却額: ¥{price * size:,.0f}")
                logger.info(f"     損益: ¥{net:+,.0f} ({pct:+.2f}%)")
                logger.info(f"     累計: ¥{self.pnl:+,.0f}")

                # LINE通知
                await send_line_notify(
                    f"\n{emoji} {result}\n"
                    f"{currency}: {size}個売却\n"
                    f"損益: ¥{net:+,.0f} ({pct:+.2f}%)\n"
                    f"累計: ¥{self.pnl:+,.0f}"
                )
                return True
            else:
                self.failed[pair] = datetime.now()
        except Exception as e:
            self.failed[pair] = datetime.now()
        return False

    async def init(self) -> bool:
        api_key = self.config.bitflyer.api_key
        api_secret = self.config.bitflyer.api_secret

        if not api_key or not api_secret:
            logger.error("❌ APIキーが設定されていません")
            return False

        logger.info("")
        logger.info("╔════════════════════════════════════════════════════════════╗")
        logger.info("║  🏆 ULTIMATE AI TRADER v8.0 - 世界最強システム             ║")
        logger.info("╠════════════════════════════════════════════════════════════╣")
        logger.info(f"║  📈 利確: {TAKE_PROFIT*100:.1f}% | 📉 損切: {STOP_LOSS*100:.1f}%                           ║")
        logger.info(f"║  🤖 AI分析: RSI + トレンド + ボラティリティ               ║")
        logger.info(f"║  📱 LINE通知: {'有効' if LINE_TOKEN else '無効'}                                        ║")
        logger.info("╚════════════════════════════════════════════════════════════╝")
        logger.info("")

        for pair, cfg in sorted(PAIRS.items(), key=lambda x: x[1].priority):
            try:
                client = BitFlyerClient(api_key, api_secret, pair)
                ticker = await asyncio.wait_for(client.get_ticker(), timeout=10)
                if ticker and ticker.ltp > 0:
                    self.clients[pair] = client
                    self.trackers[pair] = PriceTracker()
                    self.trackers[pair].add(ticker.ltp)
                    min_jpy = cfg.min_size * ticker.ltp
                    logger.info(f"  ✓ {pair}: ¥{ticker.ltp:,.0f} (最小投資: ¥{min_jpy:,.0f})")
            except:
                pass

        if not self.clients:
            logger.error("❌ 接続できるペアがありません")
            return False

        # 静かにキャンセル
        for pair in self.clients:
            await self.cancel_orders_silent(pair)
        await asyncio.sleep(0.3)

        logger.info("")
        logger.info("  【保有資産】")
        balances = await self.get_balances(force=True)
        jpy = balances.get("JPY", 0)
        logger.info(f"  💴 日本円: ¥{jpy:,.0f}")

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
                    logger.info(f"  💎 {currency}: {sellable:.0f}個 (¥{value:,.0f}) [取引可能]")
                elif amount > 0.0001:
                    logger.info(f"  💎 {currency}: {amount:.6f} (¥{value:,.0f}) [少額]")

        self.start_value = total
        logger.info("")
        logger.info(f"  📊 総資産: ¥{total:,.0f}")
        logger.info("")

        # ETH/BTC到達目標
        eth_target = 5000
        btc_target = 15000
        logger.info(f"  🎯 目標: ETHまで ¥{max(0, eth_target - total):,.0f} / BTCまで ¥{max(0, btc_target - total):,.0f}")
        logger.info("")
        return True

    def show_status(self):
        """リアルタイムステータス表示"""
        logger.info("")
        logger.info("  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        logger.info("  📊 【リアルタイム戦況】")
        logger.info("  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

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

            if sellable >= cfg.min_size:
                analysis = tracker.analyze(has_position=True)
                pnl_pct = tracker.pnl(price) * 100
                pnl_jpy = (price - tracker.entry_price) * sellable
                total_unrealized += pnl_jpy

                if pnl_pct >= 0:
                    status_emoji = "📈"
                else:
                    status_emoji = "📉"

                signal_emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡", "WAIT": "⚪"}.get(analysis.signal, "⚪")

                logger.info(f"  {status_emoji} {currency}: ¥{price:,.0f} | 損益:{pnl_pct:+.2f}% (¥{pnl_jpy:+,.0f})")
                logger.info(f"     └─ AI: {signal_emoji}{analysis.signal} ({analysis.confidence:.0f}%) - {analysis.price_prediction}")

        logger.info("  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

        win_rate = (self.wins / self.trades * 100) if self.trades > 0 else 0

        logger.info(f"  💰 含み損益: ¥{total_unrealized:+,.0f}")
        logger.info(f"  💵 確定損益: ¥{self.pnl:+,.0f}")
        logger.info(f"  📊 取引: {self.trades}回 (勝率: {win_rate:.0f}%)")

        # 現金確認
        jpy = self._balances.get("JPY", 0)
        if jpy >= 5000:
            logger.info(f"  🎉 ETH取引可能！ (現金: ¥{jpy:,.0f})")
        elif jpy >= 300:
            logger.info(f"  💴 現金: ¥{jpy:,.0f} → ETHまで: ¥{5000-jpy:,.0f}")

        logger.info("  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    async def run(self):
        if not await self.init():
            return

        self.start_time = datetime.now()
        logger.info("  🚀 トレード開始！")
        logger.info("  📱 Ctrl+C で停止")
        logger.info("")

        await send_line_notify(
            f"\n🚀 AIトレーダー起動\n"
            f"総資産: ¥{self.start_value:,.0f}\n"
            f"目標: ETH/BTC取引"
        )

        tick = 0
        while True:
            try:
                tick += 1

                balances = await self.get_balances()
                jpy = balances.get("JPY", 0)

                for pair, cfg in sorted(PAIRS.items(), key=lambda x: x[1].priority):
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

                    # AI分析
                    analysis = tracker.analyze(has_position)

                    if has_position:
                        if analysis.signal == "SELL" and analysis.confidence >= 60:
                            await self.execute_sell(pair, analysis)
                    else:
                        if tracker.entry_price > 0:
                            tracker.clear()

                        min_cost = cfg.min_size * price * 1.01
                        if jpy - MIN_CASH_RESERVE >= min_cost:
                            if analysis.signal == "BUY" and analysis.confidence >= 50:
                                await self.execute_buy(pair, analysis)

                    await asyncio.sleep(PRICE_DELAY)

                if tick % STATUS_INTERVAL == 0:
                    self.show_status()

                await asyncio.sleep(LOOP_DELAY)

            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"  ⚠️ エラー: {e}")
                await asyncio.sleep(5)

        logger.info("")
        logger.info("  🛑 システム停止")
        self.show_status()

        # 終了通知
        elapsed = datetime.now() - self.start_time if self.start_time else timedelta(0)
        await send_line_notify(
            f"\n🛑 トレーダー停止\n"
            f"稼働時間: {elapsed.seconds//3600}時間{(elapsed.seconds%3600)//60}分\n"
            f"取引: {self.trades}回\n"
            f"損益: ¥{self.pnl:+,.0f}"
        )


async def main():
    # ログ設定（エラーメッセージを抑制）
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | {message}",
        level="INFO",
        filter=lambda record: "Cancel all failed" not in record["message"]
                          and "Request error: 200" not in record["message"]
    )

    # BitFlyerクライアントのログを抑制
    import logging
    logging.getLogger("src.api.bitflyer_client").setLevel(logging.CRITICAL)

    trader = UltimateTrader()
    await trader.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass

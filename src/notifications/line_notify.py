#!/usr/bin/env python3
"""
================================================================================
    📱 LINE通知モジュール (Messaging API版)
================================================================================
    - LINE Messaging API (Broadcast)
    - 取引通知
    - 日次レポート
================================================================================
"""

import os
import asyncio
import aiohttp
import json
from typing import Optional, List
from datetime import datetime
from dataclasses import dataclass
from loguru import logger


# LINE Messaging API エンドポイント
LINE_BROADCAST_URL = "https://api.line.me/v2/bot/message/broadcast"


@dataclass
class TradeNotification:
    """取引通知データ"""
    action: str  # "BUY" or "SELL"
    pair: str
    size: float
    price: float
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None
    reason: str = ""


@dataclass
class DailyReport:
    """日次レポート"""
    total_trades: int
    wins: int
    losses: int
    total_pnl: float
    best_trade: Optional[float] = None
    worst_trade: Optional[float] = None
    start_balance: float = 0
    end_balance: float = 0


class LineNotifier:
    """LINE Messaging API通知クラス"""

    def __init__(self, token: Optional[str] = None):
        self.token = token or os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
        self.enabled = bool(self.token)

        if self.enabled:
            logger.info("  📱 LINE通知: 有効 (Messaging API)")
        else:
            logger.info("  📱 LINE通知: 無効 (LINE_CHANNEL_ACCESS_TOKEN未設定)")

    async def _send(self, message: str) -> bool:
        """ブロードキャストでメッセージ送信"""
        if not self.enabled:
            return False

        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }

        # Flex Messageまたはテキストメッセージ
        payload = {
            "messages": [
                {
                    "type": "text",
                    "text": message.strip()
                }
            ]
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    LINE_BROADCAST_URL,
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        logger.debug("  LINE通知送信成功")
                        return True
                    else:
                        text = await response.text()
                        logger.debug(f"  LINE通知エラー: {response.status} - {text}")
                        return False
        except Exception as e:
            logger.debug(f"  LINE通知例外: {e}")
            return False

    async def notify_trade(self, trade: TradeNotification) -> bool:
        """取引通知"""
        currency = trade.pair.replace("_JPY", "")

        if trade.action == "BUY":
            emoji = "🛒"
            action_text = "購入"
        else:
            emoji = "💰"
            action_text = "売却"

        lines = [
            f"{emoji} {action_text}完了！",
            "",
            f"通貨: {currency}",
            f"数量: {trade.size}",
            f"価格: ¥{trade.price:,.0f}"
        ]

        if trade.pnl is not None:
            pnl_emoji = "📈" if trade.pnl >= 0 else "📉"
            lines.append("")
            lines.append(f"{pnl_emoji} 損益: ¥{trade.pnl:+,.0f} ({trade.pnl_pct:+.2f}%)")

        if trade.reason:
            lines.append(f"理由: {trade.reason}")

        lines.append(f"時刻: {datetime.now().strftime('%H:%M:%S')}")

        return await self._send("\n".join(lines))

    async def notify_daily_report(self, report: DailyReport) -> bool:
        """日次レポート通知"""
        win_rate = (report.wins / report.total_trades * 100) if report.total_trades > 0 else 0
        pnl_emoji = "💰" if report.total_pnl >= 0 else "📉"

        lines = [
            "📊 【日次レポート】",
            "",
            f"{pnl_emoji} 本日の損益: ¥{report.total_pnl:+,.0f}",
            "",
            f"📈 取引回数: {report.total_trades}回",
            f"✅ 勝ち: {report.wins}回",
            f"❌ 負け: {report.losses}回",
            f"📊 勝率: {win_rate:.1f}%"
        ]

        if report.best_trade is not None:
            lines.append("")
            lines.append(f"🏆 最大利益: ¥{report.best_trade:+,.0f}")

        if report.worst_trade is not None:
            lines.append(f"💥 最大損失: ¥{report.worst_trade:+,.0f}")

        if report.end_balance > 0:
            change = report.end_balance - report.start_balance
            change_pct = (change / report.start_balance * 100) if report.start_balance > 0 else 0
            lines.append("")
            lines.append(f"💴 資産: ¥{report.end_balance:,.0f} ({change_pct:+.2f}%)")

        return await self._send("\n".join(lines))

    async def notify_alert(self, title: str, message: str) -> bool:
        """アラート通知"""
        lines = [
            f"🚨 {title}",
            "",
            message,
            "",
            f"時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        ]
        return await self._send("\n".join(lines))

    async def notify_start(self, balance: float, pairs: list) -> bool:
        """起動通知"""
        pairs_str = ", ".join([p.replace("_JPY", "") for p in pairs])

        lines = [
            "🚀 AI TRADER v10.0 起動！",
            "",
            f"💴 資産: ¥{balance:,.0f}",
            f"📊 対象: {pairs_str}",
            f"⏰ 開始: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "世界最強AIトレーダーが稼働開始！"
        ]
        return await self._send("\n".join(lines))

    async def notify_stop(self, total_pnl: float, trades: int) -> bool:
        """停止通知"""
        pnl_emoji = "💰" if total_pnl >= 0 else "📉"

        lines = [
            "🛑 AI TRADER 停止",
            "",
            f"{pnl_emoji} 総損益: ¥{total_pnl:+,.0f}",
            f"📊 取引回数: {trades}回",
            f"⏰ 停止: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        ]
        return await self._send("\n".join(lines))

    async def notify_milestone(self, milestone: str, balance: float, pnl: float) -> bool:
        """マイルストーン通知"""
        lines = [
            "🎉 マイルストーン達成！",
            "",
            milestone,
            "",
            f"💴 現在資産: ¥{balance:,.0f}",
            f"📈 累計損益: ¥{pnl:+,.0f}"
        ]
        return await self._send("\n".join(lines))


class SmartNotifier:
    """スマート通知 (重要度に応じて通知頻度調整)"""

    def __init__(self, token: Optional[str] = None):
        self.notifier = LineNotifier(token)
        self._last_trade_notify: Optional[datetime] = None
        self._trade_count_since_notify: int = 0
        self._cumulative_pnl_since_notify: float = 0

        # 通知間隔設定
        self.min_notify_interval = 300  # 5分
        self.notify_on_big_trade = 50  # ¥50以上の損益で即通知
        self.notify_every_n_trades = 5  # 5取引ごとに通知

    @property
    def enabled(self) -> bool:
        return self.notifier.enabled

    async def on_trade(self, trade: TradeNotification) -> bool:
        """取引発生時"""
        now = datetime.now()
        self._trade_count_since_notify += 1
        if trade.pnl:
            self._cumulative_pnl_since_notify += trade.pnl

        # 即時通知条件
        should_notify = False

        # 大きな損益
        if trade.pnl and abs(trade.pnl) >= self.notify_on_big_trade:
            should_notify = True

        # N取引ごと
        if self._trade_count_since_notify >= self.notify_every_n_trades:
            should_notify = True

        # 最小間隔チェック
        if self._last_trade_notify:
            elapsed = (now - self._last_trade_notify).total_seconds()
            if elapsed < self.min_notify_interval and not (trade.pnl and abs(trade.pnl) >= self.notify_on_big_trade * 2):
                should_notify = False

        if should_notify:
            result = await self.notifier.notify_trade(trade)
            if result:
                self._last_trade_notify = now
                self._trade_count_since_notify = 0
                self._cumulative_pnl_since_notify = 0
            return result

        return False

    async def daily_report(self, report: DailyReport) -> bool:
        """日次レポート"""
        return await self.notifier.notify_daily_report(report)

    async def start(self, balance: float, pairs: list) -> bool:
        """起動通知"""
        return await self.notifier.notify_start(balance, pairs)

    async def stop(self, total_pnl: float, trades: int) -> bool:
        """停止通知"""
        return await self.notifier.notify_stop(total_pnl, trades)

    async def alert(self, title: str, message: str) -> bool:
        """アラート"""
        return await self.notifier.notify_alert(title, message)

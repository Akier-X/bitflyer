#!/usr/bin/env python3
"""
================================================================================
    📱 LINE通知モジュール
================================================================================
    - LINE Notify API
    - 取引通知
    - 日次レポート
================================================================================
"""

import os
import asyncio
import aiohttp
from typing import Optional
from datetime import datetime
from dataclasses import dataclass
from loguru import logger


LINE_NOTIFY_URL = "https://notify-api.line.me/api/notify"


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
    """LINE通知クラス"""

    def __init__(self, token: Optional[str] = None):
        self.token = token or os.getenv("LINE_NOTIFY_TOKEN")
        self.enabled = bool(self.token)

        if not self.enabled:
            logger.info("  📱 LINE通知: 無効 (LINE_NOTIFY_TOKEN未設定)")

    async def _send(self, message: str) -> bool:
        """メッセージ送信"""
        if not self.enabled:
            return False

        headers = {
            "Authorization": f"Bearer {self.token}"
        }
        data = {
            "message": message
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    LINE_NOTIFY_URL,
                    headers=headers,
                    data=data,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        logger.debug("  LINE通知送信成功")
                        return True
                    else:
                        logger.debug(f"  LINE通知エラー: {response.status}")
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

        message = f"""
{emoji} {action_text}完了！

通貨: {currency}
数量: {trade.size}
価格: ¥{trade.price:,.0f}"""

        if trade.pnl is not None:
            pnl_emoji = "📈" if trade.pnl >= 0 else "📉"
            message += f"""

{pnl_emoji} 損益: ¥{trade.pnl:+,.0f} ({trade.pnl_pct:+.2f}%)"""

        if trade.reason:
            message += f"""
理由: {trade.reason}"""

        message += f"""
時刻: {datetime.now().strftime('%H:%M:%S')}"""

        return await self._send(message)

    async def notify_daily_report(self, report: DailyReport) -> bool:
        """日次レポート通知"""
        win_rate = (report.wins / report.total_trades * 100) if report.total_trades > 0 else 0

        pnl_emoji = "💰" if report.total_pnl >= 0 else "📉"

        message = f"""
📊 【日次レポート】

{pnl_emoji} 本日の損益: ¥{report.total_pnl:+,.0f}

📈 取引回数: {report.total_trades}回
✅ 勝ち: {report.wins}回
❌ 負け: {report.losses}回
📊 勝率: {win_rate:.1f}%"""

        if report.best_trade is not None:
            message += f"""

🏆 最大利益: ¥{report.best_trade:+,.0f}"""

        if report.worst_trade is not None:
            message += f"""
💥 最大損失: ¥{report.worst_trade:+,.0f}"""

        if report.end_balance > 0:
            change = report.end_balance - report.start_balance
            change_pct = (change / report.start_balance * 100) if report.start_balance > 0 else 0
            message += f"""

💴 資産: ¥{report.end_balance:,.0f} ({change_pct:+.2f}%)"""

        return await self._send(message)

    async def notify_alert(self, title: str, message: str) -> bool:
        """アラート通知"""
        full_message = f"""
🚨 {title}

{message}

時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"""

        return await self._send(full_message)

    async def notify_start(self, balance: float, pairs: list) -> bool:
        """起動通知"""
        pairs_str = ", ".join([p.replace("_JPY", "") for p in pairs])

        message = f"""
🚀 AI TRADER 起動！

💴 資産: ¥{balance:,.0f}
📊 対象: {pairs_str}
⏰ 開始: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

最強AIトレーダーが稼働を開始しました！"""

        return await self._send(message)

    async def notify_stop(self, total_pnl: float, trades: int) -> bool:
        """停止通知"""
        pnl_emoji = "💰" if total_pnl >= 0 else "📉"

        message = f"""
🛑 AI TRADER 停止

{pnl_emoji} 総損益: ¥{total_pnl:+,.0f}
📊 取引回数: {trades}回
⏰ 停止: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"""

        return await self._send(message)

    async def notify_milestone(self, milestone: str, balance: float, pnl: float) -> bool:
        """マイルストーン通知"""
        message = f"""
🎉 マイルストーン達成！

{milestone}

💴 現在資産: ¥{balance:,.0f}
📈 累計損益: ¥{pnl:+,.0f}"""

        return await self._send(message)


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

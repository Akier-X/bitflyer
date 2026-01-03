"""
LINE Notification System
=========================
リアルタイム取引通知をLINEに送信
"""

import asyncio
import aiohttp
import requests
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import json
from loguru import logger


class NotificationType(Enum):
    """通知タイプ"""
    TRADE_EXECUTED = "trade_executed"
    SIGNAL_GENERATED = "signal_generated"
    RISK_ALERT = "risk_alert"
    PROFIT_MILESTONE = "profit_milestone"
    SYSTEM_STATUS = "system_status"
    ERROR = "error"
    DAILY_REPORT = "daily_report"


@dataclass
class TradeNotification:
    """取引通知データ"""
    action: str  # BUY / SELL / HOLD
    symbol: str
    price: float
    size: float
    confidence: float
    pnl: float = 0.0
    reasoning: str = ""
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class LINENotifier:
    """
    LINE Notify API を使用した通知システム

    設定方法:
    1. https://notify-bot.line.me/ にアクセス
    2. ログインして「トークンを発行する」
    3. トークンをLINE_NOTIFY_TOKENに設定
    """

    # LINE Notify API endpoint
    NOTIFY_API_URL = "https://notify-api.line.me/api/notify"

    def __init__(
        self,
        token: str,
        enable_trade_notifications: bool = True,
        enable_signal_notifications: bool = True,
        enable_risk_alerts: bool = True,
        min_confidence_to_notify: float = 0.6,
        min_pnl_to_notify: float = 0.001,  # 0.1%
    ):
        self.token = token
        self.enable_trade_notifications = enable_trade_notifications
        self.enable_signal_notifications = enable_signal_notifications
        self.enable_risk_alerts = enable_risk_alerts
        self.min_confidence_to_notify = min_confidence_to_notify
        self.min_pnl_to_notify = min_pnl_to_notify

        self.headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/x-www-form-urlencoded",
        }

        # Rate limiting
        self.last_notification_time = {}
        self.rate_limit_seconds = {
            NotificationType.TRADE_EXECUTED: 5,
            NotificationType.SIGNAL_GENERATED: 30,
            NotificationType.RISK_ALERT: 60,
            NotificationType.PROFIT_MILESTONE: 300,
            NotificationType.SYSTEM_STATUS: 300,
            NotificationType.ERROR: 60,
            NotificationType.DAILY_REPORT: 3600,
        }

        # Statistics
        self.notifications_sent = 0
        self.notifications_failed = 0

        logger.info("LINE Notifier initialized")

    def _can_send(self, notification_type: NotificationType) -> bool:
        """レート制限チェック"""
        now = datetime.now()
        last_time = self.last_notification_time.get(notification_type)

        if last_time is None:
            return True

        elapsed = (now - last_time).total_seconds()
        return elapsed >= self.rate_limit_seconds.get(notification_type, 60)

    def _format_trade_message(self, trade: TradeNotification) -> str:
        """取引通知メッセージをフォーマット"""
        action_emoji = {
            "BUY": "🟢",
            "SELL": "🔴",
            "HOLD": "⚪",
        }

        emoji = action_emoji.get(trade.action.upper(), "⚪")

        pnl_str = ""
        if trade.pnl != 0:
            pnl_emoji = "📈" if trade.pnl > 0 else "📉"
            pnl_str = f"\n{pnl_emoji} P&L: {trade.pnl:+.2%}"

        message = f"""
{emoji} {trade.action.upper()} シグナル

📊 {trade.symbol}
💰 価格: ¥{trade.price:,.0f}
📏 サイズ: {trade.size:.4f}
🎯 信頼度: {trade.confidence:.1%}{pnl_str}

💡 理由: {trade.reasoning[:100]}

⏰ {trade.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
"""
        return message.strip()

    def _format_risk_alert(self, alert_type: str, details: Dict) -> str:
        """リスクアラートメッセージをフォーマット"""
        message = f"""
⚠️ リスクアラート ⚠️

🚨 タイプ: {alert_type}
📊 詳細:
"""
        for key, value in details.items():
            if isinstance(value, float):
                message += f"  • {key}: {value:.4f}\n"
            else:
                message += f"  • {key}: {value}\n"

        message += f"\n⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

        return message.strip()

    def _format_daily_report(self, stats: Dict) -> str:
        """日次レポートメッセージをフォーマット"""
        pnl = stats.get('total_pnl', 0)
        pnl_emoji = "📈" if pnl >= 0 else "📉"

        message = f"""
📊 日次トレーディングレポート 📊

{pnl_emoji} 総損益: {pnl:+.2%}
💼 取引回数: {stats.get('trade_count', 0)}
✅ 勝率: {stats.get('win_rate', 0):.1%}
📈 最大利益: {stats.get('max_profit', 0):+.2%}
📉 最大損失: {stats.get('max_loss', 0):+.2%}
💰 残高: ¥{stats.get('balance', 0):,.0f}

🤖 AIシステム状態: {stats.get('system_status', 'OK')}
🔋 信頼度平均: {stats.get('avg_confidence', 0):.1%}

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        return message.strip()

    def _format_system_status(self, status: Dict) -> str:
        """システムステータスメッセージをフォーマット"""
        status_emoji = "✅" if status.get('healthy', True) else "❌"

        message = f"""
{status_emoji} システムステータス

🖥️ CPU: {status.get('cpu_usage', 0):.1f}%
💾 メモリ: {status.get('memory_usage', 0):.1f}%
🌐 API接続: {status.get('api_status', 'Unknown')}
🤖 AIエンジン: {status.get('engine_status', 'Unknown')}
📊 処理中シグナル: {status.get('pending_signals', 0)}

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        return message.strip()

    def send_sync(self, message: str) -> bool:
        """同期的に通知を送信"""
        try:
            response = requests.post(
                self.NOTIFY_API_URL,
                headers=self.headers,
                data={"message": message},
                timeout=10,
            )

            if response.status_code == 200:
                self.notifications_sent += 1
                logger.debug(f"LINE notification sent: {message[:50]}...")
                return True
            else:
                self.notifications_failed += 1
                logger.error(f"LINE notification failed: {response.status_code}")
                return False

        except Exception as e:
            self.notifications_failed += 1
            logger.error(f"LINE notification error: {e}")
            return False

    async def send_async(self, message: str) -> bool:
        """非同期で通知を送信"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.NOTIFY_API_URL,
                    headers=self.headers,
                    data={"message": message},
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as response:
                    if response.status == 200:
                        self.notifications_sent += 1
                        logger.debug(f"LINE notification sent: {message[:50]}...")
                        return True
                    else:
                        self.notifications_failed += 1
                        logger.error(f"LINE notification failed: {response.status}")
                        return False

        except Exception as e:
            self.notifications_failed += 1
            logger.error(f"LINE notification error: {e}")
            return False

    def notify_trade(
        self,
        trade: TradeNotification,
        force: bool = False,
    ) -> bool:
        """取引通知を送信"""
        if not self.enable_trade_notifications and not force:
            return False

        if trade.confidence < self.min_confidence_to_notify and not force:
            return False

        if not self._can_send(NotificationType.TRADE_EXECUTED) and not force:
            return False

        message = self._format_trade_message(trade)
        result = self.send_sync(message)

        if result:
            self.last_notification_time[NotificationType.TRADE_EXECUTED] = datetime.now()

        return result

    def notify_signal(
        self,
        action: str,
        symbol: str,
        price: float,
        confidence: float,
        reasoning: str = "",
    ) -> bool:
        """シグナル通知を送信"""
        if not self.enable_signal_notifications:
            return False

        if confidence < self.min_confidence_to_notify:
            return False

        if not self._can_send(NotificationType.SIGNAL_GENERATED):
            return False

        trade = TradeNotification(
            action=action,
            symbol=symbol,
            price=price,
            size=0,
            confidence=confidence,
            reasoning=reasoning,
        )

        message = self._format_trade_message(trade)
        message = message.replace("シグナル\n", "検出シグナル (未執行)\n")

        result = self.send_sync(message)

        if result:
            self.last_notification_time[NotificationType.SIGNAL_GENERATED] = datetime.now()

        return result

    def notify_risk_alert(
        self,
        alert_type: str,
        details: Dict,
        force: bool = False,
    ) -> bool:
        """リスクアラートを送信"""
        if not self.enable_risk_alerts and not force:
            return False

        if not self._can_send(NotificationType.RISK_ALERT) and not force:
            return False

        message = self._format_risk_alert(alert_type, details)
        result = self.send_sync(message)

        if result:
            self.last_notification_time[NotificationType.RISK_ALERT] = datetime.now()

        return result

    def notify_daily_report(self, stats: Dict) -> bool:
        """日次レポートを送信"""
        if not self._can_send(NotificationType.DAILY_REPORT):
            return False

        message = self._format_daily_report(stats)
        result = self.send_sync(message)

        if result:
            self.last_notification_time[NotificationType.DAILY_REPORT] = datetime.now()

        return result

    def notify_system_status(self, status: Dict) -> bool:
        """システムステータスを送信"""
        if not self._can_send(NotificationType.SYSTEM_STATUS):
            return False

        message = self._format_system_status(status)
        result = self.send_sync(message)

        if result:
            self.last_notification_time[NotificationType.SYSTEM_STATUS] = datetime.now()

        return result

    def notify_error(self, error_message: str, details: Dict = None) -> bool:
        """エラー通知を送信"""
        if not self._can_send(NotificationType.ERROR):
            return False

        message = f"""
❌ エラー発生 ❌

🔴 {error_message}
"""
        if details:
            message += "\n📋 詳細:\n"
            for key, value in details.items():
                message += f"  • {key}: {value}\n"

        message += f"\n⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

        result = self.send_sync(message)

        if result:
            self.last_notification_time[NotificationType.ERROR] = datetime.now()

        return result

    def notify_profit_milestone(self, milestone: str, current_pnl: float) -> bool:
        """利益マイルストーン通知"""
        if not self._can_send(NotificationType.PROFIT_MILESTONE):
            return False

        emoji = "🎉" if current_pnl > 0 else "😢"

        message = f"""
{emoji} マイルストーン達成 {emoji}

🏆 {milestone}
💰 現在の損益: {current_pnl:+.2%}

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        result = self.send_sync(message)

        if result:
            self.last_notification_time[NotificationType.PROFIT_MILESTONE] = datetime.now()

        return result

    def get_stats(self) -> Dict:
        """通知統計を取得"""
        return {
            'notifications_sent': self.notifications_sent,
            'notifications_failed': self.notifications_failed,
            'success_rate': self.notifications_sent / max(1, self.notifications_sent + self.notifications_failed),
        }


class NotificationManager:
    """
    通知マネージャー

    複数の通知チャネルを管理
    """

    def __init__(self):
        self.notifiers: Dict[str, Any] = {}
        self.enabled = True

    def add_line_notifier(
        self,
        token: str,
        name: str = "default",
        **kwargs,
    ) -> None:
        """LINE通知を追加"""
        self.notifiers[f"line_{name}"] = LINENotifier(token, **kwargs)
        logger.info(f"LINE notifier '{name}' added")

    def notify_all(self, message: str) -> Dict[str, bool]:
        """全チャネルに通知"""
        results = {}
        for name, notifier in self.notifiers.items():
            if hasattr(notifier, 'send_sync'):
                results[name] = notifier.send_sync(message)
        return results

    def notify_trade_all(self, trade: TradeNotification) -> Dict[str, bool]:
        """全チャネルに取引通知"""
        results = {}
        for name, notifier in self.notifiers.items():
            if hasattr(notifier, 'notify_trade'):
                results[name] = notifier.notify_trade(trade)
        return results

    def disable(self) -> None:
        """通知を無効化"""
        self.enabled = False

    def enable(self) -> None:
        """通知を有効化"""
        self.enabled = True


# Convenience function
def create_line_notifier(token: str, **kwargs) -> LINENotifier:
    """LINE通知インスタンスを生成"""
    return LINENotifier(token, **kwargs)


logger.info("LINE Notification module loaded")

"""
LINE Messaging API Notification System
========================================
LINE Notify終了に伴い、Messaging APIを使用
無料枠: 月200通（2024年時点）
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


class LINEMessagingAPI:
    """
    LINE Messaging API を使用した通知システム

    設定方法:
    1. https://developers.line.biz/console/ にアクセス
    2. プロバイダー作成 → Messaging APIチャネル作成
    3. チャネルアクセストークンを発行
    4. QRコードでBotを友だち追加
    5. トークンとユーザーIDを環境変数に設定

    無料枠: 月200通（追加は有料）
    """

    # LINE Messaging API endpoint
    API_URL = "https://api.line.me/v2/bot/message"

    def __init__(
        self,
        channel_access_token: str,
        user_id: str = None,  # 特定ユーザーに送信する場合
        enable_trade_notifications: bool = True,
        enable_signal_notifications: bool = True,
        enable_risk_alerts: bool = True,
        min_confidence_to_notify: float = 0.6,
        min_pnl_to_notify: float = 0.001,
    ):
        self.channel_access_token = channel_access_token
        self.user_id = user_id
        self.enable_trade_notifications = enable_trade_notifications
        self.enable_signal_notifications = enable_signal_notifications
        self.enable_risk_alerts = enable_risk_alerts
        self.min_confidence_to_notify = min_confidence_to_notify
        self.min_pnl_to_notify = min_pnl_to_notify

        self.headers = {
            "Authorization": f"Bearer {channel_access_token}",
            "Content-Type": "application/json",
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
        self.monthly_count = 0
        self.monthly_limit = 200  # 無料枠

        logger.info("LINE Messaging API initialized")

    def _can_send(self, notification_type: NotificationType) -> bool:
        """レート制限チェック"""
        # 月間制限チェック
        if self.monthly_count >= self.monthly_limit:
            logger.warning(f"Monthly limit reached ({self.monthly_count}/{self.monthly_limit})")
            return False

        now = datetime.now()
        last_time = self.last_notification_time.get(notification_type)

        if last_time is None:
            return True

        elapsed = (now - last_time).total_seconds()
        return elapsed >= self.rate_limit_seconds.get(notification_type, 60)

    def _create_text_message(self, text: str) -> Dict:
        """テキストメッセージを作成"""
        return {
            "type": "text",
            "text": text
        }

    def _create_flex_message(
        self,
        title: str,
        body_contents: List[Dict],
        header_color: str = "#27ACB2",
    ) -> Dict:
        """Flexメッセージ（リッチなカード形式）を作成"""
        return {
            "type": "flex",
            "altText": title,
            "contents": {
                "type": "bubble",
                "header": {
                    "type": "box",
                    "layout": "vertical",
                    "contents": [
                        {
                            "type": "text",
                            "text": title,
                            "weight": "bold",
                            "size": "lg",
                            "color": "#FFFFFF"
                        }
                    ],
                    "backgroundColor": header_color,
                    "paddingAll": "15px"
                },
                "body": {
                    "type": "box",
                    "layout": "vertical",
                    "contents": body_contents,
                    "paddingAll": "15px"
                },
                "footer": {
                    "type": "box",
                    "layout": "vertical",
                    "contents": [
                        {
                            "type": "text",
                            "text": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            "size": "xs",
                            "color": "#AAAAAA",
                            "align": "end"
                        }
                    ]
                }
            }
        }

    def _format_trade_message(self, trade: TradeNotification) -> Dict:
        """取引通知Flexメッセージを作成"""
        action_colors = {
            "BUY": "#00C853",
            "SELL": "#FF1744",
            "HOLD": "#9E9E9E",
        }
        action_emoji = {
            "BUY": "🟢",
            "SELL": "🔴",
            "HOLD": "⚪",
        }

        color = action_colors.get(trade.action.upper(), "#9E9E9E")
        emoji = action_emoji.get(trade.action.upper(), "⚪")

        body_contents = [
            {
                "type": "box",
                "layout": "horizontal",
                "contents": [
                    {"type": "text", "text": "通貨", "size": "sm", "color": "#AAAAAA", "flex": 2},
                    {"type": "text", "text": trade.symbol, "size": "sm", "flex": 3}
                ]
            },
            {
                "type": "box",
                "layout": "horizontal",
                "contents": [
                    {"type": "text", "text": "価格", "size": "sm", "color": "#AAAAAA", "flex": 2},
                    {"type": "text", "text": f"¥{trade.price:,.0f}", "size": "sm", "flex": 3}
                ],
                "margin": "md"
            },
            {
                "type": "box",
                "layout": "horizontal",
                "contents": [
                    {"type": "text", "text": "数量", "size": "sm", "color": "#AAAAAA", "flex": 2},
                    {"type": "text", "text": f"{trade.size:.4f}", "size": "sm", "flex": 3}
                ],
                "margin": "md"
            },
            {
                "type": "box",
                "layout": "horizontal",
                "contents": [
                    {"type": "text", "text": "信頼度", "size": "sm", "color": "#AAAAAA", "flex": 2},
                    {"type": "text", "text": f"{trade.confidence:.1%}", "size": "sm", "flex": 3}
                ],
                "margin": "md"
            },
        ]

        if trade.pnl != 0:
            pnl_color = "#00C853" if trade.pnl > 0 else "#FF1744"
            body_contents.append({
                "type": "box",
                "layout": "horizontal",
                "contents": [
                    {"type": "text", "text": "損益", "size": "sm", "color": "#AAAAAA", "flex": 2},
                    {"type": "text", "text": f"{trade.pnl:+.2%}", "size": "sm", "color": pnl_color, "flex": 3}
                ],
                "margin": "md"
            })

        if trade.reasoning:
            body_contents.append({
                "type": "separator",
                "margin": "lg"
            })
            body_contents.append({
                "type": "text",
                "text": trade.reasoning[:100],
                "size": "xs",
                "color": "#666666",
                "wrap": True,
                "margin": "md"
            })

        return self._create_flex_message(
            title=f"{emoji} {trade.action.upper()}",
            body_contents=body_contents,
            header_color=color,
        )

    def _format_risk_alert(self, alert_type: str, details: Dict) -> Dict:
        """リスクアラートFlexメッセージ"""
        body_contents = []

        for key, value in details.items():
            if isinstance(value, float):
                value_str = f"{value:.4f}"
            else:
                value_str = str(value)

            body_contents.append({
                "type": "box",
                "layout": "horizontal",
                "contents": [
                    {"type": "text", "text": key, "size": "sm", "color": "#AAAAAA", "flex": 2},
                    {"type": "text", "text": value_str, "size": "sm", "flex": 3}
                ],
                "margin": "md"
            })

        return self._create_flex_message(
            title=f"⚠️ {alert_type}",
            body_contents=body_contents,
            header_color="#FF6D00",
        )

    def _format_daily_report(self, stats: Dict) -> Dict:
        """日次レポートFlexメッセージ"""
        pnl = stats.get('total_pnl', 0)
        pnl_color = "#00C853" if pnl >= 0 else "#FF1744"

        body_contents = [
            {
                "type": "box",
                "layout": "horizontal",
                "contents": [
                    {"type": "text", "text": "総損益", "size": "sm", "color": "#AAAAAA", "flex": 2},
                    {"type": "text", "text": f"{pnl:+.2%}", "size": "lg", "weight": "bold", "color": pnl_color, "flex": 3}
                ]
            },
            {
                "type": "separator",
                "margin": "lg"
            },
            {
                "type": "box",
                "layout": "horizontal",
                "contents": [
                    {"type": "text", "text": "取引回数", "size": "sm", "color": "#AAAAAA", "flex": 2},
                    {"type": "text", "text": str(stats.get('trade_count', 0)), "size": "sm", "flex": 3}
                ],
                "margin": "md"
            },
            {
                "type": "box",
                "layout": "horizontal",
                "contents": [
                    {"type": "text", "text": "勝率", "size": "sm", "color": "#AAAAAA", "flex": 2},
                    {"type": "text", "text": f"{stats.get('win_rate', 0):.1%}", "size": "sm", "flex": 3}
                ],
                "margin": "md"
            },
        ]

        return self._create_flex_message(
            title="📊 日次レポート",
            body_contents=body_contents,
            header_color="#1976D2",
        )

    def send_push(self, user_id: str, messages: List[Dict]) -> bool:
        """プッシュメッセージを送信"""
        try:
            response = requests.post(
                f"{self.API_URL}/push",
                headers=self.headers,
                json={
                    "to": user_id,
                    "messages": messages[:5],  # 最大5メッセージ
                },
                timeout=10,
            )

            if response.status_code == 200:
                self.notifications_sent += 1
                self.monthly_count += 1
                logger.debug(f"LINE message sent to {user_id[:10]}...")
                return True
            else:
                self.notifications_failed += 1
                logger.error(f"LINE API error {response.status_code}: {response.text}")
                return False

        except Exception as e:
            self.notifications_failed += 1
            logger.error(f"LINE notification error: {e}")
            return False

    def send_broadcast(self, messages: List[Dict]) -> bool:
        """ブロードキャストメッセージを送信（全友だちに送信）"""
        try:
            response = requests.post(
                f"{self.API_URL}/broadcast",
                headers=self.headers,
                json={"messages": messages[:5]},
                timeout=10,
            )

            if response.status_code == 200:
                self.notifications_sent += 1
                self.monthly_count += 1
                logger.debug("LINE broadcast sent")
                return True
            else:
                self.notifications_failed += 1
                logger.error(f"LINE API error {response.status_code}: {response.text}")
                return False

        except Exception as e:
            self.notifications_failed += 1
            logger.error(f"LINE notification error: {e}")
            return False

    def send_text(self, text: str) -> bool:
        """テキストメッセージを送信"""
        message = self._create_text_message(text)

        if self.user_id:
            return self.send_push(self.user_id, [message])
        else:
            return self.send_broadcast([message])

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

        if self.user_id:
            result = self.send_push(self.user_id, [message])
        else:
            result = self.send_broadcast([message])

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
            reasoning=f"[シグナル検出] {reasoning}",
        )

        return self.notify_trade(trade, force=True)

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

        if self.user_id:
            result = self.send_push(self.user_id, [message])
        else:
            result = self.send_broadcast([message])

        if result:
            self.last_notification_time[NotificationType.RISK_ALERT] = datetime.now()

        return result

    def notify_daily_report(self, stats: Dict) -> bool:
        """日次レポートを送信"""
        if not self._can_send(NotificationType.DAILY_REPORT):
            return False

        message = self._format_daily_report(stats)

        if self.user_id:
            result = self.send_push(self.user_id, [message])
        else:
            result = self.send_broadcast([message])

        if result:
            self.last_notification_time[NotificationType.DAILY_REPORT] = datetime.now()

        return result

    def notify_error(self, error_message: str, details: Dict = None) -> bool:
        """エラー通知を送信"""
        if not self._can_send(NotificationType.ERROR):
            return False

        text = f"❌ エラー発生\n\n{error_message}"

        if details:
            text += "\n\n詳細:"
            for key, value in details.items():
                text += f"\n• {key}: {value}"

        text += f"\n\n⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

        result = self.send_text(text)

        if result:
            self.last_notification_time[NotificationType.ERROR] = datetime.now()

        return result

    def notify_startup(self, config: Dict = None) -> bool:
        """起動通知を送信"""
        text = """🚀 AI Trading Bot 起動

📊 システム稼働開始
🤖 世界最強AIトレーダー

"""
        if config:
            text += f"モード: {'Paper' if config.get('paper_trading') else 'Live'}\n"
            text += f"通貨: {config.get('product', 'BTC_JPY')}\n"

        text += f"\n⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

        return self.send_text(text)

    def notify_shutdown(self, stats: Dict = None) -> bool:
        """終了通知を送信"""
        text = "🛑 AI Trading Bot 停止\n\n"

        if stats:
            text += f"取引回数: {stats.get('total_trades', 0)}\n"
            text += f"勝率: {stats.get('win_rate', 0):.1%}\n"
            text += f"総損益: {stats.get('total_pnl', 0):+.2%}\n"

        text += f"\n⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

        return self.send_text(text)

    def get_stats(self) -> Dict:
        """通知統計を取得"""
        return {
            'notifications_sent': self.notifications_sent,
            'notifications_failed': self.notifications_failed,
            'monthly_count': self.monthly_count,
            'monthly_limit': self.monthly_limit,
            'remaining': self.monthly_limit - self.monthly_count,
            'success_rate': self.notifications_sent / max(1, self.notifications_sent + self.notifications_failed),
        }


# 旧LINE Notifyとの互換性レイヤー
class LINENotifier(LINEMessagingAPI):
    """LINE Notifyとの互換性を保つエイリアス"""

    def __init__(self, token: str, **kwargs):
        # tokenがMessaging APIのチャネルアクセストークンとして使用
        super().__init__(channel_access_token=token, **kwargs)

    def send_sync(self, message: str) -> bool:
        """LINE Notify互換のsend_sync"""
        return self.send_text(message)


def create_line_notifier(
    channel_access_token: str,
    user_id: str = None,
    **kwargs,
) -> LINEMessagingAPI:
    """LINE Messaging APIインスタンスを生成"""
    return LINEMessagingAPI(
        channel_access_token=channel_access_token,
        user_id=user_id,
        **kwargs,
    )


logger.info("LINE Messaging API module loaded")

"""
Live Trading Bot
=================
リアルタイムトレーディングBotの実行エンジン
"""

import asyncio
import signal
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from collections import deque
import traceback
from loguru import logger

# Import components
sys.path.insert(0, '/home/user/bitflyer')

from config.settings import Config, get_config
from src.api.bitflyer_client import BitFlyerClient, MockBitFlyerClient, OrderSide, OrderType
from src.notifications.line_messaging import LINEMessagingAPI, TradeNotification


@dataclass
class TradeRecord:
    """取引記録"""
    timestamp: datetime
    action: str
    price: float
    size: float
    order_id: str
    pnl: float = 0.0
    confidence: float = 0.0
    reasoning: str = ""


@dataclass
class TradingStats:
    """取引統計"""
    start_time: datetime = field(default_factory=datetime.now)
    total_trades: int = 0
    winning_trades: int = 0
    total_pnl: float = 0.0
    max_pnl: float = 0.0
    min_pnl: float = 0.0
    max_drawdown: float = 0.0
    current_position: float = 0.0
    current_position_value: float = 0.0

    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return self.winning_trades / self.total_trades

    @property
    def running_hours(self) -> float:
        return (datetime.now() - self.start_time).total_seconds() / 3600

    def to_dict(self) -> Dict:
        return {
            'start_time': self.start_time.isoformat(),
            'running_hours': self.running_hours,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'win_rate': self.win_rate,
            'total_pnl': self.total_pnl,
            'max_pnl': self.max_pnl,
            'min_pnl': self.min_pnl,
            'max_drawdown': self.max_drawdown,
            'current_position': self.current_position,
        }


class LiveTrader:
    """
    ライブトレーディングBot

    AIエンジンとbitFlyer APIを接続し、リアルタイム取引を実行
    """

    def __init__(self, config: Config = None):
        self.config = config or get_config()

        # API Client
        if self.config.trading.paper_trading:
            logger.info("🧪 Paper Trading Mode")
            self.client = MockBitFlyerClient(
                initial_balance=1000000,
                product_code=self.config.bitflyer.product_code,
            )
        else:
            logger.info("💰 Live Trading Mode")
            self.client = BitFlyerClient(
                api_key=self.config.bitflyer.api_key,
                api_secret=self.config.bitflyer.api_secret,
                product_code=self.config.bitflyer.product_code,
            )

        # LINE Notification (Messaging API対応)
        self.notifier = None
        if self.config.line.is_configured:
            if self.config.line.use_messaging_api:
                # LINE Messaging API (推奨)
                self.notifier = LINEMessagingAPI(
                    channel_access_token=self.config.line.channel_access_token,
                    user_id=self.config.line.user_id if self.config.line.user_id else None,
                    enable_trade_notifications=self.config.line.enable_trade_notifications,
                    enable_signal_notifications=self.config.line.enable_signal_notifications,
                    enable_risk_alerts=self.config.line.enable_risk_alerts,
                    min_confidence_to_notify=self.config.line.min_confidence_to_notify,
                    min_pnl_to_notify=self.config.line.min_pnl_to_notify,
                )
                logger.info("📱 LINE Messaging API notifications enabled")
            else:
                # Legacy: LINE Notify (2025年3月31日終了予定)
                try:
                    from src.notifications.line_notify import LINENotifier
                    self.notifier = LINENotifier(
                        token=self.config.line.notify_token,
                        enable_trade_notifications=self.config.line.enable_trade_notifications,
                        enable_signal_notifications=self.config.line.enable_signal_notifications,
                        enable_risk_alerts=self.config.line.enable_risk_alerts,
                        min_confidence_to_notify=self.config.line.min_confidence_to_notify,
                    )
                    logger.warning("⚠️ LINE Notify enabled (終了予定: 2025年3月31日)")
                    logger.warning("⚠️ LINE Messaging APIへの移行を推奨します")
                except Exception as e:
                    logger.error(f"LINE Notify initialization failed: {e}")

        # AI Engine (lazy load)
        self.ai_engine = None

        # State
        self.running = False
        self.stats = TradingStats()
        self.trade_history: deque = deque(maxlen=1000)
        self.last_trade_time: Optional[datetime] = None
        self.trades_this_hour = 0
        self.hour_start = datetime.now()

        # Risk management
        self.daily_pnl = 0.0
        self.daily_start = datetime.now().date()
        self.peak_equity = 1.0
        self.current_equity = 1.0

        logger.info("LiveTrader initialized")

    def _load_ai_engine(self):
        """AIエンジンを遅延読み込み"""
        if self.ai_engine is None:
            try:
                from src.ml.ultimate_trading_engine import UltimateTradingEngine
                self.ai_engine = UltimateTradingEngine(
                    feature_dim=self.config.ai.feature_dim,
                    seq_len=self.config.ai.seq_len,
                )
                logger.info("🤖 AI Engine loaded")
            except Exception as e:
                logger.error(f"Failed to load AI engine: {e}")
                logger.info("Using fallback simple strategy")
                self.ai_engine = None

    async def _get_market_data(self) -> Optional[Dict]:
        """市場データを取得"""
        try:
            ticker = await self.client.get_ticker()
            if ticker is None:
                return None

            order_book = await self.client.get_order_book()

            return {
                'price': ticker.ltp,
                'bid': ticker.best_bid,
                'ask': ticker.best_ask,
                'spread': ticker.spread,
                'volume': ticker.volume,
                'order_book': {
                    'bids': order_book.bids[:10] if order_book else [],
                    'asks': order_book.asks[:10] if order_book else [],
                } if order_book else None,
                'timestamp': ticker.timestamp,
            }
        except Exception as e:
            logger.error(f"Failed to get market data: {e}")
            return None

    def _can_trade(self) -> tuple:
        """取引可能かチェック"""
        now = datetime.now()

        # Reset hourly counter
        if (now - self.hour_start).total_seconds() > 3600:
            self.trades_this_hour = 0
            self.hour_start = now

        # Reset daily PnL
        if now.date() != self.daily_start:
            self.daily_pnl = 0.0
            self.daily_start = now.date()

        # Check trade interval
        if self.last_trade_time:
            elapsed = (now - self.last_trade_time).total_seconds()
            if elapsed < self.config.trading.min_trade_interval:
                return False, f"Trade interval not met ({elapsed:.0f}s < {self.config.trading.min_trade_interval}s)"

        # Check hourly limit
        if self.trades_this_hour >= self.config.trading.max_trades_per_hour:
            return False, f"Hourly limit reached ({self.trades_this_hour})"

        # Check daily loss limit
        if self.daily_pnl < -self.config.trading.daily_loss_limit:
            return False, f"Daily loss limit reached ({self.daily_pnl:.2%})"

        # Check drawdown
        if self.peak_equity > 0:
            drawdown = (self.peak_equity - self.current_equity) / self.peak_equity
            if drawdown > self.config.trading.max_drawdown:
                return False, f"Max drawdown reached ({drawdown:.2%})"

        return True, "OK"

    async def _execute_trade(
        self,
        action: int,
        confidence: float,
        price: float,
        reasoning: str = "",
    ) -> Optional[str]:
        """取引を実行"""
        # action: 0=SELL, 1=HOLD, 2=BUY
        if action == 1:
            return None

        side = OrderSide.BUY if action == 2 else OrderSide.SELL

        # Calculate position size
        base_size = self.config.trading.max_position_size
        size = base_size * confidence * 0.5  # Scale by confidence

        # Ensure minimum size
        size = max(self.config.trading.min_order_size, size)
        size = min(self.config.trading.max_position_size, size)

        # Round to valid precision
        size = round(size, 4)

        try:
            order_id = await self.client.send_order(
                side=side,
                size=size,
                order_type=OrderType.MARKET,
            )

            if order_id:
                self.last_trade_time = datetime.now()
                self.trades_this_hour += 1
                self.stats.total_trades += 1

                # Update position
                if side == OrderSide.BUY:
                    self.stats.current_position += size
                else:
                    self.stats.current_position -= size

                # Record trade
                trade = TradeRecord(
                    timestamp=datetime.now(),
                    action=side.value,
                    price=price,
                    size=size,
                    order_id=order_id,
                    confidence=confidence,
                    reasoning=reasoning,
                )
                self.trade_history.append(trade)

                logger.info(f"✅ Trade executed: {side.value} {size} @ {price}")

                # Notify
                if self.notifier:
                    self.notifier.notify_trade(TradeNotification(
                        action=side.value,
                        symbol=self.config.bitflyer.product_code,
                        price=price,
                        size=size,
                        confidence=confidence,
                        reasoning=reasoning,
                    ))

                return order_id

        except Exception as e:
            logger.error(f"Trade execution failed: {e}")

            if self.notifier:
                self.notifier.notify_error(
                    f"Trade failed: {side.value} {size}",
                    {"error": str(e), "price": price},
                )

        return None

    async def _trading_loop(self):
        """メイン取引ループ"""
        logger.info("🚀 Trading loop started")

        self._load_ai_engine()

        # Send startup notification
        if self.notifier:
            startup_config = {
                'paper_trading': self.config.trading.paper_trading,
                'product': self.config.bitflyer.product_code,
            }
            if hasattr(self.notifier, 'notify_startup'):
                self.notifier.notify_startup(startup_config)
            elif hasattr(self.notifier, 'send_text'):
                self.notifier.send_text(f"""🚀 Trading Bot Started

📊 Product: {self.config.bitflyer.product_code}
🧪 Mode: {'Paper' if self.config.trading.paper_trading else 'Live'}
🤖 AI Engine: {'Loaded' if self.ai_engine else 'Fallback'}
⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}""")

        tick_count = 0

        while self.running:
            try:
                tick_count += 1

                # Get market data
                market_data = await self._get_market_data()
                if market_data is None:
                    await asyncio.sleep(5)
                    continue

                price = market_data['price']

                # Check if we can trade
                can_trade, reason = self._can_trade()

                if not can_trade:
                    logger.debug(f"Cannot trade: {reason}")
                    await asyncio.sleep(1)
                    continue

                # Generate signal
                if self.ai_engine:
                    signal = self.ai_engine.generate_signal(
                        price=price,
                        volume=market_data['volume'],
                        high=price * 1.001,
                        low=price * 0.999,
                        order_book=market_data.get('order_book'),
                    )

                    action = signal.action
                    confidence = signal.confidence
                    reasoning = signal.reasoning
                else:
                    # Fallback: random with hold bias
                    import random
                    action = 1  # Hold by default
                    confidence = 0.3
                    reasoning = "Fallback strategy"

                # Only trade if confidence meets threshold
                if confidence >= self.config.trading.min_confidence and action != 1:
                    await self._execute_trade(
                        action=action,
                        confidence=confidence,
                        price=price,
                        reasoning=reasoning,
                    )

                # Update equity (simplified)
                self.current_equity = 1.0 + self.stats.total_pnl
                if self.current_equity > self.peak_equity:
                    self.peak_equity = self.current_equity

                # Periodic status log
                if tick_count % 60 == 0:
                    logger.info(
                        f"📊 Status: Price={price:,.0f} | "
                        f"Trades={self.stats.total_trades} | "
                        f"Position={self.stats.current_position:.4f} | "
                        f"PnL={self.stats.total_pnl:.2%}"
                    )

                # Daily report
                if tick_count % 3600 == 0 and self.notifier:
                    self.notifier.notify_daily_report(self.stats.to_dict())

                await asyncio.sleep(1)

            except Exception as e:
                logger.error(f"Trading loop error: {e}")
                logger.error(traceback.format_exc())

                if self.notifier:
                    self.notifier.notify_error(
                        "Trading loop error",
                        {"error": str(e)},
                    )

                await asyncio.sleep(5)

        logger.info("Trading loop stopped")

    async def start(self):
        """Botを開始"""
        if self.running:
            logger.warning("Bot is already running")
            return

        self.running = True
        self.stats = TradingStats()

        logger.info("=" * 50)
        logger.info("  ULTIMATE AI TRADING BOT")
        logger.info("  世界最強AIトレーディングBot")
        logger.info("=" * 50)
        logger.info(f"  Product: {self.config.bitflyer.product_code}")
        logger.info(f"  Mode: {'Paper' if self.config.trading.paper_trading else 'Live'}")
        logger.info(f"  Max Position: {self.config.trading.max_position_size}")
        logger.info(f"  Confidence: {self.config.trading.min_confidence}-{self.config.trading.max_confidence}")
        logger.info("=" * 50)

        # Setup signal handlers (Unix only - Windows uses KeyboardInterrupt)
        if sys.platform != 'win32':
            try:
                for sig in (signal.SIGTERM, signal.SIGINT):
                    asyncio.get_event_loop().add_signal_handler(
                        sig,
                        lambda: asyncio.create_task(self.stop()),
                    )
            except NotImplementedError:
                # Windows doesn't support add_signal_handler
                pass

        # Connect WebSocket
        if hasattr(self.client, 'connect_websocket'):
            try:
                await self.client.connect_websocket()
            except Exception as e:
                logger.warning(f"WebSocket connection failed: {e}")

        # Start trading loop
        await self._trading_loop()

    async def stop(self):
        """Botを停止"""
        logger.info("Stopping bot...")
        self.running = False

        # Cancel all orders
        try:
            await self.client.cancel_all_orders()
        except Exception as e:
            logger.error(f"Failed to cancel orders: {e}")

        # Disconnect WebSocket
        if hasattr(self.client, 'disconnect_websocket'):
            await self.client.disconnect_websocket()

        # Send shutdown notification
        if self.notifier:
            shutdown_stats = self.stats.to_dict()
            if hasattr(self.notifier, 'notify_shutdown'):
                self.notifier.notify_shutdown(shutdown_stats)
            elif hasattr(self.notifier, 'send_text'):
                self.notifier.send_text(f"""🛑 Trading Bot Stopped

📊 Final Statistics:
• Total Trades: {self.stats.total_trades}
• Win Rate: {self.stats.win_rate:.1%}
• Total PnL: {self.stats.total_pnl:.2%}
• Running Time: {self.stats.running_hours:.1f} hours

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}""")

        logger.info("Bot stopped")

    def get_status(self) -> Dict:
        """ステータスを取得"""
        return {
            'running': self.running,
            'stats': self.stats.to_dict(),
            'config': self.config.to_dict(),
            'client': self.client.get_stats() if hasattr(self.client, 'get_stats') else {},
            'ai_engine': self.ai_engine is not None,
            'notifier': self.notifier is not None,
        }


async def run_trader():
    """トレーダーを実行"""
    config = get_config()
    trader = LiveTrader(config)

    try:
        await trader.start()
    except KeyboardInterrupt:
        await trader.stop()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        logger.error(traceback.format_exc())
        await trader.stop()


if __name__ == "__main__":
    asyncio.run(run_trader())

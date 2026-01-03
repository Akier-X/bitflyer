"""
Multi-Asset Trading System
===========================
全通貨ペア対応・高速利益追求型AIトレーダー

Features:
- 全bitFlyer通貨ペア同時監視
- クロスペアアービトラージ検出
- 動的資金配分
- 高頻度取引対応
- リアルタイムポートフォリオ最適化
"""

import asyncio
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import deque
from enum import Enum
import traceback
from loguru import logger

sys.path.insert(0, '/home/user/bitflyer')

from config.settings import Config, get_config
from src.api.bitflyer_client import BitFlyerClient, MockBitFlyerClient, OrderSide, OrderType
from src.notifications.line_messaging import LINEMessagingAPI, TradeNotification


class TradingPair(Enum):
    """bitFlyer取引ペア"""
    BTC_JPY = "BTC_JPY"
    ETH_JPY = "ETH_JPY"
    XRP_JPY = "XRP_JPY"
    XLM_JPY = "XLM_JPY"
    MONA_JPY = "MONA_JPY"
    ETH_BTC = "ETH_BTC"
    BCH_BTC = "BCH_BTC"
    # Lightning FX
    FX_BTC_JPY = "FX_BTC_JPY"


@dataclass
class AssetState:
    """各通貨ペアの状態"""
    pair: str
    price: float = 0.0
    bid: float = 0.0
    ask: float = 0.0
    spread: float = 0.0
    volume_24h: float = 0.0
    change_24h: float = 0.0
    position: float = 0.0
    unrealized_pnl: float = 0.0
    last_signal: int = 1  # 0=SELL, 1=HOLD, 2=BUY
    signal_confidence: float = 0.0
    last_trade_time: Optional[datetime] = None
    trade_count: int = 0
    total_pnl: float = 0.0

    @property
    def spread_pct(self) -> float:
        if self.price > 0:
            return self.spread / self.price
        return 0.0


@dataclass
class ArbitrageOpportunity:
    """アービトラージ機会"""
    pair1: str
    pair2: str
    pair3: Optional[str]
    profit_pct: float
    direction: str  # "forward" or "reverse"
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class PortfolioState:
    """ポートフォリオ状態"""
    total_value_jpy: float = 0.0
    cash_jpy: float = 0.0
    btc_value: float = 0.0
    eth_value: float = 0.0
    other_value: float = 0.0
    total_pnl: float = 0.0
    total_pnl_pct: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0


class MultiAssetTrader:
    """
    マルチアセットAIトレーダー

    全通貨ペアを同時監視し、最高の機会を狙う
    """

    # 取引可能ペア（優先度順）
    TRADING_PAIRS = [
        "BTC_JPY",      # メイン - 流動性最高
        "ETH_JPY",      # サブ - 高流動性
        "XRP_JPY",      # 高ボラティリティ
        "FX_BTC_JPY",   # Lightning FX - レバレッジ
        "MONA_JPY",     # 高ボラティリティ
        "XLM_JPY",      #
        "ETH_BTC",      # クロスペア
        "BCH_BTC",      # クロスペア
    ]

    # 各ペアの最小注文サイズ
    MIN_ORDER_SIZES = {
        "BTC_JPY": 0.001,
        "ETH_JPY": 0.01,
        "XRP_JPY": 1.0,
        "XLM_JPY": 1.0,
        "MONA_JPY": 1.0,
        "FX_BTC_JPY": 0.001,
        "ETH_BTC": 0.01,
        "BCH_BTC": 0.01,
    }

    def __init__(self, config: Config = None):
        self.config = config or get_config()

        # 取引するペアのリスト（設定から取得）
        self.active_pairs = self._get_active_pairs()

        # クライアント（ペアごと）
        self.clients: Dict[str, Any] = {}
        self._init_clients()

        # LINE通知
        self.notifier = self._init_notifier()

        # AI Engine
        self.ai_engine = None

        # 状態管理
        self.running = False
        self.asset_states: Dict[str, AssetState] = {
            pair: AssetState(pair=pair) for pair in self.active_pairs
        }
        self.portfolio = PortfolioState()
        self.arbitrage_opportunities: deque = deque(maxlen=100)

        # 取引履歴
        self.trade_history: deque = deque(maxlen=1000)

        # 統計
        self.start_time = datetime.now()
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0

        # 高速取引用
        self.tick_count = 0
        self.last_arbitrage_check = datetime.now()

        logger.info(f"MultiAssetTrader initialized with {len(self.active_pairs)} pairs")
        logger.info(f"Active pairs: {self.active_pairs}")

    def _get_active_pairs(self) -> List[str]:
        """アクティブな取引ペアを取得"""
        # 環境変数から取得、なければ全ペア
        import os
        pairs_str = os.environ.get("TRADING_PAIRS", "")
        if pairs_str:
            return [p.strip() for p in pairs_str.split(",")]
        return self.TRADING_PAIRS[:5]  # デフォルトは上位5ペア

    def _init_clients(self):
        """各ペア用のクライアントを初期化"""
        for pair in self.active_pairs:
            if self.config.trading.paper_trading:
                self.clients[pair] = MockBitFlyerClient(
                    initial_balance=1000000,
                    product_code=pair,
                )
            else:
                self.clients[pair] = BitFlyerClient(
                    api_key=self.config.bitflyer.api_key,
                    api_secret=self.config.bitflyer.api_secret,
                    product_code=pair,
                )
        logger.info(f"Initialized {len(self.clients)} trading clients")

    def _init_notifier(self) -> Optional[LINEMessagingAPI]:
        """通知システムを初期化"""
        if self.config.line.is_configured and self.config.line.use_messaging_api:
            return LINEMessagingAPI(
                channel_access_token=self.config.line.channel_access_token,
                user_id=self.config.line.user_id if self.config.line.user_id else None,
                enable_trade_notifications=self.config.line.enable_trade_notifications,
                enable_signal_notifications=self.config.line.enable_signal_notifications,
                enable_risk_alerts=self.config.line.enable_risk_alerts,
            )
        return None

    def _load_ai_engine(self):
        """AIエンジンをロード"""
        if self.ai_engine is None:
            try:
                from src.ml.ultimate_trading_engine import UltimateTradingEngine
                self.ai_engine = UltimateTradingEngine(
                    feature_dim=self.config.ai.feature_dim,
                    seq_len=self.config.ai.seq_len,
                )
                logger.info("🤖 AI Engine loaded for multi-asset trading")
            except Exception as e:
                logger.error(f"Failed to load AI engine: {e}")

    async def _fetch_all_prices(self) -> Dict[str, Dict]:
        """全ペアの価格を並列取得"""
        tasks = []
        for pair in self.active_pairs:
            tasks.append(self._fetch_price(pair))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        prices = {}
        for pair, result in zip(self.active_pairs, results):
            if isinstance(result, Exception):
                logger.warning(f"Failed to fetch {pair}: {result}")
            else:
                prices[pair] = result

        return prices

    async def _fetch_price(self, pair: str) -> Dict:
        """単一ペアの価格を取得"""
        client = self.clients.get(pair)
        if not client:
            return {}

        try:
            ticker = await client.get_ticker()
            if ticker:
                return {
                    'price': ticker.ltp,
                    'bid': ticker.best_bid,
                    'ask': ticker.best_ask,
                    'spread': ticker.spread,
                    'volume': ticker.volume,
                    'timestamp': ticker.timestamp,
                }
        except Exception as e:
            logger.debug(f"Price fetch error for {pair}: {e}")

        return {}

    def _update_asset_states(self, prices: Dict[str, Dict]):
        """アセット状態を更新"""
        for pair, price_data in prices.items():
            if pair in self.asset_states and price_data:
                state = self.asset_states[pair]
                state.price = price_data.get('price', 0)
                state.bid = price_data.get('bid', 0)
                state.ask = price_data.get('ask', 0)
                state.spread = price_data.get('spread', 0)
                state.volume_24h = price_data.get('volume', 0)

    def _detect_arbitrage(self) -> List[ArbitrageOpportunity]:
        """アービトラージ機会を検出"""
        opportunities = []

        # 三角アービトラージ: BTC/JPY - ETH/BTC - ETH/JPY
        if all(p in self.asset_states for p in ["BTC_JPY", "ETH_BTC", "ETH_JPY"]):
            btc_jpy = self.asset_states["BTC_JPY"]
            eth_btc = self.asset_states["ETH_BTC"]
            eth_jpy = self.asset_states["ETH_JPY"]

            if btc_jpy.price > 0 and eth_btc.price > 0 and eth_jpy.price > 0:
                # Forward: JPY -> BTC -> ETH -> JPY
                forward_rate = (1 / btc_jpy.ask) * (1 / eth_btc.ask) * eth_jpy.bid
                if forward_rate > 1.001:  # 0.1%以上の利益
                    opportunities.append(ArbitrageOpportunity(
                        pair1="BTC_JPY",
                        pair2="ETH_BTC",
                        pair3="ETH_JPY",
                        profit_pct=(forward_rate - 1) * 100,
                        direction="forward",
                    ))

                # Reverse: JPY -> ETH -> BTC -> JPY
                reverse_rate = (1 / eth_jpy.ask) * eth_btc.bid * btc_jpy.bid
                if reverse_rate > 1.001:
                    opportunities.append(ArbitrageOpportunity(
                        pair1="ETH_JPY",
                        pair2="ETH_BTC",
                        pair3="BTC_JPY",
                        profit_pct=(reverse_rate - 1) * 100,
                        direction="reverse",
                    ))

        return opportunities

    def _calculate_opportunity_score(self, pair: str) -> Tuple[float, str]:
        """各ペアの取引機会スコアを計算"""
        state = self.asset_states.get(pair)
        if not state or state.price == 0:
            return 0.0, "no_data"

        score = 0.0
        reasons = []

        # 1. スプレッドスコア（狭いほど良い）
        if state.spread_pct < 0.001:  # 0.1%未満
            score += 30
            reasons.append("tight_spread")
        elif state.spread_pct < 0.003:
            score += 20
        elif state.spread_pct < 0.005:
            score += 10

        # 2. ボリュームスコア
        if state.volume_24h > 1000:
            score += 20
            reasons.append("high_volume")
        elif state.volume_24h > 100:
            score += 10

        # 3. AIシグナル信頼度
        if state.signal_confidence > 0.8:
            score += 40
            reasons.append("strong_signal")
        elif state.signal_confidence > 0.6:
            score += 25
        elif state.signal_confidence > 0.5:
            score += 10

        # 4. 価格変動（ボラティリティ）
        if abs(state.change_24h) > 5:
            score += 15
            reasons.append("high_volatility")
        elif abs(state.change_24h) > 2:
            score += 10

        return score, ",".join(reasons) if reasons else "normal"

    async def _generate_signals(self, prices: Dict[str, Dict]):
        """全ペアのAIシグナルを生成"""
        if not self.ai_engine:
            return

        for pair, price_data in prices.items():
            if not price_data:
                continue

            state = self.asset_states.get(pair)
            if not state:
                continue

            try:
                signal = self.ai_engine.generate_signal(
                    price=price_data.get('price', 0),
                    volume=price_data.get('volume', 0),
                    high=price_data.get('price', 0) * 1.001,
                    low=price_data.get('price', 0) * 0.999,
                )

                state.last_signal = signal.action
                state.signal_confidence = signal.confidence

            except Exception as e:
                logger.debug(f"Signal generation error for {pair}: {e}")

    def _select_best_opportunity(self) -> Optional[Tuple[str, int, float, str]]:
        """最良の取引機会を選択"""
        best_pair = None
        best_score = 0
        best_action = 1  # HOLD
        best_confidence = 0
        best_reason = ""

        for pair in self.active_pairs:
            state = self.asset_states.get(pair)
            if not state or state.price == 0:
                continue

            # スコア計算
            score, reason = self._calculate_opportunity_score(pair)

            # シグナルがHOLDでなく、信頼度が閾値以上
            if state.last_signal != 1 and state.signal_confidence >= self.config.trading.min_confidence:
                # シグナル方向にボーナス
                score += state.signal_confidence * 50

                if score > best_score:
                    best_score = score
                    best_pair = pair
                    best_action = state.last_signal
                    best_confidence = state.signal_confidence
                    best_reason = reason

        if best_pair:
            return (best_pair, best_action, best_confidence, best_reason)
        return None

    async def _execute_trade(
        self,
        pair: str,
        action: int,
        confidence: float,
        reason: str = "",
    ) -> Optional[str]:
        """取引を実行"""
        state = self.asset_states.get(pair)
        client = self.clients.get(pair)

        if not state or not client:
            return None

        side = OrderSide.BUY if action == 2 else OrderSide.SELL

        # サイズ計算
        min_size = self.MIN_ORDER_SIZES.get(pair, 0.001)
        max_size = self.config.trading.max_position_size
        size = min(max_size, max(min_size, max_size * confidence * 0.5))
        size = round(size, 4)

        try:
            order_id = await client.send_order(
                side=side,
                size=size,
                order_type=OrderType.MARKET,
            )

            if order_id:
                state.last_trade_time = datetime.now()
                state.trade_count += 1
                self.total_trades += 1

                # ポジション更新
                if side == OrderSide.BUY:
                    state.position += size
                else:
                    state.position -= size

                logger.info(f"✅ [{pair}] {side.value} {size} @ {state.price:,.0f} (conf={confidence:.2f})")

                # 通知
                if self.notifier:
                    self.notifier.notify_trade(TradeNotification(
                        action=side.value,
                        symbol=pair,
                        price=state.price,
                        size=size,
                        confidence=confidence,
                        reasoning=reason,
                    ))

                return order_id

        except Exception as e:
            logger.error(f"Trade execution failed for {pair}: {e}")

        return None

    async def _trading_loop(self):
        """メイン取引ループ"""
        logger.info("🚀 Multi-Asset Trading Loop Started")

        self._load_ai_engine()

        # 起動通知
        if self.notifier:
            self.notifier.send_text(f"""🚀 Multi-Asset AI Trader Started

📊 Active Pairs: {len(self.active_pairs)}
{chr(10).join(f'  • {p}' for p in self.active_pairs[:5])}
{'  ...' if len(self.active_pairs) > 5 else ''}

🧪 Mode: {'Paper' if self.config.trading.paper_trading else 'Live'}
🤖 AI Engine: {'Loaded' if self.ai_engine else 'Fallback'}
⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}""")

        while self.running:
            try:
                self.tick_count += 1

                # 全ペアの価格を取得
                prices = await self._fetch_all_prices()
                if not prices:
                    await asyncio.sleep(1)
                    continue

                # 状態更新
                self._update_asset_states(prices)

                # AIシグナル生成
                await self._generate_signals(prices)

                # アービトラージ検出（10秒ごと）
                if (datetime.now() - self.last_arbitrage_check).seconds >= 10:
                    arb_opps = self._detect_arbitrage()
                    for opp in arb_opps:
                        self.arbitrage_opportunities.append(opp)
                        logger.info(f"⚡ Arbitrage: {opp.pair1}->{opp.pair2}->{opp.pair3} ({opp.profit_pct:.3f}%)")
                    self.last_arbitrage_check = datetime.now()

                # 最良機会を選択
                opportunity = self._select_best_opportunity()

                if opportunity:
                    pair, action, confidence, reason = opportunity
                    state = self.asset_states[pair]

                    # 取引間隔チェック
                    if state.last_trade_time:
                        elapsed = (datetime.now() - state.last_trade_time).seconds
                        if elapsed < self.config.trading.min_trade_interval:
                            opportunity = None

                # 取引実行
                if opportunity:
                    pair, action, confidence, reason = opportunity
                    await self._execute_trade(pair, action, confidence, reason)

                # 定期ステータス表示（60 tick）
                if self.tick_count % 60 == 0:
                    self._log_status()

                # 高速ループ（0.5秒間隔）
                await asyncio.sleep(0.5)

            except Exception as e:
                logger.error(f"Trading loop error: {e}")
                logger.debug(traceback.format_exc())
                await asyncio.sleep(1)

        logger.info("Trading loop stopped")

    def _log_status(self):
        """ステータスをログ出力"""
        active_count = sum(1 for s in self.asset_states.values() if s.price > 0)

        # 最高スコアのペア
        best_pair = None
        best_score = 0
        for pair in self.active_pairs:
            score, _ = self._calculate_opportunity_score(pair)
            if score > best_score:
                best_score = score
                best_pair = pair

        logger.info(
            f"📊 Status: Active={active_count}/{len(self.active_pairs)} | "
            f"Trades={self.total_trades} | "
            f"Best={best_pair}({best_score:.0f})"
        )

        # 各ペアの簡易表示
        for pair in self.active_pairs[:3]:
            state = self.asset_states.get(pair)
            if state and state.price > 0:
                signal_str = ["SELL", "HOLD", "BUY"][state.last_signal]
                logger.info(
                    f"  {pair}: {state.price:,.0f} | "
                    f"{signal_str}({state.signal_confidence:.2f})"
                )

    async def start(self):
        """トレーダーを開始"""
        if self.running:
            return

        self.running = True
        self.start_time = datetime.now()

        logger.info("=" * 60)
        logger.info("  MULTI-ASSET AI TRADING SYSTEM")
        logger.info("  マルチアセット世界最強AIトレーダー")
        logger.info("=" * 60)
        logger.info(f"  Active Pairs: {len(self.active_pairs)}")
        for pair in self.active_pairs:
            min_size = self.MIN_ORDER_SIZES.get(pair, 0.001)
            logger.info(f"    - {pair} (min: {min_size})")
        logger.info(f"  Mode: {'Paper' if self.config.trading.paper_trading else 'Live'}")
        logger.info("=" * 60)

        await self._trading_loop()

    async def stop(self):
        """トレーダーを停止"""
        logger.info("Stopping Multi-Asset Trader...")
        self.running = False

        # 全クライアントの注文キャンセル
        for pair, client in self.clients.items():
            try:
                await client.cancel_all_orders()
            except Exception as e:
                logger.debug(f"Cancel orders failed for {pair}: {e}")

        # 停止通知
        if self.notifier:
            running_time = (datetime.now() - self.start_time).total_seconds() / 3600
            self.notifier.send_text(f"""🛑 Multi-Asset Trader Stopped

📊 Final Statistics:
• Total Trades: {self.total_trades}
• Running Time: {running_time:.1f} hours
• Active Pairs: {len(self.active_pairs)}

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}""")

        logger.info("Multi-Asset Trader stopped")

    def get_status(self) -> Dict:
        """ステータスを取得"""
        return {
            'running': self.running,
            'active_pairs': self.active_pairs,
            'total_trades': self.total_trades,
            'asset_states': {
                pair: {
                    'price': state.price,
                    'signal': state.last_signal,
                    'confidence': state.signal_confidence,
                    'position': state.position,
                    'trades': state.trade_count,
                }
                for pair, state in self.asset_states.items()
            },
            'arbitrage_count': len(self.arbitrage_opportunities),
        }


async def run_multi_asset_trader():
    """マルチアセットトレーダーを実行"""
    config = get_config()
    trader = MultiAssetTrader(config)

    try:
        await trader.start()
    except KeyboardInterrupt:
        await trader.stop()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        await trader.stop()


if __name__ == "__main__":
    asyncio.run(run_multi_asset_trader())

"""
AI Trader - World's Strongest Trading Bot
==========================================
月利30%以上を目指す世界最強AIトレーダー
"""

import os
import time
import yaml
import threading
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from loguru import logger

from ..api.client import BitFlyerClient
from ..api.websocket_client import BitFlyerWebSocket, OrderBookAnalyzer
from ..strategies.base import BaseStrategy, Signal, SignalType, MarketData
from ..strategies.market_making import MarketMakingStrategy
from ..strategies.momentum import MomentumStrategy
from ..strategies.mean_reversion import MeanReversionStrategy
from ..strategies.breakout import BreakoutStrategy
from ..strategies.arbitrage import ArbitrageStrategy
from ..strategies.ml_strategy import MLStrategy
from ..strategies.ensemble import EnsembleStrategy
from ..risk.manager import RiskManager, RiskLimits
from ..risk.position import PositionManager
from ..execution.order_manager import OrderManager, OrderType
from ..execution.smart_executor import SmartExecutor


class AITrader:
    """
    世界最強AIトレーダー

    特徴:
    - 6つの戦略のアンサンブル
    - 高頻度取引（HFT）対応
    - 機械学習による価格予測
    - 動的リスク管理
    - スマートオーダー執行
    - リアルタイムモニタリング
    """

    def __init__(
        self,
        api_key: str = None,
        api_secret: str = None,
        config_path: str = "config/settings.yaml",
    ):
        """
        Args:
            api_key: bitFlyer APIキー
            api_secret: bitFlyer APIシークレット
            config_path: 設定ファイルパス
        """
        # 設定読み込み
        self.config = self._load_config(config_path)

        # API認証情報
        self.api_key = api_key or os.getenv("BITFLYER_API_KEY", "")
        self.api_secret = api_secret or os.getenv("BITFLYER_API_SECRET", "")

        # APIクライアント
        self.api_client = BitFlyerClient(self.api_key, self.api_secret)

        # WebSocketクライアント
        self.ws_client = BitFlyerWebSocket(
            on_ticker=self._on_ticker,
            on_executions=self._on_executions,
            on_board=self._on_board,
        )
        self.order_book_analyzer = OrderBookAnalyzer(self.ws_client)

        # リスク管理
        limits = RiskLimits(
            max_position_size=self.config['risk_management']['max_position_size'],
            max_single_trade=self.config['risk_management']['max_single_trade_size'],
            max_daily_loss=self.config['risk_management']['max_daily_loss'],
            max_drawdown=self.config['risk_management']['max_drawdown'],
            stop_loss_pct=self.config['risk_management']['stop_loss_percentage'],
            take_profit_pct=self.config['risk_management']['take_profit_percentage'],
            trailing_stop_pct=self.config['risk_management']['trailing_stop'],
        )
        self.risk_manager = RiskManager(limits=limits)
        self.position_manager = PositionManager()

        # 注文管理
        self.order_manager = OrderManager(api_client=self.api_client)
        self.smart_executor = SmartExecutor(
            order_manager=self.order_manager,
            api_client=self.api_client,
        )

        # 戦略初期化
        self.strategies: List[BaseStrategy] = []
        self.ensemble_strategy: Optional[EnsembleStrategy] = None
        self._init_strategies()

        # 取引ペア
        self.primary_pairs = self.config['trading_pairs']['primary']
        self.all_pairs = self.primary_pairs + self.config['trading_pairs'].get('secondary', [])

        # 状態
        self.is_running = False
        self.last_trade_time: Dict[str, datetime] = {}
        self.trade_count = 0
        self.daily_pnl = 0.0

        # パフォーマンス
        self.start_time: Optional[datetime] = None
        self.performance_history: List[Dict] = []

        logger.info("AI Trader initialized - World's Strongest Trading System")

    def _load_config(self, config_path: str) -> Dict:
        """設定ファイル読み込み"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config file not found: {config_path}, using defaults")
            return self._get_default_config()

    def _get_default_config(self) -> Dict:
        """デフォルト設定"""
        return {
            'trading_pairs': {
                'primary': ['BTC_JPY', 'ETH_JPY'],
                'secondary': ['XRP_JPY'],
                'leverage': ['FX_BTC_JPY'],
            },
            'strategies': {
                'market_making': {'enabled': True, 'weight': 0.25},
                'momentum': {'enabled': True, 'weight': 0.20},
                'mean_reversion': {'enabled': True, 'weight': 0.15},
                'breakout': {'enabled': True, 'weight': 0.15},
                'arbitrage': {'enabled': True, 'weight': 0.10},
                'ml_strategy': {'enabled': True, 'weight': 0.15},
            },
            'risk_management': {
                'max_position_size': 0.5,
                'max_single_trade_size': 0.1,
                'max_daily_loss': 0.05,
                'max_drawdown': 0.15,
                'stop_loss_percentage': 0.02,
                'take_profit_percentage': 0.03,
                'trailing_stop': 0.01,
            },
            'hft': {
                'enabled': True,
                'min_interval_ms': 100,
            },
            'targets': {
                'monthly_return': 0.30,
                'trades_per_day': 500,
            },
        }

    def _init_strategies(self) -> None:
        """戦略初期化"""
        strategy_config = self.config['strategies']
        primary_pairs = self.config['trading_pairs']['primary']

        # マーケットメイキング
        if strategy_config['market_making']['enabled']:
            mm_strategy = MarketMakingStrategy(
                product_codes=primary_pairs,
                spread_percentage=0.05,
                weight=strategy_config['market_making']['weight'],
            )
            self.strategies.append(mm_strategy)

        # モメンタム
        if strategy_config['momentum']['enabled']:
            mom_strategy = MomentumStrategy(
                product_codes=primary_pairs,
                lookback_periods=[5, 15, 30, 60],
                weight=strategy_config['momentum']['weight'],
            )
            self.strategies.append(mom_strategy)

        # 平均回帰
        if strategy_config['mean_reversion']['enabled']:
            mr_strategy = MeanReversionStrategy(
                product_codes=primary_pairs,
                bollinger_period=20,
                z_score_threshold=2.0,
                weight=strategy_config['mean_reversion']['weight'],
            )
            self.strategies.append(mr_strategy)

        # ブレイクアウト
        if strategy_config['breakout']['enabled']:
            bo_strategy = BreakoutStrategy(
                product_codes=primary_pairs,
                lookback_period=100,
                weight=strategy_config['breakout']['weight'],
            )
            self.strategies.append(bo_strategy)

        # アービトラージ
        if strategy_config['arbitrage']['enabled']:
            arb_strategy = ArbitrageStrategy(
                product_codes=['BTC_JPY', 'FX_BTC_JPY'],
                min_spread=0.1,
                weight=strategy_config['arbitrage']['weight'],
            )
            self.strategies.append(arb_strategy)

        # 機械学習
        if strategy_config['ml_strategy']['enabled']:
            ml_strategy = MLStrategy(
                product_codes=primary_pairs,
                model_type='ensemble',
                confidence_threshold=0.65,
                weight=strategy_config['ml_strategy']['weight'],
            )
            self.strategies.append(ml_strategy)

        # アンサンブル戦略
        self.ensemble_strategy = EnsembleStrategy(
            product_codes=primary_pairs,
            strategies=self.strategies,
            voting_method="soft",
        )

        logger.info(f"Initialized {len(self.strategies)} strategies")

    def _on_ticker(self, product_code: str, data: Dict) -> None:
        """ティッカー更新コールバック"""
        # ポジション価格更新
        prices = {product_code: data.get('ltp', 0)}
        self.position_manager.update_all_positions(prices)

        # 損切り・利確チェック
        current_price = data.get('ltp', 0)
        if self.position_manager.check_stop_loss(product_code, current_price):
            self._close_position(product_code, current_price, "stop_loss")
        elif self.position_manager.check_take_profit(product_code, current_price):
            self._close_position(product_code, current_price, "take_profit")

    def _on_executions(self, product_code: str, data: List[Dict]) -> None:
        """約定コールバック"""
        pass

    def _on_board(self, product_code: str, data: Dict) -> None:
        """板更新コールバック"""
        pass

    def start(self) -> None:
        """トレーディング開始"""
        if self.is_running:
            logger.warning("Trader already running")
            return

        self.is_running = True
        self.start_time = datetime.now()

        # 残高確認
        try:
            balance = self.api_client.get_balance()
            total_jpy = self.api_client.get_total_balance_jpy()
            self.risk_manager.initial_capital = total_jpy
            self.risk_manager.current_capital = total_jpy
            logger.info(f"Starting capital: {total_jpy:,.0f} JPY")
        except Exception as e:
            logger.error(f"Failed to get balance: {e}")

        # WebSocket接続
        self.ws_client.connect()
        for pair in self.primary_pairs:
            self.ws_client.subscribe_all(pair)

        # メインループ開始
        self._trading_thread = threading.Thread(target=self._trading_loop, daemon=True)
        self._trading_thread.start()

        logger.info("🚀 AI Trader started - Aiming for 30%+ monthly returns!")

    def stop(self) -> None:
        """トレーディング停止"""
        self.is_running = False

        # 全ポジションクローズ
        self._close_all_positions()

        # 全注文キャンセル
        self.order_manager.cancel_all_orders()

        # WebSocket切断
        self.ws_client.disconnect()

        # パフォーマンスレポート
        self._print_performance_report()

        logger.info("AI Trader stopped")

    def _trading_loop(self) -> None:
        """メイントレーディングループ"""
        min_interval = self.config['hft']['min_interval_ms'] / 1000

        while self.is_running:
            loop_start = time.time()

            try:
                # 各取引ペアで処理
                for product_code in self.primary_pairs:
                    self._process_trading_cycle(product_code)

                # リスク状態チェック
                self._check_risk_status()

                # 定期同期
                if self.trade_count % 100 == 0:
                    self.order_manager.sync_with_exchange()

            except Exception as e:
                logger.error(f"Trading loop error: {e}")

            # 最小間隔維持
            elapsed = time.time() - loop_start
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)

    def _process_trading_cycle(self, product_code: str) -> None:
        """1取引サイクルの処理"""
        # 取引頻度制限
        if product_code in self.last_trade_time:
            elapsed = (datetime.now() - self.last_trade_time[product_code]).total_seconds()
            if elapsed < 0.1:  # 100msの最小間隔
                return

        # 市場データ取得
        market_data = self._get_market_data(product_code)
        if not market_data:
            return

        # 全戦略更新
        for strategy in self.strategies:
            strategy.update(market_data)

        # アンサンブルシグナル生成
        signal = self.ensemble_strategy.generate_signal(market_data)
        if not signal:
            return

        # シグナル処理
        self._process_signal(signal, market_data)

    def _get_market_data(self, product_code: str) -> Optional[MarketData]:
        """市場データ取得"""
        try:
            # WebSocketキャッシュから取得
            ticker = self.ws_client.get_cached_ticker(product_code)
            board = self.ws_client.get_cached_board(product_code)

            if not ticker:
                ticker = self.api_client.get_ticker(product_code)

            if not ticker:
                return None

            # オーダーブック分析
            imbalance = self.order_book_analyzer.get_order_book_imbalance(product_code)
            spread_info = self.order_book_analyzer.get_spread_info(product_code)

            return MarketData(
                product_code=product_code,
                timestamp=datetime.now(),
                open=ticker.get('ltp', 0),
                high=ticker.get('best_ask', ticker.get('ltp', 0)),
                low=ticker.get('best_bid', ticker.get('ltp', 0)),
                close=ticker.get('ltp', 0),
                volume=ticker.get('volume', 0),
                vwap=self.ws_client.get_vwap(product_code, 60),
                best_bid=spread_info.get('best_bid', 0),
                best_ask=spread_info.get('best_ask', 0),
                order_book_imbalance=imbalance,
                spread=spread_info.get('spread', 0),
            )

        except Exception as e:
            logger.error(f"Failed to get market data for {product_code}: {e}")
            return None

    def _process_signal(self, signal: Signal, market_data: MarketData) -> None:
        """シグナル処理"""
        product_code = signal.product_code

        # 現在のポジション
        current_position = self.position_manager.get_position(product_code)
        current_size = current_position.size if current_position else 0

        # リスクチェック
        allowed, reason = self.risk_manager.check_trade_allowed(
            product_code=product_code,
            side=signal.signal_type.value,
            size=signal.size,
            price=signal.price,
            current_position=current_size,
        )

        if not allowed:
            logger.debug(f"Trade not allowed: {reason}")
            return

        # ポジションサイズ計算
        size = self.risk_manager.calculate_position_size(
            product_code=product_code,
            side=signal.signal_type.value,
            price=signal.price,
            confidence=signal.confidence,
            method="kelly",
        )

        # 最小サイズチェック
        if size < 0.001:
            return

        # 注文執行
        self._execute_trade(signal, size, market_data)

    def _execute_trade(
        self,
        signal: Signal,
        size: float,
        market_data: MarketData,
    ) -> None:
        """取引執行"""
        product_code = signal.product_code
        side = "BUY" if signal.signal_type == SignalType.BUY else "SELL"

        try:
            # スマート執行
            result = self.smart_executor.execute_smart(
                product_code=product_code,
                side=side,
                size=size,
                urgency="normal" if signal.confidence < 0.7 else "high",
            )

            if result.success:
                # ポジション更新
                stop_loss = self.risk_manager.calculate_stop_loss(
                    result.average_price, side
                )
                take_profit = self.risk_manager.calculate_take_profit(
                    result.average_price, side
                )

                self.position_manager.open_position(
                    product_code=product_code,
                    side=side,
                    size=result.filled_size,
                    entry_price=result.average_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                )

                self.trade_count += 1
                self.last_trade_time[product_code] = datetime.now()

                logger.info(
                    f"Trade executed: {side} {result.filled_size:.4f} {product_code} "
                    f"@ {result.average_price:.0f} (slippage: {result.slippage:.4%})"
                )

        except Exception as e:
            logger.error(f"Trade execution failed: {e}")

    def _close_position(
        self,
        product_code: str,
        price: float,
        reason: str,
    ) -> None:
        """ポジションクローズ"""
        position = self.position_manager.get_position(product_code)
        if not position:
            return

        side = "SELL" if position.side.value == "LONG" else "BUY"

        try:
            result = self.smart_executor.execute_market(
                product_code=product_code,
                side=side,
                size=position.size,
            )

            if result.success:
                pnl, _ = self.position_manager.close_position(
                    product_code, result.average_price
                )
                self.risk_manager.update_pnl(pnl)
                self.daily_pnl += pnl

                logger.info(
                    f"Position closed ({reason}): {product_code} "
                    f"PnL: {pnl:,.0f} JPY"
                )

        except Exception as e:
            logger.error(f"Position close failed: {e}")

    def _close_all_positions(self) -> None:
        """全ポジションクローズ"""
        for position in self.position_manager.get_all_positions():
            try:
                ticker = self.api_client.get_ticker(position.product_code)
                self._close_position(
                    position.product_code,
                    ticker.get('ltp', 0),
                    "shutdown",
                )
            except Exception as e:
                logger.error(f"Failed to close position: {e}")

    def _check_risk_status(self) -> None:
        """リスク状態チェック"""
        risk_summary = self.risk_manager.get_risk_summary()

        if risk_summary['risk_level'] == 'critical':
            logger.warning("CRITICAL RISK LEVEL - Closing all positions")
            self._close_all_positions()

        elif risk_summary['risk_level'] == 'high':
            logger.warning("HIGH RISK LEVEL - Reducing positions")

    def _print_performance_report(self) -> None:
        """パフォーマンスレポート出力"""
        if not self.start_time:
            return

        runtime = datetime.now() - self.start_time
        position_stats = self.position_manager.get_performance_stats()
        risk_summary = self.risk_manager.get_risk_summary()

        report = f"""
╔══════════════════════════════════════════════════════════════╗
║           🏆 AI TRADER PERFORMANCE REPORT 🏆                  ║
╠══════════════════════════════════════════════════════════════╣
║ Runtime: {str(runtime).split('.')[0]:>20}                              ║
║ Total Trades: {self.trade_count:>16,}                              ║
╠══════════════════════════════════════════════════════════════╣
║ PROFIT & LOSS                                                 ║
║ ─────────────────────────────────────────────────────────────║
║ Total Return: {risk_summary['total_return']:>15.2%}                              ║
║ Daily P&L: {self.daily_pnl:>18,.0f} JPY                         ║
║ Current Capital: {risk_summary['current_capital']:>12,.0f} JPY                   ║
╠══════════════════════════════════════════════════════════════╣
║ RISK METRICS                                                  ║
║ ─────────────────────────────────────────────────────────────║
║ Max Drawdown: {risk_summary['max_drawdown']:>15.2%}                              ║
║ Sharpe Ratio: {risk_summary['sharpe_ratio']:>15.2f}                              ║
║ VaR (95%): {risk_summary['var_95']:>18,.0f} JPY                         ║
╠══════════════════════════════════════════════════════════════╣
║ TRADE STATISTICS                                              ║
║ ─────────────────────────────────────────────────────────────║
║ Win Rate: {position_stats.get('win_rate', 0):>19.2%}                              ║
║ Profit Factor: {position_stats.get('profit_factor', 0):>14.2f}                              ║
║ Avg Win: {position_stats.get('avg_win', 0):>20,.0f} JPY                         ║
║ Avg Loss: {position_stats.get('avg_loss', 0):>19,.0f} JPY                         ║
╚══════════════════════════════════════════════════════════════╝
"""
        print(report)
        logger.info("Performance report generated")

    def get_status(self) -> Dict:
        """現在のステータス取得"""
        return {
            'is_running': self.is_running,
            'trade_count': self.trade_count,
            'daily_pnl': self.daily_pnl,
            'positions': self.position_manager.get_position_summary(),
            'risk': self.risk_manager.get_risk_summary(),
            'orders': self.order_manager.get_order_stats(),
            'strategies': self.ensemble_strategy.get_strategy_stats() if self.ensemble_strategy else {},
        }

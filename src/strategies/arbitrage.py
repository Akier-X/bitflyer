"""
Arbitrage Strategy
===================
価格差を利用したアービトラージ戦略
"""

from typing import Optional, Dict, List, Tuple
from datetime import datetime
import numpy as np
from loguru import logger

from .base import BaseStrategy, Signal, SignalType, MarketData


class ArbitrageStrategy(BaseStrategy):
    """
    アービトラージ戦略

    特徴:
    - 現物とFX（レバレッジ）間の価格差を利用
    - 三角アービトラージ（ETH/BTC/JPY）
    - リスクフリーの利益追求
    """

    def __init__(
        self,
        product_codes: List[str],
        min_spread: float = 0.1,  # 最小スプレッド（%）
        pairs: List[List[str]] = None,
        weight: float = 1.0,
    ):
        """
        Args:
            min_spread: 最小スプレッド閾値（%）
            pairs: アービトラージペアリスト
        """
        super().__init__(
            name="Arbitrage",
            product_codes=product_codes,
            weight=weight,
        )

        self.min_spread = min_spread / 100
        self.pairs = pairs or [
            ["BTC_JPY", "FX_BTC_JPY"],  # 現物-FXアービトラージ
        ]

        # 価格キャッシュ
        self.price_cache: Dict[str, Dict] = {}
        self.last_update: Dict[str, datetime] = {}

        # アービトラージ機会トラッキング
        self.opportunities: List[Dict] = []

    def update(self, market_data: MarketData) -> None:
        """市場データで内部状態更新"""
        product = market_data.product_code

        self.price_cache[product] = {
            'timestamp': market_data.timestamp,
            'close': market_data.close,
            'best_bid': market_data.best_bid,
            'best_ask': market_data.best_ask,
        }
        self.last_update[product] = datetime.now()

    def generate_signal(self, market_data: MarketData) -> Optional[Signal]:
        """シグナル生成"""
        # 各ペアでアービトラージ機会を検索
        for pair in self.pairs:
            signal = self._check_arbitrage_opportunity(pair, market_data)
            if signal:
                return signal

        return None

    def _check_arbitrage_opportunity(
        self,
        pair: List[str],
        market_data: MarketData
    ) -> Optional[Signal]:
        """アービトラージ機会チェック"""
        if len(pair) == 2:
            return self._check_two_way_arbitrage(pair, market_data)
        elif len(pair) == 3:
            return self._check_triangular_arbitrage(pair, market_data)
        return None

    def _check_two_way_arbitrage(
        self,
        pair: List[str],
        market_data: MarketData
    ) -> Optional[Signal]:
        """2者間アービトラージ（現物-FX等）"""
        product_a, product_b = pair

        # 両方の価格が利用可能か確認
        if product_a not in self.price_cache or product_b not in self.price_cache:
            return None

        price_a = self.price_cache[product_a]
        price_b = self.price_cache[product_b]

        # 価格データの鮮度確認（5秒以内）
        now = datetime.now()
        if product_a in self.last_update:
            if (now - self.last_update[product_a]).total_seconds() > 5:
                return None
        if product_b in self.last_update:
            if (now - self.last_update[product_b]).total_seconds() > 5:
                return None

        # スプレッド計算
        # ケース1: Aで買ってBで売る
        if price_a.get('best_ask') and price_b.get('best_bid'):
            buy_price = price_a['best_ask']
            sell_price = price_b['best_bid']
            spread_1 = (sell_price - buy_price) / buy_price

            if spread_1 > self.min_spread:
                return self._create_arbitrage_signal(
                    product_a, buy_price, "BUY",
                    product_b, sell_price, "SELL",
                    spread_1
                )

        # ケース2: Bで買ってAで売る
        if price_b.get('best_ask') and price_a.get('best_bid'):
            buy_price = price_b['best_ask']
            sell_price = price_a['best_bid']
            spread_2 = (sell_price - buy_price) / buy_price

            if spread_2 > self.min_spread:
                return self._create_arbitrage_signal(
                    product_b, buy_price, "BUY",
                    product_a, sell_price, "SELL",
                    spread_2
                )

        return None

    def _check_triangular_arbitrage(
        self,
        pair: List[str],
        market_data: MarketData
    ) -> Optional[Signal]:
        """三角アービトラージ（ETH/BTC/JPY等）"""
        if len(pair) != 3:
            return None

        product_a, product_b, product_c = pair

        # 全ての価格が利用可能か確認
        for p in pair:
            if p not in self.price_cache:
                return None

        # 例: ETH_JPY, ETH_BTC, BTC_JPY の場合
        # ルート1: JPY → ETH → BTC → JPY
        # ルート2: JPY → BTC → ETH → JPY

        price_a = self.price_cache[product_a]  # ETH_JPY
        price_b = self.price_cache[product_b]  # ETH_BTC
        price_c = self.price_cache[product_c]  # BTC_JPY

        # 合成レートの計算
        # ETH/JPY = ETH/BTC * BTC/JPY
        if price_b.get('close') and price_c.get('close'):
            synthetic_eth_jpy = price_b['close'] * price_c['close']
            actual_eth_jpy = price_a.get('close', 0)

            if actual_eth_jpy > 0:
                spread = (actual_eth_jpy - synthetic_eth_jpy) / synthetic_eth_jpy

                if abs(spread) > self.min_spread:
                    self.opportunities.append({
                        'timestamp': datetime.now(),
                        'pair': pair,
                        'spread': spread,
                        'direction': 'positive' if spread > 0 else 'negative',
                    })

                    # 三角アービトラージは複雑なので、シンプルな2者間から実行
                    # ここでは検出のみ記録
                    logger.info(f"Triangular arbitrage opportunity: {spread:.4%}")

        return None

    def _create_arbitrage_signal(
        self,
        buy_product: str,
        buy_price: float,
        buy_side: str,
        sell_product: str,
        sell_price: float,
        sell_side: str,
        spread: float
    ) -> Signal:
        """アービトラージシグナル作成"""
        # 買い側のシグナルを返す（売り側は別途処理）
        confidence = min(spread / (self.min_spread * 2), 1.0)
        size = 0.01 * confidence  # スプレッドに応じたサイズ

        self.signals_generated += 1

        return Signal(
            signal_type=SignalType.BUY,
            product_code=buy_product,
            price=buy_price,
            size=size,
            confidence=confidence,
            strategy_name=self.name,
            metadata={
                "arbitrage_type": "two_way",
                "buy_product": buy_product,
                "buy_price": buy_price,
                "sell_product": sell_product,
                "sell_price": sell_price,
                "spread": spread,
                "expected_profit_pct": spread * 100,
            },
        )

    def get_current_spreads(self) -> Dict[str, float]:
        """現在のスプレッド取得"""
        spreads = {}

        for pair in self.pairs:
            if len(pair) == 2:
                product_a, product_b = pair

                if product_a in self.price_cache and product_b in self.price_cache:
                    price_a = self.price_cache[product_a].get('close', 0)
                    price_b = self.price_cache[product_b].get('close', 0)

                    if price_a > 0 and price_b > 0:
                        spread = abs(price_a - price_b) / min(price_a, price_b)
                        spreads[f"{product_a}-{product_b}"] = spread

        return spreads

    def get_opportunity_history(self, limit: int = 100) -> List[Dict]:
        """アービトラージ機会履歴取得"""
        return self.opportunities[-limit:]


class StatisticalArbitrage(BaseStrategy):
    """
    統計的アービトラージ戦略

    特徴:
    - ペアトレーディング
    - 共和分関係に基づく取引
    - スプレッドの平均回帰を利用
    """

    def __init__(
        self,
        product_codes: List[str],
        lookback_period: int = 100,
        entry_z_score: float = 2.0,
        exit_z_score: float = 0.5,
        weight: float = 1.0,
    ):
        super().__init__(
            name="StatisticalArbitrage",
            product_codes=product_codes,
            weight=weight,
        )

        self.lookback_period = lookback_period
        self.entry_z_score = entry_z_score
        self.exit_z_score = exit_z_score

        # スプレッド履歴
        self.spread_history: Dict[str, List[float]] = {}

    def update(self, market_data: MarketData) -> None:
        """市場データで内部状態更新"""
        pass  # ペアの更新が必要

    def generate_signal(self, market_data: MarketData) -> Optional[Signal]:
        """シグナル生成"""
        # 統計的アービトラージのロジック
        return None

    def update_pair_spread(
        self,
        product_a: str,
        price_a: float,
        product_b: str,
        price_b: float
    ) -> None:
        """ペアスプレッド更新"""
        pair_key = f"{product_a}-{product_b}"

        if pair_key not in self.spread_history:
            self.spread_history[pair_key] = []

        # ログスプレッド計算（ペアトレーディング用）
        log_spread = np.log(price_a) - np.log(price_b)
        self.spread_history[pair_key].append(log_spread)

        # 履歴制限
        if len(self.spread_history[pair_key]) > self.lookback_period * 2:
            self.spread_history[pair_key] = self.spread_history[pair_key][-self.lookback_period * 2:]

    def get_spread_z_score(self, pair_key: str) -> Optional[float]:
        """スプレッドZスコア取得"""
        if pair_key not in self.spread_history:
            return None

        spreads = self.spread_history[pair_key]
        if len(spreads) < self.lookback_period:
            return None

        recent_spreads = np.array(spreads[-self.lookback_period:])
        current = spreads[-1]
        mean = np.mean(recent_spreads)
        std = np.std(recent_spreads)

        if std == 0:
            return 0.0

        return (current - mean) / std

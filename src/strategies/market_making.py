"""
Market Making Strategy
=======================
スプレッドを取得して利益を得るマーケットメイキング戦略
高頻度取引に最適化
"""

from typing import Optional, Dict, List, Tuple
from datetime import datetime, timedelta
import numpy as np
from loguru import logger

from .base import BaseStrategy, Signal, SignalType, MarketData


class MarketMakingStrategy(BaseStrategy):
    """
    マーケットメイキング戦略

    特徴:
    - 両サイド（買い・売り）に指値注文を出してスプレッドを取得
    - 在庫リスクを管理
    - 高頻度で注文を更新
    - Maker手数料を活用してコスト削減
    """

    def __init__(
        self,
        product_codes: List[str],
        spread_percentage: float = 0.05,
        order_depth: int = 5,
        inventory_limit: float = 0.3,
        refresh_interval_ms: int = 100,
        weight: float = 1.0,
    ):
        """
        Args:
            spread_percentage: 目標スプレッド（%）
            order_depth: 注文の深さ（レベル数）
            inventory_limit: 在庫制限（総資産に対する割合）
            refresh_interval_ms: 注文更新間隔（ミリ秒）
        """
        super().__init__(
            name="MarketMaking",
            product_codes=product_codes,
            weight=weight,
        )

        self.spread_percentage = spread_percentage / 100
        self.order_depth = order_depth
        self.inventory_limit = inventory_limit
        self.refresh_interval_ms = refresh_interval_ms

        # 内部状態
        self.inventory: Dict[str, float] = {}
        self.mid_price_history: Dict[str, List[float]] = {}
        self.volatility: Dict[str, float] = {}
        self.last_update: Dict[str, datetime] = {}

        # スプレッド履歴（動的調整用）
        self.spread_history: Dict[str, List[float]] = {}

    def update(self, market_data: MarketData) -> None:
        """市場データで内部状態更新"""
        product = market_data.product_code

        # ミッドプライス履歴更新
        if product not in self.mid_price_history:
            self.mid_price_history[product] = []

        if market_data.best_bid and market_data.best_ask:
            mid_price = (market_data.best_bid + market_data.best_ask) / 2
            self.mid_price_history[product].append(mid_price)

            # 直近1000データポイントのみ保持
            if len(self.mid_price_history[product]) > 1000:
                self.mid_price_history[product] = self.mid_price_history[product][-1000:]

            # ボラティリティ計算（直近100ポイント）
            if len(self.mid_price_history[product]) >= 100:
                prices = np.array(self.mid_price_history[product][-100:])
                returns = np.diff(prices) / prices[:-1]
                self.volatility[product] = np.std(returns)

        # スプレッド履歴更新
        if market_data.spread:
            if product not in self.spread_history:
                self.spread_history[product] = []
            self.spread_history[product].append(market_data.spread)
            if len(self.spread_history[product]) > 100:
                self.spread_history[product] = self.spread_history[product][-100:]

    def generate_signal(self, market_data: MarketData) -> Optional[Signal]:
        """シグナル生成"""
        product = market_data.product_code

        if not market_data.best_bid or not market_data.best_ask:
            return None

        # 更新間隔チェック
        now = datetime.now()
        if product in self.last_update:
            elapsed = (now - self.last_update[product]).total_seconds() * 1000
            if elapsed < self.refresh_interval_ms:
                return None

        self.last_update[product] = now

        # ミッドプライス計算
        mid_price = (market_data.best_bid + market_data.best_ask) / 2

        # 動的スプレッド調整
        optimal_spread = self._calculate_optimal_spread(product, market_data)

        # 在庫に基づく価格スキュー
        inventory = self.inventory.get(product, 0)
        skew = self._calculate_inventory_skew(inventory, mid_price)

        # 買い価格・売り価格計算
        bid_price = mid_price * (1 - optimal_spread / 2) - skew
        ask_price = mid_price * (1 + optimal_spread / 2) - skew

        # シグナル生成
        signal_type = self._determine_signal_type(
            market_data, bid_price, ask_price, inventory
        )

        if signal_type == SignalType.HOLD:
            return None

        # ポジションサイズ計算
        size = self._calculate_order_size(product, market_data)

        price = bid_price if signal_type == SignalType.BUY else ask_price

        confidence = self._calculate_confidence(product, market_data)

        self.signals_generated += 1

        return Signal(
            signal_type=signal_type,
            product_code=product,
            price=price,
            size=size,
            confidence=confidence,
            strategy_name=self.name,
            metadata={
                "mid_price": mid_price,
                "optimal_spread": optimal_spread,
                "skew": skew,
                "inventory": inventory,
                "volatility": self.volatility.get(product, 0),
            },
        )

    def _calculate_optimal_spread(
        self,
        product: str,
        market_data: MarketData
    ) -> float:
        """最適スプレッド計算"""
        base_spread = self.spread_percentage

        # ボラティリティに基づく調整
        vol = self.volatility.get(product, 0)
        vol_adjustment = vol * 10  # ボラティリティが高いほどスプレッド拡大

        # オーダーブック不均衡に基づく調整
        imbalance = market_data.order_book_imbalance or 0
        imbalance_adjustment = abs(imbalance) * 0.001  # 不均衡が大きいほど拡大

        # 現在のマーケットスプレッドを考慮
        if self.spread_history.get(product):
            avg_market_spread = np.mean(self.spread_history[product])
            mid_price = (market_data.best_bid + market_data.best_ask) / 2
            market_spread_ratio = avg_market_spread / mid_price
            # マーケットスプレッドより少し狭く設定
            base_spread = min(base_spread, market_spread_ratio * 0.8)

        optimal_spread = base_spread + vol_adjustment + imbalance_adjustment

        # 最小・最大制限
        return max(0.0001, min(optimal_spread, 0.01))

    def _calculate_inventory_skew(
        self,
        inventory: float,
        mid_price: float
    ) -> float:
        """在庫スキュー計算"""
        # 在庫が正（ロング）なら売りを促進（価格を下げる）
        # 在庫が負（ショート）なら買いを促進（価格を上げる）
        inventory_ratio = inventory / (mid_price * self.inventory_limit)
        skew = inventory_ratio * mid_price * 0.001  # 最大0.1%のスキュー
        return skew

    def _determine_signal_type(
        self,
        market_data: MarketData,
        bid_price: float,
        ask_price: float,
        inventory: float
    ) -> SignalType:
        """シグナルタイプ決定"""
        # 在庫制限チェック
        max_inventory = market_data.close * self.inventory_limit

        if inventory >= max_inventory:
            # 在庫オーバー、売りのみ
            return SignalType.SELL
        elif inventory <= -max_inventory:
            # ショートオーバー、買いのみ
            return SignalType.BUY

        # オーダーブック不均衡に基づく判断
        imbalance = market_data.order_book_imbalance or 0

        if imbalance > 0.3:
            # 買い優勢 → 売りで利益を得やすい
            return SignalType.SELL
        elif imbalance < -0.3:
            # 売り優勢 → 買いで利益を得やすい
            return SignalType.BUY

        # バランス状態では両方の注文を出すが、シグナルは1つずつ
        # 交互に出す
        if not hasattr(self, '_last_side'):
            self._last_side = SignalType.BUY

        self._last_side = (
            SignalType.SELL if self._last_side == SignalType.BUY
            else SignalType.BUY
        )
        return self._last_side

    def _calculate_order_size(
        self,
        product: str,
        market_data: MarketData
    ) -> float:
        """注文サイズ計算"""
        # 基本サイズ（最小注文単位）
        base_size = 0.001  # 0.001 BTC

        # ボラティリティに応じて調整
        vol = self.volatility.get(product, 0)
        if vol > 0.001:  # 高ボラ時はサイズ縮小
            base_size *= 0.5

        return base_size

    def _calculate_confidence(
        self,
        product: str,
        market_data: MarketData
    ) -> float:
        """信頼度計算"""
        confidence = 0.5

        # スプレッドが広いほど信頼度上昇
        if market_data.spread and market_data.best_ask:
            spread_ratio = market_data.spread / market_data.best_ask
            if spread_ratio > self.spread_percentage:
                confidence += 0.2

        # ボラティリティが低いほど信頼度上昇
        vol = self.volatility.get(product, 0)
        if vol < 0.0005:
            confidence += 0.2
        elif vol > 0.002:
            confidence -= 0.2

        # オーダーブック不均衡が低いほど信頼度上昇
        imbalance = market_data.order_book_imbalance or 0
        if abs(imbalance) < 0.2:
            confidence += 0.1

        return max(0.0, min(1.0, confidence))

    def update_inventory(self, product: str, size: float, side: SignalType) -> None:
        """在庫更新"""
        if product not in self.inventory:
            self.inventory[product] = 0

        if side == SignalType.BUY:
            self.inventory[product] += size
        elif side == SignalType.SELL:
            self.inventory[product] -= size

    def get_quote_prices(
        self,
        product: str,
        mid_price: float
    ) -> Tuple[float, float]:
        """買値・売値のクオートを取得"""
        optimal_spread = self.spread_percentage
        inventory = self.inventory.get(product, 0)
        skew = self._calculate_inventory_skew(inventory, mid_price)

        bid_price = mid_price * (1 - optimal_spread / 2) - skew
        ask_price = mid_price * (1 + optimal_spread / 2) - skew

        return bid_price, ask_price

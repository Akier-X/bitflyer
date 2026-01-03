"""
Mean Reversion Strategy
========================
平均回帰を狙う逆張り戦略
"""

from typing import Optional, Dict, List
from datetime import datetime
import pandas as pd
import numpy as np
from loguru import logger

from .base import BaseStrategy, Signal, SignalType, MarketData, TechnicalIndicators


class MeanReversionStrategy(BaseStrategy):
    """
    平均回帰戦略

    特徴:
    - ボリンジャーバンドを使用した過剰な価格変動の検出
    - Zスコアによる統計的判断
    - RSIとの組み合わせで精度向上
    """

    def __init__(
        self,
        product_codes: List[str],
        bollinger_period: int = 20,
        bollinger_std: float = 2.0,
        z_score_threshold: float = 2.0,
        weight: float = 1.0,
    ):
        """
        Args:
            bollinger_period: ボリンジャーバンド期間
            bollinger_std: ボリンジャーバンド標準偏差
            z_score_threshold: Zスコア閾値
        """
        super().__init__(
            name="MeanReversion",
            product_codes=product_codes,
            weight=weight,
        )

        self.bollinger_period = bollinger_period
        self.bollinger_std = bollinger_std
        self.z_score_threshold = z_score_threshold

        # 価格履歴
        self.price_history: Dict[str, pd.DataFrame] = {}
        self.max_history = bollinger_period * 3

    def update(self, market_data: MarketData) -> None:
        """市場データで内部状態更新"""
        product = market_data.product_code

        if product not in self.price_history:
            self.price_history[product] = pd.DataFrame(columns=[
                'timestamp', 'close', 'high', 'low', 'volume'
            ])

        new_row = pd.DataFrame([{
            'timestamp': market_data.timestamp,
            'close': market_data.close,
            'high': market_data.high,
            'low': market_data.low,
            'volume': market_data.volume,
        }])

        self.price_history[product] = pd.concat(
            [self.price_history[product], new_row],
            ignore_index=True
        )

        if len(self.price_history[product]) > self.max_history:
            self.price_history[product] = self.price_history[product].iloc[-self.max_history:]

    def generate_signal(self, market_data: MarketData) -> Optional[Signal]:
        """シグナル生成"""
        product = market_data.product_code

        if product not in self.price_history:
            return None

        df = self.price_history[product]
        if len(df) < self.bollinger_period:
            return None

        close = df['close']
        current_price = close.iloc[-1]

        # ボリンジャーバンド計算
        bb = TechnicalIndicators.bollinger_bands(
            close,
            self.bollinger_period,
            self.bollinger_std
        )

        upper = bb['upper'].iloc[-1]
        middle = bb['middle'].iloc[-1]
        lower = bb['lower'].iloc[-1]

        # Zスコア計算
        z_score = TechnicalIndicators.z_score(close, self.bollinger_period)
        current_z = z_score.iloc[-1] if not pd.isna(z_score.iloc[-1]) else 0

        # RSI計算
        rsi = TechnicalIndicators.rsi(close, 14)
        current_rsi = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50

        # %B計算（ボリンジャーバンド内の位置）
        percent_b = (current_price - lower) / (upper - lower) if upper != lower else 0.5

        # シグナル判定
        signal_type = self._determine_signal(
            current_price, upper, lower, middle,
            current_z, current_rsi, percent_b
        )

        if signal_type == SignalType.HOLD:
            return None

        # 信頼度計算
        confidence = self._calculate_confidence(
            current_z, current_rsi, percent_b
        )

        # ポジションサイズ
        base_size = 0.01
        size = base_size * confidence

        self.signals_generated += 1

        return Signal(
            signal_type=signal_type,
            product_code=product,
            price=current_price,
            size=size,
            confidence=confidence,
            strategy_name=self.name,
            metadata={
                "z_score": current_z,
                "rsi": current_rsi,
                "percent_b": percent_b,
                "bb_upper": upper,
                "bb_middle": middle,
                "bb_lower": lower,
                "distance_from_mean": (current_price - middle) / middle,
            },
        )

    def _determine_signal(
        self,
        price: float,
        upper: float,
        lower: float,
        middle: float,
        z_score: float,
        rsi: float,
        percent_b: float
    ) -> SignalType:
        """シグナルタイプ決定"""

        # 買いシグナル条件（下方への過剰な乖離）
        buy_conditions = [
            price <= lower,                      # ボリンジャー下限以下
            z_score <= -self.z_score_threshold,  # Zスコアが閾値以下
            rsi < 30,                            # RSI売られすぎ
            percent_b < 0.1,                     # %Bが低い
        ]

        # 売りシグナル条件（上方への過剰な乖離）
        sell_conditions = [
            price >= upper,                      # ボリンジャー上限以上
            z_score >= self.z_score_threshold,   # Zスコアが閾値以上
            rsi > 70,                            # RSI買われすぎ
            percent_b > 0.9,                     # %Bが高い
        ]

        buy_score = sum(buy_conditions)
        sell_score = sum(sell_conditions)

        # 2つ以上の条件が満たされた場合にシグナル生成
        if buy_score >= 2:
            return SignalType.BUY
        elif sell_score >= 2:
            return SignalType.SELL

        return SignalType.HOLD

    def _calculate_confidence(
        self,
        z_score: float,
        rsi: float,
        percent_b: float
    ) -> float:
        """信頼度計算"""
        confidence = 0.5

        # Zスコアの極値
        z_strength = min(abs(z_score) / 3.0, 0.2)
        confidence += z_strength

        # RSIの極値
        rsi_distance = abs(rsi - 50) / 50
        confidence += rsi_distance * 0.2

        # %Bの極値
        pb_distance = abs(percent_b - 0.5) * 2
        confidence += pb_distance * 0.1

        return max(0.0, min(1.0, confidence))

    def get_mean_distance(self, product: str) -> float:
        """平均からの乖離率取得"""
        if product not in self.price_history:
            return 0.0

        df = self.price_history[product]
        if len(df) < self.bollinger_period:
            return 0.0

        close = df['close']
        current = close.iloc[-1]
        mean = close.iloc[-self.bollinger_period:].mean()

        return (current - mean) / mean

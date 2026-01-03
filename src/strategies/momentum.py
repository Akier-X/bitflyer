"""
Momentum Strategy
==================
価格のトレンドを追従するモメンタム戦略
"""

from typing import Optional, Dict, List
from datetime import datetime
import pandas as pd
import numpy as np
from loguru import logger

from .base import BaseStrategy, Signal, SignalType, MarketData, TechnicalIndicators


class MomentumStrategy(BaseStrategy):
    """
    モメンタム戦略

    特徴:
    - 複数時間軸でのモメンタム確認
    - RSI, MACDを使用したトレンド確認
    - 出来高確認付き
    """

    def __init__(
        self,
        product_codes: List[str],
        lookback_periods: List[int] = [5, 15, 30, 60],
        rsi_oversold: float = 30,
        rsi_overbought: float = 70,
        macd_signal: bool = True,
        weight: float = 1.0,
    ):
        """
        Args:
            lookback_periods: 参照期間リスト
            rsi_oversold: RSI売られすぎ閾値
            rsi_overbought: RSI買われすぎ閾値
            macd_signal: MACDシグナルを使用するか
        """
        super().__init__(
            name="Momentum",
            product_codes=product_codes,
            weight=weight,
        )

        self.lookback_periods = lookback_periods
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.macd_signal = macd_signal

        # 価格履歴
        self.price_history: Dict[str, pd.DataFrame] = {}
        self.max_history = max(lookback_periods) * 3

    def update(self, market_data: MarketData) -> None:
        """市場データで内部状態更新"""
        product = market_data.product_code

        if product not in self.price_history:
            self.price_history[product] = pd.DataFrame(columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume'
            ])

        new_row = pd.DataFrame([{
            'timestamp': market_data.timestamp,
            'open': market_data.open,
            'high': market_data.high,
            'low': market_data.low,
            'close': market_data.close,
            'volume': market_data.volume,
        }])

        self.price_history[product] = pd.concat(
            [self.price_history[product], new_row],
            ignore_index=True
        )

        # 最大履歴数を超えたら古いデータを削除
        if len(self.price_history[product]) > self.max_history:
            self.price_history[product] = self.price_history[product].iloc[-self.max_history:]

    def generate_signal(self, market_data: MarketData) -> Optional[Signal]:
        """シグナル生成"""
        product = market_data.product_code

        if product not in self.price_history:
            return None

        df = self.price_history[product]
        if len(df) < max(self.lookback_periods):
            return None

        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']

        # 複数時間軸モメンタム計算
        momentum_scores = []
        for period in self.lookback_periods:
            if len(close) >= period:
                returns = (close.iloc[-1] - close.iloc[-period]) / close.iloc[-period]
                momentum_scores.append(returns)

        if not momentum_scores:
            return None

        avg_momentum = np.mean(momentum_scores)

        # RSI計算
        rsi = TechnicalIndicators.rsi(close, 14)
        current_rsi = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50

        # MACD計算
        macd_data = TechnicalIndicators.macd(close)
        macd_hist = macd_data['histogram'].iloc[-1] if not pd.isna(macd_data['histogram'].iloc[-1]) else 0

        # 出来高確認
        avg_volume = volume.iloc[-20:].mean() if len(volume) >= 20 else volume.mean()
        current_volume = volume.iloc[-1]
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1

        # シグナル判定
        signal_type = self._determine_signal(
            avg_momentum, current_rsi, macd_hist, volume_ratio
        )

        if signal_type == SignalType.HOLD:
            return None

        # 信頼度計算
        confidence = self._calculate_confidence(
            avg_momentum, current_rsi, macd_hist, volume_ratio
        )

        # ポジションサイズ（信頼度に基づく）
        base_size = 0.01  # 0.01 BTC
        size = base_size * confidence

        self.signals_generated += 1

        return Signal(
            signal_type=signal_type,
            product_code=product,
            price=market_data.close,
            size=size,
            confidence=confidence,
            strategy_name=self.name,
            metadata={
                "avg_momentum": avg_momentum,
                "rsi": current_rsi,
                "macd_histogram": macd_hist,
                "volume_ratio": volume_ratio,
                "momentum_scores": momentum_scores,
            },
        )

    def _determine_signal(
        self,
        avg_momentum: float,
        rsi: float,
        macd_hist: float,
        volume_ratio: float
    ) -> SignalType:
        """シグナルタイプ決定"""
        buy_signals = 0
        sell_signals = 0

        # モメンタムシグナル
        if avg_momentum > 0.005:  # 0.5%以上の上昇
            buy_signals += 1
        elif avg_momentum < -0.005:  # 0.5%以上の下落
            sell_signals += 1

        # RSIシグナル
        if rsi < self.rsi_oversold:
            buy_signals += 1
        elif rsi > self.rsi_overbought:
            sell_signals += 1

        # MACDシグナル
        if self.macd_signal:
            if macd_hist > 0:
                buy_signals += 1
            elif macd_hist < 0:
                sell_signals += 1

        # 出来高確認（シグナルの強化）
        volume_confirmed = volume_ratio > 1.2

        # 最終判定
        if buy_signals >= 2 and buy_signals > sell_signals:
            if volume_confirmed:
                return SignalType.BUY
            elif buy_signals >= 3:
                return SignalType.BUY
        elif sell_signals >= 2 and sell_signals > buy_signals:
            if volume_confirmed:
                return SignalType.SELL
            elif sell_signals >= 3:
                return SignalType.SELL

        return SignalType.HOLD

    def _calculate_confidence(
        self,
        avg_momentum: float,
        rsi: float,
        macd_hist: float,
        volume_ratio: float
    ) -> float:
        """信頼度計算"""
        confidence = 0.5

        # モメンタムの強さ
        momentum_strength = min(abs(avg_momentum) / 0.02, 0.2)
        confidence += momentum_strength

        # RSIの極値
        if rsi < 20 or rsi > 80:
            confidence += 0.1

        # MACDの強さ
        macd_strength = min(abs(macd_hist) / 100, 0.1)
        confidence += macd_strength

        # 出来高確認
        if volume_ratio > 1.5:
            confidence += 0.1

        return max(0.0, min(1.0, confidence))

    def get_trend_strength(self, product: str) -> float:
        """トレンドの強さを取得（-1 〜 1）"""
        if product not in self.price_history:
            return 0.0

        df = self.price_history[product]
        if len(df) < 20:
            return 0.0

        close = df['close']

        # ADX的なトレンド強度計算
        returns = close.pct_change().iloc[-20:]
        trend_direction = np.sign(returns.mean())
        trend_consistency = abs(returns.mean()) / (returns.std() + 1e-8)

        return trend_direction * min(trend_consistency, 1.0)

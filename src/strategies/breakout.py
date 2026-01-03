"""
Breakout Strategy
==================
価格のブレイクアウトを検出して利益を狙う戦略
"""

from typing import Optional, Dict, List
from datetime import datetime
import pandas as pd
import numpy as np
from loguru import logger

from .base import BaseStrategy, Signal, SignalType, MarketData, TechnicalIndicators


class BreakoutStrategy(BaseStrategy):
    """
    ブレイクアウト戦略

    特徴:
    - 価格の高値・安値ブレイクを検出
    - ATRを使用したボラティリティ考慮
    - 出来高確認による偽ブレイクアウト回避
    - フィボナッチリトレースメントを使用したターゲット設定
    """

    def __init__(
        self,
        product_codes: List[str],
        lookback_period: int = 100,
        volume_confirmation: bool = True,
        atr_multiplier: float = 2.0,
        weight: float = 1.0,
    ):
        """
        Args:
            lookback_period: 高値・安値参照期間
            volume_confirmation: 出来高確認を使用するか
            atr_multiplier: ATR倍率（フィルタリング用）
        """
        super().__init__(
            name="Breakout",
            product_codes=product_codes,
            weight=weight,
        )

        self.lookback_period = lookback_period
        self.volume_confirmation = volume_confirmation
        self.atr_multiplier = atr_multiplier

        # 価格履歴
        self.price_history: Dict[str, pd.DataFrame] = {}
        self.max_history = lookback_period * 2

        # ブレイクアウトトラッキング
        self.recent_breakouts: Dict[str, List[Dict]] = {}

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

        if len(self.price_history[product]) > self.max_history:
            self.price_history[product] = self.price_history[product].iloc[-self.max_history:]

    def generate_signal(self, market_data: MarketData) -> Optional[Signal]:
        """シグナル生成"""
        product = market_data.product_code

        if product not in self.price_history:
            return None

        df = self.price_history[product]
        if len(df) < self.lookback_period:
            return None

        high = df['high']
        low = df['low']
        close = df['close']
        volume = df['volume']

        current_price = close.iloc[-1]

        # 期間高値・安値
        period_high = high.iloc[-self.lookback_period:-1].max()
        period_low = low.iloc[-self.lookback_period:-1].min()

        # ATR計算
        atr = TechnicalIndicators.atr(high, low, close, 14)
        current_atr = atr.iloc[-1] if not pd.isna(atr.iloc[-1]) else 0

        # 出来高分析
        avg_volume = volume.iloc[-20:].mean() if len(volume) >= 20 else volume.mean()
        current_volume = volume.iloc[-1]
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1

        # ブレイクアウト判定
        signal_type = self._detect_breakout(
            current_price, period_high, period_low,
            current_atr, volume_ratio
        )

        if signal_type == SignalType.HOLD:
            return None

        # 偽ブレイクアウトフィルタリング
        if not self._validate_breakout(product, signal_type, current_price, current_atr):
            return None

        # 信頼度計算
        confidence = self._calculate_confidence(
            current_price, period_high, period_low,
            current_atr, volume_ratio
        )

        # ポジションサイズ（ATRベース）
        base_size = 0.01
        risk_adjusted_size = base_size * (1 / (current_atr / current_price * 100 + 1))
        size = risk_adjusted_size * confidence

        self.signals_generated += 1

        # ブレイクアウト記録
        self._record_breakout(product, signal_type, current_price)

        return Signal(
            signal_type=signal_type,
            product_code=product,
            price=current_price,
            size=size,
            confidence=confidence,
            strategy_name=self.name,
            metadata={
                "period_high": period_high,
                "period_low": period_low,
                "atr": current_atr,
                "volume_ratio": volume_ratio,
                "breakout_strength": self._calculate_breakout_strength(
                    current_price, period_high, period_low, signal_type
                ),
            },
        )

    def _detect_breakout(
        self,
        current_price: float,
        period_high: float,
        period_low: float,
        atr: float,
        volume_ratio: float
    ) -> SignalType:
        """ブレイクアウト検出"""
        # ATRを使用したノイズフィルタリング
        breakout_threshold = atr * self.atr_multiplier * 0.1

        # 上方ブレイクアウト
        if current_price > period_high + breakout_threshold:
            if self.volume_confirmation and volume_ratio < 1.2:
                return SignalType.HOLD  # 出来高不足
            return SignalType.BUY

        # 下方ブレイクアウト
        if current_price < period_low - breakout_threshold:
            if self.volume_confirmation and volume_ratio < 1.2:
                return SignalType.HOLD  # 出来高不足
            return SignalType.SELL

        return SignalType.HOLD

    def _validate_breakout(
        self,
        product: str,
        signal_type: SignalType,
        current_price: float,
        atr: float
    ) -> bool:
        """偽ブレイクアウトフィルタリング"""
        if product not in self.recent_breakouts:
            return True

        recent = self.recent_breakouts[product]

        # 直近のブレイクアウトと逆方向なら慎重に
        for breakout in recent[-3:]:
            if breakout['type'] != signal_type:
                # 直近に逆ブレイクアウトがあった場合、
                # 十分な価格変動があるか確認
                price_change = abs(current_price - breakout['price']) / breakout['price']
                if price_change < 0.005:  # 0.5%未満の変動は無視
                    return False

        return True

    def _record_breakout(
        self,
        product: str,
        signal_type: SignalType,
        price: float
    ) -> None:
        """ブレイクアウト記録"""
        if product not in self.recent_breakouts:
            self.recent_breakouts[product] = []

        self.recent_breakouts[product].append({
            'type': signal_type,
            'price': price,
            'timestamp': datetime.now(),
        })

        # 直近10件のみ保持
        if len(self.recent_breakouts[product]) > 10:
            self.recent_breakouts[product] = self.recent_breakouts[product][-10:]

    def _calculate_breakout_strength(
        self,
        current_price: float,
        period_high: float,
        period_low: float,
        signal_type: SignalType
    ) -> float:
        """ブレイクアウト強度計算"""
        if signal_type == SignalType.BUY:
            return (current_price - period_high) / period_high
        elif signal_type == SignalType.SELL:
            return (period_low - current_price) / period_low
        return 0.0

    def _calculate_confidence(
        self,
        current_price: float,
        period_high: float,
        period_low: float,
        atr: float,
        volume_ratio: float
    ) -> float:
        """信頼度計算"""
        confidence = 0.5

        # ブレイクアウトの明確さ
        range_size = period_high - period_low
        if range_size > 0:
            breakout_clarity = min(
                abs(current_price - (period_high + period_low) / 2) / range_size,
                0.2
            )
            confidence += breakout_clarity

        # 出来高確認
        if volume_ratio > 1.5:
            confidence += 0.15
        elif volume_ratio > 2.0:
            confidence += 0.25

        # ATRに対するブレイクアウトサイズ
        breakout_size = max(
            current_price - period_high,
            period_low - current_price
        )
        if atr > 0 and breakout_size > atr:
            confidence += 0.1

        return max(0.0, min(1.0, confidence))

    def get_support_resistance(self, product: str) -> Dict[str, float]:
        """サポート・レジスタンスレベル取得"""
        if product not in self.price_history:
            return {"support": 0, "resistance": 0}

        df = self.price_history[product]
        if len(df) < self.lookback_period:
            return {"support": 0, "resistance": 0}

        high = df['high'].iloc[-self.lookback_period:]
        low = df['low'].iloc[-self.lookback_period:]

        return {
            "resistance": high.max(),
            "support": low.min(),
            "pivot": (high.max() + low.min()) / 2,
        }

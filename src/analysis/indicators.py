#!/usr/bin/env python3
"""
================================================================================
    📊 高度テクニカル指標モジュール
================================================================================
    - RSI (Relative Strength Index)
    - MACD (Moving Average Convergence Divergence)
    - ボリンジャーバンド
    - 複数時間足分析
================================================================================
"""

import numpy as np
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass
from collections import deque
from datetime import datetime, timedelta


@dataclass
class IndicatorSignal:
    """テクニカル指標シグナル"""
    name: str
    value: float
    signal: str  # "BUY", "SELL", "NEUTRAL"
    strength: float  # 0.0 - 1.0


class TechnicalIndicators:
    """高度テクニカル指標クラス"""

    def __init__(self, max_length: int = 500):
        self.prices: deque = deque(maxlen=max_length)
        self.timestamps: deque = deque(maxlen=max_length)

        # 時間足別データ
        self.candles_1m: List[dict] = []
        self.candles_5m: List[dict] = []
        self.candles_15m: List[dict] = []

        self._last_candle_time: Dict[str, datetime] = {}

    def add_price(self, price: float, timestamp: datetime = None):
        """価格を追加"""
        if timestamp is None:
            timestamp = datetime.now()
        self.prices.append(price)
        self.timestamps.append(timestamp)
        self._update_candles(price, timestamp)

    def _update_candles(self, price: float, timestamp: datetime):
        """ローソク足を更新"""
        for interval, candles in [
            (1, self.candles_1m),
            (5, self.candles_5m),
            (15, self.candles_15m)
        ]:
            key = f"{interval}m"
            candle_time = timestamp.replace(
                minute=(timestamp.minute // interval) * interval,
                second=0, microsecond=0
            )

            if key not in self._last_candle_time or self._last_candle_time[key] != candle_time:
                # 新しいローソク足
                candles.append({
                    'time': candle_time,
                    'open': price,
                    'high': price,
                    'low': price,
                    'close': price
                })
                if len(candles) > 100:
                    candles.pop(0)
                self._last_candle_time[key] = candle_time
            else:
                # 既存のローソク足を更新
                if candles:
                    candles[-1]['high'] = max(candles[-1]['high'], price)
                    candles[-1]['low'] = min(candles[-1]['low'], price)
                    candles[-1]['close'] = price

    # =========================================================================
    # RSI
    # =========================================================================

    def rsi(self, period: int = 14) -> Optional[float]:
        """RSI計算"""
        if len(self.prices) < period + 1:
            return None

        prices = list(self.prices)[-period-1:]
        deltas = np.diff(prices)

        gains = deltas.copy()
        losses = deltas.copy()
        gains[gains < 0] = 0
        losses[losses > 0] = 0
        losses = abs(losses)

        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)

        if avg_loss < 1e-10:
            return 100.0

        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def rsi_signal(self) -> Optional[IndicatorSignal]:
        """RSIシグナル"""
        rsi = self.rsi()
        if rsi is None:
            return None

        if rsi < 30:
            return IndicatorSignal("RSI", rsi, "BUY", (30 - rsi) / 30)
        elif rsi > 70:
            return IndicatorSignal("RSI", rsi, "SELL", (rsi - 70) / 30)
        else:
            return IndicatorSignal("RSI", rsi, "NEUTRAL", 0.0)

    # =========================================================================
    # MACD
    # =========================================================================

    def ema(self, period: int, prices: List[float] = None) -> Optional[float]:
        """指数移動平均"""
        if prices is None:
            prices = list(self.prices)

        if len(prices) < period:
            return None

        multiplier = 2 / (period + 1)
        ema = prices[0]

        for price in prices[1:]:
            ema = (price - ema) * multiplier + ema

        return ema

    def macd(self, fast: int = 12, slow: int = 26, signal: int = 9) -> Optional[Tuple[float, float, float]]:
        """MACD計算 (MACD線, シグナル線, ヒストグラム)"""
        if len(self.prices) < slow + signal:
            return None

        prices = list(self.prices)

        # MACD線 = 短期EMA - 長期EMA
        ema_fast = self.ema(fast, prices)
        ema_slow = self.ema(slow, prices)

        if ema_fast is None or ema_slow is None:
            return None

        macd_line = ema_fast - ema_slow

        # シグナル線 = MACD線のEMA
        # 過去のMACD値を計算
        macd_values = []
        for i in range(signal + 5):
            if len(prices) - i < slow:
                break
            subset = prices[:len(prices)-i] if i > 0 else prices
            ef = self.ema(fast, subset)
            es = self.ema(slow, subset)
            if ef and es:
                macd_values.insert(0, ef - es)

        if len(macd_values) < signal:
            return None

        signal_line = self.ema(signal, macd_values)
        if signal_line is None:
            return None

        histogram = macd_line - signal_line

        return macd_line, signal_line, histogram

    def macd_signal(self) -> Optional[IndicatorSignal]:
        """MACDシグナル"""
        result = self.macd()
        if result is None:
            return None

        macd_line, signal_line, histogram = result

        # ヒストグラムの方向で判断
        if histogram > 0 and macd_line > signal_line:
            strength = min(abs(histogram) / 100, 1.0)
            return IndicatorSignal("MACD", histogram, "BUY", strength)
        elif histogram < 0 and macd_line < signal_line:
            strength = min(abs(histogram) / 100, 1.0)
            return IndicatorSignal("MACD", histogram, "SELL", strength)
        else:
            return IndicatorSignal("MACD", histogram, "NEUTRAL", 0.0)

    # =========================================================================
    # ボリンジャーバンド
    # =========================================================================

    def bollinger_bands(self, period: int = 20, std_dev: float = 2.0) -> Optional[Tuple[float, float, float]]:
        """ボリンジャーバンド (上限, 中央, 下限)"""
        if len(self.prices) < period:
            return None

        prices = list(self.prices)[-period:]
        middle = np.mean(prices)
        std = np.std(prices)

        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)

        return upper, middle, lower

    def bollinger_signal(self) -> Optional[IndicatorSignal]:
        """ボリンジャーバンドシグナル"""
        result = self.bollinger_bands()
        if result is None or not self.prices:
            return None

        upper, middle, lower = result
        current = self.prices[-1]

        # バンド幅に対する現在位置 (0-100)
        band_width = upper - lower
        if band_width < 1e-10:
            return None

        position = (current - lower) / band_width * 100

        if current <= lower:
            # 下限到達 = 買いシグナル
            strength = min((lower - current) / (band_width * 0.1) + 0.5, 1.0)
            return IndicatorSignal("BB", position, "BUY", strength)
        elif current >= upper:
            # 上限到達 = 売りシグナル
            strength = min((current - upper) / (band_width * 0.1) + 0.5, 1.0)
            return IndicatorSignal("BB", position, "SELL", strength)
        else:
            return IndicatorSignal("BB", position, "NEUTRAL", 0.0)

    # =========================================================================
    # 複数時間足分析
    # =========================================================================

    def multi_timeframe_trend(self) -> Optional[Dict[str, str]]:
        """複数時間足トレンド分析"""
        trends = {}

        for name, candles in [
            ("1m", self.candles_1m),
            ("5m", self.candles_5m),
            ("15m", self.candles_15m)
        ]:
            if len(candles) < 3:
                trends[name] = "UNKNOWN"
                continue

            # 直近3本のローソク足で判断
            closes = [c['close'] for c in candles[-3:]]

            if closes[-1] > closes[-2] > closes[-3]:
                trends[name] = "UP"
            elif closes[-1] < closes[-2] < closes[-3]:
                trends[name] = "DOWN"
            else:
                trends[name] = "SIDEWAYS"

        return trends

    def multi_timeframe_signal(self) -> Optional[IndicatorSignal]:
        """複数時間足統合シグナル"""
        trends = self.multi_timeframe_trend()
        if not trends:
            return None

        up_count = sum(1 for t in trends.values() if t == "UP")
        down_count = sum(1 for t in trends.values() if t == "DOWN")

        if up_count >= 2:
            strength = up_count / 3
            return IndicatorSignal("MTF", up_count, "BUY", strength)
        elif down_count >= 2:
            strength = down_count / 3
            return IndicatorSignal("MTF", down_count, "SELL", strength)
        else:
            return IndicatorSignal("MTF", 0, "NEUTRAL", 0.0)

    # =========================================================================
    # 統合シグナル
    # =========================================================================

    def get_all_signals(self) -> List[IndicatorSignal]:
        """全シグナル取得"""
        signals = []

        for method in [
            self.rsi_signal,
            self.macd_signal,
            self.bollinger_signal,
            self.multi_timeframe_signal
        ]:
            sig = method()
            if sig:
                signals.append(sig)

        return signals

    def composite_signal(self) -> Tuple[str, float, str]:
        """
        統合シグナル計算
        Returns: (シグナル, 信頼度, 理由)
        """
        signals = self.get_all_signals()
        if not signals:
            return "NEUTRAL", 0.0, "データ不足"

        buy_score = 0.0
        sell_score = 0.0
        reasons = []

        weights = {"RSI": 1.5, "MACD": 1.2, "BB": 1.0, "MTF": 1.3}

        for sig in signals:
            weight = weights.get(sig.name, 1.0)

            if sig.signal == "BUY":
                buy_score += sig.strength * weight
                if sig.strength > 0.3:
                    reasons.append(f"{sig.name}↑")
            elif sig.signal == "SELL":
                sell_score += sig.strength * weight
                if sig.strength > 0.3:
                    reasons.append(f"{sig.name}↓")

        total = buy_score + sell_score
        if total < 0.1:
            return "NEUTRAL", 0.0, "シグナルなし"

        if buy_score > sell_score * 1.2:
            confidence = min(buy_score / 3.0, 1.0)
            return "BUY", confidence, " ".join(reasons) or "買いシグナル"
        elif sell_score > buy_score * 1.2:
            confidence = min(sell_score / 3.0, 1.0)
            return "SELL", confidence, " ".join(reasons) or "売りシグナル"
        else:
            return "NEUTRAL", 0.0, "方向不明"

    # =========================================================================
    # ユーティリティ
    # =========================================================================

    def current_price(self) -> Optional[float]:
        """現在価格"""
        return self.prices[-1] if self.prices else None

    def price_change_pct(self, periods: int = 10) -> Optional[float]:
        """価格変動率"""
        if len(self.prices) < periods:
            return None
        return (self.prices[-1] - self.prices[-periods]) / self.prices[-periods] * 100

    def volatility(self, period: int = 20) -> Optional[float]:
        """ボラティリティ (標準偏差)"""
        if len(self.prices) < period:
            return None
        prices = list(self.prices)[-period:]
        return np.std(prices) / np.mean(prices) * 100

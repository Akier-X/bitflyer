"""
Base Strategy Interface
========================
全戦略の基底クラス
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, List, Any
from datetime import datetime
import pandas as pd
import numpy as np


class SignalType(Enum):
    """シグナルタイプ"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    CLOSE_LONG = "CLOSE_LONG"
    CLOSE_SHORT = "CLOSE_SHORT"


@dataclass
class Signal:
    """トレーディングシグナル"""
    signal_type: SignalType
    product_code: str
    price: float
    size: float
    confidence: float  # 0.0 - 1.0
    strategy_name: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_buy(self) -> bool:
        return self.signal_type == SignalType.BUY

    @property
    def is_sell(self) -> bool:
        return self.signal_type == SignalType.SELL

    @property
    def is_entry(self) -> bool:
        return self.signal_type in [SignalType.BUY, SignalType.SELL]

    @property
    def is_exit(self) -> bool:
        return self.signal_type in [SignalType.CLOSE_LONG, SignalType.CLOSE_SHORT]


@dataclass
class MarketData:
    """市場データ"""
    product_code: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap: Optional[float] = None

    # オーダーブック
    best_bid: Optional[float] = None
    best_ask: Optional[float] = None
    bid_volume: Optional[float] = None
    ask_volume: Optional[float] = None

    # 追加指標
    order_book_imbalance: Optional[float] = None
    spread: Optional[float] = None


class BaseStrategy(ABC):
    """戦略基底クラス"""

    def __init__(
        self,
        name: str,
        product_codes: List[str],
        weight: float = 1.0,
        enabled: bool = True,
    ):
        """
        Args:
            name: 戦略名
            product_codes: 対象取引ペアリスト
            weight: 戦略の重み
            enabled: 有効/無効
        """
        self.name = name
        self.product_codes = product_codes
        self.weight = weight
        self.enabled = enabled

        # パフォーマンス追跡
        self.signals_generated = 0
        self.successful_signals = 0
        self.total_pnl = 0.0

        # 状態
        self.position: Dict[str, float] = {}
        self.last_signal: Optional[Signal] = None

    @abstractmethod
    def generate_signal(self, market_data: MarketData) -> Optional[Signal]:
        """
        シグナル生成（サブクラスで実装）

        Args:
            market_data: 市場データ

        Returns:
            Signal or None
        """
        pass

    @abstractmethod
    def update(self, market_data: MarketData) -> None:
        """
        内部状態更新（サブクラスで実装）
        """
        pass

    def calculate_position_size(
        self,
        signal: Signal,
        capital: float,
        risk_per_trade: float = 0.02
    ) -> float:
        """ポジションサイズ計算"""
        max_position = capital * risk_per_trade / signal.price
        return min(signal.size, max_position)

    def get_performance_metrics(self) -> Dict[str, float]:
        """パフォーマンス指標取得"""
        hit_rate = (
            self.successful_signals / self.signals_generated
            if self.signals_generated > 0
            else 0.0
        )
        return {
            "signals_generated": self.signals_generated,
            "successful_signals": self.successful_signals,
            "hit_rate": hit_rate,
            "total_pnl": self.total_pnl,
        }

    def reset(self) -> None:
        """状態リセット"""
        self.position = {}
        self.last_signal = None
        self.signals_generated = 0
        self.successful_signals = 0
        self.total_pnl = 0.0


class TechnicalIndicators:
    """テクニカル指標計算ユーティリティ"""

    @staticmethod
    def sma(data: pd.Series, period: int) -> pd.Series:
        """単純移動平均"""
        return data.rolling(window=period).mean()

    @staticmethod
    def ema(data: pd.Series, period: int) -> pd.Series:
        """指数移動平均"""
        return data.ewm(span=period, adjust=False).mean()

    @staticmethod
    def rsi(data: pd.Series, period: int = 14) -> pd.Series:
        """RSI (Relative Strength Index)"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def macd(
        data: pd.Series,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9
    ) -> Dict[str, pd.Series]:
        """MACD"""
        fast_ema = TechnicalIndicators.ema(data, fast_period)
        slow_ema = TechnicalIndicators.ema(data, slow_period)

        macd_line = fast_ema - slow_ema
        signal_line = TechnicalIndicators.ema(macd_line, signal_period)
        histogram = macd_line - signal_line

        return {
            "macd": macd_line,
            "signal": signal_line,
            "histogram": histogram,
        }

    @staticmethod
    def bollinger_bands(
        data: pd.Series,
        period: int = 20,
        std_dev: float = 2.0
    ) -> Dict[str, pd.Series]:
        """ボリンジャーバンド"""
        middle = TechnicalIndicators.sma(data, period)
        std = data.rolling(window=period).std()

        return {
            "upper": middle + (std * std_dev),
            "middle": middle,
            "lower": middle - (std * std_dev),
        }

    @staticmethod
    def atr(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14
    ) -> pd.Series:
        """ATR (Average True Range)"""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()

    @staticmethod
    def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """OBV (On Balance Volume)"""
        direction = np.sign(close.diff())
        return (direction * volume).cumsum()

    @staticmethod
    def vwap(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series
    ) -> pd.Series:
        """VWAP (Volume Weighted Average Price)"""
        typical_price = (high + low + close) / 3
        return (typical_price * volume).cumsum() / volume.cumsum()

    @staticmethod
    def stochastic(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        k_period: int = 14,
        d_period: int = 3
    ) -> Dict[str, pd.Series]:
        """ストキャスティクス"""
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()

        k = 100 * (close - lowest_low) / (highest_high - lowest_low)
        d = k.rolling(window=d_period).mean()

        return {"k": k, "d": d}

    @staticmethod
    def williams_r(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14
    ) -> pd.Series:
        """ウィリアムズ%R"""
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()

        return -100 * (highest_high - close) / (highest_high - lowest_low)

    @staticmethod
    def momentum(data: pd.Series, period: int = 10) -> pd.Series:
        """モメンタム"""
        return data - data.shift(period)

    @staticmethod
    def z_score(data: pd.Series, period: int = 20) -> pd.Series:
        """Zスコア"""
        mean = data.rolling(window=period).mean()
        std = data.rolling(window=period).std()
        return (data - mean) / std

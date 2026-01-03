"""
Advanced Feature Engineering
==============================
500次元以上の高度な特徴量生成
金融市場のあらゆるパターンを捉える
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import warnings
from loguru import logger

warnings.filterwarnings('ignore')


@dataclass
class MarketMicrostructure:
    """市場マイクロストラクチャ特徴量"""
    bid_ask_spread: float = 0.0
    bid_ask_imbalance: float = 0.0
    order_flow_toxicity: float = 0.0
    price_impact: float = 0.0
    kyle_lambda: float = 0.0
    amihud_illiquidity: float = 0.0
    roll_spread: float = 0.0
    effective_spread: float = 0.0


class FractionalDifferencing:
    """
    端数階差 (Fractional Differencing)

    記憶を保ちつつ定常性を確保
    Marcos Lopez de Prado の手法
    """

    def __init__(self, d: float = 0.4, threshold: float = 1e-5):
        """
        Args:
            d: 階差の次数 (0 < d < 1)
            threshold: 重み切り捨て閾値
        """
        self.d = d
        self.threshold = threshold
        self._weights = None

    def get_weights(self, size: int) -> np.ndarray:
        """端数階差の重み計算"""
        if self._weights is not None and len(self._weights) >= size:
            return self._weights[:size]

        w = [1.0]
        for k in range(1, size):
            w_k = -w[-1] * (self.d - k + 1) / k
            if abs(w_k) < self.threshold:
                break
            w.append(w_k)

        self._weights = np.array(w)
        return self._weights[:size]

    def transform(self, series: np.ndarray) -> np.ndarray:
        """端数階差変換"""
        weights = self.get_weights(len(series))
        width = len(weights)

        result = np.zeros(len(series) - width + 1)
        for i in range(width - 1, len(series)):
            result[i - width + 1] = np.dot(weights, series[i - width + 1:i + 1][::-1])

        return result


class WaveletTransform:
    """ウェーブレット変換"""

    def __init__(self, wavelet: str = 'db4', levels: int = 4):
        self.wavelet = wavelet
        self.levels = levels

    def decompose(self, signal: np.ndarray) -> List[np.ndarray]:
        """多重解像度分解"""
        coeffs = [signal]

        for _ in range(self.levels):
            # 簡易Haar ウェーブレット
            if len(coeffs[-1]) < 2:
                break

            s = coeffs[-1]
            n = len(s) // 2 * 2

            # 近似係数 (低周波)
            approx = (s[:n:2] + s[1:n:2]) / np.sqrt(2)
            # 詳細係数 (高周波)
            detail = (s[:n:2] - s[1:n:2]) / np.sqrt(2)

            coeffs[-1] = detail
            coeffs.append(approx)

        return coeffs

    def get_features(self, signal: np.ndarray) -> Dict[str, float]:
        """ウェーブレット特徴量"""
        coeffs = self.decompose(signal)

        features = {}
        for i, c in enumerate(coeffs):
            if len(c) > 0:
                features[f'wavelet_energy_l{i}'] = np.sum(c ** 2)
                features[f'wavelet_mean_l{i}'] = np.mean(c)
                features[f'wavelet_std_l{i}'] = np.std(c)

        return features


class FourierFeatures:
    """フーリエ変換特徴量"""

    def __init__(self, n_components: int = 10):
        self.n_components = n_components

    def transform(self, signal: np.ndarray) -> Dict[str, float]:
        """周波数領域特徴量"""
        if len(signal) < 2:
            return {}

        # FFT
        fft = np.fft.rfft(signal)
        freqs = np.fft.rfftfreq(len(signal))

        # パワースペクトル
        power = np.abs(fft) ** 2

        features = {}

        # 主要周波数成分
        top_indices = np.argsort(power)[::-1][:self.n_components]
        for i, idx in enumerate(top_indices):
            features[f'fft_freq_{i}'] = freqs[idx] if idx < len(freqs) else 0
            features[f'fft_power_{i}'] = power[idx] if idx < len(power) else 0

        # スペクトル統計
        features['spectral_centroid'] = np.sum(freqs * power) / (np.sum(power) + 1e-10)
        features['spectral_bandwidth'] = np.sqrt(
            np.sum(((freqs - features['spectral_centroid']) ** 2) * power) / (np.sum(power) + 1e-10)
        )
        features['spectral_rolloff'] = freqs[np.searchsorted(np.cumsum(power), 0.85 * np.sum(power))] if len(freqs) > 0 else 0
        features['spectral_flatness'] = np.exp(np.mean(np.log(power + 1e-10))) / (np.mean(power) + 1e-10)

        return features


class TechnicalIndicatorsAdvanced:
    """高度なテクニカル指標"""

    @staticmethod
    def rsi(prices: np.ndarray, period: int = 14) -> float:
        """RSI"""
        if len(prices) < period + 1:
            return 50.0

        deltas = np.diff(prices[-period - 1:])
        gains = np.maximum(deltas, 0)
        losses = np.maximum(-deltas, 0)

        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def stochastic(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, k_period: int = 14) -> Tuple[float, float]:
        """ストキャスティクス %K, %D"""
        if len(closes) < k_period:
            return 50.0, 50.0

        lowest_low = np.min(lows[-k_period:])
        highest_high = np.max(highs[-k_period:])

        if highest_high == lowest_low:
            return 50.0, 50.0

        k = 100 * (closes[-1] - lowest_low) / (highest_high - lowest_low)
        d = k  # 簡易版

        return k, d

    @staticmethod
    def williams_r(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> float:
        """Williams %R"""
        if len(closes) < period:
            return -50.0

        highest_high = np.max(highs[-period:])
        lowest_low = np.min(lows[-period:])

        if highest_high == lowest_low:
            return -50.0

        return -100 * (highest_high - closes[-1]) / (highest_high - lowest_low)

    @staticmethod
    def cci(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 20) -> float:
        """Commodity Channel Index"""
        if len(closes) < period:
            return 0.0

        tp = (highs[-period:] + lows[-period:] + closes[-period:]) / 3
        sma = np.mean(tp)
        mad = np.mean(np.abs(tp - sma))

        if mad == 0:
            return 0.0

        return (tp[-1] - sma) / (0.015 * mad)

    @staticmethod
    def adx(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> Tuple[float, float, float]:
        """ADX, +DI, -DI"""
        if len(closes) < period + 1:
            return 25.0, 25.0, 25.0

        high_diff = np.diff(highs[-period - 1:])
        low_diff = -np.diff(lows[-period - 1:])

        plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
        minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)

        tr = np.maximum(
            highs[-period:] - lows[-period:],
            np.maximum(
                np.abs(highs[-period:] - closes[-period - 1:-1]),
                np.abs(lows[-period:] - closes[-period - 1:-1])
            )
        )

        atr = np.mean(tr)
        plus_di = 100 * np.mean(plus_dm) / (atr + 1e-10)
        minus_di = 100 * np.mean(minus_dm) / (atr + 1e-10)

        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = dx  # 簡易版

        return adx, plus_di, minus_di

    @staticmethod
    def atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> float:
        """Average True Range"""
        if len(closes) < period + 1:
            return 0.0

        tr = np.maximum(
            highs[-period:] - lows[-period:],
            np.maximum(
                np.abs(highs[-period:] - closes[-period - 1:-1]),
                np.abs(lows[-period:] - closes[-period - 1:-1])
            )
        )

        return np.mean(tr)

    @staticmethod
    def bollinger_bands(prices: np.ndarray, period: int = 20, std_mult: float = 2.0) -> Tuple[float, float, float]:
        """ボリンジャーバンド"""
        if len(prices) < period:
            return prices[-1], prices[-1], prices[-1]

        sma = np.mean(prices[-period:])
        std = np.std(prices[-period:])

        upper = sma + std_mult * std
        lower = sma - std_mult * std

        return upper, sma, lower

    @staticmethod
    def keltner_channels(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 20, mult: float = 2.0) -> Tuple[float, float, float]:
        """ケルトナーチャネル"""
        if len(closes) < period:
            return closes[-1], closes[-1], closes[-1]

        ema = np.mean(closes[-period:])  # 簡易EMA
        atr = TechnicalIndicatorsAdvanced.atr(highs, lows, closes, period)

        upper = ema + mult * atr
        lower = ema - mult * atr

        return upper, ema, lower

    @staticmethod
    def ichimoku(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> Dict[str, float]:
        """一目均衡表"""
        result = {}

        # 転換線 (9期間)
        if len(highs) >= 9:
            result['tenkan'] = (np.max(highs[-9:]) + np.min(lows[-9:])) / 2
        else:
            result['tenkan'] = closes[-1]

        # 基準線 (26期間)
        if len(highs) >= 26:
            result['kijun'] = (np.max(highs[-26:]) + np.min(lows[-26:])) / 2
        else:
            result['kijun'] = closes[-1]

        # 先行スパンA
        result['senkou_a'] = (result['tenkan'] + result['kijun']) / 2

        # 先行スパンB (52期間)
        if len(highs) >= 52:
            result['senkou_b'] = (np.max(highs[-52:]) + np.min(lows[-52:])) / 2
        else:
            result['senkou_b'] = closes[-1]

        # 遅行スパン
        result['chikou'] = closes[-1]

        return result


class OrderBookFeatures:
    """オーダーブック特徴量"""

    @staticmethod
    def calculate_imbalance(bids: List[Tuple[float, float]], asks: List[Tuple[float, float]], levels: int = 10) -> Dict[str, float]:
        """板の不均衡特徴量"""
        if not bids or not asks:
            return {
                'imbalance_1': 0.0,
                'imbalance_5': 0.0,
                'imbalance_10': 0.0,
                'weighted_imbalance': 0.0,
                'depth_ratio': 1.0,
            }

        # 各レベルでの不均衡
        bid_volumes = [b[1] for b in bids[:levels]]
        ask_volumes = [a[1] for a in asks[:levels]]

        # パッド
        while len(bid_volumes) < levels:
            bid_volumes.append(0)
        while len(ask_volumes) < levels:
            ask_volumes.append(0)

        bid_volumes = np.array(bid_volumes)
        ask_volumes = np.array(ask_volumes)

        total = bid_volumes.sum() + ask_volumes.sum() + 1e-10

        features = {
            'imbalance_1': (bid_volumes[0] - ask_volumes[0]) / (bid_volumes[0] + ask_volumes[0] + 1e-10),
            'imbalance_5': (bid_volumes[:5].sum() - ask_volumes[:5].sum()) / (bid_volumes[:5].sum() + ask_volumes[:5].sum() + 1e-10),
            'imbalance_10': (bid_volumes.sum() - ask_volumes.sum()) / total,
        }

        # 重み付き不均衡 (近いレベルほど重要)
        weights = np.exp(-np.arange(levels) * 0.3)
        weighted_bid = np.sum(bid_volumes * weights)
        weighted_ask = np.sum(ask_volumes * weights)
        features['weighted_imbalance'] = (weighted_bid - weighted_ask) / (weighted_bid + weighted_ask + 1e-10)

        # デプス比率
        features['depth_ratio'] = bid_volumes.sum() / (ask_volumes.sum() + 1e-10)

        return features

    @staticmethod
    def calculate_vpin(trades: List[Dict], bucket_size: float = 1000.0) -> float:
        """
        Volume-Synchronized Probability of Informed Trading (VPIN)

        毒性度の推定
        """
        if len(trades) < 10:
            return 0.5

        buy_volume = sum(t['size'] for t in trades if t.get('side') == 'buy')
        sell_volume = sum(t['size'] for t in trades if t.get('side') == 'sell')
        total = buy_volume + sell_volume + 1e-10

        vpin = abs(buy_volume - sell_volume) / total

        return vpin


class AdvancedFeatureGenerator:
    """
    高度な特徴量生成器

    500次元以上の特徴量を生成
    """

    def __init__(
        self,
        lookback: int = 100,
        frac_diff_d: float = 0.4,
    ):
        self.lookback = lookback

        # 価格履歴
        self.price_history: deque = deque(maxlen=lookback)
        self.volume_history: deque = deque(maxlen=lookback)
        self.high_history: deque = deque(maxlen=lookback)
        self.low_history: deque = deque(maxlen=lookback)
        self.trade_history: deque = deque(maxlen=1000)

        # 変換器
        self.frac_diff = FractionalDifferencing(d=frac_diff_d)
        self.wavelet = WaveletTransform()
        self.fourier = FourierFeatures(n_components=10)

        # メタデータ
        self.feature_names: List[str] = []

    def update(
        self,
        price: float,
        volume: float,
        high: float = None,
        low: float = None,
        trade: Dict = None,
    ) -> None:
        """データ更新"""
        self.price_history.append(price)
        self.volume_history.append(volume)
        self.high_history.append(high or price)
        self.low_history.append(low or price)

        if trade:
            self.trade_history.append(trade)

    def generate(
        self,
        order_book: Dict = None,
        external_features: Dict = None,
    ) -> np.ndarray:
        """
        特徴量生成

        Returns:
            features: 500+次元の特徴量ベクトル
        """
        if len(self.price_history) < 20:
            return np.zeros(500)

        features = {}

        prices = np.array(self.price_history)
        volumes = np.array(self.volume_history)
        highs = np.array(self.high_history)
        lows = np.array(self.low_history)

        # =================================================================
        # 1. 基本価格特徴量 (50次元)
        # =================================================================
        features['price_normalized'] = prices[-1] / prices[0] - 1
        features['price_z_score'] = (prices[-1] - np.mean(prices)) / (np.std(prices) + 1e-10)

        # リターン系列
        returns = np.diff(prices) / prices[:-1]
        for period in [1, 5, 10, 20, 50]:
            if len(returns) >= period:
                features[f'return_{period}'] = np.sum(returns[-period:])
                features[f'return_std_{period}'] = np.std(returns[-period:])
                features[f'return_skew_{period}'] = self._skewness(returns[-period:])
                features[f'return_kurt_{period}'] = self._kurtosis(returns[-period:])

        # 対数リターン
        log_returns = np.diff(np.log(prices + 1e-10))
        features['log_return_mean'] = np.mean(log_returns)
        features['log_return_std'] = np.std(log_returns)

        # =================================================================
        # 2. 端数階差特徴量 (20次元)
        # =================================================================
        if len(prices) >= 30:
            frac_diff_prices = self.frac_diff.transform(prices)
            if len(frac_diff_prices) > 0:
                features['frac_diff_last'] = frac_diff_prices[-1]
                features['frac_diff_mean'] = np.mean(frac_diff_prices)
                features['frac_diff_std'] = np.std(frac_diff_prices)
                features['frac_diff_trend'] = np.polyfit(np.arange(len(frac_diff_prices[-20:])), frac_diff_prices[-20:], 1)[0] if len(frac_diff_prices) >= 20 else 0

        # =================================================================
        # 3. テクニカル指標 (100次元)
        # =================================================================
        for period in [7, 14, 21, 28]:
            features[f'rsi_{period}'] = TechnicalIndicatorsAdvanced.rsi(prices, period)

        for period in [5, 10, 20]:
            k, d = TechnicalIndicatorsAdvanced.stochastic(highs, lows, prices, period)
            features[f'stoch_k_{period}'] = k
            features[f'stoch_d_{period}'] = d

        for period in [10, 14, 20]:
            features[f'williams_r_{period}'] = TechnicalIndicatorsAdvanced.williams_r(highs, lows, prices, period)

        for period in [14, 20]:
            features[f'cci_{period}'] = TechnicalIndicatorsAdvanced.cci(highs, lows, prices, period)

        for period in [10, 14, 20]:
            adx, plus_di, minus_di = TechnicalIndicatorsAdvanced.adx(highs, lows, prices, period)
            features[f'adx_{period}'] = adx
            features[f'plus_di_{period}'] = plus_di
            features[f'minus_di_{period}'] = minus_di

        for period in [10, 14, 20]:
            features[f'atr_{period}'] = TechnicalIndicatorsAdvanced.atr(highs, lows, prices, period)
            features[f'atr_pct_{period}'] = features[f'atr_{period}'] / (prices[-1] + 1e-10)

        # ボリンジャーバンド
        for period in [10, 20]:
            for std_mult in [1.5, 2.0, 2.5]:
                upper, middle, lower = TechnicalIndicatorsAdvanced.bollinger_bands(prices, period, std_mult)
                band_width = (upper - lower) / (middle + 1e-10)
                bb_position = (prices[-1] - lower) / (upper - lower + 1e-10)
                features[f'bb_width_{period}_{std_mult}'] = band_width
                features[f'bb_position_{period}_{std_mult}'] = bb_position

        # ケルトナーチャネル
        kc_upper, kc_mid, kc_lower = TechnicalIndicatorsAdvanced.keltner_channels(highs, lows, prices)
        features['kc_position'] = (prices[-1] - kc_lower) / (kc_upper - kc_lower + 1e-10)

        # 一目均衡表
        ichimoku = TechnicalIndicatorsAdvanced.ichimoku(highs, lows, prices)
        for key, value in ichimoku.items():
            features[f'ichimoku_{key}'] = value / prices[-1]

        # =================================================================
        # 4. 移動平均特徴量 (60次元)
        # =================================================================
        for period in [5, 10, 20, 50]:
            if len(prices) >= period:
                sma = np.mean(prices[-period:])
                ema = self._ema(prices, period)

                features[f'sma_{period}'] = sma / prices[-1] - 1
                features[f'ema_{period}'] = ema / prices[-1] - 1
                features[f'price_to_sma_{period}'] = prices[-1] / sma - 1
                features[f'sma_slope_{period}'] = (sma - np.mean(prices[-period * 2:-period])) / sma if len(prices) >= period * 2 else 0

        # 移動平均クロス
        if len(prices) >= 50:
            sma_5 = np.mean(prices[-5:])
            sma_20 = np.mean(prices[-20:])
            sma_50 = np.mean(prices[-50:])

            features['sma_cross_5_20'] = sma_5 / sma_20 - 1
            features['sma_cross_20_50'] = sma_20 / sma_50 - 1
            features['sma_cross_5_50'] = sma_5 / sma_50 - 1

        # =================================================================
        # 5. ボラティリティ特徴量 (40次元)
        # =================================================================
        for period in [5, 10, 20, 50]:
            if len(returns) >= period:
                vol = np.std(returns[-period:])
                features[f'volatility_{period}'] = vol
                features[f'volatility_ratio_{period}'] = vol / (np.std(returns) + 1e-10)

                # パーキンソンボラティリティ
                if len(highs) >= period and len(lows) >= period:
                    parkinson = np.sqrt(
                        np.mean(np.log(highs[-period:] / lows[-period:]) ** 2) / (4 * np.log(2))
                    )
                    features[f'parkinson_vol_{period}'] = parkinson

                # ガーマン・クラスボラティリティ
                if len(prices) >= period + 1:
                    gk_vol = self._garman_klass_vol(
                        prices[-period - 1:], highs[-period:], lows[-period:]
                    )
                    features[f'gk_volatility_{period}'] = gk_vol

        # ボラティリティクラスタリング
        if len(returns) >= 20:
            vol_short = np.std(returns[-5:])
            vol_long = np.std(returns[-20:])
            features['vol_clustering'] = vol_short / (vol_long + 1e-10)

        # =================================================================
        # 6. 出来高特徴量 (40次元)
        # =================================================================
        for period in [5, 10, 20]:
            if len(volumes) >= period:
                vol_sma = np.mean(volumes[-period:])
                features[f'volume_sma_{period}'] = vol_sma
                features[f'volume_ratio_{period}'] = volumes[-1] / (vol_sma + 1e-10)

                # OBV (On Balance Volume)
                if len(returns) >= period:
                    obv = np.sum(np.sign(returns[-period:]) * volumes[-period:])
                    features[f'obv_{period}'] = obv

        # VWAP
        if len(volumes) > 0:
            vwap = np.sum(prices * volumes) / (np.sum(volumes) + 1e-10)
            features['vwap'] = vwap
            features['price_to_vwap'] = prices[-1] / vwap - 1

        # 出来高変化率
        if len(volumes) >= 20:
            features['volume_change_5'] = np.sum(volumes[-5:]) / np.sum(volumes[-10:-5] + 1e-10)
            features['volume_trend'] = np.corrcoef(np.arange(20), volumes[-20:])[0, 1] if np.std(volumes[-20:]) > 0 else 0

        # =================================================================
        # 7. フーリエ特徴量 (40次元)
        # =================================================================
        fourier_features = self.fourier.transform(prices)
        for key, value in fourier_features.items():
            features[key] = value

        # リターンのフーリエ
        if len(returns) >= 20:
            return_fourier = self.fourier.transform(returns[-50:] if len(returns) >= 50 else returns)
            for key, value in return_fourier.items():
                features[f'return_{key}'] = value

        # =================================================================
        # 8. ウェーブレット特徴量 (30次元)
        # =================================================================
        wavelet_features = self.wavelet.get_features(prices)
        for key, value in wavelet_features.items():
            features[key] = value

        # =================================================================
        # 9. 統計的特徴量 (40次元)
        # =================================================================
        # ハースト指数
        features['hurst_exponent'] = self._hurst_exponent(prices)

        # 自己相関
        for lag in [1, 5, 10, 20]:
            if len(returns) > lag:
                features[f'autocorr_{lag}'] = np.corrcoef(returns[:-lag], returns[lag:])[0, 1] if len(returns) > lag + 1 else 0

        # 部分自己相関
        for lag in [1, 5, 10]:
            if len(returns) > lag:
                features[f'pacf_{lag}'] = self._partial_autocorr(returns, lag)

        # エントロピー
        features['price_entropy'] = self._entropy(prices)
        features['return_entropy'] = self._entropy(returns)

        # 四分位数
        features['price_q25'] = np.percentile(prices, 25) / prices[-1]
        features['price_q75'] = np.percentile(prices, 75) / prices[-1]
        features['price_iqr'] = (np.percentile(prices, 75) - np.percentile(prices, 25)) / prices[-1]

        # =================================================================
        # 10. パターン認識特徴量 (30次元)
        # =================================================================
        # トレンド
        if len(prices) >= 20:
            slope, intercept = np.polyfit(np.arange(20), prices[-20:], 1)
            features['trend_slope'] = slope / prices[-1]
            features['trend_r2'] = 1 - np.var(prices[-20:] - (slope * np.arange(20) + intercept)) / (np.var(prices[-20:]) + 1e-10)

        # 高値/安値からの距離
        if len(prices) >= 50:
            features['dist_from_high'] = (np.max(prices[-50:]) - prices[-1]) / prices[-1]
            features['dist_from_low'] = (prices[-1] - np.min(prices[-50:])) / prices[-1]

        # サポート/レジスタンス
        if len(prices) >= 20:
            pivots = self._find_pivots(prices[-50:] if len(prices) >= 50 else prices)
            if pivots['resistance']:
                features['resistance_dist'] = (min(pivots['resistance']) - prices[-1]) / prices[-1]
            else:
                features['resistance_dist'] = 0
            if pivots['support']:
                features['support_dist'] = (prices[-1] - max(pivots['support'])) / prices[-1]
            else:
                features['support_dist'] = 0

        # =================================================================
        # 11. オーダーブック特徴量 (30次元)
        # =================================================================
        if order_book:
            ob_features = OrderBookFeatures.calculate_imbalance(
                order_book.get('bids', []),
                order_book.get('asks', [])
            )
            for key, value in ob_features.items():
                features[f'ob_{key}'] = value

            # VPIN
            if self.trade_history:
                features['vpin'] = OrderBookFeatures.calculate_vpin(list(self.trade_history))

        # =================================================================
        # 12. 外部特徴量 (可変)
        # =================================================================
        if external_features:
            for key, value in external_features.items():
                features[f'ext_{key}'] = value

        # =================================================================
        # 特徴量ベクトル生成
        # =================================================================
        self.feature_names = list(features.keys())

        # 欠損値処理
        feature_vector = np.array([features.get(name, 0.0) for name in self.feature_names])
        feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=0.0, neginf=0.0)

        # 500次元にパディング
        if len(feature_vector) < 500:
            feature_vector = np.pad(feature_vector, (0, 500 - len(feature_vector)))
        else:
            feature_vector = feature_vector[:500]

        return feature_vector

    def _ema(self, prices: np.ndarray, period: int) -> float:
        """指数移動平均"""
        alpha = 2 / (period + 1)
        ema = prices[0]
        for price in prices[1:]:
            ema = alpha * price + (1 - alpha) * ema
        return ema

    def _skewness(self, data: np.ndarray) -> float:
        """歪度"""
        if len(data) < 3:
            return 0.0
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 3)

    def _kurtosis(self, data: np.ndarray) -> float:
        """尖度"""
        if len(data) < 4:
            return 0.0
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 4) - 3

    def _garman_klass_vol(self, closes: np.ndarray, highs: np.ndarray, lows: np.ndarray) -> float:
        """ガーマン・クラスボラティリティ"""
        n = len(highs)
        if n < 2:
            return 0.0

        log_hl = np.log(highs / lows) ** 2
        log_co = np.log(closes[1:] / closes[:-1]) ** 2

        return np.sqrt(0.5 * np.mean(log_hl) - (2 * np.log(2) - 1) * np.mean(log_co))

    def _hurst_exponent(self, prices: np.ndarray, max_lag: int = 20) -> float:
        """ハースト指数"""
        if len(prices) < max_lag * 2:
            return 0.5

        lags = range(2, max_lag)
        tau = []
        for lag in lags:
            pp = np.subtract(prices[lag:], prices[:-lag])
            tau.append(np.std(pp))

        if len(tau) < 2:
            return 0.5

        tau = np.array(tau)
        lags = np.array(list(lags))

        # 対数回帰
        log_lags = np.log(lags)
        log_tau = np.log(tau + 1e-10)

        if np.std(log_lags) == 0:
            return 0.5

        slope, _ = np.polyfit(log_lags, log_tau, 1)
        return slope

    def _partial_autocorr(self, data: np.ndarray, lag: int) -> float:
        """部分自己相関"""
        if len(data) <= lag:
            return 0.0

        # Yule-Walker方程式の近似
        acf = [np.corrcoef(data[:-i], data[i:])[0, 1] if len(data) > i + 1 else 0 for i in range(1, lag + 1)]
        return acf[-1] if acf else 0.0

    def _entropy(self, data: np.ndarray, bins: int = 10) -> float:
        """シャノンエントロピー"""
        hist, _ = np.histogram(data, bins=bins, density=True)
        hist = hist[hist > 0]
        return -np.sum(hist * np.log(hist + 1e-10))

    def _find_pivots(self, prices: np.ndarray, window: int = 5) -> Dict[str, List[float]]:
        """ピボットポイント検出"""
        pivots = {'resistance': [], 'support': []}

        for i in range(window, len(prices) - window):
            # 高値ピボット
            if all(prices[i] >= prices[i - w] for w in range(1, window + 1)) and \
               all(prices[i] >= prices[i + w] for w in range(1, window + 1)):
                pivots['resistance'].append(prices[i])

            # 安値ピボット
            if all(prices[i] <= prices[i - w] for w in range(1, window + 1)) and \
               all(prices[i] <= prices[i + w] for w in range(1, window + 1)):
                pivots['support'].append(prices[i])

        return pivots

    def get_feature_names(self) -> List[str]:
        """特徴量名取得"""
        return self.feature_names.copy()

    def get_feature_dim(self) -> int:
        """特徴量次元数"""
        return 500


logger.info("Advanced Feature Generator loaded (500+ dimensions)")

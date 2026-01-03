"""
Feature Engineering
====================
機械学習用の特徴量生成
"""

from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
from loguru import logger


class FeatureEngineer:
    """
    特徴量エンジニア

    価格予測のための高度な特徴量生成
    """

    def __init__(self, lookback_periods: List[int] = None):
        """
        Args:
            lookback_periods: 参照期間リスト
        """
        self.lookback_periods = lookback_periods or [5, 10, 20, 50, 100]

    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        全特徴量生成

        Args:
            df: OHLCV データフレーム

        Returns:
            特徴量が追加されたデータフレーム
        """
        features = df.copy()

        # 価格ベース特徴量
        features = self._add_price_features(features)

        # テクニカル指標
        features = self._add_technical_indicators(features)

        # ボラティリティ特徴量
        features = self._add_volatility_features(features)

        # 出来高特徴量
        features = self._add_volume_features(features)

        # 時間特徴量
        features = self._add_time_features(features)

        # ラグ特徴量
        features = self._add_lag_features(features)

        # 統計的特徴量
        features = self._add_statistical_features(features)

        return features

    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """価格ベース特徴量"""
        # リターン（複数期間）
        for period in self.lookback_periods:
            df[f'return_{period}'] = df['close'].pct_change(period)
            df[f'log_return_{period}'] = np.log(df['close'] / df['close'].shift(period))

        # 価格変化率
        df['price_change'] = df['close'] - df['open']
        df['price_range'] = df['high'] - df['low']
        df['price_range_pct'] = df['price_range'] / df['close']

        # 高値・安値からの距離
        df['dist_from_high'] = (df['high'] - df['close']) / df['close']
        df['dist_from_low'] = (df['close'] - df['low']) / df['close']

        # キャンドルスティックパターン
        df['body_size'] = abs(df['close'] - df['open']) / df['open']
        df['upper_shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['close']
        df['lower_shadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['close']
        df['is_bullish'] = (df['close'] > df['open']).astype(int)

        return df

    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """テクニカル指標"""
        close = df['close']
        high = df['high']
        low = df['low']

        # 移動平均
        for period in [5, 10, 20, 50]:
            df[f'sma_{period}'] = close.rolling(window=period).mean()
            df[f'ema_{period}'] = close.ewm(span=period, adjust=False).mean()
            df[f'sma_dist_{period}'] = (close - df[f'sma_{period}']) / df[f'sma_{period}']

        # RSI
        for period in [7, 14, 21]:
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / (loss + 1e-10)
            df[f'rsi_{period}'] = 100 - (100 / (1 + rs))

        # MACD
        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']

        # Bollinger Bands
        for period in [10, 20]:
            sma = close.rolling(window=period).mean()
            std = close.rolling(window=period).std()
            df[f'bb_upper_{period}'] = sma + (std * 2)
            df[f'bb_lower_{period}'] = sma - (std * 2)
            df[f'bb_width_{period}'] = (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}']) / sma
            df[f'bb_position_{period}'] = (close - df[f'bb_lower_{period}']) / (
                df[f'bb_upper_{period}'] - df[f'bb_lower_{period}'] + 1e-10
            )

        # Stochastic
        for period in [14, 21]:
            lowest_low = low.rolling(window=period).min()
            highest_high = high.rolling(window=period).max()
            df[f'stoch_k_{period}'] = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-10)
            df[f'stoch_d_{period}'] = df[f'stoch_k_{period}'].rolling(window=3).mean()

        # Williams %R
        df['williams_r'] = -100 * (high.rolling(14).max() - close) / (
            high.rolling(14).max() - low.rolling(14).min() + 1e-10
        )

        # ADX (Average Directional Index)
        df['adx'] = self._calculate_adx(high, low, close, 14)

        # CCI (Commodity Channel Index)
        typical_price = (high + low + close) / 3
        sma_tp = typical_price.rolling(window=20).mean()
        mean_deviation = typical_price.rolling(window=20).apply(
            lambda x: np.mean(np.abs(x - x.mean()))
        )
        df['cci'] = (typical_price - sma_tp) / (0.015 * mean_deviation + 1e-10)

        return df

    def _calculate_adx(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int
    ) -> pd.Series:
        """ADX計算"""
        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()

        # +DM, -DM
        up_move = high - high.shift()
        down_move = low.shift() - low

        plus_dm = ((up_move > down_move) & (up_move > 0)) * up_move
        minus_dm = ((down_move > up_move) & (down_move > 0)) * down_move

        plus_di = 100 * (plus_dm.rolling(window=period).mean() / (atr + 1e-10))
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / (atr + 1e-10))

        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = dx.rolling(window=period).mean()

        return adx

    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """ボラティリティ特徴量"""
        close = df['close']
        high = df['high']
        low = df['low']

        # 実現ボラティリティ
        for period in [5, 10, 20, 50]:
            returns = close.pct_change()
            df[f'volatility_{period}'] = returns.rolling(window=period).std()
            df[f'volatility_ann_{period}'] = df[f'volatility_{period}'] * np.sqrt(365 * 24 * 60)

        # ATR (Average True Range)
        for period in [7, 14, 21]:
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            df[f'atr_{period}'] = tr.rolling(window=period).mean()
            df[f'atr_pct_{period}'] = df[f'atr_{period}'] / close

        # Parkinson Volatility
        df['parkinson_vol'] = np.sqrt(
            (1 / (4 * np.log(2))) *
            ((np.log(high / low)) ** 2).rolling(window=20).mean()
        )

        # Garman-Klass Volatility
        df['gk_vol'] = np.sqrt(
            0.5 * (np.log(high / low) ** 2).rolling(window=20).mean() -
            (2 * np.log(2) - 1) * (np.log(close / df['open']) ** 2).rolling(window=20).mean()
        )

        return df

    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """出来高特徴量"""
        volume = df['volume']
        close = df['close']

        # 出来高移動平均
        for period in [5, 10, 20]:
            df[f'volume_sma_{period}'] = volume.rolling(window=period).mean()
            df[f'volume_ratio_{period}'] = volume / (df[f'volume_sma_{period}'] + 1e-10)

        # OBV (On Balance Volume)
        direction = np.sign(close.diff())
        df['obv'] = (direction * volume).cumsum()
        df['obv_sma_10'] = df['obv'].rolling(window=10).mean()

        # Volume Price Trend
        df['vpt'] = (volume * close.pct_change()).cumsum()

        # Money Flow Index
        typical_price = (df['high'] + df['low'] + close) / 3
        money_flow = typical_price * volume

        positive_flow = money_flow.where(typical_price > typical_price.shift(), 0)
        negative_flow = money_flow.where(typical_price < typical_price.shift(), 0)

        positive_mf = positive_flow.rolling(window=14).sum()
        negative_mf = negative_flow.rolling(window=14).sum()

        mf_ratio = positive_mf / (negative_mf + 1e-10)
        df['mfi'] = 100 - (100 / (1 + mf_ratio))

        # Volume-Price Correlation
        df['volume_price_corr'] = close.rolling(window=20).corr(volume)

        return df

    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """時間特徴量"""
        if 'timestamp' in df.columns:
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['day_of_month'] = df['timestamp'].dt.day
            df['month'] = df['timestamp'].dt.month

            # サイクリカルエンコーディング
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

            # 取引セッション
            df['is_tokyo_session'] = ((df['hour'] >= 9) & (df['hour'] < 15)).astype(int)
            df['is_london_session'] = ((df['hour'] >= 16) & (df['hour'] < 24)).astype(int)
            df['is_ny_session'] = ((df['hour'] >= 22) | (df['hour'] < 7)).astype(int)

        return df

    def _add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """ラグ特徴量"""
        # 価格ラグ
        for lag in [1, 2, 3, 5, 10]:
            df[f'close_lag_{lag}'] = df['close'].shift(lag)
            df[f'return_lag_{lag}'] = df['close'].pct_change().shift(lag)

        # ボラティリティラグ
        vol = df['close'].pct_change().rolling(window=20).std()
        for lag in [1, 5, 10]:
            df[f'vol_lag_{lag}'] = vol.shift(lag)

        return df

    def _add_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """統計的特徴量"""
        returns = df['close'].pct_change()

        # スキューネス・尖度
        for period in [20, 50]:
            df[f'skewness_{period}'] = returns.rolling(window=period).skew()
            df[f'kurtosis_{period}'] = returns.rolling(window=period).kurt()

        # Zスコア
        for period in [20, 50]:
            mean = df['close'].rolling(window=period).mean()
            std = df['close'].rolling(window=period).std()
            df[f'zscore_{period}'] = (df['close'] - mean) / (std + 1e-10)

        # パーセンタイルランク
        df['percentile_rank_20'] = df['close'].rolling(window=20).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1]
        )

        return df

    def get_feature_names(self) -> List[str]:
        """特徴量名リスト取得"""
        return [
            # 価格ベース
            'return_5', 'return_10', 'return_20', 'price_range_pct', 'body_size',
            # テクニカル
            'rsi_14', 'macd_hist', 'bb_position_20', 'stoch_k_14', 'adx', 'cci',
            # ボラティリティ
            'volatility_20', 'atr_pct_14', 'parkinson_vol',
            # 出来高
            'volume_ratio_20', 'mfi', 'volume_price_corr',
            # 時間
            'hour_sin', 'hour_cos', 'is_tokyo_session',
            # 統計
            'zscore_20', 'skewness_20',
        ]

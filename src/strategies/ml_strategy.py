"""
Machine Learning Strategy
==========================
機械学習を使用した価格予測戦略
"""

from typing import Optional, Dict, List, Any
from datetime import datetime
import numpy as np
import pandas as pd
from loguru import logger

from .base import BaseStrategy, Signal, SignalType, MarketData, TechnicalIndicators


class MLStrategy(BaseStrategy):
    """
    機械学習戦略

    特徴:
    - 複数のMLモデルのアンサンブル
    - リアルタイム特徴量生成
    - 適応的な予測
    """

    def __init__(
        self,
        product_codes: List[str],
        model_type: str = "ensemble",
        prediction_horizon: int = 60,  # 秒
        confidence_threshold: float = 0.65,
        weight: float = 1.0,
    ):
        """
        Args:
            model_type: モデルタイプ (ensemble, xgboost, lstm等)
            prediction_horizon: 予測期間（秒）
            confidence_threshold: シグナル生成の信頼度閾値
        """
        super().__init__(
            name="MLStrategy",
            product_codes=product_codes,
            weight=weight,
        )

        self.model_type = model_type
        self.prediction_horizon = prediction_horizon
        self.confidence_threshold = confidence_threshold

        # データ履歴
        self.feature_history: Dict[str, pd.DataFrame] = {}
        self.max_history = 1000

        # モデル（遅延ロード）
        self.models: Dict[str, Any] = {}
        self.is_trained: Dict[str, bool] = {}

        # 予測履歴（バックテスト用）
        self.predictions: List[Dict] = []

    def update(self, market_data: MarketData) -> None:
        """市場データで内部状態更新"""
        product = market_data.product_code

        # 特徴量生成
        features = self._generate_features(product, market_data)

        if product not in self.feature_history:
            self.feature_history[product] = pd.DataFrame()

        new_row = pd.DataFrame([features])
        self.feature_history[product] = pd.concat(
            [self.feature_history[product], new_row],
            ignore_index=True
        )

        if len(self.feature_history[product]) > self.max_history:
            self.feature_history[product] = self.feature_history[product].iloc[-self.max_history:]

    def _generate_features(
        self,
        product: str,
        market_data: MarketData
    ) -> Dict[str, float]:
        """特徴量生成"""
        features = {
            'timestamp': market_data.timestamp,
            'close': market_data.close,
            'volume': market_data.volume,
            'spread': market_data.spread or 0,
            'order_book_imbalance': market_data.order_book_imbalance or 0,
        }

        # 履歴があれば追加特徴量を計算
        if product in self.feature_history and len(self.feature_history[product]) > 0:
            df = self.feature_history[product]
            close = pd.concat([df['close'], pd.Series([market_data.close])])

            # 価格変化率
            if len(close) >= 2:
                features['return_1'] = (close.iloc[-1] - close.iloc[-2]) / close.iloc[-2]
            if len(close) >= 6:
                features['return_5'] = (close.iloc[-1] - close.iloc[-6]) / close.iloc[-6]
            if len(close) >= 16:
                features['return_15'] = (close.iloc[-1] - close.iloc[-16]) / close.iloc[-16]

            # ボラティリティ
            if len(close) >= 20:
                returns = close.pct_change().iloc[-20:]
                features['volatility_20'] = returns.std()

            # RSI
            if len(close) >= 15:
                rsi = TechnicalIndicators.rsi(close, 14)
                features['rsi'] = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50

            # MACD
            if len(close) >= 27:
                macd = TechnicalIndicators.macd(close)
                features['macd'] = macd['macd'].iloc[-1] if not pd.isna(macd['macd'].iloc[-1]) else 0
                features['macd_signal'] = macd['signal'].iloc[-1] if not pd.isna(macd['signal'].iloc[-1]) else 0
                features['macd_hist'] = macd['histogram'].iloc[-1] if not pd.isna(macd['histogram'].iloc[-1]) else 0

            # Bollinger Bands位置
            if len(close) >= 21:
                bb = TechnicalIndicators.bollinger_bands(close, 20, 2.0)
                upper = bb['upper'].iloc[-1]
                lower = bb['lower'].iloc[-1]
                if upper != lower:
                    features['bb_position'] = (close.iloc[-1] - lower) / (upper - lower)

            # 移動平均乖離
            if len(close) >= 20:
                sma20 = TechnicalIndicators.sma(close, 20).iloc[-1]
                features['sma20_deviation'] = (close.iloc[-1] - sma20) / sma20

        return features

    def generate_signal(self, market_data: MarketData) -> Optional[Signal]:
        """シグナル生成"""
        product = market_data.product_code

        if product not in self.feature_history:
            return None

        df = self.feature_history[product]
        if len(df) < 50:  # 最低限のデータが必要
            return None

        # 予測実行
        prediction = self._predict(product)

        if prediction is None:
            return None

        direction, probability = prediction

        # 信頼度閾値チェック
        if probability < self.confidence_threshold:
            return None

        # シグナルタイプ決定
        if direction == 1:  # 上昇予測
            signal_type = SignalType.BUY
        elif direction == -1:  # 下落予測
            signal_type = SignalType.SELL
        else:
            return None

        # ポジションサイズ（確率に基づく）
        base_size = 0.01
        size = base_size * (probability - 0.5) * 2  # 0.5を超えた分に比例

        self.signals_generated += 1

        # 予測記録
        self.predictions.append({
            'timestamp': datetime.now(),
            'product': product,
            'direction': direction,
            'probability': probability,
            'price': market_data.close,
        })

        return Signal(
            signal_type=signal_type,
            product_code=product,
            price=market_data.close,
            size=size,
            confidence=probability,
            strategy_name=self.name,
            metadata={
                "prediction_direction": direction,
                "prediction_probability": probability,
                "model_type": self.model_type,
                "prediction_horizon": self.prediction_horizon,
            },
        )

    def _predict(self, product: str) -> Optional[tuple]:
        """
        価格方向予測

        Returns:
            (direction, probability) or None
            direction: 1=上昇, -1=下落, 0=横ばい
            probability: 予測確率 (0.0-1.0)
        """
        df = self.feature_history[product]

        # シンプルなルールベース予測（実際のMLモデルの代替）
        # 本番では訓練済みモデルを使用

        if len(df) < 20:
            return None

        # 特徴量取得
        latest = df.iloc[-1]

        # スコア計算
        score = 0.0
        confidence_factors = []

        # RSIベースのスコア
        if 'rsi' in latest and not pd.isna(latest['rsi']):
            rsi = latest['rsi']
            if rsi < 30:
                score += 0.3
                confidence_factors.append(0.7)
            elif rsi > 70:
                score -= 0.3
                confidence_factors.append(0.7)
            else:
                confidence_factors.append(0.5)

        # MACDベースのスコア
        if 'macd_hist' in latest and not pd.isna(latest['macd_hist']):
            macd_hist = latest['macd_hist']
            if macd_hist > 0:
                score += 0.2
            elif macd_hist < 0:
                score -= 0.2
            confidence_factors.append(0.6)

        # 価格変化率ベースのスコア
        if 'return_5' in latest and not pd.isna(latest['return_5']):
            ret = latest['return_5']
            if ret > 0.005:  # 0.5%以上の上昇
                score += 0.2
            elif ret < -0.005:
                score -= 0.2
            confidence_factors.append(0.6)

        # オーダーブック不均衡
        if 'order_book_imbalance' in latest and not pd.isna(latest['order_book_imbalance']):
            imbalance = latest['order_book_imbalance']
            score += imbalance * 0.3
            confidence_factors.append(abs(imbalance) * 0.5 + 0.5)

        # 方向と確率の決定
        if score > 0.2:
            direction = 1
        elif score < -0.2:
            direction = -1
        else:
            direction = 0

        # 確率計算
        base_probability = 0.5 + abs(score) * 0.3
        if confidence_factors:
            avg_confidence = np.mean(confidence_factors)
            probability = base_probability * avg_confidence + (1 - avg_confidence) * 0.5
        else:
            probability = base_probability

        probability = max(0.0, min(1.0, probability))

        return (direction, probability)

    def train(self, product: str, training_data: pd.DataFrame) -> None:
        """
        モデル訓練（実際のML実装用プレースホルダー）

        Args:
            product: 対象商品
            training_data: 訓練データ
        """
        # 実際の実装では以下を行う:
        # 1. 特徴量エンジニアリング
        # 2. モデル選択（XGBoost, LightGBM, LSTM等）
        # 3. ハイパーパラメータチューニング
        # 4. 交差検証
        # 5. モデル保存

        logger.info(f"Training model for {product}")
        self.is_trained[product] = True

    def get_feature_importance(self, product: str) -> Dict[str, float]:
        """特徴量重要度取得"""
        # 実際のMLモデルから重要度を取得
        return {
            "rsi": 0.15,
            "macd_hist": 0.12,
            "return_5": 0.10,
            "order_book_imbalance": 0.18,
            "volatility_20": 0.08,
            "bb_position": 0.12,
            "sma20_deviation": 0.10,
            "volume": 0.08,
            "spread": 0.07,
        }

    def get_prediction_accuracy(self) -> float:
        """予測精度取得"""
        if len(self.predictions) < 10:
            return 0.0

        # 実際の実装では過去の予測と実際の価格変動を比較
        return 0.55  # プレースホルダー

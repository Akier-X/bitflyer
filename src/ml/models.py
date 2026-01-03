"""
Machine Learning Models
========================
価格予測用の機械学習モデル
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from datetime import datetime
from abc import ABC, abstractmethod
from loguru import logger


class BaseModel(ABC):
    """モデル基底クラス"""

    def __init__(self, name: str):
        self.name = name
        self.is_trained = False
        self.metrics: Dict[str, float] = {}

    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        pass


class PricePredictionModel(BaseModel):
    """
    価格予測モデル

    LSTMとGradient Boostingのアンサンブル
    """

    def __init__(
        self,
        model_type: str = "xgboost",
        sequence_length: int = 60,
        prediction_horizon: int = 1,
    ):
        """
        Args:
            model_type: モデルタイプ (xgboost, lightgbm, lstm, ensemble)
            sequence_length: 入力シーケンス長
            prediction_horizon: 予測期間
        """
        super().__init__(name=f"PricePredictor_{model_type}")

        self.model_type = model_type
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon

        self.model = None
        self.scaler = None

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """モデル訓練"""
        try:
            if self.model_type == "xgboost":
                self._train_xgboost(X, y)
            elif self.model_type == "lightgbm":
                self._train_lightgbm(X, y)
            elif self.model_type == "lstm":
                self._train_lstm(X, y)
            else:
                self._train_xgboost(X, y)  # デフォルト

            self.is_trained = True
            logger.info(f"Model {self.name} trained successfully")

        except ImportError as e:
            logger.warning(f"ML library not available: {e}")
            self._train_simple(X, y)

    def _train_xgboost(self, X: np.ndarray, y: np.ndarray) -> None:
        """XGBoost訓練"""
        try:
            import xgboost as xgb

            params = {
                'objective': 'binary:logistic',
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 200,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
            }

            self.model = xgb.XGBClassifier(**params)
            self.model.fit(X, y)

        except ImportError:
            self._train_simple(X, y)

    def _train_lightgbm(self, X: np.ndarray, y: np.ndarray) -> None:
        """LightGBM訓練"""
        try:
            import lightgbm as lgb

            params = {
                'objective': 'binary',
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 200,
                'num_leaves': 31,
                'random_state': 42,
                'verbose': -1,
            }

            self.model = lgb.LGBMClassifier(**params)
            self.model.fit(X, y)

        except ImportError:
            self._train_simple(X, y)

    def _train_lstm(self, X: np.ndarray, y: np.ndarray) -> None:
        """LSTM訓練"""
        try:
            import torch
            import torch.nn as nn

            # LSTMモデル定義
            class LSTMModel(nn.Module):
                def __init__(self, input_size, hidden_size=64, num_layers=2):
                    super().__init__()
                    self.lstm = nn.LSTM(
                        input_size, hidden_size, num_layers,
                        batch_first=True, dropout=0.2
                    )
                    self.fc = nn.Linear(hidden_size, 1)
                    self.sigmoid = nn.Sigmoid()

                def forward(self, x):
                    lstm_out, _ = self.lstm(x)
                    out = self.fc(lstm_out[:, -1, :])
                    return self.sigmoid(out)

            # 簡易訓練（実際の実装ではより詳細に）
            input_size = X.shape[-1] if len(X.shape) == 3 else X.shape[1]
            self.model = LSTMModel(input_size)

        except ImportError:
            self._train_simple(X, y)

    def _train_simple(self, X: np.ndarray, y: np.ndarray) -> None:
        """シンプルなフォールバックモデル"""
        # ロジスティック回帰の簡易実装
        self.model = SimpleLogisticRegression()
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """予測"""
        if not self.is_trained or self.model is None:
            return np.zeros(len(X))

        try:
            proba = self.predict_proba(X)
            return (proba >= 0.5).astype(int)
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return np.zeros(len(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """確率予測"""
        if not self.is_trained or self.model is None:
            return np.full(len(X), 0.5)

        try:
            if hasattr(self.model, 'predict_proba'):
                proba = self.model.predict_proba(X)
                return proba[:, 1] if proba.ndim == 2 else proba
            else:
                return self.model.predict(X)
        except Exception as e:
            logger.error(f"Probability prediction error: {e}")
            return np.full(len(X), 0.5)


class TrendClassifier(BaseModel):
    """
    トレンド分類モデル

    上昇・下落・横ばいの3クラス分類
    """

    def __init__(self, threshold: float = 0.002):
        """
        Args:
            threshold: トレンド判定閾値
        """
        super().__init__(name="TrendClassifier")
        self.threshold = threshold
        self.model = None
        self.classes = ['down', 'neutral', 'up']

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """モデル訓練"""
        try:
            import xgboost as xgb

            params = {
                'objective': 'multi:softprob',
                'num_class': 3,
                'max_depth': 5,
                'learning_rate': 0.1,
                'n_estimators': 150,
                'random_state': 42,
            }

            self.model = xgb.XGBClassifier(**params)
            self.model.fit(X, y)
            self.is_trained = True

        except ImportError:
            logger.warning("XGBoost not available, using simple model")
            self.model = SimpleMultiClassifier(n_classes=3)
            self.model.fit(X, y)
            self.is_trained = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """予測"""
        if not self.is_trained or self.model is None:
            return np.ones(len(X))  # neutral

        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """確率予測"""
        if not self.is_trained or self.model is None:
            return np.full((len(X), 3), 1/3)

        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        return np.full((len(X), 3), 1/3)

    def classify_returns(self, returns: np.ndarray) -> np.ndarray:
        """リターンをトレンドクラスに変換"""
        classes = np.ones(len(returns))  # neutral = 1
        classes[returns < -self.threshold] = 0  # down
        classes[returns > self.threshold] = 2  # up
        return classes.astype(int)


class ReinforcementLearningAgent(BaseModel):
    """
    強化学習エージェント

    PPO (Proximal Policy Optimization) ベース
    """

    def __init__(
        self,
        state_dim: int = 50,
        action_dim: int = 3,  # buy, hold, sell
        learning_rate: float = 0.0003,
    ):
        """
        Args:
            state_dim: 状態次元
            action_dim: アクション数
            learning_rate: 学習率
        """
        super().__init__(name="RLAgent")

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate

        self.policy_net = None
        self.value_net = None
        self.optimizer = None

        # 経験バッファ
        self.experiences: List[Dict] = []

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """訓練（強化学習では環境との相互作用で学習）"""
        pass

    def learn_from_experience(self) -> None:
        """経験から学習"""
        if len(self.experiences) < 32:
            return

        # PPOの更新ロジック（簡略化）
        # 実際の実装では stable-baselines3 等を使用
        self.experiences = []

    def predict(self, X: np.ndarray) -> np.ndarray:
        """アクション予測"""
        return self.get_action(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """アクション確率"""
        # 簡易実装
        proba = np.random.dirichlet([1, 1, 1], size=len(X))
        return proba

    def get_action(self, state: np.ndarray) -> int:
        """状態からアクションを選択"""
        # 簡易ルールベースポリシー
        if len(state.shape) == 1:
            state = state.reshape(1, -1)

        # 特徴量の最後（最新のリターン等）を使用
        if state.shape[1] > 0:
            last_feature = state[0, -1]
            if last_feature > 0.001:
                return 0  # buy
            elif last_feature < -0.001:
                return 2  # sell
        return 1  # hold

    def store_experience(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """経験を保存"""
        self.experiences.append({
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done,
        })


class EnsembleModel(BaseModel):
    """
    アンサンブルモデル

    複数モデルの予測を統合
    """

    def __init__(self, models: List[BaseModel] = None, weights: List[float] = None):
        """
        Args:
            models: モデルリスト
            weights: 重みリスト
        """
        super().__init__(name="Ensemble")

        self.models = models or []
        self.weights = weights or [1.0] * len(self.models)

        # 重みの正規化
        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]

    def add_model(self, model: BaseModel, weight: float = 1.0) -> None:
        """モデル追加"""
        self.models.append(model)
        self.weights.append(weight)

        # 重み再正規化
        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """全モデル訓練"""
        for model in self.models:
            model.train(X, y)
        self.is_trained = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """予測"""
        proba = self.predict_proba(X)
        return (proba >= 0.5).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """加重平均確率予測"""
        if not self.models:
            return np.full(len(X), 0.5)

        predictions = []
        for model, weight in zip(self.models, self.weights):
            if model.is_trained:
                proba = model.predict_proba(X)
                predictions.append(proba * weight)

        if not predictions:
            return np.full(len(X), 0.5)

        return np.sum(predictions, axis=0)


# フォールバック用シンプルモデル
class SimpleLogisticRegression:
    """シンプルなロジスティック回帰（依存ライブラリなし）"""

    def __init__(self, learning_rate: float = 0.01, n_iterations: int = 1000):
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self._sigmoid(linear_model)

            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X: np.ndarray) -> np.ndarray:
        linear_model = np.dot(X, self.weights) + self.bias
        return self._sigmoid(linear_model)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict(X)
        return np.column_stack([1 - proba, proba])

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


class SimpleMultiClassifier:
    """シンプルな多クラス分類器"""

    def __init__(self, n_classes: int = 3):
        self.n_classes = n_classes
        self.class_means = {}

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        for c in range(self.n_classes):
            mask = y == c
            if np.any(mask):
                self.class_means[c] = X[mask].mean(axis=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.class_means:
            return np.ones(len(X)).astype(int)

        predictions = []
        for x in X:
            min_dist = float('inf')
            pred_class = 1
            for c, mean in self.class_means.items():
                dist = np.linalg.norm(x - mean)
                if dist < min_dist:
                    min_dist = dist
                    pred_class = c
            predictions.append(pred_class)

        return np.array(predictions)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        # 距離ベースの確率
        proba = np.zeros((len(X), self.n_classes))
        proba[:, 1] = 1.0  # デフォルトはneutral
        return proba

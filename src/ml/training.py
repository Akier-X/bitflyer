"""
Model Training Pipeline
========================
モデル訓練パイプライン
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import pickle
import os
from loguru import logger

from .features import FeatureEngineer
from .models import (
    BaseModel,
    PricePredictionModel,
    TrendClassifier,
    EnsembleModel,
)


class ModelTrainer:
    """
    モデル訓練パイプライン

    データ準備、訓練、評価、保存を管理
    """

    def __init__(
        self,
        model_dir: str = "models",
        validation_split: float = 0.2,
        test_split: float = 0.1,
    ):
        """
        Args:
            model_dir: モデル保存ディレクトリ
            validation_split: 検証データ割合
            test_split: テストデータ割合
        """
        self.model_dir = model_dir
        self.validation_split = validation_split
        self.test_split = test_split

        self.feature_engineer = FeatureEngineer()
        self.models: Dict[str, BaseModel] = {}
        self.training_history: List[Dict] = []

        os.makedirs(model_dir, exist_ok=True)

    def prepare_data(
        self,
        df: pd.DataFrame,
        target_column: str = 'target',
        prediction_horizon: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        データ準備

        Args:
            df: 入力データフレーム
            target_column: ターゲット列名
            prediction_horizon: 予測期間

        Returns:
            (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # 特徴量生成
        features_df = self.feature_engineer.generate_features(df)

        # ターゲット作成（将来リターン）
        features_df['target'] = (
            df['close'].shift(-prediction_horizon) / df['close'] - 1
        )
        features_df['target_class'] = (features_df['target'] > 0).astype(int)

        # NaN削除
        features_df = features_df.dropna()

        # 特徴量選択
        feature_cols = [col for col in features_df.columns
                       if col not in ['timestamp', 'target', 'target_class', 'open', 'high', 'low', 'close', 'volume']]

        X = features_df[feature_cols].values
        y = features_df['target_class'].values

        # データ分割（時系列を考慮して順序を維持）
        n = len(X)
        train_end = int(n * (1 - self.validation_split - self.test_split))
        val_end = int(n * (1 - self.test_split))

        X_train = X[:train_end]
        y_train = y[:train_end]
        X_val = X[train_end:val_end]
        y_val = y[train_end:val_end]
        X_test = X[val_end:]
        y_test = y[val_end:]

        logger.info(f"Data prepared: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

        return X_train, X_val, X_test, y_train, y_val, y_test

    def train_model(
        self,
        model: BaseModel,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Dict[str, float]:
        """
        モデル訓練

        Args:
            model: 訓練するモデル
            X_train, y_train: 訓練データ
            X_val, y_val: 検証データ

        Returns:
            評価メトリクス
        """
        logger.info(f"Training model: {model.name}")

        start_time = datetime.now()
        model.train(X_train, y_train)
        training_time = (datetime.now() - start_time).total_seconds()

        # 評価
        metrics = self.evaluate_model(model, X_val, y_val)
        metrics['training_time'] = training_time

        # 履歴に追加
        self.training_history.append({
            'model_name': model.name,
            'timestamp': datetime.now(),
            'metrics': metrics,
        })

        # モデル保存
        self.models[model.name] = model

        logger.info(f"Model {model.name} trained: accuracy={metrics.get('accuracy', 0):.4f}")

        return metrics

    def evaluate_model(
        self,
        model: BaseModel,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Dict[str, float]:
        """
        モデル評価

        Args:
            model: 評価するモデル
            X, y: 評価データ

        Returns:
            評価メトリクス
        """
        predictions = model.predict(X)
        proba = model.predict_proba(X)

        # 精度
        accuracy = np.mean(predictions == y)

        # 適合率・再現率
        true_positives = np.sum((predictions == 1) & (y == 1))
        false_positives = np.sum((predictions == 1) & (y == 0))
        false_negatives = np.sum((predictions == 0) & (y == 1))

        precision = true_positives / (true_positives + false_positives + 1e-10)
        recall = true_positives / (true_positives + false_negatives + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)

        # ログロス
        proba_1d = proba[:, 1] if proba.ndim == 2 else proba
        proba_clipped = np.clip(proba_1d, 1e-10, 1 - 1e-10)
        log_loss = -np.mean(y * np.log(proba_clipped) + (1 - y) * np.log(1 - proba_clipped))

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'log_loss': log_loss,
        }

    def cross_validate(
        self,
        model_class: type,
        X: np.ndarray,
        y: np.ndarray,
        n_folds: int = 5,
        **model_kwargs,
    ) -> Dict[str, float]:
        """
        時系列交差検証

        Args:
            model_class: モデルクラス
            X, y: データ
            n_folds: フォールド数

        Returns:
            平均メトリクス
        """
        fold_size = len(X) // (n_folds + 1)
        all_metrics = []

        for i in range(n_folds):
            train_end = fold_size * (i + 1)
            val_start = train_end
            val_end = train_end + fold_size

            X_train = X[:train_end]
            y_train = y[:train_end]
            X_val = X[val_start:val_end]
            y_val = y[val_start:val_end]

            model = model_class(**model_kwargs)
            model.train(X_train, y_train)
            metrics = self.evaluate_model(model, X_val, y_val)
            all_metrics.append(metrics)

        # 平均メトリクス
        avg_metrics = {}
        for key in all_metrics[0].keys():
            values = [m[key] for m in all_metrics]
            avg_metrics[f'{key}_mean'] = np.mean(values)
            avg_metrics[f'{key}_std'] = np.std(values)

        return avg_metrics

    def train_ensemble(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        model_types: List[str] = None,
    ) -> Tuple[EnsembleModel, Dict[str, float]]:
        """
        アンサンブルモデル訓練

        Args:
            X_train, y_train: 訓練データ
            X_val, y_val: 検証データ
            model_types: 使用するモデルタイプのリスト

        Returns:
            (アンサンブルモデル, メトリクス)
        """
        model_types = model_types or ['xgboost', 'lightgbm']

        ensemble = EnsembleModel()
        individual_metrics = {}

        for model_type in model_types:
            model = PricePredictionModel(model_type=model_type)
            metrics = self.train_model(model, X_train, y_train, X_val, y_val)
            individual_metrics[model_type] = metrics

            # 性能に基づく重み
            weight = metrics.get('f1_score', 0.5)
            ensemble.add_model(model, weight)

        # アンサンブル評価
        ensemble.is_trained = True
        ensemble_metrics = self.evaluate_model(ensemble, X_val, y_val)

        logger.info(f"Ensemble model trained: accuracy={ensemble_metrics['accuracy']:.4f}")

        return ensemble, ensemble_metrics

    def save_model(self, model: BaseModel, filename: str = None) -> str:
        """モデル保存"""
        if filename is None:
            filename = f"{model.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"

        filepath = os.path.join(self.model_dir, filename)

        with open(filepath, 'wb') as f:
            pickle.dump(model, f)

        logger.info(f"Model saved: {filepath}")
        return filepath

    def load_model(self, filename: str) -> BaseModel:
        """モデル読み込み"""
        filepath = os.path.join(self.model_dir, filename)

        with open(filepath, 'rb') as f:
            model = pickle.load(f)

        logger.info(f"Model loaded: {filepath}")
        return model

    def get_best_model(self) -> Optional[BaseModel]:
        """最良モデル取得"""
        if not self.training_history:
            return None

        best = max(
            self.training_history,
            key=lambda x: x['metrics'].get('f1_score', 0)
        )

        return self.models.get(best['model_name'])

    def hyperparameter_search(
        self,
        model_class: type,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        param_grid: Dict[str, List],
        n_trials: int = 20,
    ) -> Tuple[Dict, Dict[str, float]]:
        """
        ハイパーパラメータ探索

        Args:
            model_class: モデルクラス
            X_train, y_train: 訓練データ
            X_val, y_val: 検証データ
            param_grid: パラメータ探索範囲
            n_trials: 試行回数

        Returns:
            (最良パラメータ, 最良メトリクス)
        """
        best_params = None
        best_metrics = {'f1_score': 0}

        for trial in range(n_trials):
            # ランダムにパラメータ選択
            params = {}
            for key, values in param_grid.items():
                params[key] = np.random.choice(values)

            try:
                model = model_class(**params)
                model.train(X_train, y_train)
                metrics = self.evaluate_model(model, X_val, y_val)

                if metrics['f1_score'] > best_metrics['f1_score']:
                    best_params = params
                    best_metrics = metrics
                    logger.info(f"Trial {trial}: New best F1={metrics['f1_score']:.4f}")

            except Exception as e:
                logger.warning(f"Trial {trial} failed: {e}")
                continue

        return best_params, best_metrics


class OnlineTrainer:
    """
    オンライン学習トレーナー

    リアルタイムでモデルを更新
    """

    def __init__(
        self,
        model: BaseModel,
        buffer_size: int = 1000,
        update_interval: int = 100,
    ):
        """
        Args:
            model: 更新するモデル
            buffer_size: データバッファサイズ
            update_interval: 更新間隔（サンプル数）
        """
        self.model = model
        self.buffer_size = buffer_size
        self.update_interval = update_interval

        self.X_buffer: List[np.ndarray] = []
        self.y_buffer: List[int] = []
        self.sample_count = 0

    def add_sample(self, x: np.ndarray, y: int) -> None:
        """サンプル追加"""
        self.X_buffer.append(x)
        self.y_buffer.append(y)
        self.sample_count += 1

        # バッファサイズ制限
        if len(self.X_buffer) > self.buffer_size:
            self.X_buffer = self.X_buffer[-self.buffer_size:]
            self.y_buffer = self.y_buffer[-self.buffer_size:]

        # 定期更新
        if self.sample_count % self.update_interval == 0:
            self.update_model()

    def update_model(self) -> None:
        """モデル更新"""
        if len(self.X_buffer) < 100:
            return

        X = np.array(self.X_buffer)
        y = np.array(self.y_buffer)

        self.model.train(X, y)
        logger.info(f"Model updated with {len(X)} samples")

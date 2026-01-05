#!/usr/bin/env python3
"""
================================================================================
    🧠 AI価格予測モジュール
================================================================================
    - LSTM風シーケンス予測 (numpy実装)
    - パターン認識
    - 適応型学習
================================================================================
"""

import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass
from collections import deque
import json
import os
from pathlib import Path


@dataclass
class Prediction:
    """予測結果"""
    direction: str  # "UP", "DOWN", "NEUTRAL"
    confidence: float  # 0.0 - 1.0
    predicted_change: float  # 予測変動率 (%)
    horizon: int  # 予測期間 (データポイント数)


class SimpleRNN:
    """シンプルなRNNセル (numpy実装)"""

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        self.hidden_size = hidden_size

        # 重み初期化 (Xavier)
        scale = np.sqrt(2.0 / (input_size + hidden_size))
        self.Wxh = np.random.randn(hidden_size, input_size) * scale
        self.Whh = np.random.randn(hidden_size, hidden_size) * scale
        self.Why = np.random.randn(output_size, hidden_size) * scale

        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((output_size, 1))

        self.h = np.zeros((hidden_size, 1))

    def forward(self, x: np.ndarray) -> np.ndarray:
        """順伝播"""
        x = x.reshape(-1, 1)
        self.h = np.tanh(self.Wxh @ x + self.Whh @ self.h + self.bh)
        y = self.Why @ self.h + self.by
        return y.flatten()

    def reset_hidden(self):
        """隠れ状態リセット"""
        self.h = np.zeros((self.hidden_size, 1))


class PatternRecognizer:
    """価格パターン認識"""

    def __init__(self):
        self.patterns = {
            'double_bottom': self._detect_double_bottom,
            'double_top': self._detect_double_top,
            'head_shoulders': self._detect_head_shoulders,
            'ascending_triangle': self._detect_ascending_triangle,
            'breakout': self._detect_breakout
        }

    def _detect_double_bottom(self, prices: List[float]) -> Tuple[bool, float]:
        """ダブルボトム検出"""
        if len(prices) < 20:
            return False, 0.0

        p = prices[-20:]
        min_idx = [i for i in range(1, len(p)-1)
                   if p[i] < p[i-1] and p[i] < p[i+1]]

        if len(min_idx) >= 2:
            if abs(p[min_idx[-1]] - p[min_idx[-2]]) / p[min_idx[-1]] < 0.02:
                return True, 0.7
        return False, 0.0

    def _detect_double_top(self, prices: List[float]) -> Tuple[bool, float]:
        """ダブルトップ検出"""
        if len(prices) < 20:
            return False, 0.0

        p = prices[-20:]
        max_idx = [i for i in range(1, len(p)-1)
                   if p[i] > p[i-1] and p[i] > p[i+1]]

        if len(max_idx) >= 2:
            if abs(p[max_idx[-1]] - p[max_idx[-2]]) / p[max_idx[-1]] < 0.02:
                return True, 0.7
        return False, 0.0

    def _detect_head_shoulders(self, prices: List[float]) -> Tuple[bool, float]:
        """ヘッドアンドショルダー検出"""
        if len(prices) < 30:
            return False, 0.0

        p = prices[-30:]
        max_idx = [i for i in range(1, len(p)-1)
                   if p[i] > p[i-1] and p[i] > p[i+1]]

        if len(max_idx) >= 3:
            # 中央が最も高い
            if p[max_idx[-2]] > p[max_idx[-1]] and p[max_idx[-2]] > p[max_idx[-3]]:
                return True, 0.8
        return False, 0.0

    def _detect_ascending_triangle(self, prices: List[float]) -> Tuple[bool, float]:
        """上昇トライアングル検出"""
        if len(prices) < 20:
            return False, 0.0

        p = prices[-20:]
        highs = [p[i] for i in range(1, len(p)-1)
                 if p[i] > p[i-1] and p[i] > p[i+1]]
        lows = [p[i] for i in range(1, len(p)-1)
                if p[i] < p[i-1] and p[i] < p[i+1]]

        if len(highs) >= 2 and len(lows) >= 2:
            # 高値が横ばい、安値が上昇
            high_stable = abs(highs[-1] - highs[-2]) / highs[-1] < 0.01
            low_rising = lows[-1] > lows[-2]

            if high_stable and low_rising:
                return True, 0.6
        return False, 0.0

    def _detect_breakout(self, prices: List[float]) -> Tuple[bool, float]:
        """ブレイクアウト検出"""
        if len(prices) < 30:
            return False, 0.0

        recent = prices[-5:]
        historical = prices[-30:-5]

        recent_avg = np.mean(recent)
        hist_high = np.max(historical)
        hist_low = np.min(historical)

        # 上方ブレイクアウト
        if recent_avg > hist_high:
            return True, 0.75

        # 下方ブレイクアウト (売りシグナル)
        if recent_avg < hist_low:
            return True, -0.75

        return False, 0.0

    def analyze(self, prices: List[float]) -> List[Tuple[str, float]]:
        """全パターン分析"""
        results = []
        for name, detector in self.patterns.items():
            detected, confidence = detector(prices)
            if detected:
                results.append((name, confidence))
        return results


class MLPredictor:
    """機械学習価格予測器"""

    def __init__(self, lookback: int = 30, hidden_size: int = 32):
        self.lookback = lookback
        self.hidden_size = hidden_size

        self.prices: deque = deque(maxlen=500)
        self.rnn = SimpleRNN(lookback, hidden_size, 3)  # 3出力: 上昇/下落/横ばい確率
        self.pattern_recognizer = PatternRecognizer()

        self.training_data: List[Tuple[np.ndarray, int]] = []
        self.is_trained = False

        # モデル保存パス
        self.model_path = Path(__file__).parent.parent.parent / "data" / "ml_model.json"

        self._load_model()

    def add_price(self, price: float):
        """価格追加"""
        self.prices.append(price)

        # オンライン学習用データ収集
        if len(self.prices) >= self.lookback + 5:
            self._collect_training_sample()

    def _normalize(self, prices: List[float]) -> np.ndarray:
        """価格正規化"""
        prices = np.array(prices)
        if len(prices) < 2:
            return prices

        # 変化率に変換
        returns = np.diff(prices) / prices[:-1]
        # -1 ~ 1 にクリップ
        returns = np.clip(returns, -0.1, 0.1) * 10
        return returns

    def _collect_training_sample(self):
        """学習サンプル収集"""
        if len(self.prices) < self.lookback + 5:
            return

        prices = list(self.prices)
        x = self._normalize(prices[-self.lookback-5:-5])

        # 5ステップ後の方向をラベルに
        future_change = (prices[-1] - prices[-5]) / prices[-5]

        if future_change > 0.002:
            label = 0  # UP
        elif future_change < -0.002:
            label = 1  # DOWN
        else:
            label = 2  # NEUTRAL

        if len(x) == self.lookback - 1:
            self.training_data.append((x, label))

            # 定期的に学習
            if len(self.training_data) >= 100 and len(self.training_data) % 50 == 0:
                self._train()

    def _train(self, epochs: int = 10, lr: float = 0.01):
        """オンライン学習"""
        if len(self.training_data) < 50:
            return

        # 最新100サンプルで学習
        samples = self.training_data[-100:]

        for _ in range(epochs):
            total_loss = 0
            for x, label in samples:
                self.rnn.reset_hidden()

                # 順伝播
                for t in range(len(x)):
                    out = self.rnn.forward(np.array([x[t]]))

                # ソフトマックス
                exp_out = np.exp(out - np.max(out))
                probs = exp_out / exp_out.sum()

                # 損失計算
                loss = -np.log(probs[label] + 1e-10)
                total_loss += loss

                # 簡易勾配更新
                grad = probs.copy()
                grad[label] -= 1

                # 重み更新 (簡易版)
                self.rnn.Why -= lr * np.outer(grad, self.rnn.h.flatten())
                self.rnn.by -= lr * grad.reshape(-1, 1)

        self.is_trained = True
        self._save_model()

    def predict(self, horizon: int = 5) -> Optional[Prediction]:
        """価格予測"""
        if len(self.prices) < self.lookback:
            return None

        prices = list(self.prices)

        # RNN予測
        x = self._normalize(prices[-self.lookback:])
        if len(x) < self.lookback - 1:
            return None

        self.rnn.reset_hidden()
        for t in range(len(x)):
            out = self.rnn.forward(np.array([x[t]]))

        # ソフトマックス
        exp_out = np.exp(out - np.max(out))
        probs = exp_out / exp_out.sum()

        # パターン認識
        patterns = self.pattern_recognizer.analyze(prices)

        # 統合スコア
        up_score = probs[0]
        down_score = probs[1]

        for pattern_name, conf in patterns:
            if pattern_name in ['double_bottom', 'ascending_triangle']:
                up_score += conf * 0.3
            elif pattern_name in ['double_top', 'head_shoulders']:
                down_score += conf * 0.3
            elif pattern_name == 'breakout':
                if conf > 0:
                    up_score += abs(conf) * 0.3
                else:
                    down_score += abs(conf) * 0.3

        # 正規化
        total = up_score + down_score + probs[2]
        up_score /= total
        down_score /= total

        # 予測変動率 (簡易推定)
        volatility = np.std(list(self.prices)[-20:]) / np.mean(list(self.prices)[-20:])
        predicted_change = volatility * 100 * (up_score - down_score)

        if up_score > down_score * 1.3:
            return Prediction("UP", up_score, predicted_change, horizon)
        elif down_score > up_score * 1.3:
            return Prediction("DOWN", down_score, -abs(predicted_change), horizon)
        else:
            return Prediction("NEUTRAL", max(probs[2], 0.3), 0.0, horizon)

    def get_pattern_signals(self) -> List[Tuple[str, float]]:
        """パターンシグナル取得"""
        if len(self.prices) < 30:
            return []
        return self.pattern_recognizer.analyze(list(self.prices))

    def _save_model(self):
        """モデル保存"""
        try:
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            model_data = {
                'Wxh': self.rnn.Wxh.tolist(),
                'Whh': self.rnn.Whh.tolist(),
                'Why': self.rnn.Why.tolist(),
                'bh': self.rnn.bh.tolist(),
                'by': self.rnn.by.tolist(),
                'is_trained': self.is_trained
            }
            with open(self.model_path, 'w') as f:
                json.dump(model_data, f)
        except:
            pass

    def _load_model(self):
        """モデル読み込み"""
        try:
            if self.model_path.exists():
                with open(self.model_path, 'r') as f:
                    model_data = json.load(f)
                self.rnn.Wxh = np.array(model_data['Wxh'])
                self.rnn.Whh = np.array(model_data['Whh'])
                self.rnn.Why = np.array(model_data['Why'])
                self.rnn.bh = np.array(model_data['bh'])
                self.rnn.by = np.array(model_data['by'])
                self.is_trained = model_data.get('is_trained', False)
        except:
            pass


class EnsemblePredictor:
    """アンサンブル予測器"""

    def __init__(self):
        self.predictors = [
            MLPredictor(lookback=20, hidden_size=16),
            MLPredictor(lookback=30, hidden_size=32),
            MLPredictor(lookback=50, hidden_size=48),
        ]

    def add_price(self, price: float):
        """全予測器に価格追加"""
        for p in self.predictors:
            p.add_price(price)

    def predict(self) -> Optional[Prediction]:
        """アンサンブル予測"""
        predictions = []

        for p in self.predictors:
            pred = p.predict()
            if pred:
                predictions.append(pred)

        if not predictions:
            return None

        # 多数決 + 信頼度加重
        up_score = sum(p.confidence for p in predictions if p.direction == "UP")
        down_score = sum(p.confidence for p in predictions if p.direction == "DOWN")
        neutral_score = sum(p.confidence for p in predictions if p.direction == "NEUTRAL")

        avg_change = np.mean([p.predicted_change for p in predictions])

        if up_score > down_score and up_score > neutral_score:
            confidence = up_score / (up_score + down_score + neutral_score)
            return Prediction("UP", confidence, avg_change, 5)
        elif down_score > up_score and down_score > neutral_score:
            confidence = down_score / (up_score + down_score + neutral_score)
            return Prediction("DOWN", confidence, avg_change, 5)
        else:
            return Prediction("NEUTRAL", 0.5, 0.0, 5)

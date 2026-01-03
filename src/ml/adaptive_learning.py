"""
Adaptive Learning System
=========================
リアルタイム自己進化・適応学習システム
市場から常に学習し、成長し続ける
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from collections import deque
import threading
import time
import pickle
import os
from loguru import logger


class ExperienceBuffer:
    """
    経験バッファ

    取引経験を保存し、学習に使用
    """

    def __init__(self, max_size: int = 100000):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.priorities = deque(maxlen=max_size)  # 優先度付き経験再生用

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        info: Dict = None,
    ) -> None:
        """経験追加"""
        experience = {
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done,
            'info': info or {},
            'timestamp': datetime.now(),
        }
        self.buffer.append(experience)
        # 初期優先度は報酬の絶対値に基づく
        self.priorities.append(abs(reward) + 0.01)

    def sample(self, batch_size: int, prioritized: bool = True) -> List[Dict]:
        """バッチサンプリング"""
        if len(self.buffer) < batch_size:
            return list(self.buffer)

        if prioritized:
            # 優先度付きサンプリング
            priorities = np.array(self.priorities)
            probs = priorities / priorities.sum()
            indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=False)
            return [self.buffer[i] for i in indices]
        else:
            indices = np.random.choice(len(self.buffer), batch_size, replace=False)
            return [self.buffer[i] for i in indices]

    def get_recent(self, n: int = 100) -> List[Dict]:
        """最近のn件取得"""
        return list(self.buffer)[-n:]

    def __len__(self) -> int:
        return len(self.buffer)


class OnlineNeuralNetwork:
    """
    オンライン学習ニューラルネットワーク

    リアルタイムで更新可能な軽量NN
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [64, 32],
        output_dim: int = 3,
        learning_rate: float = 0.001,
    ):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.lr = learning_rate

        # 重み初期化
        self.weights = []
        self.biases = []

        dims = [input_dim] + hidden_dims + [output_dim]
        for i in range(len(dims) - 1):
            # He初期化
            w = np.random.randn(dims[i], dims[i+1]) * np.sqrt(2.0 / dims[i])
            b = np.zeros(dims[i+1])
            self.weights.append(w)
            self.biases.append(b)

        # 適応的学習率 (Adam)
        self.m_w = [np.zeros_like(w) for w in self.weights]
        self.v_w = [np.zeros_like(w) for w in self.weights]
        self.m_b = [np.zeros_like(b) for b in self.biases]
        self.v_b = [np.zeros_like(b) for b in self.biases]
        self.t = 0

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        """順伝播"""
        activations = [x]

        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = np.dot(activations[-1], w) + b

            if i < len(self.weights) - 1:
                # 隠れ層: ReLU
                a = np.maximum(0, z)
            else:
                # 出力層: Softmax
                exp_z = np.exp(z - np.max(z, axis=-1, keepdims=True))
                a = exp_z / np.sum(exp_z, axis=-1, keepdims=True)

            activations.append(a)

        return activations[-1], activations

    def backward(
        self,
        activations: List[np.ndarray],
        y_true: np.ndarray,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """逆伝播"""
        m = y_true.shape[0] if len(y_true.shape) > 0 else 1

        # 出力層の勾配
        dz = activations[-1] - y_true

        dw_list = []
        db_list = []

        for i in range(len(self.weights) - 1, -1, -1):
            a_prev = activations[i]

            if len(a_prev.shape) == 1:
                a_prev = a_prev.reshape(1, -1)
            if len(dz.shape) == 1:
                dz = dz.reshape(1, -1)

            dw = np.dot(a_prev.T, dz) / m
            db = np.sum(dz, axis=0) / m

            dw_list.insert(0, dw)
            db_list.insert(0, db)

            if i > 0:
                da = np.dot(dz, self.weights[i].T)
                # ReLU微分
                dz = da * (activations[i] > 0).astype(float)

        return dw_list, db_list

    def update(self, dw_list: List[np.ndarray], db_list: List[np.ndarray]) -> None:
        """Adam最適化で重み更新"""
        self.t += 1
        beta1, beta2 = 0.9, 0.999
        epsilon = 1e-8

        for i in range(len(self.weights)):
            # モーメンタム更新
            self.m_w[i] = beta1 * self.m_w[i] + (1 - beta1) * dw_list[i]
            self.m_b[i] = beta1 * self.m_b[i] + (1 - beta1) * db_list[i]

            # 2次モーメント更新
            self.v_w[i] = beta2 * self.v_w[i] + (1 - beta2) * (dw_list[i] ** 2)
            self.v_b[i] = beta2 * self.v_b[i] + (1 - beta2) * (db_list[i] ** 2)

            # バイアス補正
            m_w_hat = self.m_w[i] / (1 - beta1 ** self.t)
            m_b_hat = self.m_b[i] / (1 - beta1 ** self.t)
            v_w_hat = self.v_w[i] / (1 - beta2 ** self.t)
            v_b_hat = self.v_b[i] / (1 - beta2 ** self.t)

            # 重み更新
            self.weights[i] -= self.lr * m_w_hat / (np.sqrt(v_w_hat) + epsilon)
            self.biases[i] -= self.lr * m_b_hat / (np.sqrt(v_b_hat) + epsilon)

    def train_step(self, x: np.ndarray, y: np.ndarray) -> float:
        """1ステップ訓練"""
        output, activations = self.forward(x)
        dw_list, db_list = self.backward(activations, y)
        self.update(dw_list, db_list)

        # 損失計算
        loss = -np.mean(np.sum(y * np.log(output + 1e-10), axis=-1))
        return loss

    def predict(self, x: np.ndarray) -> np.ndarray:
        """予測"""
        output, _ = self.forward(x)
        return output

    def predict_action(self, x: np.ndarray) -> int:
        """最良アクション予測"""
        probs = self.predict(x)
        return np.argmax(probs)


class RealTimeReinforcementLearner:
    """
    リアルタイム強化学習エージェント

    取引経験から継続的に学習
    """

    def __init__(
        self,
        state_dim: int = 50,
        action_dim: int = 3,  # BUY, HOLD, SELL
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma

        # 探索率
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # Q-Network (オンライン)
        self.q_network = OnlineNeuralNetwork(
            input_dim=state_dim,
            hidden_dims=[128, 64, 32],
            output_dim=action_dim,
            learning_rate=learning_rate,
        )

        # Target Network
        self.target_network = OnlineNeuralNetwork(
            input_dim=state_dim,
            hidden_dims=[128, 64, 32],
            output_dim=action_dim,
            learning_rate=learning_rate,
        )
        self._sync_target_network()

        # 経験バッファ
        self.experience_buffer = ExperienceBuffer(max_size=50000)

        # 学習統計
        self.total_steps = 0
        self.total_rewards = 0
        self.episode_rewards = []
        self.losses = []

        # 学習スレッド
        self.learning_thread = None
        self.is_learning = False

    def _sync_target_network(self) -> None:
        """ターゲットネットワーク同期"""
        for i in range(len(self.q_network.weights)):
            self.target_network.weights[i] = self.q_network.weights[i].copy()
            self.target_network.biases[i] = self.q_network.biases[i].copy()

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        アクション選択（ε-greedy）

        Returns:
            0: BUY, 1: HOLD, 2: SELL
        """
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)

        q_values = self.q_network.predict(state.reshape(1, -1))
        return np.argmax(q_values[0])

    def store_experience(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        info: Dict = None,
    ) -> None:
        """経験保存"""
        self.experience_buffer.add(state, action, reward, next_state, done, info)
        self.total_steps += 1
        self.total_rewards += reward

    def learn_from_batch(self, batch_size: int = 32) -> float:
        """バッチ学習"""
        if len(self.experience_buffer) < batch_size:
            return 0.0

        batch = self.experience_buffer.sample(batch_size, prioritized=True)

        states = np.array([e['state'] for e in batch])
        actions = np.array([e['action'] for e in batch])
        rewards = np.array([e['reward'] for e in batch])
        next_states = np.array([e['next_state'] for e in batch])
        dones = np.array([e['done'] for e in batch])

        # 現在のQ値
        current_q = self.q_network.predict(states)

        # ターゲットQ値 (Double DQN)
        next_q_online = self.q_network.predict(next_states)
        next_q_target = self.target_network.predict(next_states)

        best_actions = np.argmax(next_q_online, axis=1)
        next_q_values = next_q_target[np.arange(batch_size), best_actions]

        # ターゲット計算
        targets = current_q.copy()
        for i in range(batch_size):
            if dones[i]:
                targets[i, actions[i]] = rewards[i]
            else:
                targets[i, actions[i]] = rewards[i] + self.gamma * next_q_values[i]

        # 学習
        loss = self.q_network.train_step(states, targets)
        self.losses.append(loss)

        # 探索率減衰
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        # 定期的にターゲットネットワーク同期
        if self.total_steps % 100 == 0:
            self._sync_target_network()

        return loss

    def start_continuous_learning(self, interval: float = 1.0) -> None:
        """継続学習開始"""
        self.is_learning = True

        def learning_loop():
            while self.is_learning:
                if len(self.experience_buffer) >= 32:
                    loss = self.learn_from_batch(32)
                    if self.total_steps % 100 == 0:
                        logger.debug(f"RL Learning step {self.total_steps}: loss={loss:.4f}, epsilon={self.epsilon:.3f}")
                time.sleep(interval)

        self.learning_thread = threading.Thread(target=learning_loop, daemon=True)
        self.learning_thread.start()
        logger.info("Continuous RL learning started")

    def stop_learning(self) -> None:
        """学習停止"""
        self.is_learning = False
        if self.learning_thread:
            self.learning_thread.join(timeout=5)
        logger.info("RL learning stopped")

    def get_stats(self) -> Dict:
        """統計取得"""
        return {
            'total_steps': self.total_steps,
            'total_rewards': self.total_rewards,
            'epsilon': self.epsilon,
            'buffer_size': len(self.experience_buffer),
            'avg_loss': np.mean(self.losses[-100:]) if self.losses else 0,
        }


class AdaptiveStrategyOptimizer:
    """
    適応型戦略最適化

    各戦略の重みをリアルタイムで最適化
    """

    def __init__(
        self,
        strategy_names: List[str],
        initial_weights: List[float] = None,
        adaptation_rate: float = 0.1,
        lookback_window: int = 100,
    ):
        self.strategy_names = strategy_names
        self.n_strategies = len(strategy_names)

        # 初期重み
        if initial_weights:
            self.weights = np.array(initial_weights)
        else:
            self.weights = np.ones(self.n_strategies) / self.n_strategies

        self.adaptation_rate = adaptation_rate
        self.lookback_window = lookback_window

        # パフォーマンス履歴
        self.performance_history: Dict[str, List[float]] = {
            name: [] for name in strategy_names
        }

        # UCB (Upper Confidence Bound) 用
        self.strategy_counts = np.zeros(self.n_strategies)
        self.strategy_rewards = np.zeros(self.n_strategies)

    def record_performance(self, strategy_name: str, pnl: float) -> None:
        """戦略パフォーマンス記録"""
        if strategy_name not in self.performance_history:
            return

        self.performance_history[strategy_name].append(pnl)

        # 履歴制限
        if len(self.performance_history[strategy_name]) > self.lookback_window * 2:
            self.performance_history[strategy_name] = \
                self.performance_history[strategy_name][-self.lookback_window:]

        # UCB更新
        idx = self.strategy_names.index(strategy_name)
        self.strategy_counts[idx] += 1
        self.strategy_rewards[idx] += pnl

    def update_weights(self) -> np.ndarray:
        """重み更新（複数のアルゴリズムのアンサンブル）"""

        # 1. パフォーマンスベースの重み
        perf_weights = self._calculate_performance_weights()

        # 2. UCBベースの重み
        ucb_weights = self._calculate_ucb_weights()

        # 3. シャープレシオベースの重み
        sharpe_weights = self._calculate_sharpe_weights()

        # アンサンブル
        new_weights = (perf_weights + ucb_weights + sharpe_weights) / 3

        # スムーズな更新
        self.weights = (1 - self.adaptation_rate) * self.weights + \
                       self.adaptation_rate * new_weights

        # 正規化
        self.weights = np.maximum(self.weights, 0.05)  # 最小5%
        self.weights = self.weights / self.weights.sum()

        return self.weights

    def _calculate_performance_weights(self) -> np.ndarray:
        """パフォーマンスベースの重み"""
        weights = np.zeros(self.n_strategies)

        for i, name in enumerate(self.strategy_names):
            history = self.performance_history[name]
            if len(history) >= 10:
                recent = history[-self.lookback_window:]
                # 累積リターン + 一貫性ボーナス
                total_return = sum(recent)
                consistency = sum(1 for p in recent if p > 0) / len(recent)
                weights[i] = total_return * (0.5 + 0.5 * consistency)
            else:
                weights[i] = 1.0  # デフォルト

        # ソフトマックス
        weights = np.exp(weights - np.max(weights))
        return weights / weights.sum()

    def _calculate_ucb_weights(self) -> np.ndarray:
        """UCB (Upper Confidence Bound) ベースの重み"""
        total_counts = self.strategy_counts.sum() + 1

        ucb_values = np.zeros(self.n_strategies)
        for i in range(self.n_strategies):
            if self.strategy_counts[i] > 0:
                avg_reward = self.strategy_rewards[i] / self.strategy_counts[i]
                exploration = np.sqrt(2 * np.log(total_counts) / self.strategy_counts[i])
                ucb_values[i] = avg_reward + exploration
            else:
                ucb_values[i] = float('inf')

        # 正規化
        if np.any(np.isinf(ucb_values)):
            weights = np.ones(self.n_strategies) / self.n_strategies
        else:
            weights = np.exp(ucb_values - np.max(ucb_values))
            weights = weights / weights.sum()

        return weights

    def _calculate_sharpe_weights(self) -> np.ndarray:
        """シャープレシオベースの重み"""
        sharpe_ratios = np.zeros(self.n_strategies)

        for i, name in enumerate(self.strategy_names):
            history = self.performance_history[name]
            if len(history) >= 20:
                returns = np.array(history[-self.lookback_window:])
                if returns.std() > 0:
                    sharpe_ratios[i] = returns.mean() / returns.std()
                else:
                    sharpe_ratios[i] = returns.mean() if returns.mean() > 0 else 0
            else:
                sharpe_ratios[i] = 0

        # ソフトマックス
        weights = np.exp(sharpe_ratios - np.max(sharpe_ratios))
        return weights / weights.sum()

    def get_weights(self) -> Dict[str, float]:
        """現在の重み取得"""
        return dict(zip(self.strategy_names, self.weights))


class MarketRegimeDetector:
    """
    市場レジーム検出

    市場の状態を検出し、戦略を適応
    """

    def __init__(self, lookback: int = 100):
        self.lookback = lookback
        self.price_history: List[float] = []
        self.volume_history: List[float] = []
        self.current_regime = "UNKNOWN"

        # レジーム: TRENDING_UP, TRENDING_DOWN, RANGING, HIGH_VOLATILITY, LOW_VOLATILITY

    def update(self, price: float, volume: float) -> str:
        """データ更新とレジーム検出"""
        self.price_history.append(price)
        self.volume_history.append(volume)

        # 履歴制限
        if len(self.price_history) > self.lookback * 2:
            self.price_history = self.price_history[-self.lookback:]
            self.volume_history = self.volume_history[-self.lookback:]

        if len(self.price_history) < 20:
            return self.current_regime

        self.current_regime = self._detect_regime()
        return self.current_regime

    def _detect_regime(self) -> str:
        """レジーム検出ロジック"""
        prices = np.array(self.price_history[-self.lookback:])

        # リターン計算
        returns = np.diff(prices) / prices[:-1]

        # ボラティリティ
        volatility = np.std(returns)
        avg_volatility = 0.02  # 基準ボラティリティ

        # トレンド強度
        trend = (prices[-1] - prices[0]) / prices[0]

        # 方向一貫性
        positive_returns = np.sum(returns > 0) / len(returns)

        # レジーム判定
        if volatility > avg_volatility * 2:
            return "HIGH_VOLATILITY"
        elif volatility < avg_volatility * 0.5:
            return "LOW_VOLATILITY"
        elif trend > 0.02 and positive_returns > 0.6:
            return "TRENDING_UP"
        elif trend < -0.02 and positive_returns < 0.4:
            return "TRENDING_DOWN"
        else:
            return "RANGING"

    def get_strategy_adjustments(self) -> Dict[str, float]:
        """レジームに基づく戦略調整"""
        adjustments = {
            'market_making': 1.0,
            'momentum': 1.0,
            'mean_reversion': 1.0,
            'breakout': 1.0,
            'arbitrage': 1.0,
            'ml_strategy': 1.0,
        }

        if self.current_regime == "TRENDING_UP":
            adjustments['momentum'] = 1.5
            adjustments['breakout'] = 1.3
            adjustments['mean_reversion'] = 0.5

        elif self.current_regime == "TRENDING_DOWN":
            adjustments['momentum'] = 1.5
            adjustments['breakout'] = 1.3
            adjustments['mean_reversion'] = 0.5

        elif self.current_regime == "RANGING":
            adjustments['mean_reversion'] = 1.5
            adjustments['market_making'] = 1.3
            adjustments['momentum'] = 0.5
            adjustments['breakout'] = 0.5

        elif self.current_regime == "HIGH_VOLATILITY":
            adjustments['market_making'] = 0.3  # リスク回避
            adjustments['breakout'] = 1.5
            adjustments['momentum'] = 1.2

        elif self.current_regime == "LOW_VOLATILITY":
            adjustments['market_making'] = 1.5  # スプレッド取得
            adjustments['arbitrage'] = 1.3
            adjustments['breakout'] = 0.5

        return adjustments


class SelfEvolvingSystem:
    """
    自己進化システム

    全コンポーネントを統合し、継続的に進化
    """

    def __init__(
        self,
        strategy_names: List[str],
        state_dim: int = 50,
        save_dir: str = "models/evolved",
    ):
        self.strategy_names = strategy_names
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        # コンポーネント初期化
        self.rl_agent = RealTimeReinforcementLearner(
            state_dim=state_dim,
            action_dim=3,
        )

        self.strategy_optimizer = AdaptiveStrategyOptimizer(
            strategy_names=strategy_names,
        )

        self.regime_detector = MarketRegimeDetector()

        # オンライン価格予測モデル
        self.price_predictor = OnlineNeuralNetwork(
            input_dim=state_dim,
            hidden_dims=[64, 32],
            output_dim=3,  # UP, NEUTRAL, DOWN
        )

        # 学習統計
        self.evolution_history: List[Dict] = []
        self.generation = 0

        # 自動保存タイマー
        self.last_save_time = datetime.now()
        self.save_interval = timedelta(hours=1)

        logger.info("Self-Evolving System initialized")

    def start(self) -> None:
        """システム開始"""
        self.rl_agent.start_continuous_learning(interval=0.5)
        logger.info("🧬 Self-Evolution started - The system will continuously improve")

    def stop(self) -> None:
        """システム停止"""
        self.rl_agent.stop_learning()
        self.save_state()
        logger.info("Self-Evolution stopped")

    def process_market_update(
        self,
        state: np.ndarray,
        price: float,
        volume: float,
    ) -> Dict[str, Any]:
        """
        市場更新処理

        Returns:
            action: 推奨アクション
            strategy_weights: 更新された戦略重み
            regime: 現在の市場レジーム
            prediction: 価格予測
        """
        # レジーム検出
        regime = self.regime_detector.update(price, volume)
        regime_adjustments = self.regime_detector.get_strategy_adjustments()

        # RL エージェントからアクション取得
        action = self.rl_agent.select_action(state, training=True)

        # 価格予測
        prediction_probs = self.price_predictor.predict(state.reshape(1, -1))
        prediction = np.argmax(prediction_probs[0])  # 0: DOWN, 1: NEUTRAL, 2: UP

        # 戦略重み取得
        base_weights = self.strategy_optimizer.get_weights()

        # レジーム調整適用
        adjusted_weights = {}
        for strategy, weight in base_weights.items():
            adjustment = regime_adjustments.get(strategy, 1.0)
            adjusted_weights[strategy] = weight * adjustment

        # 正規化
        total = sum(adjusted_weights.values())
        adjusted_weights = {k: v / total for k, v in adjusted_weights.items()}

        return {
            'action': action,  # 0: BUY, 1: HOLD, 2: SELL
            'strategy_weights': adjusted_weights,
            'regime': regime,
            'prediction': prediction,
            'prediction_probs': prediction_probs[0],
        }

    def record_outcome(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        strategy_name: str = None,
        pnl: float = 0,
    ) -> None:
        """
        取引結果を記録し学習
        """
        # RL経験保存
        self.rl_agent.store_experience(state, action, reward, next_state, done)

        # 戦略パフォーマンス記録
        if strategy_name:
            self.strategy_optimizer.record_performance(strategy_name, pnl)

        # 価格予測モデル更新
        if not done:
            # 次の価格方向をターゲットに
            if pnl > 0:
                target = 2  # UP
            elif pnl < 0:
                target = 0  # DOWN
            else:
                target = 1  # NEUTRAL

            target_one_hot = np.zeros(3)
            target_one_hot[target] = 1
            self.price_predictor.train_step(
                state.reshape(1, -1),
                target_one_hot.reshape(1, -1)
            )

        # 定期的な重み更新
        if self.rl_agent.total_steps % 50 == 0:
            self.strategy_optimizer.update_weights()

        # 自動保存
        if datetime.now() - self.last_save_time > self.save_interval:
            self.save_state()
            self.last_save_time = datetime.now()

        # 進化履歴更新
        if self.rl_agent.total_steps % 1000 == 0:
            self.generation += 1
            self.evolution_history.append({
                'generation': self.generation,
                'timestamp': datetime.now(),
                'total_steps': self.rl_agent.total_steps,
                'total_rewards': self.rl_agent.total_rewards,
                'strategy_weights': self.strategy_optimizer.get_weights(),
                'epsilon': self.rl_agent.epsilon,
            })
            logger.info(f"🧬 Generation {self.generation}: "
                       f"Steps={self.rl_agent.total_steps}, "
                       f"Rewards={self.rl_agent.total_rewards:.2f}")

    def save_state(self) -> None:
        """状態保存"""
        state = {
            'generation': self.generation,
            'rl_weights': [(w.copy(), b.copy()) for w, b in
                          zip(self.rl_agent.q_network.weights,
                              self.rl_agent.q_network.biases)],
            'predictor_weights': [(w.copy(), b.copy()) for w, b in
                                 zip(self.price_predictor.weights,
                                     self.price_predictor.biases)],
            'strategy_weights': self.strategy_optimizer.weights.copy(),
            'epsilon': self.rl_agent.epsilon,
            'evolution_history': self.evolution_history,
        }

        filepath = os.path.join(self.save_dir, 'evolved_state.pkl')
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)

        logger.info(f"Evolution state saved: Generation {self.generation}")

    def load_state(self) -> bool:
        """状態読み込み"""
        filepath = os.path.join(self.save_dir, 'evolved_state.pkl')

        if not os.path.exists(filepath):
            return False

        try:
            with open(filepath, 'rb') as f:
                state = pickle.load(f)

            self.generation = state['generation']

            # RLネットワーク復元
            for i, (w, b) in enumerate(state['rl_weights']):
                self.rl_agent.q_network.weights[i] = w
                self.rl_agent.q_network.biases[i] = b

            # 予測モデル復元
            for i, (w, b) in enumerate(state['predictor_weights']):
                self.price_predictor.weights[i] = w
                self.price_predictor.biases[i] = b

            self.strategy_optimizer.weights = state['strategy_weights']
            self.rl_agent.epsilon = state['epsilon']
            self.evolution_history = state['evolution_history']

            logger.info(f"Evolution state loaded: Generation {self.generation}")
            return True

        except Exception as e:
            logger.error(f"Failed to load evolution state: {e}")
            return False

    def get_evolution_stats(self) -> Dict:
        """進化統計取得"""
        rl_stats = self.rl_agent.get_stats()

        return {
            'generation': self.generation,
            'regime': self.regime_detector.current_regime,
            'strategy_weights': self.strategy_optimizer.get_weights(),
            'rl_stats': rl_stats,
            'evolution_history_length': len(self.evolution_history),
        }

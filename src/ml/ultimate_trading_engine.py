"""
Ultimate Trading Engine
========================
世界最強のAIトレーディングエンジン
全てのコンポーネントを統合
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import deque
from datetime import datetime
import threading
import asyncio
from concurrent.futures import ThreadPoolExecutor
from loguru import logger

# Import all modules
from .sota_models import (
    PatchTST, Mamba, iTransformer, TemporalFusionTransformer,
    BayesianNeuralNetwork, UltimateEnsemble, create_model, DEVICE
)
from .advanced_features import AdvancedFeatureGenerator
from .advanced_rl import UnifiedRLAgent, PPO, SAC, C51, QRDQN
from .vector_db import VectorPatternDB, TemporalPatternMatcher
from .world_model import DreamerV3WorldModel, WorldModelTrainer
from .multi_agent import MultiAgentTradingSystem, HierarchicalMultiAgentSystem, AgentDecision


@dataclass
class TradingSignal:
    """トレーディングシグナル"""
    action: int  # 0: SELL, 1: HOLD, 2: BUY
    confidence: float
    position_size: float
    stop_loss: float
    take_profit: float
    urgency: float
    reasoning: str
    components: Dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class TradingState:
    """トレーディング状態"""
    position: float = 0.0
    entry_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    equity: float = 1.0
    max_equity: float = 1.0
    drawdown: float = 0.0
    trade_count: int = 0
    win_count: int = 0


class ConfidenceGate:
    """
    動的信頼度ゲート

    市場状況に応じて閾値を調整
    """

    def __init__(
        self,
        base_confidence: float = 0.60,
        min_confidence: float = 0.50,
        max_confidence: float = 0.85,
    ):
        self.base_confidence = base_confidence
        self.min_confidence = min_confidence
        self.max_confidence = max_confidence
        self.current_threshold = base_confidence

        self.trade_history = deque(maxlen=100)
        self.volatility_history = deque(maxlen=50)

    def update(
        self,
        win: Optional[bool] = None,
        volatility: float = 0.0,
        consensus: float = 0.0,
    ) -> float:
        """閾値を更新"""
        adjustments = 0.0

        # Win/loss adjustment
        if win is not None:
            self.trade_history.append(1.0 if win else 0.0)

            if len(self.trade_history) >= 10:
                recent_wr = np.mean(list(self.trade_history)[-10:])
                if recent_wr > 0.7:
                    adjustments -= 0.05  # Lower threshold if winning
                elif recent_wr < 0.3:
                    adjustments += 0.10  # Raise threshold if losing

        # Volatility adjustment
        self.volatility_history.append(volatility)
        if self.volatility_history:
            avg_vol = np.mean(self.volatility_history)
            if volatility > avg_vol * 2:
                adjustments += 0.05  # More cautious in high vol
            elif volatility < avg_vol * 0.5:
                adjustments -= 0.02  # Less cautious in low vol

        # Consensus adjustment
        if consensus > 0.8:
            adjustments -= 0.03  # Lower threshold if high consensus
        elif consensus < 0.5:
            adjustments += 0.05  # Raise threshold if low consensus

        self.current_threshold = np.clip(
            self.base_confidence + adjustments,
            self.min_confidence,
            self.max_confidence
        )

        return self.current_threshold

    def check(self, confidence: float) -> bool:
        """信頼度チェック"""
        return confidence >= self.current_threshold


class UltimateTradingEngine:
    """
    究極のトレーディングエンジン

    全AIコンポーネントを統合し、最適な取引判断を行う
    """

    def __init__(
        self,
        feature_dim: int = 500,
        seq_len: int = 60,
        enable_all_components: bool = True,
    ):
        self.feature_dim = feature_dim
        self.seq_len = seq_len

        logger.info("Initializing Ultimate Trading Engine...")

        # ========== Feature Engineering ==========
        self.feature_generator = AdvancedFeatureGenerator(
            lookback=100,
            frac_diff_d=0.4
        )

        # ========== SOTA Deep Learning Models ==========
        self.sota_ensemble = UltimateEnsemble(
            input_dim=feature_dim,
            seq_len=seq_len,
            output_dim=3,
            d_model=128,
        ).to(DEVICE)

        self.bayesian_nn = BayesianNeuralNetwork(
            input_dim=feature_dim,
            hidden_dims=[256, 128, 64],
            output_dim=3,
        ).to(DEVICE)

        # ========== Reinforcement Learning ==========
        self.rl_agent = UnifiedRLAgent(
            state_dim=feature_dim,
            action_dim=3,
            algorithms=["ppo", "sac", "c51", "qrdqn"],
        )

        # ========== World Model ==========
        self.world_model = DreamerV3WorldModel(
            obs_dim=feature_dim,
            action_dim=3,
            embed_dim=256,
            deter_dim=512,
        ).to(DEVICE)
        self.world_trainer = WorldModelTrainer(
            self.world_model,
            buffer_size=100000,
            batch_size=16,
            seq_len=50,
        )

        # ========== Vector Database ==========
        self.pattern_db = VectorPatternDB(
            dim=feature_dim,
            max_patterns=100000,
        )
        self.temporal_matcher = TemporalPatternMatcher(
            window_size=60,
            stride=10,
        )

        # ========== Multi-Agent System ==========
        self.multi_agent = MultiAgentTradingSystem(
            state_dim=feature_dim,
            use_neural_agents=True,
        )

        # ========== Confidence Gate ==========
        self.confidence_gate = ConfidenceGate(
            base_confidence=0.60,
            min_confidence=0.50,
            max_confidence=0.85,
        )

        # ========== State Management ==========
        self.trading_state = TradingState()
        self.feature_buffer = deque(maxlen=seq_len)
        self.price_history = deque(maxlen=1000)
        self.world_model_state = None

        # ========== Component Weights ==========
        self.component_weights = {
            'sota_ensemble': 0.25,
            'bayesian_nn': 0.15,
            'rl_agent': 0.20,
            'world_model': 0.15,
            'pattern_db': 0.10,
            'multi_agent': 0.15,
        }

        # ========== Threading ==========
        self.executor = ThreadPoolExecutor(max_workers=8)

        # ========== Training State ==========
        self.is_training = False
        self.training_step = 0

        logger.info("Ultimate Trading Engine initialized successfully!")
        logger.info(f"  - Feature dimension: {feature_dim}")
        logger.info(f"  - Sequence length: {seq_len}")
        logger.info(f"  - Device: {DEVICE}")

    def update_features(
        self,
        price: float,
        volume: float,
        high: float = None,
        low: float = None,
        order_book: Dict = None,
        external_features: Dict = None,
    ) -> np.ndarray:
        """特徴量を更新"""
        self.price_history.append(price)

        self.feature_generator.update(
            price=price,
            volume=volume,
            high=high or price,
            low=low or price,
        )

        features = self.feature_generator.generate(
            order_book=order_book,
            external_features=external_features,
        )

        self.feature_buffer.append(features)

        return features

    def _get_feature_sequence(self) -> torch.Tensor:
        """特徴量シーケンスを取得"""
        if len(self.feature_buffer) < self.seq_len:
            # Pad with zeros
            padding = [np.zeros(self.feature_dim)] * (self.seq_len - len(self.feature_buffer))
            sequence = list(padding) + list(self.feature_buffer)
        else:
            sequence = list(self.feature_buffer)[-self.seq_len:]

        return torch.FloatTensor(np.array(sequence)).unsqueeze(0).to(DEVICE)

    def _get_sota_prediction(self) -> Tuple[int, float, Dict]:
        """SOTAモデルの予測"""
        sequence = self._get_feature_sequence()

        with torch.no_grad():
            probs, model_outputs = self.sota_ensemble(sequence)
            action = probs.argmax(dim=-1).item()
            confidence = probs[0, action].item()

        return action, confidence, {
            'probs': probs[0].cpu().numpy(),
            'weights': model_outputs['weights'].cpu().numpy(),
        }

    def _get_bayesian_prediction(self) -> Tuple[int, float, float, Dict]:
        """ベイジアンNNの予測"""
        features = np.array(self.feature_buffer)[-1] if self.feature_buffer else np.zeros(self.feature_dim)
        state_tensor = torch.FloatTensor(features).unsqueeze(0).to(DEVICE)

        action, confidence, uncertainty = self.bayesian_nn.predict_with_uncertainty(
            state_tensor, num_samples=30
        )

        return action, confidence, uncertainty, {
            'epistemic_uncertainty': uncertainty,
        }

    def _get_rl_prediction(self) -> Tuple[int, Dict]:
        """強化学習の予測"""
        features = np.array(self.feature_buffer)[-1] if self.feature_buffer else np.zeros(self.feature_dim)
        action, info = self.rl_agent.select_action(features, eval_mode=True)
        return action, info

    def _get_world_model_prediction(self) -> Tuple[int, Dict]:
        """World Modelの予測"""
        if self.world_model_state is None:
            self.world_model_state = self.world_model.rssm.initial_state(1)

        action, info = self.world_model.select_action(self.world_model_state, eval_mode=True)

        # Simulate future
        future = self.world_model.simulate_future(
            self.world_model_state,
            [action] * 5,  # 5 steps ahead
        )

        info['future_predictions'] = future

        return action, info

    def _get_pattern_prediction(self) -> Tuple[int, float, Dict]:
        """パターンマッチングの予測"""
        if not self.feature_buffer:
            return 1, 0.0, {}

        features = np.array(self.feature_buffer)[-1]
        action, confidence, details = self.pattern_db.predict_from_patterns(
            features, k=20, min_similarity=0.6
        )

        return action, confidence, details

    def _get_multi_agent_prediction(self) -> Tuple[int, float, Dict]:
        """マルチエージェントの予測"""
        features = np.array(self.feature_buffer)[-1] if self.feature_buffer else np.zeros(self.feature_dim)

        # Build context
        context = {
            'price': self.price_history[-1] if self.price_history else 0,
            'prev_price': self.price_history[-2] if len(self.price_history) > 1 else 0,
            'volatility': np.std(list(self.price_history)[-20:]) if len(self.price_history) >= 20 else 0,
            'volume_ratio': 1.0,
            'equity': self.trading_state.equity,
            'position': self.trading_state.position,
        }

        action, confidence, details = self.multi_agent.decide(features, context)

        return action, confidence, details

    def generate_signal(
        self,
        price: float,
        volume: float,
        high: float = None,
        low: float = None,
        order_book: Dict = None,
        external_features: Dict = None,
    ) -> TradingSignal:
        """
        統合シグナルを生成

        全てのコンポーネントからの予測を統合
        """
        # Update features
        features = self.update_features(
            price, volume, high, low, order_book, external_features
        )

        if len(self.feature_buffer) < 10:
            return TradingSignal(
                action=1,
                confidence=0.0,
                position_size=0.0,
                stop_loss=0.0,
                take_profit=0.0,
                urgency=0.0,
                reasoning="Insufficient data",
            )

        # ========== Parallel Predictions ==========
        futures = {}

        futures['sota'] = self.executor.submit(self._get_sota_prediction)
        futures['bayesian'] = self.executor.submit(self._get_bayesian_prediction)
        futures['rl'] = self.executor.submit(self._get_rl_prediction)
        futures['world_model'] = self.executor.submit(self._get_world_model_prediction)
        futures['pattern'] = self.executor.submit(self._get_pattern_prediction)
        futures['multi_agent'] = self.executor.submit(self._get_multi_agent_prediction)

        # Collect results
        results = {}
        for name, future in futures.items():
            try:
                results[name] = future.result(timeout=1.0)
            except Exception as e:
                logger.warning(f"Component {name} failed: {e}")
                results[name] = (1, 0.0, {})

        # ========== Extract Predictions ==========
        predictions = {}

        # SOTA Ensemble
        sota_action, sota_conf, sota_info = results['sota']
        predictions['sota_ensemble'] = (sota_action, sota_conf)

        # Bayesian NN
        bnn_action, bnn_conf, bnn_unc, bnn_info = results['bayesian']
        # Reduce confidence if uncertainty is high
        bnn_conf_adjusted = bnn_conf * (1 - min(0.5, bnn_unc))
        predictions['bayesian_nn'] = (bnn_action, bnn_conf_adjusted)

        # RL Agent
        rl_action, rl_info = results['rl']
        rl_conf = rl_info.get('votes', {}).get(rl_action, 0.5)
        predictions['rl_agent'] = (rl_action, rl_conf)

        # World Model
        wm_action, wm_info = results['world_model']
        wm_conf = min(0.8, abs(wm_info.get('predicted_value', 0)) * 0.5 + 0.3)
        predictions['world_model'] = (wm_action, wm_conf)

        # Pattern DB
        pat_action, pat_conf, pat_info = results['pattern']
        predictions['pattern_db'] = (pat_action, pat_conf)

        # Multi-Agent
        ma_action, ma_conf, ma_info = results['multi_agent']
        predictions['multi_agent'] = (ma_action, ma_conf)

        # ========== Weighted Voting ==========
        votes = {0: 0.0, 1: 0.0, 2: 0.0}

        for component, (action, confidence) in predictions.items():
            weight = self.component_weights.get(component, 0.1)
            votes[action] += weight * confidence

        # Normalize
        total = sum(votes.values())
        if total > 0:
            for k in votes:
                votes[k] /= total

        final_action = max(votes, key=votes.get)
        final_confidence = votes[final_action]

        # ========== Consensus Analysis ==========
        action_counts = {0: 0, 1: 0, 2: 0}
        for _, (action, _) in predictions.items():
            action_counts[action] += 1

        consensus = action_counts[final_action] / len(predictions)

        # ========== Confidence Gate ==========
        volatility = np.std(list(self.price_history)[-20:]) / price if len(self.price_history) >= 20 else 0
        threshold = self.confidence_gate.update(
            volatility=volatility,
            consensus=consensus,
        )

        passes_gate = self.confidence_gate.check(final_confidence)

        if not passes_gate:
            final_action = 1  # Hold if not confident enough
            final_confidence *= 0.5

        # ========== Position Sizing ==========
        base_size = 1.0

        # Reduce size if uncertain
        if bnn_unc > 0.3:
            base_size *= 0.5

        # Reduce size if low consensus
        if consensus < 0.5:
            base_size *= 0.7

        # Reduce size if high volatility
        if volatility > 0.02:
            base_size *= 0.6

        position_size = base_size * final_confidence

        # ========== Stop Loss / Take Profit ==========
        atr = volatility * price * 14 if volatility > 0 else price * 0.01

        if final_action == 2:  # Buy
            stop_loss = price - 2 * atr
            take_profit = price + 3 * atr
        elif final_action == 0:  # Sell
            stop_loss = price + 2 * atr
            take_profit = price - 3 * atr
        else:
            stop_loss = 0
            take_profit = 0

        # ========== Urgency ==========
        urgency = 0.5

        # Increase urgency for momentum
        if results['multi_agent'][2].get('agent_decisions', {}).get('MOMENTUM', None):
            mom_decision = results['multi_agent'][2]['agent_decisions']['MOMENTUM']
            if hasattr(mom_decision, 'urgency'):
                urgency = max(urgency, mom_decision.urgency)

        # Increase urgency for strong signals
        if final_confidence > 0.8 and consensus > 0.7:
            urgency = 0.9

        # ========== Reasoning ==========
        action_names = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
        reasoning_parts = [
            f"Action: {action_names[final_action]} (conf: {final_confidence:.2%})",
            f"Consensus: {consensus:.0%} ({action_counts})",
            f"Threshold: {threshold:.2%} (passes: {passes_gate})",
        ]

        # Add component votes
        component_votes = []
        for comp, (act, conf) in predictions.items():
            component_votes.append(f"{comp}:{action_names[act]}({conf:.0%})")
        reasoning_parts.append("Votes: " + ", ".join(component_votes))

        reasoning = " | ".join(reasoning_parts)

        # ========== Create Signal ==========
        signal = TradingSignal(
            action=final_action,
            confidence=final_confidence,
            position_size=position_size,
            stop_loss=stop_loss,
            take_profit=take_profit,
            urgency=urgency,
            reasoning=reasoning,
            components={
                'sota_ensemble': sota_info,
                'bayesian_nn': {'uncertainty': bnn_unc, **bnn_info},
                'rl_agent': rl_info,
                'world_model': wm_info,
                'pattern_db': pat_info,
                'multi_agent': ma_info,
                'predictions': predictions,
                'votes': votes,
                'consensus': consensus,
                'threshold': threshold,
            }
        )

        return signal

    def update_from_trade(
        self,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool = False,
    ) -> None:
        """取引結果から学習"""
        if not self.feature_buffer:
            return

        state = np.array(self.feature_buffer)[-1]

        # Update RL agents
        self.rl_agent.store_transition(state, action, reward, next_state, done)

        # Update World Model
        self.world_trainer.step(state, action, reward, done)

        # Update Pattern DB
        outcome = 2 if reward > 0 else (0 if reward < 0 else 1)
        self.pattern_db.add_pattern(
            pattern=state,
            outcome=outcome,
            profit=reward,
            confidence=0.8,
        )

        # Update Multi-Agent
        self.multi_agent.update_from_reward(reward)

        # Update Confidence Gate
        self.confidence_gate.update(win=(reward > 0))

        # Update trading state
        self.trading_state.trade_count += 1
        if reward > 0:
            self.trading_state.win_count += 1

        self.training_step += 1

    def train_step(self) -> Dict[str, float]:
        """学習ステップ"""
        metrics = {}

        # Train RL
        if self.feature_buffer:
            next_state = np.array(self.feature_buffer)[-1]
            rl_metrics = self.rl_agent.update(next_state)
            metrics['rl'] = rl_metrics

        # Train World Model
        wm_metrics = self.world_trainer.train()
        metrics['world_model'] = wm_metrics

        # Train Pattern DB indices
        if self.training_step % 100 == 0 and len(self.pattern_db.patterns) > 100:
            self.pattern_db.train_indices()

        return metrics

    def get_diagnostics(self) -> Dict[str, Any]:
        """診断情報を取得"""
        return {
            'trading_state': {
                'position': self.trading_state.position,
                'equity': self.trading_state.equity,
                'drawdown': self.trading_state.drawdown,
                'trade_count': self.trading_state.trade_count,
                'win_rate': self.trading_state.win_count / max(1, self.trading_state.trade_count),
            },
            'confidence_gate': {
                'current_threshold': self.confidence_gate.current_threshold,
            },
            'pattern_db': self.pattern_db.get_pattern_statistics(),
            'multi_agent': self.multi_agent.get_agent_stats(),
            'component_weights': self.component_weights,
            'training_step': self.training_step,
            'device': str(DEVICE),
        }

    def save(self, path: str) -> None:
        """モデルを保存"""
        torch.save({
            'sota_ensemble': self.sota_ensemble.state_dict(),
            'bayesian_nn': self.bayesian_nn.state_dict(),
            'world_model': self.world_model.state_dict(),
            'component_weights': self.component_weights,
            'training_step': self.training_step,
        }, path)

        # Save pattern DB
        self.pattern_db.save(path.replace('.pt', '_patterns.pkl'))

        logger.info(f"Model saved to {path}")

    def load(self, path: str) -> None:
        """モデルを読み込み"""
        checkpoint = torch.load(path, map_location=DEVICE)

        self.sota_ensemble.load_state_dict(checkpoint['sota_ensemble'])
        self.bayesian_nn.load_state_dict(checkpoint['bayesian_nn'])
        self.world_model.load_state_dict(checkpoint['world_model'])
        self.component_weights = checkpoint['component_weights']
        self.training_step = checkpoint['training_step']

        # Load pattern DB
        pattern_path = path.replace('.pt', '_patterns.pkl')
        self.pattern_db.load(pattern_path)

        logger.info(f"Model loaded from {path}")


# =============================================================================
# Factory Function
# =============================================================================

def create_ultimate_trading_engine(
    feature_dim: int = 500,
    seq_len: int = 60,
) -> UltimateTradingEngine:
    """Ultimate Trading Engineを生成"""
    return UltimateTradingEngine(
        feature_dim=feature_dim,
        seq_len=seq_len,
        enable_all_components=True,
    )


logger.info("Ultimate Trading Engine module loaded")
logger.info("=" * 60)
logger.info("  WORLD'S STRONGEST AI TRADING SYSTEM")
logger.info("  世界最強AIトレーディングシステム")
logger.info("=" * 60)
logger.info("Components:")
logger.info("  [x] SOTA Deep Learning (PatchTST, Mamba, iTransformer)")
logger.info("  [x] Bayesian Neural Network (Uncertainty Estimation)")
logger.info("  [x] Advanced RL (PPO, SAC, C51, QR-DQN)")
logger.info("  [x] World Model (DreamerV3)")
logger.info("  [x] Vector Database (FAISS-style Pattern Matching)")
logger.info("  [x] Multi-Agent System (Role-based AI Group)")
logger.info("  [x] 500+ Dimension Feature Engineering")
logger.info("  [x] Dynamic Confidence Gate")
logger.info("=" * 60)

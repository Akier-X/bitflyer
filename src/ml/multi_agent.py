"""
Multi-Agent Trading System
===========================
役割分担型AI群による協調的意思決定
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import deque
from abc import ABC, abstractmethod
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from loguru import logger


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AgentRole(Enum):
    """エージェントの役割"""
    TREND_FOLLOWER = auto()      # トレンドフォロー
    MEAN_REVERTER = auto()       # 平均回帰
    MOMENTUM = auto()            # モメンタム
    SCALPER = auto()             # スキャルピング
    RISK_MANAGER = auto()        # リスク管理
    SENTIMENT_ANALYZER = auto()  # センチメント分析
    PATTERN_MATCHER = auto()     # パターンマッチング
    ARBITRAGE = auto()           # 裁定取引
    MARKET_MAKER = auto()        # マーケットメイク
    META_CONTROLLER = auto()     # メタ制御


@dataclass
class AgentDecision:
    """エージェントの決定"""
    action: int  # 0: sell, 1: hold, 2: buy
    confidence: float
    reasoning: str
    position_size: float = 1.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    urgency: float = 0.5
    metadata: Dict = field(default_factory=dict)


class BaseAgent(ABC):
    """エージェント基底クラス"""

    def __init__(
        self,
        role: AgentRole,
        state_dim: int = 500,
    ):
        self.role = role
        self.state_dim = state_dim
        self.performance_history: deque = deque(maxlen=100)
        self.trade_history: deque = deque(maxlen=1000)
        self.weight = 1.0

    @abstractmethod
    def decide(self, state: np.ndarray, context: Dict) -> AgentDecision:
        """意思決定"""
        pass

    def update_performance(self, reward: float) -> None:
        """パフォーマンス更新"""
        self.performance_history.append(reward)

    def get_performance(self) -> float:
        """平均パフォーマンス"""
        if not self.performance_history:
            return 0.0
        return np.mean(self.performance_history)

    def update_weight(self, new_weight: float) -> None:
        """重み更新"""
        self.weight = max(0.1, min(3.0, new_weight))


# =============================================================================
# Specialized Agents
# =============================================================================

class TrendFollowerAgent(BaseAgent):
    """トレンドフォローエージェント"""

    def __init__(self, state_dim: int = 500):
        super().__init__(AgentRole.TREND_FOLLOWER, state_dim)
        self.ema_short = deque(maxlen=20)
        self.ema_long = deque(maxlen=50)

    def decide(self, state: np.ndarray, context: Dict) -> AgentDecision:
        # Extract trend features from state
        price = context.get('price', 0)
        self.ema_short.append(price)
        self.ema_long.append(price)

        if len(self.ema_short) < 10:
            return AgentDecision(1, 0.3, "Insufficient data")

        ema_s = np.mean(self.ema_short)
        ema_l = np.mean(self.ema_long)

        trend_strength = (ema_s - ema_l) / (ema_l + 1e-10)

        if trend_strength > 0.002:
            action = 2  # Buy
            confidence = min(0.9, abs(trend_strength) * 50)
            reasoning = f"Uptrend detected (EMA cross: {trend_strength:.4f})"
        elif trend_strength < -0.002:
            action = 0  # Sell
            confidence = min(0.9, abs(trend_strength) * 50)
            reasoning = f"Downtrend detected (EMA cross: {trend_strength:.4f})"
        else:
            action = 1  # Hold
            confidence = 0.5
            reasoning = "No clear trend"

        return AgentDecision(
            action=action,
            confidence=confidence,
            reasoning=reasoning,
            position_size=min(1.0, confidence),
            metadata={'trend_strength': trend_strength}
        )


class MeanReverterAgent(BaseAgent):
    """平均回帰エージェント"""

    def __init__(self, state_dim: int = 500):
        super().__init__(AgentRole.MEAN_REVERTER, state_dim)
        self.price_history = deque(maxlen=100)

    def decide(self, state: np.ndarray, context: Dict) -> AgentDecision:
        price = context.get('price', 0)
        self.price_history.append(price)

        if len(self.price_history) < 30:
            return AgentDecision(1, 0.3, "Insufficient data")

        mean = np.mean(self.price_history)
        std = np.std(self.price_history)
        z_score = (price - mean) / (std + 1e-10)

        if z_score > 2.0:
            action = 0  # Sell (price too high)
            confidence = min(0.9, abs(z_score) / 4)
            reasoning = f"Overbought (z-score: {z_score:.2f})"
        elif z_score < -2.0:
            action = 2  # Buy (price too low)
            confidence = min(0.9, abs(z_score) / 4)
            reasoning = f"Oversold (z-score: {z_score:.2f})"
        else:
            action = 1
            confidence = 0.4
            reasoning = f"Within normal range (z-score: {z_score:.2f})"

        return AgentDecision(
            action=action,
            confidence=confidence,
            reasoning=reasoning,
            position_size=min(1.0, abs(z_score) / 3),
            metadata={'z_score': z_score}
        )


class MomentumAgent(BaseAgent):
    """モメンタムエージェント"""

    def __init__(self, state_dim: int = 500):
        super().__init__(AgentRole.MOMENTUM, state_dim)
        self.returns = deque(maxlen=20)

    def decide(self, state: np.ndarray, context: Dict) -> AgentDecision:
        price = context.get('price', 0)
        prev_price = context.get('prev_price', price)

        if prev_price > 0:
            ret = (price - prev_price) / prev_price
            self.returns.append(ret)

        if len(self.returns) < 5:
            return AgentDecision(1, 0.3, "Insufficient data")

        momentum = sum(self.returns)
        momentum_strength = abs(momentum)

        if momentum > 0.005:
            action = 2
            confidence = min(0.85, momentum_strength * 30)
            reasoning = f"Positive momentum ({momentum:.4f})"
        elif momentum < -0.005:
            action = 0
            confidence = min(0.85, momentum_strength * 30)
            reasoning = f"Negative momentum ({momentum:.4f})"
        else:
            action = 1
            confidence = 0.4
            reasoning = "Weak momentum"

        return AgentDecision(
            action=action,
            confidence=confidence,
            reasoning=reasoning,
            urgency=min(1.0, momentum_strength * 50),
            metadata={'momentum': momentum}
        )


class ScalperAgent(BaseAgent):
    """スキャルピングエージェント"""

    def __init__(self, state_dim: int = 500):
        super().__init__(AgentRole.SCALPER, state_dim)
        self.tick_history = deque(maxlen=50)
        self.last_trade_time = 0

    def decide(self, state: np.ndarray, context: Dict) -> AgentDecision:
        price = context.get('price', 0)
        spread = context.get('spread', 0)
        timestamp = context.get('timestamp', 0)

        self.tick_history.append(price)

        if len(self.tick_history) < 10:
            return AgentDecision(1, 0.3, "Insufficient data")

        # Micro trend
        recent_5 = list(self.tick_history)[-5:]
        micro_trend = (recent_5[-1] - recent_5[0]) / (recent_5[0] + 1e-10)

        # Volatility
        volatility = np.std(self.tick_history)

        # Spread check
        spread_ok = spread < 0.001

        if micro_trend > 0.001 and spread_ok:
            action = 2
            confidence = 0.7
            reasoning = "Quick upward move"
            urgency = 0.9
        elif micro_trend < -0.001 and spread_ok:
            action = 0
            confidence = 0.7
            reasoning = "Quick downward move"
            urgency = 0.9
        else:
            action = 1
            confidence = 0.3
            reasoning = "No scalp opportunity"
            urgency = 0.2

        return AgentDecision(
            action=action,
            confidence=confidence,
            reasoning=reasoning,
            position_size=0.3,  # Small positions
            urgency=urgency,
            take_profit=0.001,
            stop_loss=0.0005,
            metadata={'micro_trend': micro_trend, 'volatility': volatility}
        )


class RiskManagerAgent(BaseAgent):
    """リスク管理エージェント"""

    def __init__(self, state_dim: int = 500):
        super().__init__(AgentRole.RISK_MANAGER, state_dim)
        self.volatility_history = deque(maxlen=100)
        self.drawdown = 0.0
        self.max_equity = 0.0

    def decide(self, state: np.ndarray, context: Dict) -> AgentDecision:
        volatility = context.get('volatility', 0)
        equity = context.get('equity', 1.0)
        position = context.get('position', 0)

        self.volatility_history.append(volatility)

        # Update drawdown
        if equity > self.max_equity:
            self.max_equity = equity
        self.drawdown = (self.max_equity - equity) / self.max_equity

        # Risk assessment
        avg_vol = np.mean(self.volatility_history) if self.volatility_history else 0
        vol_spike = volatility > avg_vol * 2

        risk_level = 0.0

        if self.drawdown > 0.1:
            risk_level += 0.5
        if vol_spike:
            risk_level += 0.3
        if abs(position) > 0.8:
            risk_level += 0.2

        # Risk management decisions
        if risk_level > 0.7:
            action = 1  # Force hold / reduce
            confidence = 0.9
            reasoning = f"HIGH RISK: DD={self.drawdown:.2%}, Vol spike={vol_spike}"
            position_size = 0.3
        elif risk_level > 0.4:
            action = 1
            confidence = 0.7
            reasoning = f"Medium risk: DD={self.drawdown:.2%}"
            position_size = 0.6
        else:
            action = 1  # Let others decide
            confidence = 0.3
            reasoning = "Risk within limits"
            position_size = 1.0

        return AgentDecision(
            action=action,
            confidence=confidence,
            reasoning=reasoning,
            position_size=position_size,
            metadata={
                'risk_level': risk_level,
                'drawdown': self.drawdown,
                'vol_spike': vol_spike
            }
        )


class SentimentAgent(BaseAgent):
    """センチメント分析エージェント"""

    def __init__(self, state_dim: int = 500):
        super().__init__(AgentRole.SENTIMENT_ANALYZER, state_dim)
        self.fear_greed_index = 50.0

    def decide(self, state: np.ndarray, context: Dict) -> AgentDecision:
        # Extract sentiment from state features
        volume_ratio = context.get('volume_ratio', 1.0)
        price_change = context.get('price_change', 0)
        volatility = context.get('volatility', 0)

        # Simple fear/greed calculation
        greed = 0.0
        fear = 0.0

        if price_change > 0.01:
            greed += 20
        elif price_change < -0.01:
            fear += 20

        if volume_ratio > 1.5:
            greed += 15
        elif volume_ratio < 0.5:
            fear += 10

        if volatility > 0.02:
            fear += 25

        self.fear_greed_index = 50 + greed - fear
        self.fear_greed_index = max(0, min(100, self.fear_greed_index))

        if self.fear_greed_index > 75:
            action = 0  # Extreme greed -> Sell
            confidence = 0.6
            reasoning = f"Extreme greed ({self.fear_greed_index:.0f})"
        elif self.fear_greed_index < 25:
            action = 2  # Extreme fear -> Buy
            confidence = 0.6
            reasoning = f"Extreme fear ({self.fear_greed_index:.0f})"
        else:
            action = 1
            confidence = 0.4
            reasoning = f"Neutral sentiment ({self.fear_greed_index:.0f})"

        return AgentDecision(
            action=action,
            confidence=confidence,
            reasoning=reasoning,
            metadata={'fear_greed_index': self.fear_greed_index}
        )


class PatternMatcherAgent(BaseAgent):
    """パターンマッチングエージェント"""

    def __init__(self, state_dim: int = 500):
        super().__init__(AgentRole.PATTERN_MATCHER, state_dim)
        self.patterns_db = []

    def decide(self, state: np.ndarray, context: Dict) -> AgentDecision:
        # Use vector similarity from context or compute
        similar_patterns = context.get('similar_patterns', [])

        if not similar_patterns:
            return AgentDecision(1, 0.3, "No matching patterns")

        # Vote based on historical outcomes
        votes = {0: 0.0, 1: 0.0, 2: 0.0}

        for pattern, similarity, outcome in similar_patterns:
            votes[outcome] += similarity

        total = sum(votes.values())
        if total > 0:
            for k in votes:
                votes[k] /= total

        action = max(votes, key=votes.get)
        confidence = votes[action]

        return AgentDecision(
            action=action,
            confidence=confidence,
            reasoning=f"Pattern match: {len(similar_patterns)} similar patterns",
            metadata={'pattern_votes': votes}
        )


class NeuralAgent(BaseAgent):
    """ニューラルネットワークベースのエージェント"""

    def __init__(
        self,
        role: AgentRole,
        state_dim: int = 500,
        hidden_dims: List[int] = [256, 128],
    ):
        super().__init__(role, state_dim)

        # Build network
        layers = []
        dims = [state_dim] + hidden_dims

        for i in range(len(dims) - 1):
            layers.extend([
                nn.Linear(dims[i], dims[i + 1]),
                nn.LayerNorm(dims[i + 1]),
                nn.ReLU(),
                nn.Dropout(0.1),
            ])

        layers.append(nn.Linear(hidden_dims[-1], 3))

        self.network = nn.Sequential(*layers).to(DEVICE)

    def decide(self, state: np.ndarray, context: Dict) -> AgentDecision:
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = self.network(state_tensor)
            probs = F.softmax(logits, dim=-1)
            action = probs.argmax(dim=-1).item()
            confidence = probs[0, action].item()

        return AgentDecision(
            action=action,
            confidence=confidence,
            reasoning=f"Neural prediction (probs: {probs[0].cpu().numpy()})",
            metadata={'probabilities': probs[0].cpu().numpy()}
        )


# =============================================================================
# Meta Controller
# =============================================================================

class MetaController(nn.Module):
    """
    メタコントローラー

    全エージェントの決定を統合し、最終判断を行う
    """

    def __init__(
        self,
        num_agents: int,
        hidden_dim: int = 128,
    ):
        super().__init__()

        # Agent decision encoder
        self.decision_encoder = nn.Sequential(
            nn.Linear(num_agents * 4, hidden_dim),  # action, confidence, urgency, position_size
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Weight predictor
        self.weight_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_agents),
            nn.Softmax(dim=-1),
        )

        # Action predictor
        self.action_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3),
        )

    def forward(
        self,
        decisions: List[AgentDecision],
    ) -> Tuple[int, float, Dict]:
        """
        統合決定

        Returns:
            action: 最終アクション
            confidence: 信頼度
            info: 詳細情報
        """
        # Encode decisions
        decision_features = []
        for d in decisions:
            features = [
                d.action / 2.0,  # Normalize to [0, 1]
                d.confidence,
                d.urgency,
                d.position_size,
            ]
            decision_features.extend(features)

        x = torch.FloatTensor(decision_features).unsqueeze(0).to(DEVICE)

        encoded = self.decision_encoder(x)
        agent_weights = self.weight_predictor(encoded)[0]
        action_logits = self.action_predictor(encoded)[0]

        action_probs = F.softmax(action_logits, dim=-1)
        final_action = action_probs.argmax().item()
        confidence = action_probs[final_action].item()

        return final_action, confidence, {
            'agent_weights': agent_weights.cpu().detach().numpy(),
            'action_probs': action_probs.cpu().detach().numpy(),
        }


# =============================================================================
# Multi-Agent System
# =============================================================================

class MultiAgentTradingSystem:
    """
    マルチエージェントトレーディングシステム

    複数の専門エージェントが協調して意思決定
    """

    def __init__(
        self,
        state_dim: int = 500,
        use_neural_agents: bool = True,
    ):
        self.state_dim = state_dim

        # Create specialized agents
        self.agents: Dict[AgentRole, BaseAgent] = {
            AgentRole.TREND_FOLLOWER: TrendFollowerAgent(state_dim),
            AgentRole.MEAN_REVERTER: MeanReverterAgent(state_dim),
            AgentRole.MOMENTUM: MomentumAgent(state_dim),
            AgentRole.SCALPER: ScalperAgent(state_dim),
            AgentRole.RISK_MANAGER: RiskManagerAgent(state_dim),
            AgentRole.SENTIMENT_ANALYZER: SentimentAgent(state_dim),
            AgentRole.PATTERN_MATCHER: PatternMatcherAgent(state_dim),
        }

        if use_neural_agents:
            # Add neural agents for each role
            self.agents[AgentRole.META_CONTROLLER] = NeuralAgent(
                AgentRole.META_CONTROLLER, state_dim
            )

        # Meta controller
        self.meta_controller = MetaController(len(self.agents)).to(DEVICE)
        self.meta_optimizer = torch.optim.Adam(self.meta_controller.parameters(), lr=1e-4)

        # Decision history
        self.decision_history: deque = deque(maxlen=1000)

        # Thread pool for parallel execution
        self.executor = ThreadPoolExecutor(max_workers=len(self.agents))

        logger.info(f"MultiAgentTradingSystem initialized with {len(self.agents)} agents")

    def _get_agent_decision(
        self,
        agent: BaseAgent,
        state: np.ndarray,
        context: Dict,
    ) -> Tuple[AgentRole, AgentDecision]:
        """エージェントの決定を取得"""
        try:
            decision = agent.decide(state, context)
            return agent.role, decision
        except Exception as e:
            logger.error(f"Agent {agent.role} error: {e}")
            return agent.role, AgentDecision(1, 0.0, f"Error: {e}")

    def decide(
        self,
        state: np.ndarray,
        context: Dict,
        parallel: bool = True,
    ) -> Tuple[int, float, Dict]:
        """
        統合意思決定

        Args:
            state: 状態ベクトル
            context: コンテキスト情報
            parallel: 並列実行するか

        Returns:
            action: 最終アクション
            confidence: 信頼度
            details: 詳細情報
        """
        decisions = {}

        if parallel:
            # Parallel execution
            futures = {
                self.executor.submit(
                    self._get_agent_decision, agent, state, context
                ): role
                for role, agent in self.agents.items()
            }

            for future in as_completed(futures):
                role, decision = future.result()
                decisions[role] = decision
        else:
            # Sequential execution
            for role, agent in self.agents.items():
                _, decision = self._get_agent_decision(agent, state, context)
                decisions[role] = decision

        # Risk manager has veto power
        risk_decision = decisions.get(AgentRole.RISK_MANAGER)
        if risk_decision and risk_decision.metadata.get('risk_level', 0) > 0.7:
            # High risk - force position reduction
            return 1, 0.9, {
                'decisions': decisions,
                'veto_reason': 'High risk detected',
                'risk_level': risk_decision.metadata.get('risk_level'),
            }

        # Collect decisions for meta controller
        decision_list = [decisions[role] for role in sorted(decisions.keys(), key=lambda x: x.value)]

        # Meta controller decision
        final_action, confidence, meta_info = self.meta_controller(decision_list)

        # Weighted voting as backup
        votes = {0: 0.0, 1: 0.0, 2: 0.0}
        for role, decision in decisions.items():
            weight = self.agents[role].weight * decision.confidence
            votes[decision.action] += weight

        total = sum(votes.values())
        if total > 0:
            for k in votes:
                votes[k] /= total

        vote_action = max(votes, key=votes.get)
        vote_confidence = votes[vote_action]

        # Combine meta controller and voting
        if confidence > 0.6:
            action = final_action
            final_confidence = confidence
        else:
            action = vote_action
            final_confidence = vote_confidence

        # Store decision
        self.decision_history.append({
            'state': state,
            'context': context,
            'decisions': decisions,
            'final_action': action,
            'confidence': final_confidence,
        })

        return action, final_confidence, {
            'agent_decisions': {role.name: d for role, d in decisions.items()},
            'votes': votes,
            'meta_weights': meta_info['agent_weights'],
            'action_probs': meta_info['action_probs'],
        }

    def update_from_reward(self, reward: float) -> None:
        """報酬に基づいて学習"""
        if not self.decision_history:
            return

        last_decision = self.decision_history[-1]

        # Update individual agents
        for role, decision in last_decision['decisions'].items():
            agent = self.agents[role]

            # Performance update
            if decision.action == last_decision['final_action']:
                # This agent's vote was used
                agent.update_performance(reward * decision.confidence)
            else:
                # Different vote
                if reward > 0:
                    # Final decision was good, this agent was wrong
                    agent.update_performance(-abs(reward) * 0.5)
                else:
                    # Final decision was bad, maybe this agent was right
                    agent.update_performance(abs(reward) * 0.3)

            # Update weight based on recent performance
            perf = agent.get_performance()
            new_weight = 1.0 + perf * 0.5
            agent.update_weight(new_weight)

        # Update meta controller
        self._train_meta_controller(reward)

    def _train_meta_controller(self, reward: float) -> None:
        """メタコントローラーを学習"""
        if len(self.decision_history) < 10:
            return

        # Use recent decisions for training
        recent = list(self.decision_history)[-10:]

        for entry in recent:
            decisions = [entry['decisions'][role]
                        for role in sorted(entry['decisions'].keys(), key=lambda x: x.value)]

            decision_features = []
            for d in decisions:
                features = [d.action / 2.0, d.confidence, d.urgency, d.position_size]
                decision_features.extend(features)

            x = torch.FloatTensor(decision_features).unsqueeze(0).to(DEVICE)

            encoded = self.meta_controller.decision_encoder(x)
            action_logits = self.meta_controller.action_predictor(encoded)

            # Create target based on reward
            target = torch.zeros(3, device=DEVICE)
            if reward > 0:
                target[entry['final_action']] = 1.0
            else:
                # Wrong action, discourage it
                target = torch.ones(3, device=DEVICE) / 3
                target[entry['final_action']] = 0.0
                target /= target.sum()

            loss = F.cross_entropy(action_logits, target.unsqueeze(0))

            self.meta_optimizer.zero_grad()
            loss.backward()
            self.meta_optimizer.step()

    def get_agent_stats(self) -> Dict[str, Dict]:
        """エージェント統計"""
        stats = {}
        for role, agent in self.agents.items():
            stats[role.name] = {
                'weight': agent.weight,
                'performance': agent.get_performance(),
                'trade_count': len(agent.trade_history),
            }
        return stats

    def get_consensus(self, state: np.ndarray, context: Dict) -> Dict:
        """
        エージェント間のコンセンサス分析

        Returns:
            consensus_info: コンセンサス情報
        """
        decisions = {}
        for role, agent in self.agents.items():
            _, decision = self._get_agent_decision(agent, state, context)
            decisions[role] = decision

        # Analyze consensus
        action_counts = {0: 0, 1: 0, 2: 0}
        confidence_sum = {0: 0.0, 1: 0.0, 2: 0.0}

        for decision in decisions.values():
            action_counts[decision.action] += 1
            confidence_sum[decision.action] += decision.confidence

        total_agents = len(decisions)
        consensus_action = max(action_counts, key=action_counts.get)
        consensus_strength = action_counts[consensus_action] / total_agents
        avg_confidence = confidence_sum[consensus_action] / max(1, action_counts[consensus_action])

        # Disagreement analysis
        disagreement = 1 - consensus_strength

        return {
            'consensus_action': consensus_action,
            'consensus_strength': consensus_strength,
            'avg_confidence': avg_confidence,
            'disagreement': disagreement,
            'action_distribution': {k: v / total_agents for k, v in action_counts.items()},
            'individual_decisions': {role.name: d.action for role, d in decisions.items()},
        }


class HierarchicalMultiAgentSystem:
    """
    階層型マルチエージェントシステム

    複数のMultiAgentTradingSystemを階層的に組織
    """

    def __init__(self, state_dim: int = 500):
        # Level 1: Specialized groups
        self.technical_group = MultiAgentTradingSystem(state_dim, use_neural_agents=False)
        self.fundamental_group = MultiAgentTradingSystem(state_dim, use_neural_agents=False)
        self.risk_group = MultiAgentTradingSystem(state_dim, use_neural_agents=False)

        # Level 2: Group coordinator
        self.coordinator = MetaController(num_agents=3).to(DEVICE)

        logger.info("HierarchicalMultiAgentSystem initialized")

    def decide(self, state: np.ndarray, context: Dict) -> Tuple[int, float, Dict]:
        """階層的意思決定"""
        # Get decisions from each group
        tech_action, tech_conf, tech_info = self.technical_group.decide(state, context)
        fund_action, fund_conf, fund_info = self.fundamental_group.decide(state, context)
        risk_action, risk_conf, risk_info = self.risk_group.decide(state, context)

        # Create group decisions
        group_decisions = [
            AgentDecision(tech_action, tech_conf, "Technical analysis"),
            AgentDecision(fund_action, fund_conf, "Fundamental analysis"),
            AgentDecision(risk_action, risk_conf, "Risk analysis"),
        ]

        # Coordinator decision
        final_action, final_conf, coord_info = self.coordinator(group_decisions)

        return final_action, final_conf, {
            'technical': tech_info,
            'fundamental': fund_info,
            'risk': risk_info,
            'coordinator': coord_info,
        }


logger.info("Multi-Agent Trading System module loaded")

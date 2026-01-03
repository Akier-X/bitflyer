"""
Advanced Reinforcement Learning Module
=======================================
世界最強の強化学習アルゴリズム群
PPO, SAC, C51, QR-DQN, IQN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Categorical
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, NamedTuple
from dataclasses import dataclass, field
from collections import deque, namedtuple
import random
import math
from loguru import logger


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# Experience Replay
# =============================================================================

class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay (PER)

    TD誤差に基づく優先度付きサンプリング
    """

    def __init__(
        self,
        capacity: int = 100000,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment: float = 0.001,
    ):
        self.capacity = capacity
        self.alpha = alpha  # 優先度の指数
        self.beta = beta    # 重要度サンプリングの指数
        self.beta_increment = beta_increment

        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.max_priority = 1.0

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """経験を追加"""
        experience = (state, action, reward, next_state, done)

        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience

        self.priorities[self.position] = self.max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> Tuple[List, np.ndarray, np.ndarray]:
        """優先度に基づくサンプリング"""
        n = len(self.buffer)
        priorities = self.priorities[:n]

        # 優先度を確率に変換
        probs = priorities ** self.alpha
        probs /= probs.sum()

        # サンプリング
        indices = np.random.choice(n, batch_size, p=probs, replace=False)

        # 重要度サンプリング重み
        weights = (n * probs[indices]) ** (-self.beta)
        weights /= weights.max()

        # Beta を徐々に増加
        self.beta = min(1.0, self.beta + self.beta_increment)

        experiences = [self.buffer[i] for i in indices]

        return experiences, indices, weights

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        """TD誤差に基づいて優先度を更新"""
        for idx, td_error in zip(indices, td_errors):
            priority = abs(td_error) + 1e-6
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)

    def __len__(self) -> int:
        return len(self.buffer)


# =============================================================================
# Actor-Critic Networks
# =============================================================================

class ActorNetwork(nn.Module):
    """Actor Network for PPO/SAC"""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256],
        log_std_min: float = -20,
        log_std_max: float = 2,
    ):
        super().__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # Feature extractor
        layers = []
        dims = [state_dim] + hidden_dims
        for i in range(len(dims) - 1):
            layers.extend([
                nn.Linear(dims[i], dims[i + 1]),
                nn.LayerNorm(dims[i + 1]),
                nn.ReLU(),
            ])
        self.features = nn.Sequential(*layers)

        # Mean and log_std heads
        self.mean_head = nn.Linear(hidden_dims[-1], action_dim)
        self.log_std_head = nn.Linear(hidden_dims[-1], action_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.features(state)
        mean = self.mean_head(features)
        log_std = self.log_std_head(features)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """アクションをサンプリング"""
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = Normal(mean, std)
        x_t = normal.rsample()  # Reparameterization trick
        action = torch.tanh(x_t)

        # Log probability with tanh squashing
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob


class CriticNetwork(nn.Module):
    """Critic Network (Q-function)"""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256],
    ):
        super().__init__()

        # Q1
        layers1 = []
        dims = [state_dim + action_dim] + hidden_dims + [1]
        for i in range(len(dims) - 1):
            layers1.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers1.extend([nn.LayerNorm(dims[i + 1]), nn.ReLU()])
        self.q1 = nn.Sequential(*layers1)

        # Q2 (Double Q-learning)
        layers2 = []
        for i in range(len(dims) - 1):
            layers2.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers2.extend([nn.LayerNorm(dims[i + 1]), nn.ReLU()])
        self.q2 = nn.Sequential(*layers2)

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([state, action], dim=-1)
        return self.q1(x), self.q2(x)


class ValueNetwork(nn.Module):
    """Value Network for PPO"""

    def __init__(
        self,
        state_dim: int,
        hidden_dims: List[int] = [256, 256],
    ):
        super().__init__()

        layers = []
        dims = [state_dim] + hidden_dims + [1]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.extend([nn.LayerNorm(dims[i + 1]), nn.ReLU()])
        self.network = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)


# =============================================================================
# PPO (Proximal Policy Optimization)
# =============================================================================

class PPO:
    """
    PPO - Proximal Policy Optimization

    安定した方策勾配法
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        ppo_epochs: int = 10,
        mini_batch_size: int = 64,
    ):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size

        # Networks
        self.actor = ActorNetwork(state_dim, action_dim).to(DEVICE)
        self.critic = ValueNetwork(state_dim).to(DEVICE)

        # Optimizers
        self.actor_optimizer = optim.AdamW(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.AdamW(self.critic.parameters(), lr=lr)

        # Scaler for mixed precision
        self.scaler = GradScaler()

        # Buffer
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []

    def select_action(self, state: np.ndarray) -> Tuple[int, float]:
        """アクション選択"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            action, log_prob = self.actor.sample(state_tensor)
            value = self.critic(state_tensor)

        action = action.cpu().numpy()[0]

        # 離散アクションに変換 (0: sell, 1: hold, 2: buy)
        action_discrete = int((action[0] + 1) / 2 * 2.99)  # [-1, 1] -> [0, 2]
        action_discrete = max(0, min(2, action_discrete))

        self.log_probs.append(log_prob.cpu().numpy()[0])
        self.values.append(value.cpu().numpy()[0])

        return action_discrete, float(log_prob.cpu().numpy()[0])

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        done: bool,
    ) -> None:
        """遷移を保存"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)

    def compute_gae(
        self,
        rewards: List[float],
        values: List[float],
        dones: List[bool],
        next_value: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """GAE (Generalized Advantage Estimation) 計算"""
        advantages = []
        returns = []
        gae = 0

        values = values + [next_value]

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])

        return np.array(advantages), np.array(returns)

    def update(self, next_state: np.ndarray) -> Dict[str, float]:
        """PPO更新"""
        if len(self.states) < self.mini_batch_size:
            return {}

        # Next value for GAE
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            next_value = self.critic(next_state_tensor).cpu().numpy()[0, 0]

        # GAE計算
        advantages, returns = self.compute_gae(
            self.rewards, [v[0] for v in self.values], self.dones, next_value
        )

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Convert to tensors
        states = torch.FloatTensor(np.array(self.states)).to(DEVICE)
        actions = torch.LongTensor(self.actions).to(DEVICE)
        old_log_probs = torch.FloatTensor(np.array(self.log_probs)).to(DEVICE)
        advantages = torch.FloatTensor(advantages).to(DEVICE)
        returns = torch.FloatTensor(returns).to(DEVICE)

        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy = 0

        # PPO epochs
        for _ in range(self.ppo_epochs):
            # Mini-batch
            indices = np.random.permutation(len(self.states))

            for start in range(0, len(self.states), self.mini_batch_size):
                end = start + self.mini_batch_size
                batch_indices = indices[start:end]

                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                with autocast():
                    # Actor loss
                    mean, log_std = self.actor(batch_states)
                    std = log_std.exp()

                    # 離散アクションを連続値に変換
                    continuous_actions = (batch_actions.float() / 2.0 * 2 - 1).unsqueeze(-1)

                    normal = Normal(mean, std)
                    new_log_probs = normal.log_prob(continuous_actions).sum(dim=-1)
                    entropy = normal.entropy().sum(dim=-1).mean()

                    ratio = torch.exp(new_log_probs - batch_old_log_probs.squeeze())

                    surr1 = ratio * batch_advantages
                    surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages

                    actor_loss = -torch.min(surr1, surr2).mean()
                    actor_loss -= self.entropy_coef * entropy

                    # Critic loss
                    values = self.critic(batch_states).squeeze()
                    critic_loss = F.mse_loss(values, batch_returns)

                # Update actor
                self.actor_optimizer.zero_grad()
                self.scaler.scale(actor_loss).backward()
                self.scaler.unscale_(self.actor_optimizer)
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.scaler.step(self.actor_optimizer)

                # Update critic
                self.critic_optimizer.zero_grad()
                self.scaler.scale(critic_loss).backward()
                self.scaler.unscale_(self.critic_optimizer)
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.scaler.step(self.critic_optimizer)

                self.scaler.update()

                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy += entropy.item()

        # Clear buffer
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.log_probs.clear()
        self.values.clear()

        n_updates = self.ppo_epochs * (len(indices) // self.mini_batch_size + 1)

        return {
            'actor_loss': total_actor_loss / n_updates,
            'critic_loss': total_critic_loss / n_updates,
            'entropy': total_entropy / n_updates,
        }


# =============================================================================
# SAC (Soft Actor-Critic)
# =============================================================================

class SAC:
    """
    SAC - Soft Actor-Critic

    最大エントロピー強化学習
    探索と活用のバランスを自動調整
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        auto_alpha: bool = True,
        buffer_size: int = 100000,
        batch_size: int = 256,
    ):
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.auto_alpha = auto_alpha

        # Networks
        self.actor = ActorNetwork(state_dim, action_dim).to(DEVICE)
        self.critic = CriticNetwork(state_dim, action_dim).to(DEVICE)
        self.critic_target = CriticNetwork(state_dim, action_dim).to(DEVICE)

        # Copy weights to target
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Optimizers
        self.actor_optimizer = optim.AdamW(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.AdamW(self.critic.parameters(), lr=lr)

        # Alpha (temperature)
        if auto_alpha:
            self.target_entropy = -action_dim
            self.log_alpha = torch.zeros(1, requires_grad=True, device=DEVICE)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
            self.alpha = self.log_alpha.exp().item()
        else:
            self.alpha = alpha

        # Replay buffer
        self.replay_buffer = PrioritizedReplayBuffer(buffer_size)

        # Scaler
        self.scaler = GradScaler()

    def select_action(self, state: np.ndarray, eval_mode: bool = False) -> int:
        """アクション選択"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            if eval_mode:
                mean, _ = self.actor(state_tensor)
                action = torch.tanh(mean)
            else:
                action, _ = self.actor.sample(state_tensor)

        action = action.cpu().numpy()[0]

        # 離散アクションに変換
        action_discrete = int((action[0] + 1) / 2 * 2.99)
        return max(0, min(2, action_discrete))

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """遷移を保存"""
        self.replay_buffer.push(state, action, reward, next_state, done)

    def update(self) -> Dict[str, float]:
        """SAC更新"""
        if len(self.replay_buffer) < self.batch_size:
            return {}

        # Sample from buffer
        experiences, indices, weights = self.replay_buffer.sample(self.batch_size)

        states, actions, rewards, next_states, dones = zip(*experiences)

        states = torch.FloatTensor(np.array(states)).to(DEVICE)
        actions = torch.LongTensor(actions).unsqueeze(-1).to(DEVICE)
        rewards = torch.FloatTensor(rewards).unsqueeze(-1).to(DEVICE)
        next_states = torch.FloatTensor(np.array(next_states)).to(DEVICE)
        dones = torch.FloatTensor(dones).unsqueeze(-1).to(DEVICE)
        weights = torch.FloatTensor(weights).unsqueeze(-1).to(DEVICE)

        # 離散アクションを連続値に変換
        continuous_actions = (actions.float() / 2.0 * 2 - 1)

        with torch.no_grad():
            # Target Q values
            next_actions, next_log_probs = self.actor.sample(next_states)
            q1_target, q2_target = self.critic_target(next_states, next_actions)
            q_target = torch.min(q1_target, q2_target) - self.alpha * next_log_probs
            target_q = rewards + (1 - dones) * self.gamma * q_target

        # Critic loss
        q1, q2 = self.critic(states, continuous_actions)
        critic_loss = (weights * (F.mse_loss(q1, target_q, reduction='none') +
                                   F.mse_loss(q2, target_q, reduction='none'))).mean()

        self.critic_optimizer.zero_grad()
        self.scaler.scale(critic_loss).backward()
        self.scaler.step(self.critic_optimizer)

        # Actor loss
        new_actions, log_probs = self.actor.sample(states)
        q1_new, q2_new = self.critic(states, new_actions)
        q_new = torch.min(q1_new, q2_new)

        actor_loss = (self.alpha * log_probs - q_new).mean()

        self.actor_optimizer.zero_grad()
        self.scaler.scale(actor_loss).backward()
        self.scaler.step(self.actor_optimizer)

        # Alpha loss
        alpha_loss = 0
        if self.auto_alpha:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            self.alpha = self.log_alpha.exp().item()

        self.scaler.update()

        # Update target network
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        # Update priorities
        td_errors = (target_q - q1).detach().cpu().numpy().squeeze()
        self.replay_buffer.update_priorities(indices, td_errors)

        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha': self.alpha,
        }


# =============================================================================
# C51 (Categorical DQN)
# =============================================================================

class C51Network(nn.Module):
    """C51 Network - 分布を出力"""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        n_atoms: int = 51,
        v_min: float = -10.0,
        v_max: float = 10.0,
        hidden_dims: List[int] = [256, 256],
    ):
        super().__init__()

        self.action_dim = action_dim
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max

        self.support = torch.linspace(v_min, v_max, n_atoms).to(DEVICE)
        self.delta_z = (v_max - v_min) / (n_atoms - 1)

        # Feature extractor
        layers = []
        dims = [state_dim] + hidden_dims
        for i in range(len(dims) - 1):
            layers.extend([
                nn.Linear(dims[i], dims[i + 1]),
                nn.LayerNorm(dims[i + 1]),
                nn.ReLU(),
            ])
        self.features = nn.Sequential(*layers)

        # Distribution head
        self.dist_head = nn.Linear(hidden_dims[-1], action_dim * n_atoms)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """分布を出力"""
        features = self.features(state)
        dist = self.dist_head(features)
        dist = dist.view(-1, self.action_dim, self.n_atoms)
        dist = F.softmax(dist, dim=-1)
        return dist

    def get_q_values(self, state: torch.Tensor) -> torch.Tensor:
        """Q値を計算"""
        dist = self.forward(state)
        q_values = (dist * self.support).sum(dim=-1)
        return q_values


class C51:
    """
    C51 - Categorical DQN

    リターンの分布を学習
    リスク管理に優れる
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int = 3,
        n_atoms: int = 51,
        v_min: float = -10.0,
        v_max: float = 10.0,
        lr: float = 1e-4,
        gamma: float = 0.99,
        buffer_size: int = 100000,
        batch_size: int = 64,
        target_update_freq: int = 100,
    ):
        self.action_dim = action_dim
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        self.support = torch.linspace(v_min, v_max, n_atoms).to(DEVICE)
        self.delta_z = (v_max - v_min) / (n_atoms - 1)

        # Networks
        self.network = C51Network(state_dim, action_dim, n_atoms, v_min, v_max).to(DEVICE)
        self.target_network = C51Network(state_dim, action_dim, n_atoms, v_min, v_max).to(DEVICE)
        self.target_network.load_state_dict(self.network.state_dict())

        # Optimizer
        self.optimizer = optim.AdamW(self.network.parameters(), lr=lr)

        # Replay buffer
        self.replay_buffer = PrioritizedReplayBuffer(buffer_size)

        self.update_count = 0
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

    def select_action(self, state: np.ndarray, eval_mode: bool = False) -> int:
        """アクション選択"""
        if not eval_mode and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            q_values = self.network.get_q_values(state_tensor)
            action = q_values.argmax(dim=-1).item()

        return action

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """遷移を保存"""
        self.replay_buffer.push(state, action, reward, next_state, done)

    def update(self) -> Dict[str, float]:
        """C51更新"""
        if len(self.replay_buffer) < self.batch_size:
            return {}

        # Sample
        experiences, indices, weights = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*experiences)

        states = torch.FloatTensor(np.array(states)).to(DEVICE)
        actions = torch.LongTensor(actions).to(DEVICE)
        rewards = torch.FloatTensor(rewards).to(DEVICE)
        next_states = torch.FloatTensor(np.array(next_states)).to(DEVICE)
        dones = torch.FloatTensor(dones).to(DEVICE)
        weights = torch.FloatTensor(weights).to(DEVICE)

        # Current distribution
        current_dist = self.network(states)
        current_dist = current_dist[range(self.batch_size), actions]

        with torch.no_grad():
            # Next distribution (Double DQN style)
            next_q = self.network.get_q_values(next_states)
            next_actions = next_q.argmax(dim=-1)

            next_dist = self.target_network(next_states)
            next_dist = next_dist[range(self.batch_size), next_actions]

            # Projection
            Tz = rewards.unsqueeze(-1) + self.gamma * (1 - dones.unsqueeze(-1)) * self.support
            Tz = Tz.clamp(self.v_min, self.v_max)

            b = (Tz - self.v_min) / self.delta_z
            l = b.floor().long()
            u = b.ceil().long()

            # Handle edge cases
            l[(u > 0) * (l == u)] -= 1
            u[(l < (self.n_atoms - 1)) * (l == u)] += 1

            # Distribute probability
            target_dist = torch.zeros_like(next_dist)
            offset = torch.linspace(0, (self.batch_size - 1) * self.n_atoms, self.batch_size).long().unsqueeze(-1).to(DEVICE)

            target_dist.view(-1).index_add_(
                0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1)
            )
            target_dist.view(-1).index_add_(
                0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1)
            )

        # Cross-entropy loss
        loss = -(target_dist * (current_dist + 1e-8).log()).sum(dim=-1)
        weighted_loss = (weights * loss).mean()

        self.optimizer.zero_grad()
        weighted_loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), 10.0)
        self.optimizer.step()

        # Update priorities
        self.replay_buffer.update_priorities(indices, loss.detach().cpu().numpy())

        # Update target network
        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.network.state_dict())

        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return {
            'loss': weighted_loss.item(),
            'epsilon': self.epsilon,
        }

    def get_value_distribution(self, state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """価値分布を取得"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            dist = self.network(state_tensor)
            q_values = self.network.get_q_values(state_tensor)

        return dist.cpu().numpy()[0], q_values.cpu().numpy()[0]


# =============================================================================
# QR-DQN (Quantile Regression DQN)
# =============================================================================

class QRDQNNetwork(nn.Module):
    """QR-DQN Network - 分位点を出力"""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        n_quantiles: int = 64,
        hidden_dims: List[int] = [256, 256],
    ):
        super().__init__()

        self.action_dim = action_dim
        self.n_quantiles = n_quantiles

        # Feature extractor
        layers = []
        dims = [state_dim] + hidden_dims
        for i in range(len(dims) - 1):
            layers.extend([
                nn.Linear(dims[i], dims[i + 1]),
                nn.LayerNorm(dims[i + 1]),
                nn.ReLU(),
            ])
        self.features = nn.Sequential(*layers)

        # Quantile head
        self.quantile_head = nn.Linear(hidden_dims[-1], action_dim * n_quantiles)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """分位点を出力"""
        features = self.features(state)
        quantiles = self.quantile_head(features)
        quantiles = quantiles.view(-1, self.action_dim, self.n_quantiles)
        return quantiles

    def get_q_values(self, state: torch.Tensor) -> torch.Tensor:
        """Q値 (分位点の平均)"""
        quantiles = self.forward(state)
        return quantiles.mean(dim=-1)


class QRDQN:
    """
    QR-DQN - Quantile Regression DQN

    分位点回帰で価値分布を学習
    リスク感度が調整可能
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int = 3,
        n_quantiles: int = 64,
        lr: float = 1e-4,
        gamma: float = 0.99,
        buffer_size: int = 100000,
        batch_size: int = 64,
        target_update_freq: int = 100,
        risk_distortion: str = "neutral",  # "neutral", "risk_averse", "risk_seeking"
    ):
        self.action_dim = action_dim
        self.n_quantiles = n_quantiles
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.risk_distortion = risk_distortion

        # Quantile midpoints
        self.tau = torch.FloatTensor(
            [(2 * i + 1) / (2 * n_quantiles) for i in range(n_quantiles)]
        ).to(DEVICE)

        # Networks
        self.network = QRDQNNetwork(state_dim, action_dim, n_quantiles).to(DEVICE)
        self.target_network = QRDQNNetwork(state_dim, action_dim, n_quantiles).to(DEVICE)
        self.target_network.load_state_dict(self.network.state_dict())

        # Optimizer
        self.optimizer = optim.AdamW(self.network.parameters(), lr=lr)

        # Replay buffer
        self.replay_buffer = PrioritizedReplayBuffer(buffer_size)

        self.update_count = 0
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

    def _distort_quantiles(self, quantiles: torch.Tensor) -> torch.Tensor:
        """リスク歪曲"""
        if self.risk_distortion == "neutral":
            return quantiles.mean(dim=-1)
        elif self.risk_distortion == "risk_averse":
            # CVaR (下位25%の平均)
            n = self.n_quantiles // 4
            return quantiles[..., :n].mean(dim=-1)
        elif self.risk_distortion == "risk_seeking":
            # 上位25%の平均
            n = self.n_quantiles // 4
            return quantiles[..., -n:].mean(dim=-1)
        else:
            return quantiles.mean(dim=-1)

    def select_action(self, state: np.ndarray, eval_mode: bool = False) -> int:
        """アクション選択"""
        if not eval_mode and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            quantiles = self.network(state_tensor)
            q_values = self._distort_quantiles(quantiles)
            action = q_values.argmax(dim=-1).item()

        return action

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """遷移を保存"""
        self.replay_buffer.push(state, action, reward, next_state, done)

    def update(self) -> Dict[str, float]:
        """QR-DQN更新"""
        if len(self.replay_buffer) < self.batch_size:
            return {}

        # Sample
        experiences, indices, weights = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*experiences)

        states = torch.FloatTensor(np.array(states)).to(DEVICE)
        actions = torch.LongTensor(actions).to(DEVICE)
        rewards = torch.FloatTensor(rewards).unsqueeze(-1).to(DEVICE)
        next_states = torch.FloatTensor(np.array(next_states)).to(DEVICE)
        dones = torch.FloatTensor(dones).unsqueeze(-1).to(DEVICE)
        weights = torch.FloatTensor(weights).to(DEVICE)

        # Current quantiles
        current_quantiles = self.network(states)
        current_quantiles = current_quantiles[range(self.batch_size), actions]

        with torch.no_grad():
            # Next quantiles (Double DQN)
            next_q = self.network.get_q_values(next_states)
            next_actions = next_q.argmax(dim=-1)

            next_quantiles = self.target_network(next_states)
            next_quantiles = next_quantiles[range(self.batch_size), next_actions]

            # Target quantiles
            target_quantiles = rewards + self.gamma * (1 - dones) * next_quantiles

        # Quantile Huber loss
        td_errors = target_quantiles.unsqueeze(-2) - current_quantiles.unsqueeze(-1)

        huber_loss = torch.where(
            td_errors.abs() <= 1.0,
            0.5 * td_errors.pow(2),
            td_errors.abs() - 0.5
        )

        quantile_loss = (
            (self.tau.view(1, -1, 1) - (td_errors < 0).float()).abs() * huber_loss
        ).mean(dim=-1).sum(dim=-1)

        weighted_loss = (weights * quantile_loss).mean()

        self.optimizer.zero_grad()
        weighted_loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), 10.0)
        self.optimizer.step()

        # Update priorities
        self.replay_buffer.update_priorities(indices, quantile_loss.detach().cpu().numpy())

        # Update target
        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.network.state_dict())

        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return {
            'loss': weighted_loss.item(),
            'epsilon': self.epsilon,
        }

    def get_risk_metrics(self, state: np.ndarray) -> Dict[str, float]:
        """リスク指標を取得"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            quantiles = self.network(state_tensor)[0]  # (action_dim, n_quantiles)

        metrics = {}
        for a in range(self.action_dim):
            q = quantiles[a].cpu().numpy()
            metrics[f'action_{a}_mean'] = q.mean()
            metrics[f'action_{a}_std'] = q.std()
            metrics[f'action_{a}_VaR_5'] = np.percentile(q, 5)
            metrics[f'action_{a}_CVaR_5'] = q[q <= np.percentile(q, 5)].mean()

        return metrics


# =============================================================================
# Unified RL Agent
# =============================================================================

class UnifiedRLAgent:
    """
    統合強化学習エージェント

    複数のRLアルゴリズムを統合
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int = 3,
        algorithms: List[str] = ["ppo", "sac", "c51", "qrdqn"],
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.algorithms = algorithms

        self.agents = {}

        if "ppo" in algorithms:
            self.agents["ppo"] = PPO(state_dim, action_dim)

        if "sac" in algorithms:
            self.agents["sac"] = SAC(state_dim, action_dim)

        if "c51" in algorithms:
            self.agents["c51"] = C51(state_dim, action_dim)

        if "qrdqn" in algorithms:
            self.agents["qrdqn"] = QRDQN(state_dim, action_dim)

        # Voting weights
        self.weights = {alg: 1.0 / len(algorithms) for alg in algorithms}

        # Performance tracking
        self.performance = {alg: deque(maxlen=100) for alg in algorithms}

        logger.info(f"UnifiedRLAgent initialized with: {algorithms}")

    def select_action(self, state: np.ndarray, eval_mode: bool = False) -> Tuple[int, Dict]:
        """
        アクション選択

        全エージェントの投票で決定
        """
        votes = {0: 0.0, 1: 0.0, 2: 0.0}  # sell, hold, buy
        agent_actions = {}

        for name, agent in self.agents.items():
            if name == "ppo":
                action, _ = agent.select_action(state)
            else:
                action = agent.select_action(state, eval_mode)

            agent_actions[name] = action
            votes[action] += self.weights[name]

        # 最多得票のアクション
        final_action = max(votes, key=votes.get)

        return final_action, {
            'agent_actions': agent_actions,
            'votes': votes,
            'weights': self.weights.copy(),
        }

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """遷移を保存"""
        for name, agent in self.agents.items():
            if name == "ppo":
                agent.store_transition(state, action, reward, done)
            else:
                agent.store_transition(state, action, reward, next_state, done)

    def update(self, next_state: np.ndarray = None) -> Dict[str, Any]:
        """全エージェントを更新"""
        results = {}

        for name, agent in self.agents.items():
            if name == "ppo" and next_state is not None:
                results[name] = agent.update(next_state)
            elif name != "ppo":
                results[name] = agent.update()

        return results

    def update_weights(self, rewards: Dict[str, float]) -> None:
        """パフォーマンスに基づいて重みを更新"""
        for name, reward in rewards.items():
            if name in self.performance:
                self.performance[name].append(reward)

        # 平均パフォーマンスで重み更新
        total = 0
        for name in self.algorithms:
            if len(self.performance[name]) > 0:
                avg = np.mean(self.performance[name])
                self.weights[name] = max(0.1, avg + 1.0)  # 最低0.1
                total += self.weights[name]

        # 正規化
        if total > 0:
            for name in self.weights:
                self.weights[name] /= total


logger.info("Advanced RL module loaded (PPO, SAC, C51, QR-DQN)")

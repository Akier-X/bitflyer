"""
World Model - DreamerV3 Style
==============================
将来の市場状態を予測し、想像空間で学習
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Categorical, OneHotCategorical
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import deque
from loguru import logger


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# Symlog Transform (DreamerV3)
# =============================================================================

def symlog(x: torch.Tensor) -> torch.Tensor:
    """Symmetric logarithm"""
    return torch.sign(x) * torch.log(torch.abs(x) + 1)


def symexp(x: torch.Tensor) -> torch.Tensor:
    """Symmetric exponential (inverse of symlog)"""
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)


# =============================================================================
# RSSM (Recurrent State Space Model)
# =============================================================================

class GRUCell(nn.Module):
    """GRU Cell with LayerNorm"""

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size

        self.linear_ih = nn.Linear(input_size, 3 * hidden_size)
        self.linear_hh = nn.Linear(hidden_size, 3 * hidden_size)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        x: torch.Tensor,
        h: torch.Tensor,
    ) -> torch.Tensor:
        gates_i = self.linear_ih(x)
        gates_h = self.linear_hh(h)

        r_i, z_i, n_i = gates_i.chunk(3, dim=-1)
        r_h, z_h, n_h = gates_h.chunk(3, dim=-1)

        r = torch.sigmoid(r_i + r_h)
        z = torch.sigmoid(z_i + z_h)
        n = torch.tanh(n_i + r * n_h)

        h_new = (1 - z) * n + z * h
        h_new = self.norm(h_new)

        return h_new


class RSSM(nn.Module):
    """
    Recurrent State Space Model

    DreamerV3の核心：決定的状態 + 確率的状態
    """

    def __init__(
        self,
        embed_dim: int = 256,
        deter_dim: int = 512,
        stoch_dim: int = 32,
        discrete_dim: int = 32,
        action_dim: int = 3,
        hidden_dim: int = 512,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.deter_dim = deter_dim
        self.stoch_dim = stoch_dim
        self.discrete_dim = discrete_dim
        self.state_dim = deter_dim + stoch_dim * discrete_dim

        # Sequence model (GRU)
        self.gru = GRUCell(stoch_dim * discrete_dim + action_dim, deter_dim)

        # Prior (予測)
        self.prior_net = nn.Sequential(
            nn.Linear(deter_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, stoch_dim * discrete_dim),
        )

        # Posterior (推論)
        self.posterior_net = nn.Sequential(
            nn.Linear(deter_dim + embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, stoch_dim * discrete_dim),
        )

    def initial_state(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """初期状態"""
        return {
            'deter': torch.zeros(batch_size, self.deter_dim, device=DEVICE),
            'stoch': torch.zeros(batch_size, self.stoch_dim, self.discrete_dim, device=DEVICE),
        }

    def get_feat(self, state: Dict[str, torch.Tensor]) -> torch.Tensor:
        """状態から特徴量を取得"""
        stoch = state['stoch'].flatten(start_dim=-2)
        return torch.cat([state['deter'], stoch], dim=-1)

    def prior(self, deter: torch.Tensor) -> Dict[str, torch.Tensor]:
        """事前分布 p(s_t | h_t)"""
        logits = self.prior_net(deter)
        logits = logits.view(-1, self.stoch_dim, self.discrete_dim)

        dist = OneHotCategorical(logits=logits)
        stoch = dist.sample() + dist.probs - dist.probs.detach()  # Straight-through

        return {'stoch': stoch, 'logits': logits, 'dist': dist}

    def posterior(
        self,
        deter: torch.Tensor,
        embed: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """事後分布 q(s_t | h_t, o_t)"""
        x = torch.cat([deter, embed], dim=-1)
        logits = self.posterior_net(x)
        logits = logits.view(-1, self.stoch_dim, self.discrete_dim)

        dist = OneHotCategorical(logits=logits)
        stoch = dist.sample() + dist.probs - dist.probs.detach()

        return {'stoch': stoch, 'logits': logits, 'dist': dist}

    def forward(
        self,
        prev_state: Dict[str, torch.Tensor],
        action: torch.Tensor,
        embed: Optional[torch.Tensor] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        1ステップ遷移

        Args:
            prev_state: 前の状態
            action: アクション (one-hot)
            embed: 観測の埋め込み (training時のみ)

        Returns:
            state: 新しい状態
            prior_info: 事前分布情報
        """
        prev_stoch = prev_state['stoch'].flatten(start_dim=-2)
        x = torch.cat([prev_stoch, action], dim=-1)

        # Deterministic state
        deter = self.gru(x, prev_state['deter'])

        # Prior
        prior_info = self.prior(deter)

        if embed is not None:
            # Posterior (training)
            post_info = self.posterior(deter, embed)
            stoch = post_info['stoch']
            prior_info['posterior'] = post_info
        else:
            # Prior (imagination)
            stoch = prior_info['stoch']

        state = {'deter': deter, 'stoch': stoch}

        return state, prior_info

    def imagine(
        self,
        initial_state: Dict[str, torch.Tensor],
        actions: torch.Tensor,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        想像モード - 将来の状態を予測

        Args:
            initial_state: 初期状態
            actions: アクション系列 (T, batch, action_dim)

        Returns:
            states: 予測状態系列
        """
        states = []
        state = initial_state

        for t in range(len(actions)):
            state, _ = self.forward(state, actions[t], embed=None)
            states.append(state)

        return states


# =============================================================================
# Encoder / Decoder
# =============================================================================

class Encoder(nn.Module):
    """観測エンコーダ"""

    def __init__(
        self,
        obs_dim: int,
        embed_dim: int = 256,
        hidden_dims: List[int] = [256, 256],
    ):
        super().__init__()

        layers = []
        dims = [obs_dim] + hidden_dims + [embed_dim]

        for i in range(len(dims) - 1):
            layers.extend([
                nn.Linear(dims[i], dims[i + 1]),
                nn.LayerNorm(dims[i + 1]),
                nn.SiLU(),
            ])

        self.network = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.network(symlog(obs))


class Decoder(nn.Module):
    """観測デコーダ"""

    def __init__(
        self,
        state_dim: int,
        obs_dim: int,
        hidden_dims: List[int] = [256, 256],
    ):
        super().__init__()

        layers = []
        dims = [state_dim] + hidden_dims + [obs_dim]

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.extend([nn.LayerNorm(dims[i + 1]), nn.SiLU()])

        self.network = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)


class RewardPredictor(nn.Module):
    """報酬予測器 (Twohot encoding for DreamerV3)"""

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 256,
        num_bins: int = 255,
    ):
        super().__init__()

        self.num_bins = num_bins

        # Bin boundaries
        self.register_buffer(
            'bins',
            torch.linspace(-20, 20, num_bins)
        )

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_bins),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        logits = self.network(state)
        return logits

    def predict(self, state: torch.Tensor) -> torch.Tensor:
        """報酬を予測"""
        logits = self.forward(state)
        probs = F.softmax(logits, dim=-1)
        reward = (probs * self.bins).sum(dim=-1)
        return symexp(reward)

    def loss(self, state: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Twohot loss"""
        logits = self.forward(state)
        target = symlog(target)

        # Find bin indices
        below = (self.bins <= target.unsqueeze(-1)).sum(dim=-1) - 1
        below = below.clamp(0, self.num_bins - 2)
        above = below + 1

        # Twohot weights
        below_weight = (self.bins[above] - target) / (self.bins[above] - self.bins[below] + 1e-8)
        above_weight = 1 - below_weight

        # Create target distribution
        target_dist = torch.zeros_like(logits)
        target_dist.scatter_(-1, below.unsqueeze(-1), below_weight.unsqueeze(-1))
        target_dist.scatter_(-1, above.unsqueeze(-1), above_weight.unsqueeze(-1))

        # Cross-entropy loss
        log_probs = F.log_softmax(logits, dim=-1)
        loss = -(target_dist * log_probs).sum(dim=-1)

        return loss


class ContinuePredictor(nn.Module):
    """継続予測器 (エピソード終了予測)"""

    def __init__(self, state_dim: int, hidden_dim: int = 256):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)

    def predict(self, state: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(state))


# =============================================================================
# Actor-Critic for World Model
# =============================================================================

class WorldModelActor(nn.Module):
    """Actor for imagination-based learning"""

    def __init__(
        self,
        state_dim: int,
        action_dim: int = 3,
        hidden_dim: int = 256,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        logits = self.network(state)
        return logits

    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """アクションをサンプリング"""
        logits = self.forward(state)
        dist = Categorical(logits=logits)
        action = dist.sample()

        # One-hot encoding
        action_onehot = F.one_hot(action, num_classes=logits.size(-1)).float()

        # Straight-through
        action_onehot = action_onehot + dist.probs - dist.probs.detach()

        log_prob = dist.log_prob(action)

        return action_onehot, log_prob

    def entropy(self, state: torch.Tensor) -> torch.Tensor:
        """エントロピー"""
        logits = self.forward(state)
        dist = Categorical(logits=logits)
        return dist.entropy()


class WorldModelCritic(nn.Module):
    """Critic for imagination-based learning (Twohot)"""

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 256,
        num_bins: int = 255,
    ):
        super().__init__()

        self.num_bins = num_bins

        self.register_buffer(
            'bins',
            torch.linspace(-20, 20, num_bins)
        )

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_bins),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)

    def predict(self, state: torch.Tensor) -> torch.Tensor:
        """価値を予測"""
        logits = self.forward(state)
        probs = F.softmax(logits, dim=-1)
        value = (probs * self.bins).sum(dim=-1)
        return symexp(value)

    def loss(self, state: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Twohot loss"""
        logits = self.forward(state)
        target = symlog(target)

        below = (self.bins <= target.unsqueeze(-1)).sum(dim=-1) - 1
        below = below.clamp(0, self.num_bins - 2)
        above = below + 1

        below_weight = (self.bins[above] - target) / (self.bins[above] - self.bins[below] + 1e-8)
        above_weight = 1 - below_weight

        target_dist = torch.zeros_like(logits)
        target_dist.scatter_(-1, below.unsqueeze(-1), below_weight.unsqueeze(-1))
        target_dist.scatter_(-1, above.unsqueeze(-1), above_weight.unsqueeze(-1))

        log_probs = F.log_softmax(logits, dim=-1)
        loss = -(target_dist * log_probs).sum(dim=-1)

        return loss


# =============================================================================
# World Model (DreamerV3)
# =============================================================================

class DreamerV3WorldModel(nn.Module):
    """
    DreamerV3 World Model

    市場の動態モデルを学習し、想像空間で強化学習
    """

    def __init__(
        self,
        obs_dim: int = 500,
        action_dim: int = 3,
        embed_dim: int = 256,
        deter_dim: int = 512,
        stoch_dim: int = 32,
        discrete_dim: int = 32,
        hidden_dim: int = 512,
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        state_dim = deter_dim + stoch_dim * discrete_dim

        # World Model components
        self.encoder = Encoder(obs_dim, embed_dim)
        self.rssm = RSSM(embed_dim, deter_dim, stoch_dim, discrete_dim, action_dim, hidden_dim)
        self.decoder = Decoder(state_dim, obs_dim)
        self.reward_predictor = RewardPredictor(state_dim, hidden_dim)
        self.continue_predictor = ContinuePredictor(state_dim, hidden_dim)

        # Actor-Critic
        self.actor = WorldModelActor(state_dim, action_dim, hidden_dim)
        self.critic = WorldModelCritic(state_dim, hidden_dim)

        # Optimizers
        self.world_optimizer = optim.AdamW(
            list(self.encoder.parameters()) +
            list(self.rssm.parameters()) +
            list(self.decoder.parameters()) +
            list(self.reward_predictor.parameters()) +
            list(self.continue_predictor.parameters()),
            lr=1e-4,
            weight_decay=0.01,
        )

        self.actor_optimizer = optim.AdamW(self.actor.parameters(), lr=3e-5)
        self.critic_optimizer = optim.AdamW(self.critic.parameters(), lr=3e-5)

        self.scaler = GradScaler()

        # Hyperparameters
        self.gamma = 0.997
        self.lambda_ = 0.95
        self.imagination_horizon = 15
        self.kl_free = 1.0
        self.kl_scale = 0.1

        logger.info(f"DreamerV3 World Model initialized (state_dim={state_dim})")

    def to(self, device):
        super().to(device)
        return self

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        """観測をエンコード"""
        return self.encoder(obs)

    def get_state_feat(self, state: Dict[str, torch.Tensor]) -> torch.Tensor:
        """状態特徴量を取得"""
        return self.rssm.get_feat(state)

    def world_step(
        self,
        prev_state: Dict[str, torch.Tensor],
        action: torch.Tensor,
        obs: Optional[torch.Tensor] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """World modelを1ステップ進める"""
        embed = self.encode(obs) if obs is not None else None
        state, info = self.rssm(prev_state, action, embed)
        return state, info

    def imagine_trajectory(
        self,
        initial_state: Dict[str, torch.Tensor],
        horizon: int = None,
    ) -> Dict[str, torch.Tensor]:
        """想像軌道を生成"""
        if horizon is None:
            horizon = self.imagination_horizon

        states_deter = [initial_state['deter']]
        states_stoch = [initial_state['stoch']]
        actions = []
        log_probs = []

        state = initial_state

        for _ in range(horizon):
            feat = self.get_state_feat(state)
            action, log_prob = self.actor.sample(feat)

            state, _ = self.rssm(state, action, embed=None)

            states_deter.append(state['deter'])
            states_stoch.append(state['stoch'])
            actions.append(action)
            log_probs.append(log_prob)

        # Stack
        trajectory = {
            'deter': torch.stack(states_deter[:-1]),  # T, B, D
            'stoch': torch.stack(states_stoch[:-1]),
            'actions': torch.stack(actions),
            'log_probs': torch.stack(log_probs),
            'next_deter': torch.stack(states_deter[1:]),
            'next_stoch': torch.stack(states_stoch[1:]),
        }

        return trajectory

    def compute_returns(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        continues: torch.Tensor,
    ) -> torch.Tensor:
        """λ-returnsを計算"""
        T = len(rewards)

        returns = torch.zeros_like(rewards)
        last_return = values[-1]

        for t in reversed(range(T)):
            returns[t] = rewards[t] + continues[t] * self.gamma * (
                (1 - self.lambda_) * values[t + 1] if t < T - 1 else 0 +
                self.lambda_ * last_return
            )
            last_return = returns[t]

        return returns

    def train_world_model(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
    ) -> Dict[str, float]:
        """World Modelを学習"""
        batch_size, seq_len = observations.shape[:2]

        # One-hot actions
        actions_onehot = F.one_hot(actions, num_classes=self.action_dim).float()

        # Initial state
        state = self.rssm.initial_state(batch_size)

        # Forward pass
        prior_logits = []
        posterior_logits = []
        recon_obs = []
        pred_rewards = []
        pred_continues = []

        for t in range(seq_len):
            embed = self.encode(observations[:, t])
            state, info = self.rssm(state, actions_onehot[:, t], embed)

            feat = self.get_state_feat(state)

            prior_logits.append(info['logits'])
            posterior_logits.append(info['posterior']['logits'])
            recon_obs.append(self.decoder(feat))
            pred_rewards.append(self.reward_predictor.forward(feat))
            pred_continues.append(self.continue_predictor.forward(feat))

        # Stack predictions
        prior_logits = torch.stack(prior_logits, dim=1)
        posterior_logits = torch.stack(posterior_logits, dim=1)
        recon_obs = torch.stack(recon_obs, dim=1)

        # Losses
        # Reconstruction loss
        recon_loss = F.mse_loss(recon_obs, symlog(observations))

        # KL loss
        prior_dist = OneHotCategorical(logits=prior_logits)
        posterior_dist = OneHotCategorical(logits=posterior_logits)
        kl = torch.distributions.kl.kl_divergence(posterior_dist, prior_dist)
        kl = kl.sum(dim=-1).mean()

        # Free bits
        kl_loss = torch.max(kl, torch.tensor(self.kl_free, device=kl.device))
        kl_loss = self.kl_scale * kl_loss

        # Reward loss
        reward_loss = 0
        for t, pred_r in enumerate(pred_rewards):
            reward_loss += self.reward_predictor.loss(
                self.get_state_feat({'deter': state['deter'], 'stoch': state['stoch']}),
                rewards[:, t]
            ).mean()
        reward_loss /= seq_len

        # Continue loss
        continue_loss = 0
        for t, pred_c in enumerate(pred_continues):
            continue_loss += F.binary_cross_entropy_with_logits(
                pred_c.squeeze(-1),
                1 - dones[:, t].float()
            )
        continue_loss /= seq_len

        # Total loss
        total_loss = recon_loss + kl_loss + reward_loss + continue_loss

        # Optimize
        self.world_optimizer.zero_grad()
        self.scaler.scale(total_loss).backward()
        self.scaler.unscale_(self.world_optimizer)
        nn.utils.clip_grad_norm_(
            list(self.encoder.parameters()) +
            list(self.rssm.parameters()) +
            list(self.decoder.parameters()),
            100.0
        )
        self.scaler.step(self.world_optimizer)
        self.scaler.update()

        return {
            'recon_loss': recon_loss.item(),
            'kl_loss': kl_loss.item(),
            'reward_loss': reward_loss.item(),
            'continue_loss': continue_loss.item(),
            'total_loss': total_loss.item(),
        }

    def train_actor_critic(
        self,
        initial_state: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """Actor-Criticを想像空間で学習"""
        # Imagine trajectory
        with torch.no_grad():
            trajectory = self.imagine_trajectory(initial_state)

        T, B = trajectory['deter'].shape[:2]

        # Get features
        feats = []
        next_feats = []
        for t in range(T):
            feat = self.get_state_feat({
                'deter': trajectory['deter'][t],
                'stoch': trajectory['stoch'][t],
            })
            next_feat = self.get_state_feat({
                'deter': trajectory['next_deter'][t],
                'stoch': trajectory['next_stoch'][t],
            })
            feats.append(feat)
            next_feats.append(next_feat)

        feats = torch.stack(feats)
        next_feats = torch.stack(next_feats)

        # Predict rewards and continues
        with torch.no_grad():
            rewards = self.reward_predictor.predict(feats)
            continues = self.continue_predictor.predict(next_feats)

        # Predict values
        values = self.critic.predict(feats)
        next_values = self.critic.predict(next_feats)

        # Compute returns
        returns = self.compute_returns(rewards, values, continues)

        # Critic loss
        critic_loss = self.critic.loss(feats, returns.detach()).mean()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 100.0)
        self.critic_optimizer.step()

        # Actor loss
        advantages = (returns - values).detach()

        # Reinforce + entropy bonus
        log_probs = trajectory['log_probs']
        entropy = self.actor.entropy(feats).mean()

        actor_loss = -(log_probs * advantages).mean() - 0.003 * entropy

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 100.0)
        self.actor_optimizer.step()

        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'entropy': entropy.item(),
            'mean_return': returns.mean().item(),
        }

    def select_action(
        self,
        state: Dict[str, torch.Tensor],
        eval_mode: bool = False,
    ) -> Tuple[int, Dict]:
        """アクション選択"""
        feat = self.get_state_feat(state)

        if eval_mode:
            logits = self.actor(feat)
            action = logits.argmax(dim=-1).item()
            probs = F.softmax(logits, dim=-1)
        else:
            action_onehot, _ = self.actor.sample(feat)
            action = action_onehot.argmax(dim=-1).item()
            probs = action_onehot

        # 予測報酬
        pred_reward = self.reward_predictor.predict(feat).item()
        pred_value = self.critic.predict(feat).item()

        return action, {
            'predicted_reward': pred_reward,
            'predicted_value': pred_value,
            'action_probs': probs.detach().cpu().numpy(),
        }

    def simulate_future(
        self,
        initial_state: Dict[str, torch.Tensor],
        actions: List[int],
    ) -> List[Dict[str, float]]:
        """将来をシミュレート"""
        state = initial_state
        predictions = []

        for action in actions:
            action_onehot = F.one_hot(
                torch.tensor([action], device=DEVICE),
                num_classes=self.action_dim
            ).float()

            state, _ = self.rssm(state, action_onehot, embed=None)
            feat = self.get_state_feat(state)

            predictions.append({
                'reward': self.reward_predictor.predict(feat).item(),
                'value': self.critic.predict(feat).item(),
                'continue_prob': self.continue_predictor.predict(feat).item(),
            })

        return predictions


class WorldModelTrainer:
    """World Modelの学習管理"""

    def __init__(
        self,
        model: DreamerV3WorldModel,
        buffer_size: int = 100000,
        batch_size: int = 16,
        seq_len: int = 50,
    ):
        self.model = model
        self.batch_size = batch_size
        self.seq_len = seq_len

        # Experience buffer
        self.obs_buffer = deque(maxlen=buffer_size)
        self.action_buffer = deque(maxlen=buffer_size)
        self.reward_buffer = deque(maxlen=buffer_size)
        self.done_buffer = deque(maxlen=buffer_size)

        # Current episode
        self.current_episode = {
            'obs': [],
            'actions': [],
            'rewards': [],
            'dones': [],
        }

        # Current state
        self.current_state = None

    def reset(self) -> None:
        """エピソードリセット"""
        if len(self.current_episode['obs']) > self.seq_len:
            # Save episode to buffer
            self.obs_buffer.extend(self.current_episode['obs'])
            self.action_buffer.extend(self.current_episode['actions'])
            self.reward_buffer.extend(self.current_episode['rewards'])
            self.done_buffer.extend(self.current_episode['dones'])

        self.current_episode = {
            'obs': [],
            'actions': [],
            'rewards': [],
            'dones': [],
        }

        self.current_state = self.model.rssm.initial_state(1)

    def step(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        done: bool,
    ) -> Tuple[int, Dict]:
        """1ステップ"""
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
        action_tensor = F.one_hot(
            torch.tensor([action], device=DEVICE),
            num_classes=self.model.action_dim
        ).float()

        # Update state
        self.current_state, _ = self.model.world_step(
            self.current_state,
            action_tensor,
            obs_tensor,
        )

        # Store
        self.current_episode['obs'].append(obs)
        self.current_episode['actions'].append(action)
        self.current_episode['rewards'].append(reward)
        self.current_episode['dones'].append(done)

        # Select next action
        next_action, info = self.model.select_action(self.current_state)

        if done:
            self.reset()

        return next_action, info

    def train(self) -> Dict[str, float]:
        """学習"""
        if len(self.obs_buffer) < self.batch_size * self.seq_len:
            return {}

        # Sample sequences
        obs_batch = []
        action_batch = []
        reward_batch = []
        done_batch = []

        buffer_len = len(self.obs_buffer)

        for _ in range(self.batch_size):
            start = np.random.randint(0, buffer_len - self.seq_len)

            obs_seq = [self.obs_buffer[start + i] for i in range(self.seq_len)]
            action_seq = [self.action_buffer[start + i] for i in range(self.seq_len)]
            reward_seq = [self.reward_buffer[start + i] for i in range(self.seq_len)]
            done_seq = [self.done_buffer[start + i] for i in range(self.seq_len)]

            obs_batch.append(obs_seq)
            action_batch.append(action_seq)
            reward_batch.append(reward_seq)
            done_batch.append(done_seq)

        obs_batch = torch.FloatTensor(np.array(obs_batch)).to(DEVICE)
        action_batch = torch.LongTensor(np.array(action_batch)).to(DEVICE)
        reward_batch = torch.FloatTensor(np.array(reward_batch)).to(DEVICE)
        done_batch = torch.FloatTensor(np.array(done_batch)).to(DEVICE)

        # Train world model
        world_metrics = self.model.train_world_model(
            obs_batch, action_batch, reward_batch, done_batch
        )

        # Train actor-critic
        initial_state = self.model.rssm.initial_state(self.batch_size)
        ac_metrics = self.model.train_actor_critic(initial_state)

        return {**world_metrics, **ac_metrics}


logger.info("World Model (DreamerV3) module loaded")

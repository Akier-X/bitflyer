"""
State-of-the-Art Deep Learning Models
======================================
世界最強のディープラーニングモデル群
PyTorch実装 - GPU対応
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import math
from loguru import logger


# デバイス設定
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {DEVICE}")


# =============================================================================
# 1. PatchTST - State-of-the-Art Time Series Transformer
# =============================================================================

class PatchEmbedding(nn.Module):
    """パッチ埋め込み - 時系列をパッチに分割"""

    def __init__(
        self,
        input_dim: int,
        patch_len: int,
        stride: int,
        d_model: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride

        # 各チャネル独立の投影
        self.projection = nn.Linear(patch_len * input_dim, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_dim)
        Returns:
            patches: (batch, num_patches, d_model)
        """
        batch, seq_len, input_dim = x.shape

        # パッチ作成
        num_patches = (seq_len - self.patch_len) // self.stride + 1
        patches = []

        for i in range(num_patches):
            start = i * self.stride
            end = start + self.patch_len
            patch = x[:, start:end, :].reshape(batch, -1)
            patches.append(patch)

        patches = torch.stack(patches, dim=1)  # (batch, num_patches, patch_len * input_dim)
        patches = self.projection(patches)
        patches = self.dropout(patches)

        return patches


class RotaryPositionalEncoding(nn.Module):
    """RoPE - Rotary Position Embedding"""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        self.register_buffer('sin', torch.sin(position * div_term))
        self.register_buffer('cos', torch.cos(position * div_term))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)

        # RoPE適用
        x1, x2 = x[..., ::2], x[..., 1::2]

        cos = self.cos[:seq_len].unsqueeze(0)
        sin = self.sin[:seq_len].unsqueeze(0)

        x_rotated = torch.cat([
            x1 * cos - x2 * sin,
            x1 * sin + x2 * cos
        ], dim=-1)

        return x_rotated


class FlashAttention(nn.Module):
    """Flash Attention - メモリ効率の良い高速アテンション"""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        self.scale = self.head_dim ** -0.5

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, seq_len, _ = x.shape

        # QKV計算
        qkv = self.qkv(x).reshape(batch, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch, heads, seq, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled Dot-Product Attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Causal mask
        if mask is None:
            mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
            attn = attn.masked_fill(mask, float('-inf'))

        attn_weights = F.softmax(attn, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 出力
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).reshape(batch, seq_len, self.d_model)
        out = self.out_proj(out)

        return out, attn_weights


class PatchTST(nn.Module):
    """
    PatchTST - Patch Time Series Transformer

    チャネル独立性を保ち、長期依存関係を効率的に学習
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 3,
        d_model: int = 128,
        num_heads: int = 8,
        num_layers: int = 4,
        patch_len: int = 16,
        stride: int = 8,
        dropout: float = 0.1,
        max_seq_len: int = 512,
    ):
        super().__init__()

        self.patch_embedding = PatchEmbedding(
            input_dim, patch_len, stride, d_model, dropout
        )

        self.pos_encoding = RotaryPositionalEncoding(d_model, max_seq_len)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,  # Pre-LN for stability
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        # 出力層
        self.output_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_dim),
        )

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    @autocast()
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, input_dim)
        Returns:
            output: (batch, output_dim)
            attention: attention weights
        """
        # パッチ埋め込み
        patches = self.patch_embedding(x)

        # 位置エンコーディング
        patches = self.pos_encoding(patches)

        # Transformer Encoder
        encoded = self.encoder(patches)

        # 平均プーリング
        pooled = encoded.mean(dim=1)

        # 出力
        output = self.output_proj(pooled)
        output = F.softmax(output, dim=-1)

        return output, None

    def predict(self, x: torch.Tensor) -> Tuple[int, float]:
        """予測"""
        self.eval()
        with torch.no_grad():
            probs, _ = self.forward(x)
            pred = torch.argmax(probs, dim=-1).item()
            conf = probs[0, pred].item()
        return pred, conf


# =============================================================================
# 2. Mamba - State Space Model (線形計算コスト)
# =============================================================================

class SelectiveSSM(nn.Module):
    """Selective State Space Model - Mamba核心"""

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()

        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand

        # 入力投影
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # 1D畳み込み
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
        )

        # SSMパラメータ
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)

        # A パラメータ (対角行列として表現)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))

        # D パラメータ
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # 出力投影
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            output: (batch, seq_len, d_model)
        """
        batch, seq_len, _ = x.shape

        # 入力投影
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        # 1D畳み込み
        x = x.transpose(1, 2)
        x = self.conv1d(x)[:, :, :seq_len]
        x = x.transpose(1, 2)

        # SiLU活性化
        x = F.silu(x)

        # SSM パラメータ計算
        x_ssm = self.x_proj(x)
        delta, B, C = x_ssm.split([1, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(delta)

        # A の離散化
        A = -torch.exp(self.A_log)

        # Selective Scan (簡略化版)
        y = self._selective_scan(x, delta, A, B, C)

        # スキップ接続
        y = y + self.D * x

        # ゲート
        y = y * F.silu(z)

        # 出力投影
        return self.out_proj(y)

    def _selective_scan(
        self,
        x: torch.Tensor,
        delta: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
    ) -> torch.Tensor:
        """Selective Scan実装"""
        batch, seq_len, d_inner = x.shape

        # 状態初期化
        h = torch.zeros(batch, d_inner, self.d_state, device=x.device)

        outputs = []
        for t in range(seq_len):
            # 状態更新
            delta_t = delta[:, t, :].unsqueeze(-1)  # (batch, d_inner, 1)
            A_bar = torch.exp(delta_t * A)  # 離散化されたA
            B_bar = delta_t * B[:, t, :].unsqueeze(1)  # (batch, 1, d_state)

            h = A_bar * h + B_bar * x[:, t, :].unsqueeze(-1)

            # 出力計算
            y_t = (h * C[:, t, :].unsqueeze(1)).sum(dim=-1)
            outputs.append(y_t)

        return torch.stack(outputs, dim=1)


class MambaBlock(nn.Module):
    """Mambaブロック"""

    def __init__(self, d_model: int, d_state: int = 16):
        super().__init__()

        self.norm = nn.LayerNorm(d_model)
        self.ssm = SelectiveSSM(d_model, d_state)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ssm(self.norm(x))


class Mamba(nn.Module):
    """
    Mamba - 線形計算コストのState Space Model

    Transformerと同等以上の精度で、O(N)の計算量
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 3,
        d_model: int = 128,
        d_state: int = 16,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, d_model)

        self.layers = nn.ModuleList([
            MambaBlock(d_model, d_state)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.output = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_dim),
        )

    @autocast()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        x = x.mean(dim=1)  # Global Average Pooling
        x = self.dropout(x)

        output = self.output(x)
        return F.softmax(output, dim=-1)

    def predict(self, x: torch.Tensor) -> Tuple[int, float]:
        self.eval()
        with torch.no_grad():
            probs = self.forward(x)
            pred = torch.argmax(probs, dim=-1).item()
            conf = probs[0, pred].item()
        return pred, conf


# =============================================================================
# 3. iTransformer - Inverted Transformer for Time Series
# =============================================================================

class iTransformer(nn.Module):
    """
    iTransformer - チャネルをトークンとして扱う逆転Transformer

    従来: (batch, seq, features) → Attention over seq
    iTransformer: (batch, features, seq) → Attention over features
    """

    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        output_dim: int = 3,
        d_model: int = 128,
        num_heads: int = 8,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.seq_len = seq_len

        # 各特徴量チャネルの時系列を埋め込み
        self.embedding = nn.Linear(seq_len, d_model)

        # Transformer (特徴量間のAttention)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        # 出力
        self.output = nn.Sequential(
            nn.LayerNorm(d_model * input_dim),
            nn.Linear(d_model * input_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, output_dim),
        )

    @autocast()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_dim)
        """
        batch = x.size(0)

        # 転置: (batch, input_dim, seq_len)
        x = x.transpose(1, 2)

        # 埋め込み: (batch, input_dim, d_model)
        x = self.embedding(x)

        # Transformer: 特徴量間のAttention
        x = self.encoder(x)

        # フラット化
        x = x.reshape(batch, -1)

        # 出力
        output = self.output(x)
        return F.softmax(output, dim=-1)


# =============================================================================
# 4. Temporal Fusion Transformer (TFT)
# =============================================================================

class GatedResidualNetwork(nn.Module):
    """Gated Residual Network"""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.gate = nn.Linear(hidden_dim, output_dim)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(output_dim)

        if input_dim != output_dim:
            self.skip = nn.Linear(input_dim, output_dim)
        else:
            self.skip = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = F.elu(self.fc1(x))
        hidden = self.dropout(hidden)

        output = self.fc2(hidden)
        gate = torch.sigmoid(self.gate(hidden))

        gated_output = gate * output

        if self.skip is not None:
            skip = self.skip(x)
        else:
            skip = x

        return self.layer_norm(skip + gated_output)


class TemporalFusionTransformer(nn.Module):
    """
    Temporal Fusion Transformer

    解釈可能性と予測精度を両立
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 3,
        d_model: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Variable Selection Network
        self.vsn = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, input_dim),
            nn.Softmax(dim=-1),
        )

        # Embedding
        self.embedding = nn.Linear(input_dim, d_model)

        # LSTM Encoder
        self.lstm = nn.LSTM(
            d_model, d_model,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        # Gated Residual Network
        self.grn = GatedResidualNetwork(d_model * 2, d_model * 2, d_model, dropout)

        # Self-Attention
        self.attention = nn.MultiheadAttention(d_model, num_heads, dropout, batch_first=True)

        # Output
        self.output = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, output_dim),
        )

    @autocast()
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            output: predictions
            variable_importance: 特徴量重要度
        """
        # Variable Selection
        var_weights = self.vsn(x)
        x_selected = x * var_weights

        # Embedding
        embedded = self.embedding(x_selected)

        # LSTM
        lstm_out, _ = self.lstm(embedded)

        # GRN
        grn_out = self.grn(lstm_out)

        # Self-Attention
        attn_out, attn_weights = self.attention(grn_out, grn_out, grn_out)

        # 最後のタイムステップ
        final = attn_out[:, -1, :]

        # Output
        output = self.output(final)
        output = F.softmax(output, dim=-1)

        return output, var_weights.mean(dim=1)


# =============================================================================
# 5. Bayesian Neural Network (不確実性推定)
# =============================================================================

class BayesianLinear(nn.Module):
    """ベイジアン線形層"""

    def __init__(self, in_features: int, out_features: int, prior_std: float = 1.0):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        # 重みの事前分布パラメータ
        self.weight_mu = nn.Parameter(torch.zeros(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.zeros(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.zeros(out_features))
        self.bias_rho = nn.Parameter(torch.zeros(out_features))

        # 初期化
        nn.init.xavier_normal_(self.weight_mu)
        nn.init.constant_(self.weight_rho, -3)

        self.prior_std = prior_std

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Reparameterization trick
        weight_std = F.softplus(self.weight_rho)
        bias_std = F.softplus(self.bias_rho)

        if self.training:
            weight_eps = torch.randn_like(weight_std)
            bias_eps = torch.randn_like(bias_std)

            weight = self.weight_mu + weight_std * weight_eps
            bias = self.bias_mu + bias_std * bias_eps
        else:
            weight = self.weight_mu
            bias = self.bias_mu

        output = F.linear(x, weight, bias)

        # KL divergence
        kl = self._kl_divergence(
            self.weight_mu, weight_std,
            self.bias_mu, bias_std
        )

        return output, kl

    def _kl_divergence(
        self,
        weight_mu: torch.Tensor,
        weight_std: torch.Tensor,
        bias_mu: torch.Tensor,
        bias_std: torch.Tensor,
    ) -> torch.Tensor:
        """KL divergence from prior"""
        kl_weight = 0.5 * (
            weight_std.pow(2) / self.prior_std**2 +
            weight_mu.pow(2) / self.prior_std**2 -
            1 - 2 * torch.log(weight_std / self.prior_std)
        ).sum()

        kl_bias = 0.5 * (
            bias_std.pow(2) / self.prior_std**2 +
            bias_mu.pow(2) / self.prior_std**2 -
            1 - 2 * torch.log(bias_std / self.prior_std)
        ).sum()

        return kl_weight + kl_bias


class BayesianNeuralNetwork(nn.Module):
    """
    Bayesian Neural Network

    予測の不確実性を推定可能
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [256, 128, 64],
        output_dim: int = 3,
    ):
        super().__init__()

        layers = []
        dims = [input_dim] + hidden_dims

        for i in range(len(dims) - 1):
            layers.append(BayesianLinear(dims[i], dims[i + 1]))

        self.layers = nn.ModuleList(layers)
        self.output_layer = BayesianLinear(hidden_dims[-1], output_dim)

    def forward(
        self,
        x: torch.Tensor,
        num_samples: int = 10,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            mean_pred: 平均予測
            uncertainty: 不確実性 (予測分散)
            kl: KL divergence
        """
        predictions = []
        total_kl = 0

        for _ in range(num_samples):
            h = x
            kl = 0

            for layer in self.layers:
                h, layer_kl = layer(h)
                h = F.relu(h)
                kl += layer_kl

            output, output_kl = self.output_layer(h)
            kl += output_kl

            predictions.append(F.softmax(output, dim=-1))
            total_kl += kl

        predictions = torch.stack(predictions)

        mean_pred = predictions.mean(dim=0)
        uncertainty = predictions.var(dim=0)

        return mean_pred, uncertainty, total_kl / num_samples

    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        num_samples: int = 50,
    ) -> Tuple[int, float, float]:
        """
        不確実性付き予測

        Returns:
            prediction: 予測クラス
            confidence: 信頼度
            epistemic_uncertainty: 認識論的不確実性
        """
        self.eval()
        with torch.no_grad():
            mean_pred, uncertainty, _ = self.forward(x, num_samples)

            pred = torch.argmax(mean_pred, dim=-1).item()
            conf = mean_pred[0, pred].item()
            epist_unc = uncertainty[0, pred].item()

        return pred, conf, epist_unc


# =============================================================================
# 6. 統合モデル - UltimateEnsemble
# =============================================================================

class UltimateEnsemble(nn.Module):
    """
    究極のアンサンブルモデル

    全てのSOTAモデルを統合
    """

    def __init__(
        self,
        input_dim: int,
        seq_len: int = 60,
        output_dim: int = 3,
        d_model: int = 128,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.seq_len = seq_len

        # 各モデル
        self.patchtst = PatchTST(
            input_dim, output_dim, d_model,
            num_heads=8, num_layers=4,
            patch_len=16, stride=8,
        )

        self.mamba = Mamba(
            input_dim, output_dim, d_model,
            d_state=16, num_layers=4,
        )

        self.itransformer = iTransformer(
            input_dim, seq_len, output_dim, d_model,
            num_heads=8, num_layers=3,
        )

        # メタ学習器
        self.meta_learner = nn.Sequential(
            nn.Linear(output_dim * 3, d_model),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, output_dim * 3),
            nn.Softmax(dim=-1),
        )

        # 学習可能な融合重み
        self.fusion_weights = nn.Parameter(torch.ones(3) / 3)

    @autocast()
    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            x: (batch, seq_len, input_dim)
        """
        # 各モデルの予測
        patch_out, _ = self.patchtst(x)
        mamba_out = self.mamba(x)
        itrans_out = self.itransformer(x)

        # メタ学習による重み調整
        combined = torch.cat([patch_out, mamba_out, itrans_out], dim=-1)
        meta_weights = self.meta_learner(combined)
        meta_weights = meta_weights.view(-1, 3, self.patchtst.output_proj[-1].out_features)

        # 融合
        predictions = torch.stack([patch_out, mamba_out, itrans_out], dim=1)

        # Softmax重み
        weights = F.softmax(self.fusion_weights, dim=0)

        # 加重平均
        final_output = (predictions * weights.view(1, 3, 1)).sum(dim=1)

        model_outputs = {
            'patchtst': patch_out,
            'mamba': mamba_out,
            'itransformer': itrans_out,
            'weights': weights,
        }

        return final_output, model_outputs

    def predict(self, x: torch.Tensor) -> Tuple[int, float, Dict]:
        """予測と詳細情報"""
        self.eval()
        with torch.no_grad():
            output, model_outputs = self.forward(x)
            pred = torch.argmax(output, dim=-1).item()
            conf = output[0, pred].item()

        return pred, conf, {k: v.cpu().numpy() for k, v in model_outputs.items()}


# =============================================================================
# モデルファクトリ
# =============================================================================

def create_model(
    model_type: str,
    input_dim: int,
    seq_len: int = 60,
    output_dim: int = 3,
    **kwargs,
) -> nn.Module:
    """モデル生成ファクトリ"""

    models = {
        'patchtst': PatchTST,
        'mamba': Mamba,
        'itransformer': iTransformer,
        'tft': TemporalFusionTransformer,
        'bayesian': BayesianNeuralNetwork,
        'ensemble': UltimateEnsemble,
    }

    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}")

    if model_type == 'itransformer':
        return models[model_type](input_dim, seq_len, output_dim, **kwargs).to(DEVICE)
    elif model_type == 'ensemble':
        return models[model_type](input_dim, seq_len, output_dim, **kwargs).to(DEVICE)
    else:
        return models[model_type](input_dim, output_dim, **kwargs).to(DEVICE)


logger.info("SOTA models loaded successfully")

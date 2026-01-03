"""
Deep Learning Models
=====================
最先端のディープラーニングモデル
Transformer, Attention, TCN実装
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from loguru import logger


class TemporalAttention:
    """
    時系列アテンション機構

    価格パターンの重要な部分に注目
    """

    def __init__(self, hidden_dim: int, num_heads: int = 4):
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # Query, Key, Value の重み
        self.W_q = np.random.randn(hidden_dim, hidden_dim) * 0.02
        self.W_k = np.random.randn(hidden_dim, hidden_dim) * 0.02
        self.W_v = np.random.randn(hidden_dim, hidden_dim) * 0.02
        self.W_o = np.random.randn(hidden_dim, hidden_dim) * 0.02

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Multi-Head Self Attention

        Args:
            x: (batch_size, seq_len, hidden_dim)

        Returns:
            attended: (batch_size, seq_len, hidden_dim)
        """
        batch_size, seq_len, _ = x.shape

        # Q, K, V 計算
        Q = np.dot(x, self.W_q)  # (batch, seq, hidden)
        K = np.dot(x, self.W_k)
        V = np.dot(x, self.W_v)

        # Multi-head reshape
        Q = Q.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        K = K.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        V = V.reshape(batch_size, seq_len, self.num_heads, self.head_dim)

        # Transpose for attention: (batch, heads, seq, head_dim)
        Q = Q.transpose(0, 2, 1, 3)
        K = K.transpose(0, 2, 1, 3)
        V = V.transpose(0, 2, 1, 3)

        # Scaled dot-product attention
        scale = np.sqrt(self.head_dim)
        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / scale  # (batch, heads, seq, seq)

        # Causal mask (未来の情報を見ない)
        mask = np.triu(np.ones((seq_len, seq_len)), k=1) * -1e9
        scores = scores + mask

        # Softmax
        attention_weights = self._softmax(scores)

        # Apply attention to V
        context = np.matmul(attention_weights, V)  # (batch, heads, seq, head_dim)

        # Reshape back
        context = context.transpose(0, 2, 1, 3)  # (batch, seq, heads, head_dim)
        context = context.reshape(batch_size, seq_len, self.hidden_dim)

        # Output projection
        output = np.dot(context, self.W_o)

        return output, attention_weights

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


class PositionalEncoding:
    """位置エンコーディング"""

    def __init__(self, max_len: int, d_model: int):
        self.encoding = np.zeros((max_len, d_model))

        position = np.arange(max_len).reshape(-1, 1)
        div_term = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))

        self.encoding[:, 0::2] = np.sin(position * div_term)
        self.encoding[:, 1::2] = np.cos(position * div_term)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """位置エンコーディング追加"""
        seq_len = x.shape[1]
        return x + self.encoding[:seq_len]


class TransformerBlock:
    """
    Transformerブロック

    Self-Attention + Feed Forward
    """

    def __init__(self, hidden_dim: int, num_heads: int = 4, ff_dim: int = 256):
        self.attention = TemporalAttention(hidden_dim, num_heads)

        # Feed Forward Network
        self.W1 = np.random.randn(hidden_dim, ff_dim) * 0.02
        self.b1 = np.zeros(ff_dim)
        self.W2 = np.random.randn(ff_dim, hidden_dim) * 0.02
        self.b2 = np.zeros(hidden_dim)

        # Layer Normalization parameters
        self.gamma1 = np.ones(hidden_dim)
        self.beta1 = np.zeros(hidden_dim)
        self.gamma2 = np.ones(hidden_dim)
        self.beta2 = np.zeros(hidden_dim)

    def forward(self, x: np.ndarray) -> np.ndarray:
        # Self-Attention with residual
        attended, _ = self.attention.forward(x)
        x = self._layer_norm(x + attended, self.gamma1, self.beta1)

        # Feed Forward with residual
        ff_out = self._feed_forward(x)
        x = self._layer_norm(x + ff_out, self.gamma2, self.beta2)

        return x

    def _feed_forward(self, x: np.ndarray) -> np.ndarray:
        """Feed Forward with GELU activation"""
        hidden = np.dot(x, self.W1) + self.b1
        hidden = self._gelu(hidden)
        output = np.dot(hidden, self.W2) + self.b2
        return output

    def _gelu(self, x: np.ndarray) -> np.ndarray:
        """GELU activation"""
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

    def _layer_norm(self, x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """Layer Normalization"""
        mean = np.mean(x, axis=-1, keepdims=True)
        std = np.std(x, axis=-1, keepdims=True)
        return gamma * (x - mean) / (std + eps) + beta


class TemporalConvNet:
    """
    Temporal Convolutional Network (TCN)

    時系列の長期依存関係を捉える
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        kernel_size: int = 3,
        num_layers: int = 4,
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers

        # Dilated Causal Convolution weights
        self.conv_weights = []
        for i in range(num_layers):
            in_channels = input_dim if i == 0 else hidden_dim
            # (out_channels, in_channels, kernel_size)
            w = np.random.randn(hidden_dim, in_channels, kernel_size) * 0.1
            self.conv_weights.append(w)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        TCN forward pass

        Args:
            x: (batch_size, seq_len, input_dim)

        Returns:
            output: (batch_size, seq_len, hidden_dim)
        """
        # Transpose for conv: (batch, channels, seq_len)
        x = x.transpose(0, 2, 1)

        for layer_idx, weights in enumerate(self.conv_weights):
            dilation = 2 ** layer_idx

            # Dilated causal convolution
            x = self._dilated_causal_conv(x, weights, dilation)

            # ReLU activation
            x = np.maximum(0, x)

        # Transpose back: (batch, seq_len, channels)
        return x.transpose(0, 2, 1)

    def _dilated_causal_conv(
        self,
        x: np.ndarray,
        weights: np.ndarray,
        dilation: int,
    ) -> np.ndarray:
        """Dilated Causal Convolution"""
        batch_size, in_channels, seq_len = x.shape
        out_channels, _, kernel_size = weights.shape

        # Causal padding
        padding = (kernel_size - 1) * dilation
        x_padded = np.pad(x, ((0, 0), (0, 0), (padding, 0)), mode='constant')

        # Output
        output = np.zeros((batch_size, out_channels, seq_len))

        for t in range(seq_len):
            for k in range(kernel_size):
                idx = padding + t - k * dilation
                if idx >= 0:
                    output[:, :, t] += np.tensordot(
                        x_padded[:, :, idx],
                        weights[:, :, k],
                        axes=([1], [1])
                    )

        return output


class TransformerPredictor:
    """
    Transformer価格予測モデル

    最先端のTransformerアーキテクチャで価格を予測
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 3,
        output_dim: int = 3,
        max_seq_len: int = 100,
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Input projection
        self.input_proj = np.random.randn(input_dim, hidden_dim) * 0.02

        # Positional encoding
        self.pos_encoding = PositionalEncoding(max_seq_len, hidden_dim)

        # Transformer blocks
        self.transformer_blocks = [
            TransformerBlock(hidden_dim, num_heads, hidden_dim * 4)
            for _ in range(num_layers)
        ]

        # Output layer
        self.output_proj = np.random.randn(hidden_dim, output_dim) * 0.02
        self.output_bias = np.zeros(output_dim)

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass

        Args:
            x: (batch_size, seq_len, input_dim)

        Returns:
            output: (batch_size, output_dim) - 予測確率
            attention: attention weights for interpretability
        """
        # Input projection
        x = np.dot(x, self.input_proj)

        # Add positional encoding
        x = self.pos_encoding.forward(x)

        # Transformer blocks
        for block in self.transformer_blocks:
            x = block.forward(x)

        # Use last position for prediction
        x = x[:, -1, :]  # (batch, hidden_dim)

        # Output projection
        logits = np.dot(x, self.output_proj) + self.output_bias

        # Softmax
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

        return probs

    def predict(self, x: np.ndarray) -> int:
        """予測クラス取得"""
        if len(x.shape) == 2:
            x = x.reshape(1, x.shape[0], x.shape[1])

        probs = self.forward(x)
        return np.argmax(probs[0])


class HybridDeepModel:
    """
    ハイブリッド深層学習モデル

    TCN + Transformer + Attention の組み合わせ
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 3,
        seq_len: int = 60,
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # TCN for local patterns
        self.tcn = TemporalConvNet(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            kernel_size=3,
            num_layers=3,
        )

        # Transformer for global patterns
        self.transformer = TransformerPredictor(
            input_dim=hidden_dim,  # TCN output
            hidden_dim=hidden_dim,
            num_heads=4,
            num_layers=2,
            output_dim=hidden_dim,
            max_seq_len=seq_len,
        )

        # Final classification
        self.final_layer = np.random.randn(hidden_dim * 2, output_dim) * 0.02
        self.final_bias = np.zeros(output_dim)

        # Learnable fusion weights
        self.fusion_weight = 0.5

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Hybrid forward pass

        Args:
            x: (batch_size, seq_len, input_dim)

        Returns:
            probs: (batch_size, output_dim)
        """
        # TCN path
        tcn_out = self.tcn.forward(x)  # (batch, seq, hidden)

        # Transformer path
        transformer_out = self.transformer.forward(tcn_out)  # (batch, hidden)

        # TCN last output
        tcn_last = tcn_out[:, -1, :]  # (batch, hidden)

        # Fusion
        if len(transformer_out.shape) == 1:
            transformer_out = transformer_out.reshape(1, -1)

        fused = np.concatenate([tcn_last, transformer_out], axis=-1)

        # Classification
        logits = np.dot(fused, self.final_layer) + self.final_bias

        # Softmax
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

        return probs

    def predict(self, x: np.ndarray) -> Tuple[int, float]:
        """
        予測

        Returns:
            (predicted_class, confidence)
        """
        if len(x.shape) == 2:
            x = x.reshape(1, x.shape[0], x.shape[1])

        probs = self.forward(x)
        predicted = np.argmax(probs[0])
        confidence = probs[0, predicted]

        return predicted, confidence


class MetaLearner:
    """
    メタ学習器

    複数のモデルから学習し、最適なモデル選択を行う
    """

    def __init__(self, num_models: int = 5):
        self.num_models = num_models
        self.model_weights = np.ones(num_models) / num_models
        self.model_performance = np.zeros(num_models)
        self.update_count = np.zeros(num_models)

    def update(self, model_idx: int, reward: float) -> None:
        """モデルパフォーマンス更新"""
        self.update_count[model_idx] += 1
        # 指数移動平均
        alpha = 0.1
        self.model_performance[model_idx] = (
            (1 - alpha) * self.model_performance[model_idx] + alpha * reward
        )
        self._update_weights()

    def _update_weights(self) -> None:
        """重み更新（Softmax）"""
        temperature = 1.0
        exp_perf = np.exp(self.model_performance / temperature)
        self.model_weights = exp_perf / exp_perf.sum()

    def select_model(self) -> int:
        """モデル選択（Thompson Sampling風）"""
        # 探索と活用のバランス
        if np.random.random() < 0.1:
            return np.random.randint(self.num_models)
        return np.argmax(self.model_weights)

    def get_ensemble_weights(self) -> np.ndarray:
        """アンサンブル重み取得"""
        return self.model_weights.copy()

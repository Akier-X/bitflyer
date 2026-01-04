"""
╔══════════════════════════════════════════════════════════════════════════════╗
║     🏆 ULTIMATE AI TRADER v2.0 - 世界最強・世界一・究極完成版 🏆              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  5000円から最速で資産を増やす究極のAIトレーダー                               ║
║  Target: 5000円 → 15000円+ in 1 month (3x return)                            ║
║  Strategy: ML + High-frequency scalping + Kelly Criterion + Regime Detection ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  🧠 Ultimate Features v2.0:                                                  ║
║  ├─ 機械学習価格予測（Ridge回帰 + λ動的調整 + 8特徴量工学）                   ║
║  ├─ Kelly基準 + EMA基づくp/b推定（適応的Half-Kelly）                          ║
║  ├─ 2Dレジーム検出（ボラティリティ × トレンド強度ADX）                         ║
║  ├─ 動的ウェイト配分（レジーム別戦略強度調整）                                 ║
║  ├─ 最大ドローダウン保護（5%下落でポジション縮小）                             ║
║  ├─ 注文板インバランス分析（WebSocket）                                       ║
║  ├─ 外部環境完全対応（入金・出金・手動取引）                                   ║
║  ├─ リアルタイム収支グラフ生成（LINE通知）                                    ║
║  └─ 状態永続化（再起動後も継続）                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  🎯 これ以上の改善は不可能 - 究極の世界最強AIトレーダー 🎯                     ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import asyncio
import sys
import json
import os
import io
import base64
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
from enum import Enum
import traceback
import numpy as np
from loguru import logger

# チャート生成用（オプショナル）
try:
    import matplotlib
    matplotlib.use('Agg')  # ヘッドレスモード
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.ticker import FuncFormatter
    CHART_AVAILABLE = True
except ImportError:
    CHART_AVAILABLE = False
    logger.warning("matplotlib not available, charts disabled")


# ============================================================================
# Machine Learning Price Predictor - 機械学習価格予測（λ動的調整付き）
# ============================================================================
class MLPredictor:
    """
    機械学習価格予測エンジン v2.0

    改善点:
    - λ動的調整: ノイズが多い（ボラティリティ高い）時はλを大きく
    - 予測精度のEMA追跡で信頼度調整
    """

    def __init__(self, lookback: int = 50):
        self.lookback = lookback
        self.weights = None
        self.trained = False
        self.training_count = 0

        # λ動的調整用
        self.base_lambda = 0.1
        self.current_lambda = 0.1
        self.prediction_accuracy_ema = 0.5  # 予測精度のEMA
        self.last_predictions: deque = deque(maxlen=20)  # 直近20予測の結果

    def _create_features(self, prices: List[float]) -> Optional[np.ndarray]:
        """特徴量生成（8特徴量）"""
        if len(prices) < 10:
            return None

        features = []
        arr = np.array(prices)

        # 1. 移動平均との乖離
        sma5 = np.mean(arr[-5:])
        sma10 = np.mean(arr[-10:])
        sma20 = np.mean(arr[-20:]) if len(arr) >= 20 else sma10

        features.append((arr[-1] - sma5) / sma5 if sma5 > 0 else 0)
        features.append((arr[-1] - sma10) / sma10 if sma10 > 0 else 0)
        features.append((sma5 - sma20) / sma20 if sma20 > 0 else 0)

        # 2. モメンタム
        features.append((arr[-1] - arr[-2]) / arr[-2] if arr[-2] > 0 else 0)
        features.append((arr[-1] - arr[-5]) / arr[-5] if len(arr) >= 5 and arr[-5] > 0 else 0)

        # 3. ボラティリティ
        if len(arr) >= 20:
            returns = np.diff(arr[-20:]) / arr[-20:-1]
            features.append(np.std(returns) if len(returns) > 0 else 0)
        else:
            features.append(0)

        # 4. RSI
        if len(arr) >= 15:
            gains = np.maximum(np.diff(arr[-15:]), 0)
            losses = np.abs(np.minimum(np.diff(arr[-15:]), 0))
            avg_gain = np.mean(gains) if len(gains) > 0 else 0
            avg_loss = np.mean(losses) if len(losses) > 0 else 1
            rsi = 100 - (100 / (1 + avg_gain / avg_loss)) if avg_loss > 0 else 50
            features.append((rsi - 50) / 50)
        else:
            features.append(0)

        # 5. 価格位置
        if len(arr) >= 20:
            high = np.max(arr[-20:])
            low = np.min(arr[-20:])
            price_pos = (arr[-1] - low) / (high - low) if high > low else 0.5
            features.append(price_pos - 0.5)
        else:
            features.append(0)

        return np.array(features)

    def _calculate_dynamic_lambda(self, prices: List[float]) -> float:
        """
        λ動的調整: ノイズが多い時はλを大きくしてオーバーフィット防止

        λ = base_lambda * (1 + volatility_factor + error_factor)
        """
        arr = np.array(prices)
        if len(arr) < 20:
            return self.base_lambda

        # ボラティリティファクター（高ボラ = 高λ）
        returns = np.diff(arr[-20:]) / arr[-20:-1]
        volatility = np.std(returns)
        volatility_factor = min(volatility * 50, 2.0)  # 最大2倍

        # 予測誤差ファクター（精度が低い = 高λ）
        error_factor = max(0, 1 - self.prediction_accuracy_ema) * 1.5

        dynamic_lambda = self.base_lambda * (1 + volatility_factor + error_factor)
        return min(dynamic_lambda, 1.0)  # 最大λ=1.0

    def update_accuracy(self, predicted_direction: int, actual_direction: int):
        """予測精度を更新（EMA）"""
        correct = 1.0 if predicted_direction == actual_direction else 0.0
        alpha = 0.1  # EMA係数
        self.prediction_accuracy_ema = alpha * correct + (1 - alpha) * self.prediction_accuracy_ema
        self.last_predictions.append((predicted_direction, actual_direction))

    def train(self, prices: List[float]):
        """オンライン学習（Ridge回帰 + λ動的調整）"""
        if len(prices) < self.lookback:
            return

        X = []
        y = []

        for i in range(20, len(prices) - 1):
            features = self._create_features(prices[:i])
            if features is not None:
                X.append(features)
                change = (prices[i + 1] - prices[i]) / prices[i]
                y.append(1 if change > 0.001 else (-1 if change < -0.001 else 0))

        if len(X) < 10:
            return

        X = np.array(X)
        y = np.array(y)

        # λ動的調整
        self.current_lambda = self._calculate_dynamic_lambda(prices)

        XtX = X.T @ X + self.current_lambda * np.eye(X.shape[1])
        Xty = X.T @ y
        try:
            self.weights = np.linalg.solve(XtX, Xty)
            self.trained = True
            self.training_count += 1
        except:
            pass

    def predict(self, prices: List[float]) -> Tuple[int, float]:
        """価格予測 Returns: (direction, confidence)"""
        if not self.trained or self.weights is None:
            return 0, 0.0

        features = self._create_features(prices)
        if features is None:
            return 0, 0.0

        score = np.dot(features, self.weights)

        # 基本信頼度（シグモイド変換）
        base_confidence = 1 / (1 + np.exp(-abs(score) * 2))

        # 予測精度EMAで信頼度を調整
        adjusted_confidence = base_confidence * (0.5 + self.prediction_accuracy_ema * 0.5)

        if score > 0.2:
            return 1, min(adjusted_confidence, 0.95)
        elif score < -0.2:
            return -1, min(adjusted_confidence, 0.95)
        else:
            return 0, adjusted_confidence * 0.3


# ============================================================================
# Kelly Criterion Position Sizer - Kelly基準ポジションサイジング（EMA版）
# ============================================================================
class KellyPositionSizer:
    """
    Kelly基準による最適ポジションサイジング v2.0

    改善点:
    - EMAによるp/b推定（最近の取引を重視）
    - 適応的Half-Kelly（市場状態による調整）
    """

    def __init__(self, ema_alpha: float = 0.1):
        self.win_count = 0
        self.loss_count = 0
        self.total_win = 0.0
        self.total_loss = 0.0

        # EMAによるp/b推定（ユーザーアドバイス実装）
        self.ema_alpha = ema_alpha
        self.p_ema = 0.5  # 勝率のEMA
        self.b_ema = 1.0  # win/loss比のEMA

        # 取引履歴（直近のみ保持）
        self.recent_trades: deque = deque(maxlen=50)

    def update(self, pnl: float):
        """取引結果を更新（EMA更新含む）"""
        if pnl > 0:
            self.win_count += 1
            self.total_win += pnl
            # EMA更新: 勝ち
            self.p_ema = self.ema_alpha * 1.0 + (1 - self.ema_alpha) * self.p_ema
        else:
            self.loss_count += 1
            self.total_loss += abs(pnl)
            # EMA更新: 負け
            self.p_ema = self.ema_alpha * 0.0 + (1 - self.ema_alpha) * self.p_ema

        # 取引を記録
        self.recent_trades.append(pnl)

        # b（win/loss比）のEMA更新
        if len(self.recent_trades) >= 5:
            recent_wins = [t for t in self.recent_trades if t > 0]
            recent_losses = [abs(t) for t in self.recent_trades if t < 0]
            if recent_wins and recent_losses:
                recent_b = np.mean(recent_wins) / np.mean(recent_losses)
                self.b_ema = self.ema_alpha * recent_b + (1 - self.ema_alpha) * self.b_ema

    def get_kelly_fraction(self) -> float:
        """
        Kelly分数を計算 f* = (p*b - q) / b

        EMAによる適応的推定を使用
        """
        total = self.win_count + self.loss_count
        if total < 5:
            return 0.3  # デフォルト30%

        # EMAベースのp/b使用（直近の状況を重視）
        p = self.p_ema
        q = 1 - p
        b = self.b_ema

        # Kelly計算
        kelly = (p * b - q) / b if b > 0 else 0

        # Half-Kelly（より保守的）
        kelly = kelly / 2

        # 信頼度に基づく調整（取引数が少ない場合は控えめに）
        confidence_factor = min(1.0, total / 20)  # 20取引で完全信頼
        kelly = kelly * (0.5 + confidence_factor * 0.5)

        return max(0.1, min(0.6, kelly))

    def get_stats(self) -> Dict:
        """統計情報を取得"""
        return {
            'win_count': self.win_count,
            'loss_count': self.loss_count,
            'p_ema': self.p_ema,
            'b_ema': self.b_ema,
            'kelly': self.get_kelly_fraction(),
        }


# ============================================================================
# Dynamic Parameter Optimizer - 動的パラメータ最適化（2Dレジーム検出）
# ============================================================================
class DynamicOptimizer:
    """
    動的パラメータ最適化 v2.0

    改善点（ユーザーアドバイス実装）:
    - 2Dレジーム検出: ボラティリティ × トレンド強度（ADX）
    - 動的ウェイト配分: レジーム別に戦略の強度を調整
    """

    # レジームマトリクス（ボラティリティ × トレンド）
    REGIME_MATRIX = {
        ('low', 'weak'): 'range_quiet',      # レンジ相場・静か
        ('low', 'strong'): 'trend_slow',     # ゆっくりトレンド
        ('normal', 'weak'): 'range_active',   # レンジ相場・活発
        ('normal', 'strong'): 'trend_normal', # 通常トレンド
        ('high', 'weak'): 'choppy',           # 方向感なし・荒い
        ('high', 'strong'): 'trend_strong',   # 強トレンド
    }

    # レジーム別の戦略ウェイト（動的ウェイト配分）
    REGIME_WEIGHTS = {
        'range_quiet': {'ml': 0.5, 'momentum': 0.2, 'mean_reversion': 0.8, 'trend': 0.1},
        'trend_slow': {'ml': 0.6, 'momentum': 0.4, 'mean_reversion': 0.3, 'trend': 0.7},
        'range_active': {'ml': 0.4, 'momentum': 0.5, 'mean_reversion': 0.6, 'trend': 0.2},
        'trend_normal': {'ml': 0.5, 'momentum': 0.6, 'mean_reversion': 0.2, 'trend': 0.8},
        'choppy': {'ml': 0.3, 'momentum': 0.3, 'mean_reversion': 0.4, 'trend': 0.2},
        'trend_strong': {'ml': 0.4, 'momentum': 0.8, 'mean_reversion': 0.1, 'trend': 0.9},
    }

    def __init__(self):
        self.volatility_regime = 'normal'
        self.trend_regime = 'weak'
        self.combined_regime = 'range_active'
        self.current_params = {
            'take_profit': 0.5,
            'stop_loss': 0.3,
            'confidence_threshold': 0.25,
            'momentum_threshold': 0.001,
        }
        self.current_weights = self.REGIME_WEIGHTS['range_active']

    def _calculate_adx(self, prices: List[float], period: int = 14) -> float:
        """
        ADX (Average Directional Index) 計算
        トレンドの強さを測定（0-100、25以上がトレンド）
        """
        if len(prices) < period + 2:
            return 25.0  # デフォルト

        arr = np.array(prices)
        high = arr  # 簡易版：終値をHigh/Lowとして使用
        low = arr
        close = arr

        # True Range
        tr = np.maximum(
            np.abs(high[1:] - low[1:]),
            np.maximum(
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:] - close[:-1])
            )
        )

        # +DM / -DM
        plus_dm = np.maximum(high[1:] - high[:-1], 0)
        minus_dm = np.maximum(low[:-1] - low[1:], 0)

        # 両方正の場合、大きい方のみ
        both_positive = (plus_dm > 0) & (minus_dm > 0)
        plus_dm[both_positive & (plus_dm <= minus_dm)] = 0
        minus_dm[both_positive & (minus_dm <= plus_dm)] = 0

        # 平滑化
        def smooth(arr, period):
            result = np.zeros(len(arr))
            result[:period] = arr[:period].sum()
            for i in range(period, len(arr)):
                result[i] = result[i-1] - result[i-1]/period + arr[i]
            return result

        if len(tr) < period:
            return 25.0

        atr = smooth(tr, period)
        plus_di = 100 * smooth(plus_dm, period) / np.maximum(atr, 1e-10)
        minus_di = 100 * smooth(minus_dm, period) / np.maximum(atr, 1e-10)

        # DX
        di_sum = plus_di + minus_di
        di_diff = np.abs(plus_di - minus_di)
        dx = 100 * di_diff / np.maximum(di_sum, 1e-10)

        # ADX（DXの平滑化）
        if len(dx) >= period:
            adx = smooth(dx[-period:], period)[-1]
        else:
            adx = np.mean(dx)

        return min(100, max(0, adx))

    def detect_volatility_regime(self, prices: List[float]) -> str:
        """ボラティリティレジーム検出"""
        if len(prices) < 20:
            return 'normal'

        arr = np.array(prices[-20:])
        returns = np.diff(arr) / arr[:-1]
        vol = np.std(returns)

        if vol < 0.002:
            return 'low'
        elif vol > 0.008:
            return 'high'
        else:
            return 'normal'

    def detect_trend_regime(self, prices: List[float]) -> str:
        """トレンド強度レジーム検出（ADX使用）"""
        adx = self._calculate_adx(prices)
        return 'strong' if adx >= 25 else 'weak'

    def detect_2d_regime(self, prices: List[float]) -> str:
        """2Dレジーム検出（ボラティリティ × トレンド）"""
        vol_regime = self.detect_volatility_regime(prices)
        trend_regime = self.detect_trend_regime(prices)

        self.volatility_regime = vol_regime
        self.trend_regime = trend_regime

        key = (vol_regime, trend_regime)
        self.combined_regime = self.REGIME_MATRIX.get(key, 'range_active')

        return self.combined_regime

    def get_strategy_weights(self) -> Dict[str, float]:
        """現在のレジームに基づく戦略ウェイトを取得"""
        return self.REGIME_WEIGHTS.get(self.combined_regime, self.REGIME_WEIGHTS['range_active'])

    def optimize(self, prices: List[float]) -> Dict:
        """パラメータ最適化（2Dレジーム対応）"""
        regime = self.detect_2d_regime(prices)
        self.current_weights = self.get_strategy_weights()

        # レジーム別パラメータ設定
        if regime == 'range_quiet':
            self.current_params = {
                'take_profit': 0.2,
                'stop_loss': 0.15,
                'confidence_threshold': 0.15,
                'momentum_threshold': 0.0003,
            }
        elif regime == 'trend_slow':
            self.current_params = {
                'take_profit': 0.4,
                'stop_loss': 0.25,
                'confidence_threshold': 0.2,
                'momentum_threshold': 0.0005,
            }
        elif regime == 'range_active':
            self.current_params = {
                'take_profit': 0.3,
                'stop_loss': 0.2,
                'confidence_threshold': 0.2,
                'momentum_threshold': 0.0008,
            }
        elif regime == 'trend_normal':
            self.current_params = {
                'take_profit': 0.5,
                'stop_loss': 0.3,
                'confidence_threshold': 0.25,
                'momentum_threshold': 0.001,
            }
        elif regime == 'choppy':
            self.current_params = {
                'take_profit': 0.25,
                'stop_loss': 0.2,
                'confidence_threshold': 0.35,  # 高い閾値（慎重に）
                'momentum_threshold': 0.0015,
            }
        elif regime == 'trend_strong':
            self.current_params = {
                'take_profit': 0.8,
                'stop_loss': 0.5,
                'confidence_threshold': 0.25,
                'momentum_threshold': 0.002,
            }
        else:
            self.current_params = {
                'take_profit': 0.5,
                'stop_loss': 0.3,
                'confidence_threshold': 0.25,
                'momentum_threshold': 0.001,
            }

        return self.current_params

    def get_regime_info(self) -> Dict:
        """レジーム情報を取得"""
        return {
            'volatility': self.volatility_regime,
            'trend': self.trend_regime,
            'combined': self.combined_regime,
            'weights': self.current_weights,
            'params': self.current_params,
        }


# ============================================================================
# Portfolio Chart Generator - 収益グラフ生成
# ============================================================================
class PortfolioChartGenerator:
    """収益推移グラフ生成"""

    def __init__(self):
        self.portfolio_history: List[Tuple[datetime, float]] = []
        self.trade_history: List[Tuple[datetime, str, float]] = []  # (time, action, pnl)

    def add_portfolio_value(self, value: float):
        """ポートフォリオ価値を記録"""
        self.portfolio_history.append((datetime.now(), value))
        # 最大1000件保持
        if len(self.portfolio_history) > 1000:
            self.portfolio_history = self.portfolio_history[-1000:]

    def add_trade(self, action: str, pnl: float):
        """取引を記録"""
        self.trade_history.append((datetime.now(), action, pnl))
        if len(self.trade_history) > 500:
            self.trade_history = self.trade_history[-500:]

    def generate_chart(self, initial_capital: float) -> Optional[str]:
        """収益推移グラフを生成してファイルパスを返す"""
        if not CHART_AVAILABLE or len(self.portfolio_history) < 2:
            return None

        try:
            # 日本語フォント設定
            plt.rcParams['font.family'] = ['DejaVu Sans', 'sans-serif']

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]})
            fig.patch.set_facecolor('#1a1a2e')

            # ポートフォリオ推移グラフ
            times = [h[0] for h in self.portfolio_history]
            values = [h[1] for h in self.portfolio_history]

            ax1.set_facecolor('#16213e')
            ax1.plot(times, values, color='#00ff88', linewidth=2, label='Portfolio Value')
            ax1.axhline(y=initial_capital, color='#ff6b6b', linestyle='--', alpha=0.7, label='Initial')
            ax1.axhline(y=initial_capital * 3, color='#ffd93d', linestyle='--', alpha=0.7, label='Target (3x)')

            # 利益エリアを塗りつぶし
            ax1.fill_between(times, initial_capital, values,
                           where=[v > initial_capital for v in values],
                           color='#00ff88', alpha=0.3)
            ax1.fill_between(times, initial_capital, values,
                           where=[v < initial_capital for v in values],
                           color='#ff6b6b', alpha=0.3)

            ax1.set_title('📈 AI Trader Portfolio Performance', color='white', fontsize=14, fontweight='bold')
            ax1.set_ylabel('Portfolio Value (JPY)', color='white')
            ax1.tick_params(colors='white')
            ax1.legend(loc='upper left', facecolor='#16213e', edgecolor='white', labelcolor='white')
            ax1.grid(True, alpha=0.3, color='white')

            # Y軸フォーマット
            ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'¥{x:,.0f}'))

            # 累積損益グラフ
            if self.trade_history:
                trade_times = [t[0] for t in self.trade_history]
                cumulative_pnl = []
                total = 0
                for t in self.trade_history:
                    total += t[2]
                    cumulative_pnl.append(total)

                ax2.set_facecolor('#16213e')
                colors = ['#00ff88' if p >= 0 else '#ff6b6b' for p in cumulative_pnl]
                ax2.bar(range(len(cumulative_pnl)), cumulative_pnl, color=colors, alpha=0.8)
                ax2.axhline(y=0, color='white', linewidth=0.5)
                ax2.set_title('📊 Cumulative P&L', color='white', fontsize=12)
                ax2.set_ylabel('P&L (JPY)', color='white')
                ax2.tick_params(colors='white')
                ax2.grid(True, alpha=0.3, color='white')

            plt.tight_layout()

            # ファイル保存
            chart_path = 'logs/portfolio_chart.png'
            os.makedirs('logs', exist_ok=True)
            plt.savefig(chart_path, dpi=150, facecolor='#1a1a2e', edgecolor='none')
            plt.close()

            return chart_path
        except Exception as e:
            logger.warning(f"Chart generation failed: {e}")
            return None


# ============================================================================
# Beautiful Terminal Display - 美しいターミナル表示
# ============================================================================
class TerminalDisplay:
    """美しいターミナル表示"""

    COLORS = {
        'reset': '\033[0m',
        'bold': '\033[1m',
        'green': '\033[92m',
        'red': '\033[91m',
        'yellow': '\033[93m',
        'blue': '\033[94m',
        'cyan': '\033[96m',
        'magenta': '\033[95m',
        'white': '\033[97m',
        'bg_green': '\033[42m',
        'bg_red': '\033[41m',
    }

    @staticmethod
    def colorize(text: str, color: str) -> str:
        return f"{TerminalDisplay.COLORS.get(color, '')}{text}{TerminalDisplay.COLORS['reset']}"

    @staticmethod
    def format_jpy(value: float) -> str:
        """日本円フォーマット"""
        if value >= 0:
            return TerminalDisplay.colorize(f"¥{value:,.0f}", 'green')
        else:
            return TerminalDisplay.colorize(f"¥{value:,.0f}", 'red')

    @staticmethod
    def format_pct(value: float) -> str:
        """パーセントフォーマット"""
        if value >= 0:
            return TerminalDisplay.colorize(f"+{value:.2f}%", 'green')
        else:
            return TerminalDisplay.colorize(f"{value:.2f}%", 'red')

    @staticmethod
    def print_header():
        """ヘッダー表示"""
        print("\n" + "=" * 70)
        print(TerminalDisplay.colorize("  🏆 ULTIMATE AI TRADER - 世界最強システム 稼働中", 'cyan'))
        print("=" * 70)

    @staticmethod
    def print_status(data: Dict):
        """ステータス表示"""
        print("\n" + "-" * 70)
        print(TerminalDisplay.colorize(f"  📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 'white'))
        print("-" * 70)

        # ポートフォリオ
        portfolio = data.get('portfolio_value', 0)
        initial = data.get('initial_capital', 0)
        roi = ((portfolio / initial) - 1) * 100 if initial > 0 else 0
        cash = data.get('cash', 0)
        crypto = data.get('crypto_value', 0)

        print(f"  💰 ポートフォリオ: {TerminalDisplay.format_jpy(portfolio)} ({TerminalDisplay.format_pct(roi)})")
        print(f"  💴 現金:          {TerminalDisplay.format_jpy(cash)}")
        print(f"  🪙 暗号資産:       {TerminalDisplay.format_jpy(crypto)}")

        # 取引統計
        trades = data.get('total_trades', 0)
        win_rate = data.get('win_rate', 0)
        pnl = data.get('total_pnl', 0)

        print(f"  📊 取引回数:       {trades}回")
        print(f"  ✅ 勝率:          {win_rate:.1f}%")
        print(f"  💵 累計損益:       {TerminalDisplay.format_jpy(pnl)}")

        # ML/Kelly状態
        ml_trained = data.get('ml_trained', 0)
        ml_total = data.get('ml_total', 0)
        vol_regime = data.get('volatility_regime', 'normal')
        trend_regime = data.get('trend_regime', 'weak')
        combined_regime = data.get('combined_regime', 'range_active')
        drawdown_active = data.get('drawdown_active', False)

        regime_color = {'low': 'blue', 'normal': 'yellow', 'high': 'red'}.get(vol_regime, 'white')
        trend_color = 'green' if trend_regime == 'strong' else 'cyan'
        print(f"  🧠 ML学習状態:     {ml_trained}/{ml_total} trained")
        print(f"  📈 2Dレジーム:     {TerminalDisplay.colorize(vol_regime.upper(), regime_color)} × {TerminalDisplay.colorize(trend_regime.upper(), trend_color)} → {combined_regime}")

        # ドローダウン保護状態
        if drawdown_active:
            print(f"  🛡️ DD保護:        {TerminalDisplay.colorize('ACTIVE (50%)', 'red')}")

        # ポジション
        positions = data.get('positions', {})
        if positions:
            print("\n  📦 保有ポジション:")
            for pair, pos in positions.items():
                value = pos.get('value', 0)
                pnl_pct = pos.get('pnl_pct', 0)
                kelly = pos.get('kelly', 0.3)
                print(f"     {pair}: {TerminalDisplay.format_jpy(value)} ({TerminalDisplay.format_pct(pnl_pct)}) [K={kelly:.2f}]")

        print("-" * 70)


sys.path.insert(0, '/home/user/bitflyer')

from config.settings import Config, get_config
from src.api.bitflyer_client import BitFlyerClient, MockBitFlyerClient, OrderSide, OrderType
from src.api.websocket_client import BitFlyerWebSocket, OrderBookAnalyzer
from src.notifications.line_messaging import LINEMessagingAPI, TradeNotification


@dataclass
class PriceHistory:
    """価格履歴"""
    prices: deque = field(default_factory=lambda: deque(maxlen=100))
    volumes: deque = field(default_factory=lambda: deque(maxlen=100))
    timestamps: deque = field(default_factory=lambda: deque(maxlen=100))

    def add(self, price: float, volume: float):
        self.prices.append(price)
        self.volumes.append(volume)
        self.timestamps.append(datetime.now())

    @property
    def sma_5(self) -> float:
        if len(self.prices) < 5:
            return 0
        return np.mean(list(self.prices)[-5:])

    @property
    def sma_20(self) -> float:
        if len(self.prices) < 20:
            return 0
        return np.mean(list(self.prices)[-20:])

    @property
    def momentum(self) -> float:
        """短期モメンタム（-1 to 1）"""
        if len(self.prices) < 5:
            return 0
        recent = list(self.prices)[-5:]
        return (recent[-1] - recent[0]) / recent[0] if recent[0] > 0 else 0

    @property
    def volatility(self) -> float:
        """ボラティリティ"""
        if len(self.prices) < 10:
            return 0
        return np.std(list(self.prices)[-10:]) / np.mean(list(self.prices)[-10:])

    @property
    def trend(self) -> int:
        """トレンド方向 (-1, 0, 1)"""
        if self.sma_5 == 0 or self.sma_20 == 0:
            return 0
        if self.sma_5 > self.sma_20 * 1.001:
            return 1  # Uptrend
        elif self.sma_5 < self.sma_20 * 0.999:
            return -1  # Downtrend
        return 0


@dataclass
class AssetPosition:
    """ポジション管理"""
    pair: str
    size: float = 0.0
    entry_price: float = 0.0
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0
    entry_time: Optional[datetime] = None

    def update(self, current_price: float):
        self.current_price = current_price
        if self.size != 0 and self.entry_price > 0:
            if self.size > 0:  # Long
                self.unrealized_pnl = (current_price - self.entry_price) * self.size
                self.unrealized_pnl_pct = (current_price / self.entry_price - 1) * 100
            else:  # Short (FX only)
                self.unrealized_pnl = (self.entry_price - current_price) * abs(self.size)
                self.unrealized_pnl_pct = (self.entry_price / current_price - 1) * 100


class AggressiveTrader:
    """
    超攻撃的スモールアカウントトレーダー

    5000円スタートで最速利益追求
    """

    # 5000円で取引可能なペア（最小注文額順）
    SMALL_ACCOUNT_PAIRS = [
        ("XRP_JPY", 1.0, 100),      # 1 XRP ≈ 100円
        ("MONA_JPY", 1.0, 50),      # 1 MONA ≈ 50円
        ("XLM_JPY", 1.0, 50),       # 1 XLM ≈ 50円
        ("ETH_JPY", 0.01, 5000),    # 0.01 ETH ≈ 5000円
    ]

    # 利確・損切りライン（超攻撃的設定）
    TAKE_PROFIT_PCT = 0.5    # 0.5%で即利確（高速回転）
    STOP_LOSS_PCT = -0.3     # -0.3%で即損切り（損失最小化）
    TRAILING_STOP_PCT = 0.2  # 0.2%トレーリングストップ

    # 取引頻度（超高速設定）
    MIN_INTERVAL_SECONDS = 0.5  # 500ms間隔（ミリ秒単位の高速取引）
    MAX_TRADES_PER_HOUR = 1000  # 1時間1000回まで（無制限に近い）

    def __init__(self, config: Config = None, initial_capital: float = None):
        self.config = config or get_config()
        # 初期資本は後でAPIから取得
        self.initial_capital = initial_capital or 0
        self.current_capital = self.initial_capital

        # API Client for balance check
        self.balance_client = None
        if not self.config.trading.paper_trading:
            self.balance_client = BitFlyerClient(
                api_key=self.config.bitflyer.api_key,
                api_secret=self.config.bitflyer.api_secret,
                product_code="BTC_JPY",
            )

        # アクティブペア（初期化後に更新される）
        self.active_pairs = self._select_pairs_for_capital(initial_capital or 5000)

        # クライアント
        self.clients: Dict[str, any] = {}
        self._init_clients()

        # 通知
        self.notifier = self._init_notifier()

        # 価格履歴
        self.price_history: Dict[str, PriceHistory] = {
            pair: PriceHistory() for pair, _, _ in self.SMALL_ACCOUNT_PAIRS
        }

        # ポジション
        self.positions: Dict[str, AssetPosition] = {
            pair: AssetPosition(pair=pair) for pair, _, _ in self.SMALL_ACCOUNT_PAIRS
        }

        # 統計
        self.running = False
        self.start_time = datetime.now()
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0
        self.max_capital = initial_capital
        self.trades_this_hour = 0
        self.last_hour_reset = datetime.now()
        self.last_trade_time: Dict[str, datetime] = {}

        # 取引履歴
        self.trade_history: deque = deque(maxlen=500)

        # 状態保存ファイル
        self.state_file = "logs/trader_state.json"
        self.last_state_save = datetime.now()
        self.state_save_interval = 60  # 60秒ごとに保存

        # ============================================
        # 世界最強コンポーネント初期化
        # ============================================
        # 機械学習予測エンジン（各ペアごと）
        self.ml_predictors: Dict[str, MLPredictor] = {
            pair: MLPredictor(lookback=50) for pair, _, _ in self.SMALL_ACCOUNT_PAIRS
        }

        # Kelly基準ポジションサイジング（各ペアごと）
        self.kelly_sizers: Dict[str, KellyPositionSizer] = {
            pair: KellyPositionSizer() for pair, _, _ in self.SMALL_ACCOUNT_PAIRS
        }

        # 動的パラメータ最適化
        self.dynamic_optimizer = DynamicOptimizer()

        # MLトレーニング間隔
        self.last_ml_training = datetime.now()
        self.ml_train_interval = 300  # 5分ごとにトレーニング

        # ============================================
        # WebSocketリアルタイムデータ
        # ============================================
        self.ws_client: Optional[BitFlyerWebSocket] = None
        self.order_book_analyzer: Optional[OrderBookAnalyzer] = None
        self.use_websocket = not self.config.trading.paper_trading  # 本番時のみWebSocket使用

        if self.use_websocket:
            self._init_websocket()

        # ============================================
        # チャート生成器・ターミナル表示
        # ============================================
        self.chart_generator = PortfolioChartGenerator()
        self.terminal_display = TerminalDisplay()
        self.last_chart_send = datetime.now()
        self.chart_send_interval = 3600  # 1時間ごとにチャート送信

        # ============================================
        # 外部同期（入金・出金・手動取引対応）
        # ============================================
        self.last_external_sync = datetime.now()
        self.external_sync_interval = 60  # 1分ごとに外部状態をチェック
        self.last_known_api_balance = 0
        self.last_known_positions: Dict[str, float] = {}

        # LINE通知間隔
        self.last_line_notify = datetime.now()
        self.line_notify_interval = 1800  # 30分ごとに定期通知

        # ============================================
        # ドローダウン保護（ユーザーアドバイス実装）
        # ============================================
        self.max_drawdown_pct = 5.0  # 最大ドローダウン5%
        self.peak_portfolio_value = initial_capital or 0  # ピーク値
        self.drawdown_protection_active = False
        self.drawdown_position_scale = 1.0  # ポジションスケール（0.5で50%縮小）

        # 起動時のヘッダー表示
        self.terminal_display.print_header()
        logger.info("=" * 60)
        logger.info("  🏆 ULTIMATE AI TRADER v2.0 - 世界最強・究極完成版")
        logger.info("=" * 60)
        logger.info(f"  💰 Initial Capital: ¥{initial_capital:,.0f}")
        logger.info(f"  🎯 Target: ¥{initial_capital * 3:,.0f} (3x)")
        logger.info(f"  📊 Active Pairs: {self.active_pairs}")
        logger.info("  🧠 v2.0 Features:")
        logger.info("    ├─ ML Price Prediction (Ridge + λ動的調整)")
        logger.info("    ├─ Kelly Criterion (EMA-based p/b推定)")
        logger.info("    ├─ 2D Regime Detection (Vol × Trend ADX)")
        logger.info("    ├─ Dynamic Weight Allocation (レジーム別)")
        logger.info("    ├─ Drawdown Protection (5% → 50%縮小)")
        logger.info("    ├─ Order Book Imbalance Analysis")
        logger.info("    ├─ External Sync (入金/出金/手動取引)")
        logger.info("    ├─ Portfolio Chart Generation (LINE)")
        logger.info(f"    └─ WebSocket Real-time Data: {'✓' if self.use_websocket else '✗'}")
        logger.info("=" * 60)

    def _select_pairs_for_capital(self, capital: float) -> List[str]:
        """資本に応じたペアを選択"""
        pairs = []
        for pair, min_size, approx_cost in self.SMALL_ACCOUNT_PAIRS:
            if approx_cost <= capital * 0.9:  # 資本の90%以下
                pairs.append(pair)
        return pairs if pairs else ["XRP_JPY", "MONA_JPY"]  # フォールバック

    def _init_clients(self):
        """クライアント初期化"""
        for pair, _, _ in self.SMALL_ACCOUNT_PAIRS:
            if self.config.trading.paper_trading:
                self.clients[pair] = MockBitFlyerClient(
                    initial_balance=self.initial_capital,
                    product_code=pair,
                )
            else:
                self.clients[pair] = BitFlyerClient(
                    api_key=self.config.bitflyer.api_key,
                    api_secret=self.config.bitflyer.api_secret,
                    product_code=pair,
                )

    def _init_notifier(self) -> Optional[LINEMessagingAPI]:
        """通知システム初期化"""
        if self.config.line.is_configured and self.config.line.use_messaging_api:
            return LINEMessagingAPI(
                channel_access_token=self.config.line.channel_access_token,
                user_id=self.config.line.user_id,
            )
        return None

    def _init_websocket(self):
        """WebSocketクライアント初期化"""
        try:
            self.ws_client = BitFlyerWebSocket(
                on_ticker=self._on_ws_ticker,
                on_executions=self._on_ws_executions,
            )

            # 各ペアを購読
            for pair in self.active_pairs:
                self.ws_client.subscribe_ticker(pair)
                self.ws_client.subscribe_executions(pair)
                self.ws_client.subscribe_board(pair)  # 注文板購読

            self.order_book_analyzer = OrderBookAnalyzer(self.ws_client)
            self.ws_client.connect()
            logger.info("📡 WebSocket initialized for real-time data")
        except Exception as e:
            logger.warning(f"WebSocket init failed: {e}")
            self.use_websocket = False

    def _on_ws_ticker(self, product_code: str, data: Dict):
        """WebSocketティッカー受信時（リアルタイム価格更新）"""
        try:
            if product_code in self.price_history:
                price = data.get('ltp', 0)
                volume = data.get('volume', 0)
                if price > 0:
                    self.price_history[product_code].add(price, volume)
                    self.positions[product_code].update(price)
        except Exception as e:
            pass  # サイレント失敗

    def _on_ws_executions(self, product_code: str, data: List):
        """WebSocket約定受信時"""
        # 約定データはOrderBookAnalyzerで使用
        pass

    async def _sync_external_state(self):
        """
        外部状態同期（入金・出金・手動取引対応）

        ユーザーがWebサイトから取引した場合や、
        入金・出金があった場合に状態を同期する

        重要: APIの値を常に正として扱う
        """
        if self.config.trading.paper_trading:
            return

        try:
            # APIから最新残高を取得
            api_balance, holdings = await self._fetch_balance_from_api()

            # 現金残高の変化を検出・通知
            if self.last_known_api_balance > 0:
                balance_diff = api_balance - self.last_known_api_balance

                if abs(balance_diff) > 100:  # ¥100以上の変化
                    if balance_diff > 0:
                        logger.info(f"💹 外部変化検出: +¥{balance_diff:,.0f} (入金または売却)")
                        if self.notifier:
                            self.notifier.send_text(
                                f"💹 外部変化を検出しました\n\n"
                                f"💰 残高変化: +¥{balance_diff:,.0f}\n"
                                f"📊 新残高: ¥{api_balance:,.0f}\n"
                                f"⏰ {datetime.now().strftime('%H:%M:%S')}"
                            )
                    else:
                        logger.info(f"💸 外部変化検出: ¥{balance_diff:,.0f} (出金または購入)")
                        if self.notifier:
                            self.notifier.send_text(
                                f"💸 外部変化を検出しました\n\n"
                                f"💰 残高変化: ¥{balance_diff:,.0f}\n"
                                f"📊 新残高: ¥{api_balance:,.0f}\n"
                                f"⏰ {datetime.now().strftime('%H:%M:%S')}"
                            )

            # ============================================
            # ポジションを完全にAPIから同期（重要！）
            # ============================================
            currency_to_pair = {
                'XRP': 'XRP_JPY',
                'MONA': 'MONA_JPY',
                'XLM': 'XLM_JPY',
                'ETH': 'ETH_JPY',
                'BTC': 'BTC_JPY',
            }

            # 全ペアのポジションをリセット
            for pair in self.positions:
                # API上に存在しない通貨はゼロにリセット
                currency = pair.replace('_JPY', '')
                if currency not in holdings:
                    if self.positions[pair].size > 0:
                        logger.info(f"📤 ポジションリセット: {pair} (APIに存在しない)")
                    self.positions[pair].size = 0
                    self.positions[pair].entry_price = 0

            # APIの保有量でポジションを更新
            for currency, amount in holdings.items():
                pair = currency_to_pair.get(currency)
                if pair and pair in self.positions:
                    old_size = self.positions[pair].size

                    # APIの値を正として設定
                    self.positions[pair].size = amount

                    # 現在価格を取得してエントリー価格を更新
                    if pair in self.clients and amount > 0:
                        try:
                            ticker = await self.clients[pair].get_ticker()
                            if ticker:
                                self.positions[pair].current_price = ticker.ltp
                                # エントリー価格が0の場合は現在価格で初期化
                                if self.positions[pair].entry_price == 0:
                                    self.positions[pair].entry_price = ticker.ltp
                        except:
                            pass

                    # 変化を検出してログ
                    if abs(amount - old_size) > 0.001:
                        if amount > old_size:
                            logger.info(f"📦 外部購入検出: {currency} +{amount - old_size:.4f} (合計: {amount:.4f})")
                        elif amount < old_size:
                            logger.info(f"📤 外部売却検出: {currency} -{old_size - amount:.4f} (合計: {amount:.4f})")

            # 現在の状態を保存
            self.last_known_api_balance = api_balance
            self.current_capital = api_balance  # 現金残高を同期
            self.last_known_positions = holdings.copy()

            logger.debug(f"🔄 External sync: JPY=¥{api_balance:,.0f}, holdings={len(holdings)}")

        except Exception as e:
            logger.debug(f"External sync failed: {e}")

    def _send_line_status(self, include_chart: bool = False):
        """
        LINE定期ステータス通知（日本語）
        """
        if not self.notifier:
            return

        try:
            portfolio_value = self._calculate_current_portfolio_value()
            roi = ((portfolio_value / self.initial_capital) - 1) * 100 if self.initial_capital > 0 else 0
            win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0

            # 運用時間計算
            runtime = datetime.now() - self.start_time
            hours = runtime.total_seconds() / 3600
            days = int(hours // 24)
            remaining_hours = int(hours % 24)

            # 日本語メッセージ
            status_emoji = "📈" if roi >= 0 else "📉"
            profit_emoji = "💰" if self.total_pnl >= 0 else "💸"

            message = (
                f"🏆 AI Trader 定期レポート\n"
                f"{'━' * 20}\n\n"
                f"💼 ポートフォリオ\n"
                f"   現在価値: ¥{portfolio_value:,.0f}\n"
                f"   {status_emoji} 収益率: {roi:+.2f}%\n"
                f"   {profit_emoji} 累計損益: ¥{self.total_pnl:,.0f}\n\n"
                f"📊 取引統計\n"
                f"   取引回数: {self.total_trades}回\n"
                f"   勝率: {win_rate:.1f}%\n"
                f"   勝ち: {self.winning_trades}回\n\n"
                f"⏱️ 稼働時間\n"
                f"   {days}日 {remaining_hours}時間\n\n"
                f"🧠 AI状態\n"
                f"   ML学習: {sum(1 for p in self.ml_predictors.values() if p.trained)}/{len(self.ml_predictors)}\n"
                f"   ボラ: {self.dynamic_optimizer.volatility_regime}\n\n"
                f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            )

            self.notifier.send_text(message)

            # チャート送信（対応している場合）
            if include_chart:
                chart_path = self.chart_generator.generate_chart(self.initial_capital)
                if chart_path and os.path.exists(chart_path):
                    # チャートファイルをLINEに送信（画像送信対応の場合）
                    try:
                        self.notifier.send_image(chart_path)
                        logger.info("📊 Chart sent to LINE")
                    except Exception:
                        # 画像送信非対応の場合は無視
                        pass

        except Exception as e:
            logger.warning(f"LINE notification failed: {e}")

    def _send_trade_notification(self, pair: str, side: str, size: float, price: float, pnl: float = None):
        """取引通知（日本語）"""
        if not self.notifier:
            return

        try:
            action = "買い" if side == "BUY" else "売り"
            emoji = "🟢" if side == "BUY" else "🔴"

            message = (
                f"{emoji} 取引完了\n\n"
                f"📌 {pair}\n"
                f"   {action}: {size:.4f}\n"
                f"   価格: ¥{price:,.0f}\n"
            )

            if pnl is not None:
                pnl_emoji = "💰" if pnl >= 0 else "💸"
                message += f"   {pnl_emoji} 損益: ¥{pnl:,.0f}\n"

            message += f"\n⏰ {datetime.now().strftime('%H:%M:%S')}"

            self.notifier.send_text(message)
        except Exception:
            pass

    async def _fetch_prices(self) -> Dict[str, Dict]:
        """全ペアの価格を取得"""
        tasks = []
        pairs = []
        for pair in self.active_pairs:
            if pair in self.clients:
                tasks.append(self.clients[pair].get_ticker())
                pairs.append(pair)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        prices = {}
        for pair, result in zip(pairs, results):
            if not isinstance(result, Exception) and result:
                prices[pair] = {
                    'price': result.ltp,
                    'bid': result.best_bid,
                    'ask': result.best_ask,
                    'spread': result.spread,
                    'volume': result.volume,
                }
                # 価格履歴更新
                self.price_history[pair].add(result.ltp, result.volume)
                # ポジション更新
                self.positions[pair].update(result.ltp)

        return prices

    def _generate_signal(self, pair: str, prices: Dict) -> Tuple[int, float, str]:
        """
        シグナル生成（世界最強 - ML + テクニカル + 動的最適化）

        Returns: (action, confidence, reason)
            action: 0=SELL, 1=HOLD, 2=BUY
        """
        history = self.price_history.get(pair)
        if not history or len(history.prices) < 10:
            return 1, 0.0, "insufficient_data"

        price_data = prices.get(pair, {})
        if not price_data:
            return 1, 0.0, "no_price"

        current_price = price_data['price']
        spread_pct = price_data['spread'] / current_price if current_price > 0 else 1

        # スプレッドが広すぎる場合はスキップ
        if spread_pct > 0.005:  # 0.5%以上
            return 1, 0.0, "wide_spread"

        # 現金不足チェック - 買いは禁止、売りのみ許可
        min_size = 1.0
        for p, ms, _ in self.SMALL_ACCOUNT_PAIRS:
            if p == pair:
                min_size = ms
                break
        min_order_cost = min_size * current_price
        can_buy = self.current_capital >= min_order_cost

        # ============================================
        # 動的パラメータ最適化 + 2Dレジーム検出
        # ============================================
        price_list = list(history.prices)
        params = self.dynamic_optimizer.optimize(price_list)
        dynamic_take_profit = params['take_profit']
        dynamic_stop_loss = params['stop_loss']
        dynamic_confidence_threshold = params['confidence_threshold']
        dynamic_momentum_threshold = params['momentum_threshold']

        # 動的ウェイト取得（2Dレジーム基づく）
        strategy_weights = self.dynamic_optimizer.get_strategy_weights()
        ml_weight = strategy_weights.get('ml', 0.5)
        momentum_weight = strategy_weights.get('momentum', 0.5)
        trend_weight = strategy_weights.get('trend', 0.5)
        # mean_reversion_weight = strategy_weights.get('mean_reversion', 0.5)

        momentum = history.momentum
        volatility = history.volatility
        trend = history.trend

        action = 1  # HOLD
        confidence = 0.0
        reasons = []

        # ポジション取得（早期に取得）
        position = self.positions.get(pair)

        # ============================================
        # 機械学習予測シグナル（動的ウェイト適用）
        # ============================================
        ml_predictor = self.ml_predictors.get(pair)
        ml_direction = 0
        ml_confidence = 0.0
        if ml_predictor and ml_predictor.trained:
            ml_direction, ml_confidence = ml_predictor.predict(price_list)
            # 動的ウェイト適用（レジームに基づく）
            weighted_ml_confidence = ml_confidence * ml_weight
            if ml_direction == 1:
                action = 2  # BUY
                confidence += weighted_ml_confidence
                reasons.append(f"ml_buy({ml_confidence:.2f}×{ml_weight:.1f})")
            elif ml_direction == -1:
                action = 0  # SELL
                confidence += weighted_ml_confidence
                reasons.append(f"ml_sell({ml_confidence:.2f}×{ml_weight:.1f})")

        # ============================================
        # 注文板インバランス分析（WebSocket使用時）
        # ============================================
        order_imbalance = 0.0
        market_pressure = 0.0
        if self.order_book_analyzer and self.use_websocket:
            try:
                order_imbalance = self.order_book_analyzer.get_order_book_imbalance(pair, depth=10)
                pressure = self.order_book_analyzer.get_market_pressure(pair)
                market_pressure = pressure.get('net_pressure', 0)

                # インバランスが大きい場合はシグナル強化
                if order_imbalance > 0.3:  # 買い優勢
                    confidence += 0.15
                    reasons.append(f"order_imbalance({order_imbalance:.2f})")
                elif order_imbalance < -0.3:  # 売り優勢
                    confidence += 0.15
                    reasons.append(f"sell_pressure({order_imbalance:.2f})")

                # 市場圧力
                if market_pressure > 0.3:
                    confidence += 0.1
                    reasons.append("buy_flow")
                elif market_pressure < -0.3:
                    confidence += 0.1
                    reasons.append("sell_flow")
            except Exception:
                pass

        # === 超攻撃的シグナル生成（世界最強設定 + 動的ウェイト） ===

        # 1. モメンタムシグナル（動的閾値 + 動的ウェイト）
        if momentum > dynamic_momentum_threshold:  # 動的閾値で上昇判定
            if action != 0:  # MLがSELLでない場合
                action = 2  # BUY
            confidence += 0.35 * momentum_weight  # 動的ウェイト適用
            reasons.append(f"momentum_up×{momentum_weight:.1f}")
        elif momentum < -dynamic_momentum_threshold:  # 動的閾値で下落判定
            if action != 2:  # MLがBUYでない場合
                action = 0  # SELL
            confidence += 0.35 * momentum_weight  # 動的ウェイト適用
            reasons.append(f"momentum_down×{momentum_weight:.1f}")

        # 2. トレンドフォロー（強化 + 動的ウェイト）
        if trend == 1:
            if action == 2:
                confidence += 0.3 * trend_weight  # 動的ウェイト適用
            elif action == 1:  # HOLDでもトレンド中は買い
                action = 2
                confidence += 0.25 * trend_weight  # 動的ウェイト適用
            reasons.append(f"uptrend×{trend_weight:.1f}")
        elif trend == -1:
            if action == 0:
                confidence += 0.3 * trend_weight  # 動的ウェイト適用
            reasons.append(f"downtrend×{trend_weight:.1f}")

        # 3. ボラティリティボーナス（低閾値）
        if volatility > 0.003:  # 低ボラでも反応
            confidence += 0.25
            reasons.append("volatility_opportunity")

        # 4. ゴールデンクロス/デッドクロス（高感度）
        sma_5 = history.sma_5
        sma_20 = history.sma_20
        if sma_5 > 0 and sma_20 > 0:
            if sma_5 > sma_20 * 1.001 and action != 0:  # 0.1%差で反応
                action = 2
                confidence += 0.2
                reasons.append("golden_cross")
            elif sma_5 < sma_20 * 0.999 and action != 2:
                action = 0
                confidence += 0.2
                reasons.append("dead_cross")

        # 5. 価格の急変検出（超高感度）
        if len(history.prices) >= 3:
            last_3 = list(history.prices)[-3:]
            quick_change = (last_3[-1] - last_3[0]) / last_3[0]
            if quick_change > 0.002:  # 0.2%急騰で即買い
                action = 2
                confidence += 0.3
                reasons.append("quick_pump")
            elif quick_change < -0.002:  # 0.2%急落で即売り
                action = 0
                confidence += 0.3
                reasons.append("quick_dump")

        # 6. スキャルピングボーナス（ポジションなしで買いやすく）
        if action == 1 and (not position or position.size == 0):
            # ポジションがなく、わずかでも上昇傾向なら買い
            if momentum > 0.0005:
                action = 2
                confidence += 0.2
                reasons.append("scalp_entry")

        # ポジションチェック（利確・損切り - 動的パラメータ使用）
        if position and position.size != 0:
            # 利確チェック（動的閾値）
            if position.unrealized_pnl_pct >= dynamic_take_profit:
                action = 0 if position.size > 0 else 2
                confidence = 0.95
                reasons = [f"take_profit({dynamic_take_profit:.1f}%)"]
            # 損切りチェック（動的閾値）
            elif position.unrealized_pnl_pct <= -dynamic_stop_loss:
                action = 0 if position.size > 0 else 2
                confidence = 0.9
                reasons = [f"stop_loss({dynamic_stop_loss:.1f}%)"]

        # 現金不足時はBUYをブロック
        if action == 2 and not can_buy:
            # ポジションがあれば売りを検討
            if position and position.size > 0:
                action = 0  # SELL
                reasons = ["low_cash_sell"]
            else:
                action = 1  # HOLD
                reasons = ["insufficient_cash"]
                confidence = 0

        return action, min(confidence, 1.0), ",".join(reasons)

    def _check_drawdown_protection(self) -> bool:
        """
        ドローダウン保護チェック（ユーザーアドバイス実装）

        ポートフォリオがピークから5%下落したらポジションを50%縮小
        Returns: True if protection is active
        """
        current_value = self._calculate_current_portfolio_value()

        # ピーク更新
        if current_value > self.peak_portfolio_value:
            self.peak_portfolio_value = current_value
            self.drawdown_protection_active = False
            self.drawdown_position_scale = 1.0
            return False

        # ドローダウン計算
        if self.peak_portfolio_value > 0:
            drawdown_pct = ((self.peak_portfolio_value - current_value) / self.peak_portfolio_value) * 100

            if drawdown_pct >= self.max_drawdown_pct:
                if not self.drawdown_protection_active:
                    # 保護発動！
                    self.drawdown_protection_active = True
                    self.drawdown_position_scale = 0.5  # ポジション50%縮小
                    logger.warning(f"⚠️ DRAWDOWN PROTECTION ACTIVATED! DD={drawdown_pct:.1f}%")
                    logger.warning(f"  Peak: ¥{self.peak_portfolio_value:,.0f} → Current: ¥{current_value:,.0f}")

                    # LINE通知
                    if self.notifier:
                        self.notifier.send_text(
                            f"⚠️ ドローダウン保護発動\n\n"
                            f"📉 下落率: {drawdown_pct:.1f}%\n"
                            f"💰 ピーク: ¥{self.peak_portfolio_value:,.0f}\n"
                            f"💴 現在: ¥{current_value:,.0f}\n\n"
                            f"🛡️ ポジションを50%に縮小\n"
                            f"⏰ {datetime.now().strftime('%H:%M:%S')}"
                        )
                return True
            elif drawdown_pct < self.max_drawdown_pct * 0.5:
                # 回復時（ドローダウンが半分以下になったら）
                if self.drawdown_protection_active:
                    self.drawdown_protection_active = False
                    self.drawdown_position_scale = 1.0
                    logger.info(f"✅ Drawdown protection lifted. DD recovered to {drawdown_pct:.1f}%")

        return self.drawdown_protection_active

    def _calculate_order_size(self, pair: str, price: float, is_buy: bool, confidence: float = 0.5) -> float:
        """
        注文サイズを計算（Kelly基準 + ドローダウン保護）

        Kelly Criterion: f* = (p*b - q) / b
        where p = win probability, q = 1-p, b = win/loss ratio
        """
        # ペアの最小サイズを取得
        min_size = 1.0
        for p, ms, _ in self.SMALL_ACCOUNT_PAIRS:
            if p == pair:
                min_size = ms
                break

        if is_buy:
            # 最小注文金額を計算
            min_order_cost = min_size * price

            # 現金が最小注文金額未満なら買えない
            if self.current_capital < min_order_cost:
                return 0  # 資金不足

            # ============================================
            # Kelly基準によるポジションサイジング
            # ============================================
            kelly_sizer = self.kelly_sizers.get(pair)
            if kelly_sizer:
                kelly_fraction = kelly_sizer.get_kelly_fraction()
            else:
                kelly_fraction = 0.3  # デフォルト30%

            # 信頼度でKelly分数を調整（高信頼度 = 大きいポジション）
            adjusted_kelly = kelly_fraction * (0.5 + confidence * 0.5)  # 0.5〜1.0倍

            # ============================================
            # ドローダウン保護によるスケーリング
            # ============================================
            adjusted_kelly *= self.drawdown_position_scale

            # 資本のKelly分数を使用
            available = self.current_capital * adjusted_kelly

            # 最低でも最小注文金額は確保（保護発動中でも最小は維持）
            if not self.drawdown_protection_active:
                available = max(available, min_order_cost)

            # 全資金を使用可能（制限なし）
            available = min(available, self.current_capital)

            max_size = available / price if price > 0 else 0

            # 最小サイズ以上かチェック
            if max_size < min_size:
                return 0  # 資金不足

            size = max_size

            if self.drawdown_protection_active:
                logger.debug(f"Kelly sizing (DD protected): {pair} kelly={kelly_fraction:.2f} scale={self.drawdown_position_scale:.1f} size={size:.4f}")
            else:
                logger.debug(f"Kelly sizing: {pair} kelly={kelly_fraction:.2f} adj={adjusted_kelly:.2f} size={size:.4f}")
        else:
            # 売り: 保有ポジションのみ
            position = self.positions.get(pair)
            if not position or position.size <= 0:
                return 0  # ポジションなし
            size = position.size

        return round(size, 4)

    async def _execute_trade(
        self,
        pair: str,
        action: int,
        confidence: float,
        reason: str,
        price: float,
    ) -> bool:
        """取引実行"""
        # 取引間隔チェック
        last_trade = self.last_trade_time.get(pair)
        if last_trade:
            elapsed = (datetime.now() - last_trade).total_seconds()
            if elapsed < self.MIN_INTERVAL_SECONDS:
                return False

        # 1時間あたりの取引数チェック
        if (datetime.now() - self.last_hour_reset).seconds >= 3600:
            self.trades_this_hour = 0
            self.last_hour_reset = datetime.now()

        if self.trades_this_hour >= self.MAX_TRADES_PER_HOUR:
            return False

        side = OrderSide.BUY if action == 2 else OrderSide.SELL
        is_buy = (action == 2)
        size = self._calculate_order_size(pair, price, is_buy, confidence)

        # サイズが0なら取引しない
        if size <= 0:
            return False

        client = self.clients.get(pair)
        if not client:
            return False

        try:
            order_id = await client.send_order(
                side=side,
                size=size,
                order_type=OrderType.MARKET,
            )

            if order_id:
                self.last_trade_time[pair] = datetime.now()
                self.trades_this_hour += 1
                self.total_trades += 1

                # ============================================
                # ポジション更新（APIと同期するため最小限に）
                # ============================================
                position = self.positions[pair]
                cost = price * size

                if side == OrderSide.BUY:
                    # 買い: 資本から購入コストを引く（概算）
                    # 注: 実際の残高は外部同期で更新される
                    self.current_capital -= cost
                    # エントリー価格を記録（外部同期でサイズは更新される）
                    if position.entry_price == 0:
                        position.entry_price = price
                    position.entry_time = datetime.now()
                    logger.info(f"  💰 Estimated Available: ¥{self.current_capital:,.0f}")
                else:
                    # 売り: 売却収入を資本に加算（概算）
                    self.current_capital += cost
                    # 決済PnL計算
                    if position.size > 0 and position.entry_price > 0:
                        pnl = (price - position.entry_price) * min(size, position.size)
                        self.total_pnl += pnl
                        if pnl > 0:
                            self.winning_trades += 1
                        logger.info(f"  💵 PnL: ¥{pnl:,.0f}")

                        # Kelly sizerに結果を記録
                        kelly_sizer = self.kelly_sizers.get(pair)
                        if kelly_sizer:
                            kelly_sizer.update(pnl)

                # 注: position.sizeは外部同期(_sync_external_state)で
                # APIから取得した正確な値に更新される
                # ローカルでの加減算は行わない（二重カウント防止）

                # 最大資本更新
                if self.current_capital > self.max_capital:
                    self.max_capital = self.current_capital

                logger.info(
                    f"✅ [{pair}] {side.value} {size} @ ¥{price:,.1f} "
                    f"(conf={confidence:.2f}, {reason})"
                )

                # 取引後に状態を保存
                self._save_state()

                # チャート用に取引を記録
                trade_pnl = 0
                if side == OrderSide.SELL and position.entry_price > 0:
                    trade_pnl = (price - position.entry_price) * size
                self.chart_generator.add_trade(side.value, trade_pnl)

                # 通知（日本語・10回ごと）
                if self.notifier and self.total_trades % 10 == 0:
                    self._send_trade_notification(pair, side.value, size, price, trade_pnl if trade_pnl != 0 else None)

                # 取引後に即座に外部同期をスケジュール（次のループで実行）
                self.last_external_sync = datetime.now() - timedelta(seconds=self.external_sync_interval + 1)

                return True

        except Exception as e:
            logger.error(f"Trade failed: {e}")

        return False

    async def _trading_loop(self):
        """メイン取引ループ（超高速モード）"""
        logger.info("🚀 ULTRA AGGRESSIVE Trading Loop Started - 世界最強モード")

        tick = 0
        rate_limit_backoff = 0  # レート制限バックオフ（秒）
        poll_interval = 1.0  # 高速ポーリング（1秒間隔）

        while self.running:
            try:
                tick += 1

                # レート制限バックオフ中は待機
                if rate_limit_backoff > 0:
                    logger.warning(f"⏳ Rate limit backoff: {rate_limit_backoff}s")
                    await asyncio.sleep(rate_limit_backoff)
                    rate_limit_backoff = max(0, rate_limit_backoff - 5)

                # 価格取得（順次取得でレート制限回避）
                prices = {}

                # ペア別の現実的な価格範囲（異常値検出用）
                PRICE_RANGES = {
                    "BTC_JPY": (1000000, 20000000),   # ¥1M - ¥20M
                    "ETH_JPY": (50000, 1000000),      # ¥50K - ¥1M
                    "XRP_JPY": (10, 500),             # ¥10 - ¥500
                    "XLM_JPY": (5, 200),              # ¥5 - ¥200
                    "MONA_JPY": (10, 500),            # ¥10 - ¥500
                }

                for pair in self.active_pairs:
                    try:
                        if pair in self.clients:
                            ticker = await self.clients[pair].get_ticker()
                            if ticker:
                                price = ticker.ltp

                                # 価格妥当性チェック
                                price_range = PRICE_RANGES.get(pair, (1, 100000000))
                                if not (price_range[0] <= price <= price_range[1]):
                                    logger.warning(f"⚠️ Abnormal price detected: {pair} = ¥{price:,.0f} (expected: ¥{price_range[0]:,.0f} - ¥{price_range[1]:,.0f})")
                                    # 異常価格は無視
                                    continue

                                prices[pair] = {
                                    'price': price,
                                    'bid': ticker.best_bid,
                                    'ask': ticker.best_ask,
                                    'spread': ticker.spread,
                                    'volume': ticker.volume,
                                }
                                self.price_history[pair].add(price, ticker.volume)
                                self.positions[pair].update(price)
                        await asyncio.sleep(0.15)  # 超高速API間隔（150ms）
                    except Exception as e:
                        if "429" in str(e) or "rate" in str(e).lower():
                            rate_limit_backoff = min(rate_limit_backoff + 5, 30)
                            logger.warning(f"Rate limit hit, backing off {rate_limit_backoff}s")
                            break

                if not prices:
                    await asyncio.sleep(poll_interval)
                    continue

                # ============================================
                # ドローダウン保護チェック（毎tick）
                # ============================================
                self._check_drawdown_protection()

                # 各ペアでシグナル生成・取引（超攻撃的）
                for pair in self.active_pairs:
                    if pair not in prices:
                        continue

                    action, confidence, reason = self._generate_signal(pair, prices)

                    # 信頼度閾値（超攻撃的: 0.25）- わずかなチャンスも逃さない
                    if action != 1 and confidence >= 0.25:
                        price = prices[pair]['price']
                        await self._execute_trade(pair, action, confidence, reason, price)

                # ============================================
                # 定期ステータス・チャート記録（30秒ごと）
                # ============================================
                if tick % 30 == 0:
                    self._log_status()

                    # ポートフォリオ価値をチャート用に記録
                    portfolio_value = self._calculate_current_portfolio_value()
                    self.chart_generator.add_portfolio_value(portfolio_value)

                    # 美しいターミナル表示
                    status_data = self._get_terminal_status_data()
                    self.terminal_display.print_status(status_data)

                # ============================================
                # 定期MLトレーニング（5分ごと）
                # ============================================
                if (datetime.now() - self.last_ml_training).seconds >= self.ml_train_interval:
                    for pair in self.active_pairs:
                        history = self.price_history.get(pair)
                        if history and len(history.prices) >= 50:
                            ml_predictor = self.ml_predictors.get(pair)
                            if ml_predictor:
                                ml_predictor.train(list(history.prices))
                                if ml_predictor.trained:
                                    logger.info(f"🧠 ML trained: {pair} (count={ml_predictor.training_count})")
                    self.last_ml_training = datetime.now()

                # ============================================
                # 外部状態同期（1分ごと - 入金・出金・手動取引対応）
                # ============================================
                if (datetime.now() - self.last_external_sync).seconds >= self.external_sync_interval:
                    await self._sync_external_state()
                    self.last_external_sync = datetime.now()

                # ============================================
                # LINE定期通知（30分ごと）
                # ============================================
                if (datetime.now() - self.last_line_notify).seconds >= self.line_notify_interval:
                    self._send_line_status(include_chart=False)
                    self.last_line_notify = datetime.now()

                # ============================================
                # チャート付きLINE通知（1時間ごと）
                # ============================================
                if (datetime.now() - self.last_chart_send).seconds >= self.chart_send_interval:
                    self._send_line_status(include_chart=True)
                    self.last_chart_send = datetime.now()

                # 定期状態保存（60秒ごと）
                if (datetime.now() - self.last_state_save).seconds >= self.state_save_interval:
                    self._save_state()
                    self.last_state_save = datetime.now()

                # 目標達成チェック
                if self.current_capital >= self.initial_capital * 3:
                    logger.info(f"🎉 TARGET ACHIEVED! Capital: ¥{self.current_capital:,.0f}")
                    if self.notifier:
                        # 目標達成時はチャート付き通知
                        self._send_line_status(include_chart=True)
                        self.notifier.send_text(
                            f"🎉🎉🎉 目標達成！🎉🎉🎉\n\n"
                            f"💰 資本: ¥{self.current_capital:,.0f}\n"
                            f"📈 利益: ¥{self.total_pnl:,.0f}\n"
                            f"🔄 取引数: {self.total_trades}\n\n"
                            f"おめでとうございます！\n"
                            f"目標の3倍を達成しました！"
                        )

                # ループ間隔（2秒 - レート制限対応）
                await asyncio.sleep(poll_interval)

            except Exception as e:
                if "429" in str(e):
                    rate_limit_backoff = min(rate_limit_backoff + 15, 60)
                    logger.warning(f"Rate limit in loop, backing off {rate_limit_backoff}s")
                else:
                    logger.error(f"Loop error: {e}")
                await asyncio.sleep(3)

    def _calculate_current_portfolio_value(self) -> float:
        """現在のポートフォリオ価値を計算（現金 + ポジションの時価）"""
        total = self.current_capital

        for pair, pos in self.positions.items():
            if pos.size > 0 and pos.current_price > 0:
                total += pos.size * pos.current_price

        return total

    def _get_terminal_status_data(self) -> Dict:
        """ターミナル表示用のステータスデータ"""
        portfolio_value = self._calculate_current_portfolio_value()
        position_value = portfolio_value - self.current_capital

        positions_data = {}
        for pair in self.active_pairs:
            pos = self.positions.get(pair)
            if pos and pos.size > 0:
                kelly = self.kelly_sizers.get(pair)
                positions_data[pair] = {
                    'value': pos.size * pos.current_price,
                    'pnl_pct': pos.unrealized_pnl_pct,
                    'kelly': kelly.get_kelly_fraction() if kelly else 0.3,
                }

        return {
            'portfolio_value': portfolio_value,
            'initial_capital': self.initial_capital,
            'cash': self.current_capital,
            'crypto_value': position_value,
            'total_trades': self.total_trades,
            'win_rate': (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0,
            'total_pnl': self.total_pnl,
            'ml_trained': sum(1 for p in self.ml_predictors.values() if p.trained),
            'ml_total': len(self.ml_predictors),
            'volatility_regime': self.dynamic_optimizer.volatility_regime,
            'trend_regime': self.dynamic_optimizer.trend_regime,
            'combined_regime': self.dynamic_optimizer.combined_regime,
            'drawdown_active': self.drawdown_protection_active,
            'positions': positions_data,
        }

    def _log_status(self):
        """ステータスログ（世界最強情報表示）"""
        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0

        # 全ポートフォリオ価値を計算
        portfolio_value = self._calculate_current_portfolio_value()
        roi = ((portfolio_value / self.initial_capital) - 1) * 100

        # ポジション価値
        position_value = portfolio_value - self.current_capital

        logger.info(
            f"📊 Portfolio: ¥{portfolio_value:,.0f} ({roi:+.1f}%) | "
            f"Cash: ¥{self.current_capital:,.0f} | "
            f"Crypto: ¥{position_value:,.0f}"
        )
        logger.info(
            f"   Trades: {self.total_trades} | "
            f"Win: {win_rate:.0f}% | "
            f"PnL: ¥{self.total_pnl:,.0f}"
        )

        # ML/Kelly状態
        ml_trained = sum(1 for p in self.ml_predictors.values() if p.trained)
        total_kelly_trades = sum(k.win_count + k.loss_count for k in self.kelly_sizers.values())
        vol_regime = self.dynamic_optimizer.volatility_regime
        logger.info(
            f"   🧠 ML: {ml_trained}/{len(self.ml_predictors)} trained | "
            f"Kelly trades: {total_kelly_trades} | "
            f"Volatility: {vol_regime}"
        )

        for pair in self.active_pairs[:4]:
            pos = self.positions.get(pair)
            if pos and pos.size > 0:
                value = pos.size * pos.current_price
                kelly = self.kelly_sizers.get(pair)
                kelly_f = kelly.get_kelly_fraction() if kelly else 0.3
                logger.info(
                    f"  {pair}: {pos.size:.4f} @ ¥{pos.current_price:,.0f} "
                    f"= ¥{value:,.0f} ({pos.unrealized_pnl_pct:+.2f}%) [K={kelly_f:.2f}]"
                )

    async def _fetch_balance_from_api(self) -> Tuple[float, Dict[str, float]]:
        """APIから残高と保有コインを取得"""
        jpy_balance = 0
        crypto_holdings = {}

        if self.balance_client:
            try:
                balances = await self.balance_client.get_balance()
                if balances and not isinstance(balances, dict):
                    for balance in balances:
                        currency = balance.get('currency_code', '')
                        available = float(balance.get('available', 0))

                        if currency == 'JPY':
                            jpy_balance = available
                            logger.info(f"💰 JPY Balance: ¥{jpy_balance:,.0f}")
                        elif available > 0:
                            crypto_holdings[currency] = available
                            logger.info(f"🪙 {currency}: {available:.4f}")

                elif isinstance(balances, dict) and 'error' not in balances:
                    for balance in balances if isinstance(balances, list) else [balances]:
                        currency = balance.get('currency_code', '')
                        available = float(balance.get('available', 0))
                        if currency == 'JPY':
                            jpy_balance = available
                        elif available > 0:
                            crypto_holdings[currency] = available

            except Exception as e:
                logger.warning(f"Failed to fetch balance from API: {e}")

        return jpy_balance, crypto_holdings

    async def _calculate_total_portfolio_value(self, jpy_balance: float, holdings: Dict[str, float]) -> float:
        """全資産の合計価値（JPY換算）を計算"""
        total_value = jpy_balance

        currency_to_pair = {
            'XRP': 'XRP_JPY',
            'MONA': 'MONA_JPY',
            'XLM': 'XLM_JPY',
            'ETH': 'ETH_JPY',
            'BTC': 'BTC_JPY',
        }

        for currency, amount in holdings.items():
            pair = currency_to_pair.get(currency)
            if pair and pair in self.clients:
                try:
                    ticker = await self.clients[pair].get_ticker()
                    if ticker:
                        value = amount * ticker.ltp
                        total_value += value
                        logger.info(f"  {currency}: {amount:.4f} × ¥{ticker.ltp:,.0f} = ¥{value:,.0f}")
                except Exception:
                    pass

        return total_value

    def _save_state(self):
        """状態をファイルに保存（再起動後も継続可能）"""
        try:
            state = {
                'saved_at': datetime.now().isoformat(),
                'initial_capital': self.initial_capital,
                'current_capital': self.current_capital,
                'max_capital': self.max_capital,
                'total_trades': self.total_trades,
                'winning_trades': self.winning_trades,
                'total_pnl': self.total_pnl,
                'start_time': self.start_time.isoformat() if self.start_time else None,
                'positions': {
                    pair: {
                        'size': pos.size,
                        'entry_price': pos.entry_price,
                        'entry_time': pos.entry_time.isoformat() if pos.entry_time else None,
                    }
                    for pair, pos in self.positions.items() if pos.size > 0
                },
            }

            os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)

            logger.debug(f"💾 State saved: {len(state['positions'])} positions")
        except Exception as e:
            logger.warning(f"Failed to save state: {e}")

    def _load_state(self) -> bool:
        """保存された状態を読み込み"""
        try:
            if not os.path.exists(self.state_file):
                logger.info("📂 No saved state found, starting fresh")
                return False

            with open(self.state_file, 'r') as f:
                state = json.load(f)

            # 状態が古すぎる場合はスキップ（24時間以上前）
            saved_at = datetime.fromisoformat(state['saved_at'])
            age_hours = (datetime.now() - saved_at).total_seconds() / 3600
            if age_hours > 24:
                logger.warning(f"⚠️ Saved state is {age_hours:.1f} hours old, ignoring")
                return False

            # 統計情報を復元
            self.total_trades = state.get('total_trades', 0)
            self.winning_trades = state.get('winning_trades', 0)
            self.total_pnl = state.get('total_pnl', 0.0)

            # ポジションを復元
            saved_positions = state.get('positions', {})
            for pair, pos_data in saved_positions.items():
                if pair in self.positions:
                    self.positions[pair].size = pos_data.get('size', 0)
                    self.positions[pair].entry_price = pos_data.get('entry_price', 0)
                    entry_time = pos_data.get('entry_time')
                    if entry_time:
                        self.positions[pair].entry_time = datetime.fromisoformat(entry_time)

            logger.info(f"📥 State loaded from {saved_at.strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info(f"   Trades: {self.total_trades}, PnL: ¥{self.total_pnl:,.0f}")
            logger.info(f"   Positions: {len(saved_positions)}")

            return True
        except Exception as e:
            logger.warning(f"Failed to load state: {e}")
            return False

    async def _init_positions_from_holdings(self, holdings: Dict[str, float]):
        """既存の保有コインからポジションを初期化"""
        # 通貨コードとペアのマッピング
        currency_to_pair = {
            'XRP': 'XRP_JPY',
            'MONA': 'MONA_JPY',
            'XLM': 'XLM_JPY',
            'ETH': 'ETH_JPY',
            'BTC': 'BTC_JPY',
        }

        for currency, amount in holdings.items():
            pair = currency_to_pair.get(currency)
            if pair and pair in self.positions:
                # 現在価格を取得して平均取得単価として使用
                try:
                    if pair in self.clients:
                        ticker = await self.clients[pair].get_ticker()
                        if ticker:
                            self.positions[pair].size = amount
                            self.positions[pair].entry_price = ticker.ltp
                            self.positions[pair].current_price = ticker.ltp
                            logger.info(f"📦 Position loaded: {pair} = {amount:.2f} @ ¥{ticker.ltp:,.0f}")
                except Exception as e:
                    logger.warning(f"Failed to get price for {pair}: {e}")

    async def start(self):
        """トレーダー開始"""
        if self.running:
            return

        self.running = True
        self.start_time = datetime.now()

        # 保存された状態を読み込み
        state_loaded = self._load_state()
        if state_loaded:
            logger.info("📥 Resuming from saved state...")

        # APIから残高と保有コインを取得
        if not self.config.trading.paper_trading:
            logger.info("📡 Fetching balance and holdings from bitFlyer API...")
            api_balance, holdings = await self._fetch_balance_from_api()

            # 保有コインをポジションに反映
            if holdings:
                await self._init_positions_from_holdings(holdings)
                logger.info(f"📦 Loaded {len(holdings)} existing positions")

            if self.initial_capital == 0:
                if api_balance > 0 or holdings:
                    # 全資産の合計価値を計算（JPY + 保有コインのJPY換算額）
                    logger.info("📊 Calculating total portfolio value...")
                    total_portfolio = await self._calculate_total_portfolio_value(api_balance, holdings)

                    self.initial_capital = total_portfolio
                    self.current_capital = api_balance  # 取引可能な現金のみ
                    self.max_capital = total_portfolio

                    # 保有コインの価値を記録
                    crypto_value = total_portfolio - api_balance

                    logger.info("=" * 50)
                    logger.info("💼 PORTFOLIO SUMMARY")
                    logger.info("=" * 50)
                    logger.info(f"  💴 JPY Balance:    ¥{api_balance:,.0f}")
                    logger.info(f"  🪙 Crypto Value:   ¥{crypto_value:,.0f}")
                    logger.info(f"  💰 TOTAL VALUE:    ¥{total_portfolio:,.0f}")
                    logger.info("=" * 50)

                    # アクティブペアを全ポートフォリオ価値で選択
                    self.active_pairs = self._select_pairs_for_capital(total_portfolio)
                else:
                    logger.warning("⚠️ Could not fetch balance, using default 5000")
                    self.initial_capital = 5000
                    self.current_capital = 5000
                    self.max_capital = 5000

        # Paper mode default
        if self.initial_capital == 0:
            self.initial_capital = 5000
            self.current_capital = 5000
            self.max_capital = 5000

        logger.info("=" * 60)
        logger.info("  AGGRESSIVE SMALL ACCOUNT TRADER")
        logger.info("  超攻撃的スモールアカウントトレーダー")
        logger.info("=" * 60)
        logger.info(f"  Initial: ¥{self.initial_capital:,.0f}")
        logger.info(f"  Target:  ¥{self.initial_capital * 3:,.0f} (3x in 1 month)")
        logger.info(f"  Pairs:   {', '.join(self.active_pairs)}")
        logger.info(f"  Mode:    {'Paper' if self.config.trading.paper_trading else 'LIVE'}")
        logger.info("=" * 60)

        # 開始通知
        if self.notifier:
            self.notifier.send_text(
                f"🚀 Aggressive Trader Started\n\n"
                f"💰 Initial: ¥{self.initial_capital:,.0f}\n"
                f"🎯 Target: ¥{self.initial_capital * 3:,.0f}\n"
                f"📊 Pairs: {', '.join(self.active_pairs)}\n"
                f"⚡ Mode: {'Paper' if self.config.trading.paper_trading else 'LIVE'}"
            )

        await self._trading_loop()

    async def stop(self):
        """トレーダー停止"""
        logger.info("Stopping Aggressive Trader...")
        self.running = False

        # WebSocket切断
        if self.ws_client:
            self.ws_client.disconnect()
            logger.info("📡 WebSocket disconnected")

        # 停止時に状態を保存
        self._save_state()
        logger.info("💾 State saved for resume on restart")

        # 全ポジション決済
        for pair, position in self.positions.items():
            if position.size > 0:
                try:
                    client = self.clients.get(pair)
                    if client:
                        await client.send_order(
                            side=OrderSide.SELL,
                            size=position.size,
                            order_type=OrderType.MARKET,
                        )
                except Exception:
                    pass

        # 最終レポート
        running_hours = (datetime.now() - self.start_time).total_seconds() / 3600
        roi = ((self.current_capital / self.initial_capital) - 1) * 100
        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0

        logger.info("=" * 60)
        logger.info("  FINAL REPORT")
        logger.info("=" * 60)
        logger.info(f"  Runtime: {running_hours:.1f} hours")
        logger.info(f"  Capital: ¥{self.initial_capital:,.0f} → ¥{self.current_capital:,.0f}")
        logger.info(f"  ROI: {roi:+.1f}%")
        logger.info(f"  Trades: {self.total_trades}")
        logger.info(f"  Win Rate: {win_rate:.1f}%")
        logger.info(f"  Total PnL: ¥{self.total_pnl:,.0f}")
        logger.info("=" * 60)

        if self.notifier:
            self.notifier.send_text(
                f"🛑 Trader Stopped\n\n"
                f"📊 Final Report:\n"
                f"💰 ¥{self.initial_capital:,.0f} → ¥{self.current_capital:,.0f}\n"
                f"📈 ROI: {roi:+.1f}%\n"
                f"🔄 Trades: {self.total_trades}\n"
                f"✅ Win Rate: {win_rate:.1f}%"
            )

    def get_status(self) -> Dict:
        """ステータス取得"""
        portfolio_value = self._calculate_current_portfolio_value()
        position_value = portfolio_value - self.current_capital

        return {
            'running': self.running,
            'portfolio_value': portfolio_value,
            'cash': self.current_capital,
            'crypto_value': position_value,
            'initial': self.initial_capital,
            'pnl': self.total_pnl,
            'roi_pct': ((portfolio_value / self.initial_capital) - 1) * 100 if self.initial_capital > 0 else 0,
            'trades': self.total_trades,
            'win_rate': (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0,
            'positions': {
                pair: {
                    'size': pos.size,
                    'entry': pos.entry_price,
                    'current': pos.current_price,
                    'value': pos.size * pos.current_price,
                    'pnl_pct': pos.unrealized_pnl_pct,
                }
                for pair, pos in self.positions.items() if pos.size > 0
            },
        }


async def run_aggressive_trader(initial_capital: float = 5000):
    """攻撃的トレーダー実行"""
    config = get_config()
    trader = AggressiveTrader(config, initial_capital)

    try:
        await trader.start()
    except KeyboardInterrupt:
        await trader.stop()


if __name__ == "__main__":
    asyncio.run(run_aggressive_trader(5000))

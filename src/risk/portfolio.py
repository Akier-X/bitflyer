"""
Portfolio Optimizer
====================
ポートフォリオ最適化
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from datetime import datetime
from loguru import logger


class PortfolioOptimizer:
    """
    ポートフォリオ最適化システム

    特徴:
    - Mean-Variance Optimization
    - リスクパリティ
    - 動的リバランス
    - 相関管理
    """

    def __init__(
        self,
        products: List[str],
        risk_free_rate: float = 0.001,
    ):
        """
        Args:
            products: 対象商品リスト
            risk_free_rate: 無リスク金利
        """
        self.products = products
        self.risk_free_rate = risk_free_rate

        # リターン履歴
        self.returns_history: Dict[str, List[float]] = {p: [] for p in products}

        # 最適ウェイト
        self.optimal_weights: Dict[str, float] = {}

    def update_returns(self, returns: Dict[str, float]) -> None:
        """
        リターン更新

        Args:
            returns: 商品 → リターンのマップ
        """
        for product, ret in returns.items():
            if product in self.returns_history:
                self.returns_history[product].append(ret)

                # 履歴制限
                if len(self.returns_history[product]) > 1000:
                    self.returns_history[product] = self.returns_history[product][-1000:]

    def calculate_covariance_matrix(self) -> np.ndarray:
        """共分散行列計算"""
        returns_data = []
        for product in self.products:
            if len(self.returns_history[product]) >= 20:
                returns_data.append(self.returns_history[product][-100:])

        if not returns_data or len(returns_data) != len(self.products):
            return np.eye(len(self.products))

        returns_array = np.array(returns_data)
        return np.cov(returns_array)

    def calculate_correlation_matrix(self) -> np.ndarray:
        """相関行列計算"""
        cov_matrix = self.calculate_covariance_matrix()
        std_devs = np.sqrt(np.diag(cov_matrix))
        std_outer = np.outer(std_devs, std_devs)
        return cov_matrix / (std_outer + 1e-10)

    def optimize_mean_variance(
        self,
        target_return: float = None,
        max_weight: float = 0.4,
    ) -> Dict[str, float]:
        """
        Mean-Variance最適化

        Args:
            target_return: 目標リターン
            max_weight: 最大ウェイト

        Returns:
            最適ウェイト
        """
        n = len(self.products)

        # 期待リターン計算
        expected_returns = []
        for product in self.products:
            if len(self.returns_history[product]) >= 20:
                expected_returns.append(np.mean(self.returns_history[product][-100:]))
            else:
                expected_returns.append(0.0)

        expected_returns = np.array(expected_returns)

        # 共分散行列
        cov_matrix = self.calculate_covariance_matrix()

        # 単純な最適化（等ウェイトからスタート）
        weights = np.ones(n) / n

        # シャープレシオ最大化の簡易実装
        best_sharpe = -np.inf
        best_weights = weights.copy()

        # ランダムサーチ
        for _ in range(1000):
            # ランダムウェイト生成
            random_weights = np.random.random(n)
            random_weights = random_weights / random_weights.sum()

            # 最大ウェイト制約
            random_weights = np.minimum(random_weights, max_weight)
            random_weights = random_weights / random_weights.sum()

            # ポートフォリオリターン
            portfolio_return = np.dot(expected_returns, random_weights)

            # ポートフォリオリスク
            portfolio_risk = np.sqrt(np.dot(random_weights.T, np.dot(cov_matrix, random_weights)))

            # シャープレシオ
            if portfolio_risk > 0:
                sharpe = (portfolio_return - self.risk_free_rate) / portfolio_risk
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_weights = random_weights.copy()

        self.optimal_weights = dict(zip(self.products, best_weights))
        return self.optimal_weights

    def optimize_risk_parity(self) -> Dict[str, float]:
        """
        リスクパリティ最適化

        各資産のリスク貢献度を均等化

        Returns:
            最適ウェイト
        """
        n = len(self.products)
        cov_matrix = self.calculate_covariance_matrix()

        # 逆ボラティリティウェイト（簡易版）
        volatilities = np.sqrt(np.diag(cov_matrix))
        inv_vol_weights = 1 / (volatilities + 1e-10)
        weights = inv_vol_weights / inv_vol_weights.sum()

        self.optimal_weights = dict(zip(self.products, weights))
        return self.optimal_weights

    def optimize_minimum_variance(self) -> Dict[str, float]:
        """
        最小分散ポートフォリオ

        Returns:
            最適ウェイト
        """
        n = len(self.products)
        cov_matrix = self.calculate_covariance_matrix()

        # 最小分散の解析解
        ones = np.ones(n)
        try:
            cov_inv = np.linalg.inv(cov_matrix)
            weights = np.dot(cov_inv, ones) / np.dot(ones.T, np.dot(cov_inv, ones))
        except np.linalg.LinAlgError:
            weights = ones / n

        # 負のウェイトを0にクリップ
        weights = np.maximum(weights, 0)
        weights = weights / weights.sum()

        self.optimal_weights = dict(zip(self.products, weights))
        return self.optimal_weights

    def calculate_rebalance_trades(
        self,
        current_positions: Dict[str, float],
        total_capital: float,
        prices: Dict[str, float],
        threshold: float = 0.05,
    ) -> List[Dict]:
        """
        リバランス取引計算

        Args:
            current_positions: 現在のポジション（商品 → サイズ）
            total_capital: 総資本
            prices: 現在価格
            threshold: リバランス閾値

        Returns:
            取引リスト
        """
        if not self.optimal_weights:
            return []

        trades = []

        for product in self.products:
            target_weight = self.optimal_weights.get(product, 0)
            price = prices.get(product, 0)

            if price <= 0:
                continue

            # 現在のウェイト計算
            current_size = current_positions.get(product, 0)
            current_value = current_size * price
            current_weight = current_value / total_capital if total_capital > 0 else 0

            # 差分チェック
            weight_diff = target_weight - current_weight

            if abs(weight_diff) > threshold:
                # 目標サイズ
                target_value = total_capital * target_weight
                target_size = target_value / price

                size_diff = target_size - current_size

                trades.append({
                    'product': product,
                    'side': 'BUY' if size_diff > 0 else 'SELL',
                    'size': abs(size_diff),
                    'current_weight': current_weight,
                    'target_weight': target_weight,
                    'weight_diff': weight_diff,
                })

        return trades

    def get_portfolio_metrics(
        self,
        weights: Dict[str, float] = None,
    ) -> Dict[str, float]:
        """
        ポートフォリオメトリクス計算

        Args:
            weights: ウェイト（Noneで最適ウェイト使用）

        Returns:
            メトリクス
        """
        weights = weights or self.optimal_weights
        if not weights:
            return {}

        weight_array = np.array([weights.get(p, 0) for p in self.products])

        # 期待リターン
        expected_returns = []
        for product in self.products:
            if len(self.returns_history[product]) >= 20:
                expected_returns.append(np.mean(self.returns_history[product][-100:]))
            else:
                expected_returns.append(0.0)

        expected_returns = np.array(expected_returns)
        cov_matrix = self.calculate_covariance_matrix()

        # ポートフォリオリターン
        portfolio_return = np.dot(expected_returns, weight_array)

        # ポートフォリオリスク
        portfolio_risk = np.sqrt(np.dot(weight_array.T, np.dot(cov_matrix, weight_array)))

        # シャープレシオ
        sharpe = (portfolio_return - self.risk_free_rate) / (portfolio_risk + 1e-10)

        # 分散度
        herfindahl = np.sum(weight_array ** 2)
        diversification = 1 - herfindahl

        return {
            'expected_return': portfolio_return,
            'expected_risk': portfolio_risk,
            'sharpe_ratio': sharpe,
            'diversification': diversification,
            'max_weight': max(weight_array),
            'min_weight': min(weight_array),
        }

    def check_correlation_limits(
        self,
        max_correlation: float = 0.7,
    ) -> List[Tuple[str, str, float]]:
        """
        相関制限チェック

        Args:
            max_correlation: 最大許容相関

        Returns:
            高相関ペアリスト
        """
        correlation_matrix = self.calculate_correlation_matrix()
        high_correlation_pairs = []

        n = len(self.products)
        for i in range(n):
            for j in range(i + 1, n):
                corr = correlation_matrix[i, j]
                if abs(corr) > max_correlation:
                    high_correlation_pairs.append(
                        (self.products[i], self.products[j], corr)
                    )

        return high_correlation_pairs

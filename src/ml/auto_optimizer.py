"""
Auto Optimizer
===============
自動ハイパーパラメータ最適化
リアルタイムでシステム全体を最適化
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from datetime import datetime, timedelta
from collections import defaultdict
import threading
import time
from loguru import logger


class BayesianOptimizer:
    """
    ベイズ最適化

    ガウス過程を使用したハイパーパラメータ最適化
    """

    def __init__(
        self,
        param_bounds: Dict[str, Tuple[float, float]],
        n_initial_points: int = 10,
    ):
        """
        Args:
            param_bounds: パラメータ名 → (min, max) のマップ
            n_initial_points: 初期ランダムサンプル数
        """
        self.param_bounds = param_bounds
        self.param_names = list(param_bounds.keys())
        self.n_params = len(param_bounds)
        self.n_initial_points = n_initial_points

        # 観測データ
        self.X: List[np.ndarray] = []  # パラメータ
        self.y: List[float] = []  # スコア

        # ガウス過程のパラメータ
        self.length_scale = 1.0
        self.signal_variance = 1.0
        self.noise_variance = 0.1

    def suggest(self) -> Dict[str, float]:
        """次に試すパラメータを提案"""
        if len(self.X) < self.n_initial_points:
            # 初期探索: ランダムサンプリング
            return self._random_sample()

        # ベイズ最適化: 獲得関数最大化
        best_params = None
        best_acquisition = -np.inf

        # ランダムサーチで獲得関数を最大化
        for _ in range(100):
            candidate = self._random_sample()
            x = self._params_to_array(candidate)
            acquisition = self._expected_improvement(x)

            if acquisition > best_acquisition:
                best_acquisition = acquisition
                best_params = candidate

        return best_params

    def observe(self, params: Dict[str, float], score: float) -> None:
        """観測結果を記録"""
        x = self._params_to_array(params)
        self.X.append(x)
        self.y.append(score)

    def _random_sample(self) -> Dict[str, float]:
        """ランダムサンプル"""
        params = {}
        for name, (low, high) in self.param_bounds.items():
            params[name] = np.random.uniform(low, high)
        return params

    def _params_to_array(self, params: Dict[str, float]) -> np.ndarray:
        """パラメータを配列に変換（正規化）"""
        arr = np.zeros(self.n_params)
        for i, name in enumerate(self.param_names):
            low, high = self.param_bounds[name]
            arr[i] = (params[name] - low) / (high - low)
        return arr

    def _array_to_params(self, arr: np.ndarray) -> Dict[str, float]:
        """配列をパラメータに変換"""
        params = {}
        for i, name in enumerate(self.param_names):
            low, high = self.param_bounds[name]
            params[name] = arr[i] * (high - low) + low
        return params

    def _rbf_kernel(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """RBFカーネル"""
        dist = np.sum((x1 - x2) ** 2)
        return self.signal_variance * np.exp(-0.5 * dist / (self.length_scale ** 2))

    def _compute_kernel_matrix(self, X: List[np.ndarray]) -> np.ndarray:
        """カーネル行列計算"""
        n = len(X)
        K = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                K[i, j] = self._rbf_kernel(X[i], X[j])
            K[i, i] += self.noise_variance
        return K

    def _predict(self, x: np.ndarray) -> Tuple[float, float]:
        """ガウス過程予測"""
        if len(self.X) == 0:
            return 0.0, 1.0

        X = np.array(self.X)
        y = np.array(self.y)

        K = self._compute_kernel_matrix(self.X)
        K_inv = np.linalg.inv(K + 1e-6 * np.eye(len(K)))

        k_star = np.array([self._rbf_kernel(x, xi) for xi in self.X])
        k_star_star = self._rbf_kernel(x, x)

        # 予測平均
        mu = np.dot(k_star, np.dot(K_inv, y))

        # 予測分散
        var = k_star_star - np.dot(k_star, np.dot(K_inv, k_star))
        var = max(var, 1e-6)

        return mu, np.sqrt(var)

    def _expected_improvement(self, x: np.ndarray, xi: float = 0.01) -> float:
        """Expected Improvement 獲得関数"""
        if len(self.y) == 0:
            return 0.0

        mu, sigma = self._predict(x)
        y_best = max(self.y)

        if sigma == 0:
            return 0.0

        z = (mu - y_best - xi) / sigma
        ei = (mu - y_best - xi) * self._norm_cdf(z) + sigma * self._norm_pdf(z)

        return ei

    def _norm_cdf(self, x: float) -> float:
        """標準正規分布の累積分布関数"""
        return 0.5 * (1 + np.tanh(x * 0.7978845608))

    def _norm_pdf(self, x: float) -> float:
        """標準正規分布の確率密度関数"""
        return np.exp(-0.5 * x ** 2) / np.sqrt(2 * np.pi)

    def get_best(self) -> Tuple[Dict[str, float], float]:
        """最良のパラメータと値を取得"""
        if not self.y:
            return {}, 0.0

        best_idx = np.argmax(self.y)
        best_params = self._array_to_params(self.X[best_idx])
        return best_params, self.y[best_idx]


class OnlineHyperparameterTuner:
    """
    オンラインハイパーパラメータチューナー

    取引しながらリアルタイムでパラメータを最適化
    """

    def __init__(self):
        self.optimizers: Dict[str, BayesianOptimizer] = {}
        self.current_params: Dict[str, Dict[str, float]] = {}
        self.performance_window: Dict[str, List[float]] = defaultdict(list)
        self.window_size = 100

        # デフォルトのパラメータ範囲
        self.default_bounds = {
            'strategy': {
                'momentum_lookback': (5, 100),
                'rsi_oversold': (20, 40),
                'rsi_overbought': (60, 80),
                'bollinger_std': (1.5, 3.0),
                'breakout_threshold': (0.5, 3.0),
            },
            'risk': {
                'stop_loss_pct': (0.01, 0.05),
                'take_profit_pct': (0.02, 0.08),
                'position_size_pct': (0.05, 0.20),
                'max_drawdown': (0.10, 0.25),
            },
            'execution': {
                'slippage_tolerance': (0.0005, 0.002),
                'order_timeout': (10, 120),
                'twap_slices': (3, 20),
            },
        }

    def register_component(
        self,
        component_name: str,
        param_bounds: Dict[str, Tuple[float, float]] = None,
    ) -> None:
        """コンポーネント登録"""
        bounds = param_bounds or self.default_bounds.get(component_name, {})
        if bounds:
            self.optimizers[component_name] = BayesianOptimizer(bounds)
            self.current_params[component_name] = self.optimizers[component_name].suggest()
            logger.info(f"Registered optimizer for {component_name}")

    def get_params(self, component_name: str) -> Dict[str, float]:
        """現在のパラメータ取得"""
        return self.current_params.get(component_name, {})

    def report_performance(self, component_name: str, score: float) -> None:
        """パフォーマンス報告"""
        if component_name not in self.optimizers:
            return

        self.performance_window[component_name].append(score)

        # ウィンドウサイズ制限
        if len(self.performance_window[component_name]) > self.window_size:
            self.performance_window[component_name] = \
                self.performance_window[component_name][-self.window_size:]

        # 十分なサンプルが集まったら最適化
        if len(self.performance_window[component_name]) >= 20:
            avg_score = np.mean(self.performance_window[component_name][-20:])
            self.optimizers[component_name].observe(
                self.current_params[component_name],
                avg_score
            )

            # 新しいパラメータを提案
            self.current_params[component_name] = \
                self.optimizers[component_name].suggest()

            logger.debug(f"Updated params for {component_name}: {self.current_params[component_name]}")

    def get_optimization_status(self) -> Dict[str, Dict]:
        """最適化状態取得"""
        status = {}
        for name, optimizer in self.optimizers.items():
            best_params, best_score = optimizer.get_best()
            status[name] = {
                'current_params': self.current_params[name],
                'best_params': best_params,
                'best_score': best_score,
                'n_trials': len(optimizer.y),
            }
        return status


class GeneticOptimizer:
    """
    遺伝的アルゴリズム最適化

    戦略パラメータの進化的最適化
    """

    def __init__(
        self,
        param_bounds: Dict[str, Tuple[float, float]],
        population_size: int = 20,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.7,
    ):
        self.param_bounds = param_bounds
        self.param_names = list(param_bounds.keys())
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

        # 個体群初期化
        self.population: List[Dict[str, float]] = []
        self.fitness: List[float] = []
        self._initialize_population()

        self.generation = 0
        self.best_individual: Optional[Dict[str, float]] = None
        self.best_fitness = -np.inf

    def _initialize_population(self) -> None:
        """個体群初期化"""
        self.population = []
        for _ in range(self.population_size):
            individual = {}
            for name, (low, high) in self.param_bounds.items():
                individual[name] = np.random.uniform(low, high)
            self.population.append(individual)
        self.fitness = [0.0] * self.population_size

    def evaluate(self, idx: int, fitness: float) -> None:
        """個体の適応度を設定"""
        self.fitness[idx] = fitness

        if fitness > self.best_fitness:
            self.best_fitness = fitness
            self.best_individual = self.population[idx].copy()

    def evolve(self) -> None:
        """次世代へ進化"""
        # 選択（トーナメント選択）
        new_population = []

        # エリート保存
        elite_idx = np.argmax(self.fitness)
        new_population.append(self.population[elite_idx].copy())

        while len(new_population) < self.population_size:
            # 親選択
            parent1 = self._tournament_select()
            parent2 = self._tournament_select()

            # 交叉
            if np.random.random() < self.crossover_rate:
                child1, child2 = self._crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()

            # 突然変異
            child1 = self._mutate(child1)
            child2 = self._mutate(child2)

            new_population.append(child1)
            if len(new_population) < self.population_size:
                new_population.append(child2)

        self.population = new_population
        self.fitness = [0.0] * self.population_size
        self.generation += 1

        logger.info(f"Generation {self.generation}: Best fitness = {self.best_fitness:.4f}")

    def _tournament_select(self, k: int = 3) -> Dict[str, float]:
        """トーナメント選択"""
        indices = np.random.choice(self.population_size, k, replace=False)
        best_idx = indices[np.argmax([self.fitness[i] for i in indices])]
        return self.population[best_idx].copy()

    def _crossover(
        self,
        parent1: Dict[str, float],
        parent2: Dict[str, float],
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """一様交叉"""
        child1, child2 = {}, {}

        for name in self.param_names:
            if np.random.random() < 0.5:
                child1[name] = parent1[name]
                child2[name] = parent2[name]
            else:
                child1[name] = parent2[name]
                child2[name] = parent1[name]

        return child1, child2

    def _mutate(self, individual: Dict[str, float]) -> Dict[str, float]:
        """突然変異"""
        mutated = individual.copy()

        for name, (low, high) in self.param_bounds.items():
            if np.random.random() < self.mutation_rate:
                # ガウスノイズ
                noise = np.random.normal(0, (high - low) * 0.1)
                mutated[name] = np.clip(individual[name] + noise, low, high)

        return mutated

    def get_individual(self, idx: int) -> Dict[str, float]:
        """個体取得"""
        return self.population[idx].copy()

    def get_best(self) -> Tuple[Dict[str, float], float]:
        """最良個体取得"""
        return self.best_individual.copy() if self.best_individual else {}, self.best_fitness


class AutoMLSystem:
    """
    自動機械学習システム

    全自動でモデル選択・最適化を行う
    """

    def __init__(self):
        self.hyperparameter_tuner = OnlineHyperparameterTuner()
        self.genetic_optimizer: Optional[GeneticOptimizer] = None

        # モデルパフォーマンス追跡
        self.model_performance: Dict[str, List[float]] = defaultdict(list)
        self.best_models: Dict[str, str] = {}

        # 自動最適化スレッド
        self.is_running = False
        self.optimization_thread: Optional[threading.Thread] = None

    def register_for_optimization(
        self,
        component_name: str,
        param_bounds: Dict[str, Tuple[float, float]],
    ) -> None:
        """最適化対象として登録"""
        self.hyperparameter_tuner.register_component(component_name, param_bounds)

    def start_evolution(
        self,
        param_bounds: Dict[str, Tuple[float, float]],
        population_size: int = 20,
    ) -> None:
        """遺伝的アルゴリズム開始"""
        self.genetic_optimizer = GeneticOptimizer(
            param_bounds=param_bounds,
            population_size=population_size,
        )

    def report_model_performance(
        self,
        model_name: str,
        metric_name: str,
        value: float,
    ) -> None:
        """モデルパフォーマンス報告"""
        key = f"{model_name}_{metric_name}"
        self.model_performance[key].append(value)

        # 履歴制限
        if len(self.model_performance[key]) > 1000:
            self.model_performance[key] = self.model_performance[key][-500:]

    def get_model_rankings(self) -> Dict[str, float]:
        """モデルランキング取得"""
        rankings = {}
        for key, values in self.model_performance.items():
            if len(values) >= 10:
                rankings[key] = np.mean(values[-50:])
        return dict(sorted(rankings.items(), key=lambda x: x[1], reverse=True))

    def start_continuous_optimization(self, interval: float = 60.0) -> None:
        """継続的最適化開始"""
        self.is_running = True

        def optimization_loop():
            while self.is_running:
                # 遺伝的アルゴリズムの世代更新
                if self.genetic_optimizer and self.genetic_optimizer.generation > 0:
                    # すべての個体が評価されたら進化
                    if all(f != 0 for f in self.genetic_optimizer.fitness):
                        self.genetic_optimizer.evolve()

                time.sleep(interval)

        self.optimization_thread = threading.Thread(target=optimization_loop, daemon=True)
        self.optimization_thread.start()
        logger.info("Continuous optimization started")

    def stop(self) -> None:
        """最適化停止"""
        self.is_running = False
        if self.optimization_thread:
            self.optimization_thread.join(timeout=5)

    def get_optimization_report(self) -> Dict:
        """最適化レポート取得"""
        report = {
            'hyperparameter_status': self.hyperparameter_tuner.get_optimization_status(),
            'model_rankings': self.get_model_rankings(),
        }

        if self.genetic_optimizer:
            best_individual, best_fitness = self.genetic_optimizer.get_best()
            report['genetic_optimization'] = {
                'generation': self.genetic_optimizer.generation,
                'best_individual': best_individual,
                'best_fitness': best_fitness,
            }

        return report

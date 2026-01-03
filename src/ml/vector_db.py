"""
Vector Database for Pattern Matching
=====================================
FAISS風の高速ベクトル検索
過去の類似パターンから学習
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import deque
import pickle
import os
from datetime import datetime
from loguru import logger


@dataclass
class PatternRecord:
    """パターン記録"""
    pattern: np.ndarray
    timestamp: datetime
    outcome: int  # 0: 下落, 1: 横ばい, 2: 上昇
    profit: float  # 実際の利益率
    confidence: float
    metadata: Dict = field(default_factory=dict)


class LSHIndex:
    """
    Locality Sensitive Hashing (LSH)

    近似最近傍探索のためのハッシュインデックス
    """

    def __init__(
        self,
        dim: int,
        num_tables: int = 10,
        num_hashes: int = 8,
    ):
        self.dim = dim
        self.num_tables = num_tables
        self.num_hashes = num_hashes

        # ランダムハイパープレーン
        self.hyperplanes = [
            np.random.randn(num_hashes, dim)
            for _ in range(num_tables)
        ]

        # ハッシュテーブル
        self.tables = [{} for _ in range(num_tables)]
        self.vectors = []
        self.ids = []

    def _hash_vector(self, vector: np.ndarray, table_idx: int) -> str:
        """ベクトルをハッシュ"""
        projections = self.hyperplanes[table_idx] @ vector
        bits = (projections > 0).astype(int)
        return ''.join(map(str, bits))

    def add(self, vector: np.ndarray, idx: int) -> None:
        """ベクトルを追加"""
        self.vectors.append(vector)
        self.ids.append(idx)

        for t in range(self.num_tables):
            hash_key = self._hash_vector(vector, t)
            if hash_key not in self.tables[t]:
                self.tables[t][hash_key] = []
            self.tables[t][hash_key].append(idx)

    def query(
        self,
        vector: np.ndarray,
        k: int = 10,
    ) -> List[int]:
        """k近傍を検索"""
        candidates = set()

        for t in range(self.num_tables):
            hash_key = self._hash_vector(vector, t)
            if hash_key in self.tables[t]:
                candidates.update(self.tables[t][hash_key])

        if not candidates:
            return []

        # 候補から距離でソート
        distances = []
        for idx in candidates:
            dist = np.linalg.norm(vector - self.vectors[idx])
            distances.append((idx, dist))

        distances.sort(key=lambda x: x[1])

        return [idx for idx, _ in distances[:k]]


class IVFIndex:
    """
    Inverted File Index (IVF)

    クラスタリングベースの近似最近傍探索
    """

    def __init__(
        self,
        dim: int,
        n_clusters: int = 100,
        n_probe: int = 10,
    ):
        self.dim = dim
        self.n_clusters = n_clusters
        self.n_probe = n_probe

        self.centroids = None
        self.inverted_lists = [[] for _ in range(n_clusters)]
        self.vectors = []
        self.is_trained = False

    def train(self, vectors: np.ndarray) -> None:
        """k-meansでセントロイドを学習"""
        if len(vectors) < self.n_clusters:
            self.n_clusters = max(1, len(vectors) // 2)
            self.inverted_lists = [[] for _ in range(self.n_clusters)]

        # Simple k-means
        indices = np.random.choice(len(vectors), self.n_clusters, replace=False)
        self.centroids = vectors[indices].copy()

        for _ in range(10):  # 10 iterations
            # Assign to clusters
            assignments = []
            for v in vectors:
                dists = np.linalg.norm(self.centroids - v, axis=1)
                assignments.append(np.argmin(dists))

            # Update centroids
            for c in range(self.n_clusters):
                cluster_vectors = [v for v, a in zip(vectors, assignments) if a == c]
                if cluster_vectors:
                    self.centroids[c] = np.mean(cluster_vectors, axis=0)

        self.is_trained = True
        logger.info(f"IVF Index trained with {self.n_clusters} clusters")

    def add(self, vector: np.ndarray, idx: int) -> None:
        """ベクトルを追加"""
        self.vectors.append(vector)

        if self.centroids is None:
            return

        # 最も近いクラスタに追加
        dists = np.linalg.norm(self.centroids - vector, axis=1)
        cluster = np.argmin(dists)
        self.inverted_lists[cluster].append(idx)

    def query(
        self,
        vector: np.ndarray,
        k: int = 10,
    ) -> List[Tuple[int, float]]:
        """k近傍を検索"""
        if self.centroids is None or not self.vectors:
            return []

        # 最も近いn_probe個のクラスタを探す
        dists = np.linalg.norm(self.centroids - vector, axis=1)
        probe_clusters = np.argsort(dists)[:self.n_probe]

        # 候補を収集
        candidates = []
        for cluster in probe_clusters:
            candidates.extend(self.inverted_lists[cluster])

        if not candidates:
            return []

        # 距離を計算してソート
        results = []
        for idx in candidates:
            dist = np.linalg.norm(vector - self.vectors[idx])
            results.append((idx, dist))

        results.sort(key=lambda x: x[1])

        return results[:k]


class ProductQuantizer:
    """
    Product Quantization (PQ)

    ベクトルを圧縮して高速検索
    """

    def __init__(
        self,
        dim: int,
        n_subspaces: int = 8,
        n_centroids: int = 256,
    ):
        self.dim = dim
        self.n_subspaces = n_subspaces
        self.n_centroids = n_centroids
        self.subspace_dim = dim // n_subspaces

        self.codebooks = None
        self.codes = []

    def train(self, vectors: np.ndarray) -> None:
        """コードブックを学習"""
        self.codebooks = []

        for m in range(self.n_subspaces):
            start = m * self.subspace_dim
            end = start + self.subspace_dim
            subvectors = vectors[:, start:end]

            # k-means for each subspace
            n_samples = min(self.n_centroids, len(subvectors))
            indices = np.random.choice(len(subvectors), n_samples, replace=False)
            centroids = subvectors[indices].copy()

            for _ in range(10):
                assignments = np.argmin(
                    np.linalg.norm(subvectors[:, None] - centroids, axis=2),
                    axis=1
                )
                for c in range(len(centroids)):
                    mask = assignments == c
                    if mask.sum() > 0:
                        centroids[c] = subvectors[mask].mean(axis=0)

            self.codebooks.append(centroids)

        logger.info(f"PQ trained with {self.n_subspaces} subspaces")

    def encode(self, vector: np.ndarray) -> np.ndarray:
        """ベクトルをコードに変換"""
        codes = []
        for m in range(self.n_subspaces):
            start = m * self.subspace_dim
            end = start + self.subspace_dim
            subvector = vector[start:end]

            dists = np.linalg.norm(self.codebooks[m] - subvector, axis=1)
            codes.append(np.argmin(dists))

        return np.array(codes, dtype=np.uint8)

    def add(self, vector: np.ndarray) -> None:
        """ベクトルを追加"""
        code = self.encode(vector)
        self.codes.append(code)

    def asymmetric_distance(
        self,
        query: np.ndarray,
        idx: int,
    ) -> float:
        """非対称距離を計算"""
        distance = 0
        code = self.codes[idx]

        for m in range(self.n_subspaces):
            start = m * self.subspace_dim
            end = start + self.subspace_dim
            subquery = query[start:end]
            centroid = self.codebooks[m][code[m]]
            distance += np.sum((subquery - centroid) ** 2)

        return np.sqrt(distance)


class VectorPatternDB:
    """
    Vector Pattern Database

    過去のパターンを保存し、類似パターンを高速検索
    """

    def __init__(
        self,
        dim: int = 500,
        max_patterns: int = 100000,
        use_lsh: bool = True,
        use_ivf: bool = True,
        use_pq: bool = True,
    ):
        self.dim = dim
        self.max_patterns = max_patterns

        # パターン保存
        self.patterns: List[PatternRecord] = []
        self.pattern_vectors: List[np.ndarray] = []

        # インデックス
        self.lsh_index = LSHIndex(dim) if use_lsh else None
        self.ivf_index = IVFIndex(dim) if use_ivf else None
        self.pq = ProductQuantizer(dim) if use_pq else None

        self.use_lsh = use_lsh
        self.use_ivf = use_ivf
        self.use_pq = use_pq

        # 統計
        self.query_count = 0
        self.hit_count = 0

        logger.info(f"VectorPatternDB initialized (dim={dim})")

    def add_pattern(
        self,
        pattern: np.ndarray,
        outcome: int,
        profit: float,
        confidence: float = 1.0,
        metadata: Dict = None,
    ) -> int:
        """パターンを追加"""
        if len(pattern) != self.dim:
            pattern = np.pad(pattern, (0, self.dim - len(pattern)))[:self.dim]

        # 正規化
        norm = np.linalg.norm(pattern)
        if norm > 0:
            pattern = pattern / norm

        record = PatternRecord(
            pattern=pattern,
            timestamp=datetime.now(),
            outcome=outcome,
            profit=profit,
            confidence=confidence,
            metadata=metadata or {},
        )

        idx = len(self.patterns)
        self.patterns.append(record)
        self.pattern_vectors.append(pattern)

        # インデックスに追加
        if self.lsh_index:
            self.lsh_index.add(pattern, idx)

        if self.ivf_index and self.ivf_index.is_trained:
            self.ivf_index.add(pattern, idx)

        if self.pq and self.pq.codebooks is not None:
            self.pq.add(pattern)

        # 容量超過時は古いパターンを削除
        if len(self.patterns) > self.max_patterns:
            self._prune_old_patterns()

        return idx

    def _prune_old_patterns(self) -> None:
        """古いパターンを削除"""
        # 最新の80%を保持
        keep_count = int(self.max_patterns * 0.8)
        self.patterns = self.patterns[-keep_count:]
        self.pattern_vectors = self.pattern_vectors[-keep_count:]

        # インデックス再構築が必要
        self._rebuild_indices()

    def _rebuild_indices(self) -> None:
        """インデックスを再構築"""
        vectors = np.array(self.pattern_vectors)

        if self.lsh_index:
            self.lsh_index = LSHIndex(self.dim)
            for i, v in enumerate(vectors):
                self.lsh_index.add(v, i)

        if self.ivf_index and len(vectors) > 100:
            self.ivf_index = IVFIndex(self.dim)
            self.ivf_index.train(vectors)
            for i, v in enumerate(vectors):
                self.ivf_index.add(v, i)

        if self.pq and len(vectors) > 100:
            self.pq = ProductQuantizer(self.dim)
            self.pq.train(vectors)
            for v in vectors:
                self.pq.add(v)

    def train_indices(self) -> None:
        """インデックスを学習"""
        if len(self.pattern_vectors) < 100:
            logger.warning("Not enough patterns for training")
            return

        vectors = np.array(self.pattern_vectors)

        if self.ivf_index:
            self.ivf_index.train(vectors)
            for i, v in enumerate(vectors):
                self.ivf_index.add(v, i)

        if self.pq:
            self.pq.train(vectors)
            for v in vectors:
                self.pq.add(v)

        logger.info(f"Indices trained with {len(vectors)} patterns")

    def search(
        self,
        query: np.ndarray,
        k: int = 10,
        min_similarity: float = 0.7,
    ) -> List[Tuple[PatternRecord, float]]:
        """類似パターンを検索"""
        if len(query) != self.dim:
            query = np.pad(query, (0, self.dim - len(query)))[:self.dim]

        # 正規化
        norm = np.linalg.norm(query)
        if norm > 0:
            query = query / norm

        self.query_count += 1

        candidates = set()

        # LSHで候補を取得
        if self.lsh_index:
            lsh_results = self.lsh_index.query(query, k * 3)
            candidates.update(lsh_results)

        # IVFで候補を取得
        if self.ivf_index and self.ivf_index.is_trained:
            ivf_results = self.ivf_index.query(query, k * 3)
            candidates.update([idx for idx, _ in ivf_results])

        # 候補がない場合は全探索
        if not candidates:
            candidates = set(range(min(len(self.patterns), 1000)))

        # コサイン類似度でランキング
        results = []
        for idx in candidates:
            if idx < len(self.pattern_vectors):
                similarity = np.dot(query, self.pattern_vectors[idx])
                if similarity >= min_similarity:
                    results.append((self.patterns[idx], similarity))
                    self.hit_count += 1

        results.sort(key=lambda x: x[1], reverse=True)

        return results[:k]

    def predict_from_patterns(
        self,
        query: np.ndarray,
        k: int = 20,
        min_similarity: float = 0.6,
    ) -> Tuple[int, float, Dict]:
        """
        類似パターンから予測

        Returns:
            prediction: 予測アクション (0: 売り, 1: ホールド, 2: 買い)
            confidence: 信頼度
            details: 詳細情報
        """
        similar_patterns = self.search(query, k, min_similarity)

        if not similar_patterns:
            return 1, 0.0, {'patterns_found': 0}

        # 重み付き投票
        votes = {0: 0.0, 1: 0.0, 2: 0.0}
        total_profit = 0
        total_weight = 0

        for pattern, similarity in similar_patterns:
            weight = similarity * pattern.confidence
            votes[pattern.outcome] += weight
            total_profit += pattern.profit * weight
            total_weight += weight

        # 正規化
        total_votes = sum(votes.values())
        if total_votes > 0:
            for k in votes:
                votes[k] /= total_votes

        prediction = max(votes, key=votes.get)
        confidence = votes[prediction]

        # 期待利益
        expected_profit = total_profit / total_weight if total_weight > 0 else 0

        details = {
            'patterns_found': len(similar_patterns),
            'votes': votes,
            'expected_profit': expected_profit,
            'avg_similarity': np.mean([s for _, s in similar_patterns]),
            'hit_rate': self.hit_count / max(1, self.query_count),
        }

        return prediction, confidence, details

    def get_pattern_statistics(self) -> Dict:
        """パターン統計を取得"""
        if not self.patterns:
            return {}

        outcomes = [p.outcome for p in self.patterns]
        profits = [p.profit for p in self.patterns]

        return {
            'total_patterns': len(self.patterns),
            'outcome_distribution': {
                0: outcomes.count(0) / len(outcomes),
                1: outcomes.count(1) / len(outcomes),
                2: outcomes.count(2) / len(outcomes),
            },
            'avg_profit': np.mean(profits),
            'profit_std': np.std(profits),
            'query_count': self.query_count,
            'hit_rate': self.hit_count / max(1, self.query_count),
        }

    def save(self, path: str) -> None:
        """データベースを保存"""
        data = {
            'patterns': self.patterns,
            'pattern_vectors': self.pattern_vectors,
            'dim': self.dim,
        }

        with open(path, 'wb') as f:
            pickle.dump(data, f)

        logger.info(f"VectorPatternDB saved to {path}")

    def load(self, path: str) -> None:
        """データベースを読み込み"""
        if not os.path.exists(path):
            logger.warning(f"File not found: {path}")
            return

        with open(path, 'rb') as f:
            data = pickle.load(f)

        self.patterns = data['patterns']
        self.pattern_vectors = data['pattern_vectors']

        # インデックス再構築
        if len(self.pattern_vectors) > 100:
            self._rebuild_indices()

        logger.info(f"VectorPatternDB loaded from {path} ({len(self.patterns)} patterns)")


class TemporalPatternMatcher:
    """
    時系列パターンマッチャー

    動的時間伸縮法 (DTW) などを使用
    """

    def __init__(
        self,
        window_size: int = 60,
        stride: int = 10,
    ):
        self.window_size = window_size
        self.stride = stride
        self.patterns = []
        self.outcomes = []

    @staticmethod
    def dtw_distance(s1: np.ndarray, s2: np.ndarray) -> float:
        """Dynamic Time Warping距離"""
        n, m = len(s1), len(s2)
        dtw = np.full((n + 1, m + 1), np.inf)
        dtw[0, 0] = 0

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = abs(s1[i - 1] - s2[j - 1])
                dtw[i, j] = cost + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])

        return dtw[n, m]

    @staticmethod
    def fast_dtw_distance(s1: np.ndarray, s2: np.ndarray, radius: int = 5) -> float:
        """Fast DTW (近似)"""
        n, m = len(s1), len(s2)

        if n < radius * 2 or m < radius * 2:
            return TemporalPatternMatcher.dtw_distance(s1, s2)

        # ダウンサンプリング
        s1_down = s1[::2]
        s2_down = s2[::2]

        # 再帰的に計算
        path = []
        low_res_dist = TemporalPatternMatcher.fast_dtw_distance(s1_down, s2_down, radius)

        return low_res_dist * 2

    def add_pattern(
        self,
        time_series: np.ndarray,
        outcome: int,
    ) -> None:
        """パターンを追加"""
        # 正規化
        ts = (time_series - np.mean(time_series)) / (np.std(time_series) + 1e-10)
        self.patterns.append(ts)
        self.outcomes.append(outcome)

    def find_similar(
        self,
        query: np.ndarray,
        k: int = 5,
    ) -> List[Tuple[int, float, int]]:
        """類似パターンを検索"""
        if not self.patterns:
            return []

        # 正規化
        query = (query - np.mean(query)) / (np.std(query) + 1e-10)

        distances = []
        for i, pattern in enumerate(self.patterns):
            # サイズ調整
            if len(pattern) != len(query):
                if len(pattern) > len(query):
                    pattern = pattern[:len(query)]
                else:
                    pattern = np.pad(pattern, (0, len(query) - len(pattern)))

            dist = self.fast_dtw_distance(query, pattern)
            distances.append((i, dist, self.outcomes[i]))

        distances.sort(key=lambda x: x[1])

        return distances[:k]

    def predict(self, query: np.ndarray, k: int = 10) -> Tuple[int, float]:
        """予測"""
        similar = self.find_similar(query, k)

        if not similar:
            return 1, 0.0

        votes = {0: 0, 1: 0, 2: 0}
        total_weight = 0

        for _, dist, outcome in similar:
            weight = 1 / (1 + dist)
            votes[outcome] += weight
            total_weight += weight

        prediction = max(votes, key=votes.get)
        confidence = votes[prediction] / total_weight if total_weight > 0 else 0

        return prediction, confidence


logger.info("Vector Database module loaded")

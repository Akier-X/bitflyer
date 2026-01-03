"""
Unbeatable Trading System
==========================
勝率99%以上を目指す無敗システム
負けたら即座に進化し、同じ失敗を二度と繰り返さない
動的閾値による高頻度取引対応
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass, field
import hashlib
import pickle
import os
from loguru import logger


@dataclass
class TradePattern:
    """取引パターン"""
    features: np.ndarray
    action: int  # 0: BUY, 1: HOLD, 2: SELL
    result: float  # P&L
    was_win: bool
    timestamp: datetime
    pattern_hash: str = ""

    def __post_init__(self):
        # パターンハッシュ生成
        feature_bytes = self.features.tobytes()
        self.pattern_hash = hashlib.md5(feature_bytes).hexdigest()[:16]


class LossMemory:
    """
    負けパターン記憶システム

    過去の負けを記憶し、同じ失敗を繰り返さない
    """

    def __init__(self, similarity_threshold: float = 0.95):
        self.similarity_threshold = similarity_threshold
        self.loss_patterns: List[TradePattern] = []
        self.loss_hashes: set = set()

        # パターンクラスタリング
        self.loss_clusters: Dict[str, List[TradePattern]] = defaultdict(list)

    def remember_loss(self, pattern: TradePattern) -> None:
        """負けパターンを記憶"""
        self.loss_patterns.append(pattern)
        self.loss_hashes.add(pattern.pattern_hash)

        # クラスタに追加
        cluster_key = self._get_cluster_key(pattern.features)
        self.loss_clusters[cluster_key].append(pattern)

        # 履歴制限
        if len(self.loss_patterns) > 10000:
            self.loss_patterns = self.loss_patterns[-5000:]

        logger.warning(f"📝 Loss pattern memorized: {pattern.pattern_hash}")

    def is_similar_to_loss(self, features: np.ndarray) -> Tuple[bool, float]:
        """
        負けパターンに類似しているかチェック

        Returns:
            (is_similar, similarity_score)
        """
        if not self.loss_patterns:
            return False, 0.0

        # クイックチェック（ハッシュ）
        test_hash = hashlib.md5(features.tobytes()).hexdigest()[:16]
        if test_hash in self.loss_hashes:
            return True, 1.0

        # クラスタベースのチェック
        cluster_key = self._get_cluster_key(features)
        if cluster_key in self.loss_clusters:
            for loss_pattern in self.loss_clusters[cluster_key]:
                similarity = self._calculate_similarity(features, loss_pattern.features)
                if similarity >= self.similarity_threshold:
                    return True, similarity

        # 直近の負けパターンとの詳細比較
        for loss_pattern in self.loss_patterns[-100:]:
            similarity = self._calculate_similarity(features, loss_pattern.features)
            if similarity >= self.similarity_threshold:
                return True, similarity

        return False, 0.0

    def _get_cluster_key(self, features: np.ndarray) -> str:
        """クラスタキー生成"""
        # 特徴量を離散化してクラスタキーを生成
        discretized = np.round(features * 10).astype(int)
        return hashlib.md5(discretized.tobytes()).hexdigest()[:8]

    def _calculate_similarity(self, f1: np.ndarray, f2: np.ndarray) -> float:
        """コサイン類似度計算"""
        norm1 = np.linalg.norm(f1)
        norm2 = np.linalg.norm(f2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return np.dot(f1, f2) / (norm1 * norm2)


class MultiModelConsensus:
    """
    マルチモデル合意システム

    複数のモデルが合意した時のみ取引
    """

    def __init__(self, min_agreement: float = 0.9):
        """
        Args:
            min_agreement: 最低合意率（0.9 = 90%のモデルが合意）
        """
        self.min_agreement = min_agreement
        self.model_predictions: Dict[str, int] = {}
        self.model_confidences: Dict[str, float] = {}
        self.model_weights: Dict[str, float] = {}

    def register_prediction(
        self,
        model_name: str,
        prediction: int,  # 0: BUY, 1: HOLD, 2: SELL
        confidence: float,
        weight: float = 1.0,
    ) -> None:
        """モデル予測を登録"""
        self.model_predictions[model_name] = prediction
        self.model_confidences[model_name] = confidence
        self.model_weights[model_name] = weight

    def get_consensus(self) -> Tuple[Optional[int], float, bool]:
        """
        合意を取得

        Returns:
            (action, confidence, has_consensus)
            action: 0=BUY, 1=HOLD, 2=SELL, None=合意なし
        """
        if not self.model_predictions:
            return None, 0.0, False

        # 重み付き投票
        votes = {0: 0.0, 1: 0.0, 2: 0.0}

        for model_name, prediction in self.model_predictions.items():
            weight = self.model_weights.get(model_name, 1.0)
            confidence = self.model_confidences.get(model_name, 0.5)
            votes[prediction] += weight * confidence

        total_weight = sum(votes.values())
        if total_weight == 0:
            return 1, 0.0, False  # HOLD

        # 正規化
        for k in votes:
            votes[k] /= total_weight

        # 最大票
        best_action = max(votes, key=votes.get)
        agreement = votes[best_action]

        # 合意チェック
        has_consensus = agreement >= self.min_agreement

        # HOLDは常にOK（リスク回避）
        if best_action == 1:
            has_consensus = True

        return best_action, agreement, has_consensus

    def clear(self) -> None:
        """予測をクリア"""
        self.model_predictions.clear()
        self.model_confidences.clear()


class DynamicConfidenceGate:
    """
    動的確信度ゲート

    取引頻度と勝率に応じて閾値を動的に調整
    高頻度取引と高勝率のバランスを取る
    """

    def __init__(
        self,
        base_confidence: float = 0.60,
        min_confidence: float = 0.50,
        max_confidence: float = 0.85,
        min_indicators_aligned: int = 3,
        target_trades_per_hour: int = 50,
    ):
        """
        Args:
            base_confidence: ベース確信度
            min_confidence: 最低確信度
            max_confidence: 最大確信度
            min_indicators_aligned: 最低一致インジケータ数
            target_trades_per_hour: 目標時間当たり取引数
        """
        self.base_confidence = base_confidence
        self.min_confidence = min_confidence
        self.max_confidence = max_confidence
        self.min_indicators_aligned = min_indicators_aligned
        self.target_trades_per_hour = target_trades_per_hour

        # 動的閾値
        self.current_threshold = base_confidence

        # インジケータチェックリスト
        self.indicators: Dict[str, bool] = {}
        self.indicator_confidences: Dict[str, float] = {}

        # 取引履歴
        self.trade_times: deque = deque(maxlen=1000)
        self.recent_results: deque = deque(maxlen=100)

        # 連勝/連敗トラッキング
        self.win_streak = 0
        self.loss_streak = 0

    def update_threshold(self, volatility: float = 0.02) -> float:
        """
        閾値を動的に更新

        Args:
            volatility: 現在のボラティリティ

        Returns:
            更新された閾値
        """
        threshold = self.base_confidence

        # 取引頻度による調整
        trades_last_hour = self._count_trades_last_hour()

        if trades_last_hour < self.target_trades_per_hour * 0.5:
            # 取引少なすぎ → 閾値下げる
            threshold -= 0.08
        elif trades_last_hour < self.target_trades_per_hour * 0.8:
            threshold -= 0.04
        elif trades_last_hour > self.target_trades_per_hour * 1.5:
            # 取引多すぎ → 閾値上げる
            threshold += 0.05

        # 勝率による調整
        if len(self.recent_results) >= 10:
            recent_win_rate = sum(1 for r in self.recent_results if r > 0) / len(self.recent_results)
            if recent_win_rate > 0.75:
                # 勝率高い → 積極的に
                threshold -= 0.05
            elif recent_win_rate > 0.65:
                threshold -= 0.02
            elif recent_win_rate < 0.50:
                # 勝率低い → 慎重に
                threshold += 0.08
            elif recent_win_rate < 0.55:
                threshold += 0.04

        # 連勝/連敗による調整
        if self.win_streak >= 5:
            threshold -= 0.03
        if self.loss_streak >= 2:
            threshold += 0.10

        # ボラティリティによる調整
        if volatility > 0.03:
            # 高ボラ時は利益チャンス大
            threshold -= 0.05
        elif volatility < 0.01:
            # 低ボラ時は厳しめ
            threshold += 0.03

        # 範囲内に収める
        self.current_threshold = np.clip(threshold, self.min_confidence, self.max_confidence)
        return self.current_threshold

    def record_trade(self, profit: float) -> None:
        """取引結果記録"""
        self.trade_times.append(datetime.now())
        self.recent_results.append(profit)

        if profit > 0:
            self.win_streak += 1
            self.loss_streak = 0
        else:
            self.loss_streak += 1
            self.win_streak = 0

    def _count_trades_last_hour(self) -> int:
        """直近1時間の取引数"""
        cutoff = datetime.now() - timedelta(hours=1)
        return sum(1 for t in self.trade_times if t > cutoff)

    def check_indicator(
        self,
        name: str,
        signal: bool,
        confidence: float,
    ) -> None:
        """インジケータをチェック"""
        self.indicators[name] = signal
        self.indicator_confidences[name] = confidence

    def is_gate_open(self, volatility: float = 0.02) -> Tuple[bool, float, Dict[str, Any]]:
        """
        ゲートが開いているか（取引可能か）

        動的閾値を使用して判定

        Args:
            volatility: 現在のボラティリティ

        Returns:
            (is_open, overall_confidence, details)
        """
        if not self.indicators:
            return False, 0.0, {"reason": "No indicators"}

        # 動的閾値更新
        dynamic_threshold = self.update_threshold(volatility)

        # 一致インジケータをカウント
        aligned = sum(1 for v in self.indicators.values() if v)

        # 加重平均確信度
        total_weight = 0.0
        weighted_confidence = 0.0

        for name, signal in self.indicators.items():
            if signal:
                conf = self.indicator_confidences.get(name, 0.5)
                weighted_confidence += conf
                total_weight += 1

        overall_confidence = weighted_confidence / total_weight if total_weight > 0 else 0

        details = {
            "aligned_indicators": aligned,
            "total_indicators": len(self.indicators),
            "overall_confidence": overall_confidence,
            "dynamic_threshold": dynamic_threshold,
            "trades_last_hour": self._count_trades_last_hour(),
            "win_streak": self.win_streak,
            "loss_streak": self.loss_streak,
            "indicators": self.indicators.copy(),
        }

        # ゲート判定（動的閾値を使用）
        is_open = (
            aligned >= self.min_indicators_aligned and
            overall_confidence >= dynamic_threshold
        )

        return is_open, overall_confidence, details

    def clear(self) -> None:
        """インジケータをクリア"""
        self.indicators.clear()
        self.indicator_confidences.clear()


class InstantEvolution:
    """
    即時進化システム

    負けた瞬間に学習し、即座に進化
    """

    def __init__(self, evolution_speed: float = 0.5):
        """
        Args:
            evolution_speed: 進化速度（0-1、高いほど急激に変化）
        """
        self.evolution_speed = evolution_speed

        # 戦略パラメータ
        self.strategy_params: Dict[str, float] = {}

        # 負けパターンへの対策
        self.countermeasures: Dict[str, Dict[str, float]] = {}

        # 進化履歴
        self.evolution_history: List[Dict] = []
        self.generation = 0

    def evolve_from_loss(
        self,
        pattern: TradePattern,
        market_conditions: Dict[str, float],
    ) -> Dict[str, float]:
        """
        負けから進化

        Args:
            pattern: 負けたパターン
            market_conditions: 市場状況

        Returns:
            更新されたパラメータ
        """
        self.generation += 1

        # 負けた条件を分析
        loss_features = pattern.features

        # 対策を生成
        countermeasure = self._generate_countermeasure(
            loss_features, pattern.action, market_conditions
        )

        # パターンハッシュで保存
        self.countermeasures[pattern.pattern_hash] = countermeasure

        # 戦略パラメータを更新
        self._update_strategy_params(countermeasure)

        # 進化履歴
        self.evolution_history.append({
            "generation": self.generation,
            "timestamp": datetime.now(),
            "pattern_hash": pattern.pattern_hash,
            "countermeasure": countermeasure,
            "loss_amount": pattern.result,
        })

        logger.info(f"🧬 Evolution #{self.generation}: Evolved from loss pattern {pattern.pattern_hash}")

        return self.strategy_params

    def _generate_countermeasure(
        self,
        loss_features: np.ndarray,
        action: int,
        market_conditions: Dict[str, float],
    ) -> Dict[str, float]:
        """対策生成"""
        countermeasure = {}

        # 失敗したアクションを避ける
        countermeasure['avoid_action'] = action

        # 特徴量の逆の条件を設定
        countermeasure['feature_threshold_adjustment'] = -np.mean(loss_features) * 0.1

        # 市場条件に基づく調整
        if market_conditions.get('volatility', 0) > 0.02:
            countermeasure['reduce_position_size'] = 0.5
        else:
            countermeasure['reduce_position_size'] = 0.8

        # 確信度閾値を上げる
        countermeasure['increase_confidence_threshold'] = 0.05

        return countermeasure

    def _update_strategy_params(self, countermeasure: Dict[str, float]) -> None:
        """戦略パラメータ更新"""
        for key, value in countermeasure.items():
            if key not in self.strategy_params:
                self.strategy_params[key] = value
            else:
                # 指数移動平均で更新
                self.strategy_params[key] = (
                    (1 - self.evolution_speed) * self.strategy_params[key] +
                    self.evolution_speed * value
                )

    def get_adjusted_confidence_threshold(self, base_threshold: float) -> float:
        """調整された確信度閾値取得"""
        adjustment = self.strategy_params.get('increase_confidence_threshold', 0)
        return min(0.99, base_threshold + adjustment * self.generation * 0.01)


class UnbeatableSystem:
    """
    無敗システム

    勝率99%以上を目指す究極のシステム
    負けたら即座に進化し、同じ失敗を二度と繰り返さない
    動的閾値による高頻度取引対応
    """

    def __init__(
        self,
        feature_dim: int = 50,
        save_dir: str = "models/unbeatable",
        target_trades_per_hour: int = 50,
    ):
        self.feature_dim = feature_dim
        self.save_dir = save_dir
        self.target_trades_per_hour = target_trades_per_hour
        os.makedirs(save_dir, exist_ok=True)

        # コンポーネント
        self.loss_memory = LossMemory(similarity_threshold=0.90)
        self.consensus = MultiModelConsensus(min_agreement=0.70)  # 70%合意で取引
        self.confidence_gate = DynamicConfidenceGate(
            base_confidence=0.60,
            min_confidence=0.50,
            max_confidence=0.85,
            min_indicators_aligned=3,
            target_trades_per_hour=target_trades_per_hour,
        )
        self.evolution = InstantEvolution(evolution_speed=0.3)

        # 統計
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.consecutive_wins = 0
        self.max_consecutive_wins = 0

        # 履歴
        self.trade_history: List[TradePattern] = []

        # 状態
        self.is_locked = False  # 連続負け後のロック
        self.lock_until: Optional[datetime] = None

        logger.info("🏆 Unbeatable System initialized - High-frequency profit mode")

    def should_trade(
        self,
        features: np.ndarray,
        model_predictions: Dict[str, Tuple[int, float]],
        indicator_signals: Dict[str, Tuple[bool, float]],
        market_conditions: Dict[str, float] = None,
    ) -> Tuple[bool, int, float, str]:
        """
        取引すべきか判断

        Args:
            features: 特徴量
            model_predictions: モデル名 → (予測, 確信度)
            indicator_signals: インジケータ名 → (シグナル, 確信度)
            market_conditions: 市場状況

        Returns:
            (should_trade, action, confidence, reason)
        """
        market_conditions = market_conditions or {}

        # ロックチェック
        if self.is_locked:
            if self.lock_until and datetime.now() < self.lock_until:
                return False, 1, 0.0, "System locked after consecutive losses"
            self.is_locked = False

        # 1. 負けパターンチェック
        is_similar, similarity = self.loss_memory.is_similar_to_loss(features)
        if is_similar:
            logger.debug(f"⚠️ Similar to loss pattern (similarity: {similarity:.2%})")
            return False, 1, 0.0, f"Similar to loss pattern ({similarity:.2%})"

        # 2. マルチモデル合意
        self.consensus.clear()
        for model_name, (pred, conf) in model_predictions.items():
            self.consensus.register_prediction(model_name, pred, conf)

        action, agreement, has_consensus = self.consensus.get_consensus()

        if not has_consensus:
            return False, 1, 0.0, f"No model consensus ({agreement:.2%})"

        # HOLDの場合は取引しない
        if action == 1:
            return False, 1, agreement, "Consensus is HOLD"

        # 3. インジケータゲート（動的閾値を使用）
        volatility = market_conditions.get('volatility', 0.02)
        self.confidence_gate.clear()
        for indicator_name, (signal, conf) in indicator_signals.items():
            self.confidence_gate.check_indicator(indicator_name, signal, conf)

        gate_open, gate_confidence, gate_details = self.confidence_gate.is_gate_open(volatility)

        if not gate_open:
            return False, 1, 0.0, f"Gate closed (conf: {gate_confidence:.2%}, threshold: {gate_details.get('dynamic_threshold', 0):.2%})"

        # 4. 最終確信度計算
        final_confidence = min(agreement, gate_confidence)

        # 動的閾値との比較（ゲートが開いていれば基本的にOK）
        dynamic_threshold = gate_details.get('dynamic_threshold', 0.60)

        # 進化による微調整（負けが多い時のみ閾値上昇）
        if self.evolution.generation > 0 and len(self.trade_history) >= 10:
            recent_losses = sum(1 for p in self.trade_history[-10:] if not p.was_win)
            if recent_losses >= 3:
                dynamic_threshold += 0.05  # 連敗時は少し厳しく

        if final_confidence < dynamic_threshold:
            return False, 1, final_confidence, f"Below dynamic threshold ({dynamic_threshold:.2%})"

        # 5. 利益期待値チェック（手数料考慮）
        # 最低0.50の確信度があれば取引許可（高頻度取引のため）
        if final_confidence < 0.50:
            return False, 1, final_confidence, "Confidence below minimum 50%"

        return True, action, final_confidence, f"GO: conf={final_confidence:.2%}, trades/hr={gate_details.get('trades_last_hour', 0)}"

    def record_trade_result(
        self,
        features: np.ndarray,
        action: int,
        pnl: float,
        market_conditions: Dict[str, float] = None,
    ) -> None:
        """
        取引結果を記録

        Args:
            features: 特徴量
            action: 取ったアクション
            pnl: 損益
            market_conditions: 市場状況
        """
        market_conditions = market_conditions or {}
        was_win = pnl > 0

        pattern = TradePattern(
            features=features,
            action=action,
            result=pnl,
            was_win=was_win,
            timestamp=datetime.now(),
        )

        self.trade_history.append(pattern)
        self.total_trades += 1

        # 動的閾値マネージャーにも記録
        self.confidence_gate.record_trade(pnl)

        if was_win:
            self.winning_trades += 1
            self.consecutive_wins += 1
            self.max_consecutive_wins = max(self.max_consecutive_wins, self.consecutive_wins)
            logger.info(f"✅ Win #{self.winning_trades} (Streak: {self.consecutive_wins})")
        else:
            self.losing_trades += 1
            self.consecutive_wins = 0

            # 負けを記憶
            self.loss_memory.remember_loss(pattern)

            # 即座に進化
            self.evolution.evolve_from_loss(pattern, market_conditions)

            # 連続負けでロック（3連敗でロック、時間を短縮）
            recent_losses = sum(1 for p in self.trade_history[-5:] if not p.was_win)
            if recent_losses >= 3:
                self.is_locked = True
                self.lock_until = datetime.now() + timedelta(minutes=2)  # 2分に短縮
                logger.warning(f"🔒 System locked for 2 minutes after {recent_losses} recent losses")

            logger.warning(f"❌ Loss recorded - Evolving to prevent recurrence")

        # 定期保存
        if self.total_trades % 100 == 0:
            self.save_state()

    def get_stats(self) -> Dict[str, Any]:
        """統計取得"""
        win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0
        trades_per_hour = self.confidence_gate._count_trades_last_hour()

        return {
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": win_rate,
            "win_rate_pct": f"{win_rate * 100:.2f}%",
            "consecutive_wins": self.consecutive_wins,
            "max_consecutive_wins": self.max_consecutive_wins,
            "loss_patterns_memorized": len(self.loss_memory.loss_patterns),
            "evolution_generation": self.evolution.generation,
            "is_locked": self.is_locked,
            "trades_per_hour": trades_per_hour,
            "target_trades_per_hour": self.target_trades_per_hour,
            "current_threshold": self.confidence_gate.current_threshold,
            "win_streak": self.confidence_gate.win_streak,
            "loss_streak": self.confidence_gate.loss_streak,
            "target_achieved": win_rate >= 0.70 and trades_per_hour >= self.target_trades_per_hour * 0.5,
        }

    def save_state(self) -> None:
        """状態保存"""
        state = {
            "loss_patterns": self.loss_memory.loss_patterns,
            "loss_hashes": self.loss_memory.loss_hashes,
            "evolution_params": self.evolution.strategy_params,
            "evolution_generation": self.evolution.generation,
            "countermeasures": self.evolution.countermeasures,
            "stats": self.get_stats(),
        }

        filepath = os.path.join(self.save_dir, "unbeatable_state.pkl")
        with open(filepath, "wb") as f:
            pickle.dump(state, f)

        logger.info(f"💾 Unbeatable system state saved (Generation {self.evolution.generation})")

    def load_state(self) -> bool:
        """状態読み込み"""
        filepath = os.path.join(self.save_dir, "unbeatable_state.pkl")

        if not os.path.exists(filepath):
            return False

        try:
            with open(filepath, "rb") as f:
                state = pickle.load(f)

            self.loss_memory.loss_patterns = state["loss_patterns"]
            self.loss_memory.loss_hashes = state["loss_hashes"]
            self.evolution.strategy_params = state["evolution_params"]
            self.evolution.generation = state["evolution_generation"]
            self.evolution.countermeasures = state["countermeasures"]

            logger.info(f"📂 Unbeatable system state loaded (Generation {self.evolution.generation})")
            return True

        except Exception as e:
            logger.error(f"Failed to load state: {e}")
            return False


class UltimateDecisionMaker:
    """
    究極の意思決定システム

    高頻度取引と高利益のバランスを取る最終判断システム
    動的閾値により取引頻度を確保しつつ利益を最大化
    """

    def __init__(self, target_trades_per_hour: int = 50):
        self.unbeatable = UnbeatableSystem(target_trades_per_hour=target_trades_per_hour)

        # 追加のセーフティチェック
        self.safety_checks: List[callable] = []

        # 最低確信度（動的閾値の下限）
        self.min_confidence = 0.50

    def add_safety_check(self, check_func: callable) -> None:
        """セーフティチェック追加"""
        self.safety_checks.append(check_func)

    def make_decision(
        self,
        features: np.ndarray,
        model_predictions: Dict[str, Tuple[int, float]],
        indicator_signals: Dict[str, Tuple[bool, float]],
        market_conditions: Dict[str, float] = None,
    ) -> Tuple[int, float, bool, str]:
        """
        最終決定

        Returns:
            (action, confidence, should_execute, reason)
        """
        # 無敗システムのチェック（動的閾値を使用）
        should_trade, action, confidence, reason = self.unbeatable.should_trade(
            features, model_predictions, indicator_signals, market_conditions
        )

        if not should_trade:
            return 1, 0.0, False, reason  # HOLD

        # 追加セーフティチェック
        for i, check_func in enumerate(self.safety_checks):
            try:
                passed = check_func(features, action, confidence)
                if not passed:
                    return 1, 0.0, False, f"Failed safety check #{i+1}"
            except Exception as e:
                logger.error(f"Safety check error: {e}")
                return 1, 0.0, False, f"Safety check error"

        # 最低確信度チェック
        if confidence < self.min_confidence:
            return 1, confidence, False, f"Below minimum confidence ({self.min_confidence})"

        return action, confidence, True, reason

    def record_result(
        self,
        features: np.ndarray,
        action: int,
        pnl: float,
        market_conditions: Dict[str, float] = None,
    ) -> None:
        """結果記録"""
        self.unbeatable.record_trade_result(features, action, pnl, market_conditions)

    def get_performance_report(self) -> str:
        """パフォーマンスレポート"""
        stats = self.unbeatable.get_stats()

        target_status = '✅ TARGET OK!' if stats['target_achieved'] else '⏳ Optimizing...'

        report = f"""
╔══════════════════════════════════════════════════════════════════╗
║       🚀 HIGH-FREQUENCY PROFIT SYSTEM PERFORMANCE 🚀              ║
╠══════════════════════════════════════════════════════════════════╣
║ WIN RATE: {stats['win_rate_pct']:>10}  {target_status:>25}║
╠══════════════════════════════════════════════════════════════════╣
║ TRADING FREQUENCY                                                 ║
║ Trades/Hour:      {stats['trades_per_hour']:>10,} / {stats['target_trades_per_hour']:>4} target                       ║
║ Current Threshold:{stats['current_threshold']:>9.1%}                                    ║
╠══════════════════════════════════════════════════════════════════╣
║ TRADE STATISTICS                                                  ║
║ Total Trades:     {stats['total_trades']:>10,}                                    ║
║ Winning Trades:   {stats['winning_trades']:>10,}                                    ║
║ Losing Trades:    {stats['losing_trades']:>10,}                                    ║
║ Win Streak:       {stats['win_streak']:>10,}                                    ║
║ Loss Streak:      {stats['loss_streak']:>10,}                                    ║
║ Max Win Streak:   {stats['max_consecutive_wins']:>10,}                                    ║
╠══════════════════════════════════════════════════════════════════╣
║ EVOLUTION STATUS                                                  ║
║ Generation:       {stats['evolution_generation']:>10,}                                    ║
║ Patterns Memorized:{stats['loss_patterns_memorized']:>9,}                                    ║
║ System Lock:      {'🔒 LOCKED' if stats['is_locked'] else '🔓 ACTIVE':>10}                                    ║
╚══════════════════════════════════════════════════════════════════╝
"""
        return report

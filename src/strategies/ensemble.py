"""
Ensemble Strategy
==================
複数戦略を統合したアンサンブル戦略
"""

from typing import Optional, Dict, List
from datetime import datetime
import numpy as np
from loguru import logger

from .base import BaseStrategy, Signal, SignalType, MarketData


class EnsembleStrategy(BaseStrategy):
    """
    アンサンブル戦略

    特徴:
    - 複数の戦略シグナルを統合
    - 重み付き投票
    - 動的重み調整
    - リスク調整済みシグナル生成
    """

    def __init__(
        self,
        product_codes: List[str],
        strategies: List[BaseStrategy] = None,
        voting_method: str = "soft",  # soft or hard
        weight: float = 1.0,
    ):
        """
        Args:
            strategies: 統合する戦略のリスト
            voting_method: 投票方法 (soft=確率加重, hard=多数決)
        """
        super().__init__(
            name="Ensemble",
            product_codes=product_codes,
            weight=weight,
        )

        self.strategies = strategies or []
        self.voting_method = voting_method

        # 戦略パフォーマンス追跡
        self.strategy_performance: Dict[str, Dict] = {}
        self.signal_history: List[Dict] = []

        # 動的重み（パフォーマンスに基づいて更新）
        self.dynamic_weights: Dict[str, float] = {}

    def add_strategy(self, strategy: BaseStrategy) -> None:
        """戦略追加"""
        self.strategies.append(strategy)
        self.strategy_performance[strategy.name] = {
            "total_signals": 0,
            "profitable_signals": 0,
            "total_pnl": 0.0,
        }
        self.dynamic_weights[strategy.name] = strategy.weight

    def update(self, market_data: MarketData) -> None:
        """市場データで全戦略を更新"""
        for strategy in self.strategies:
            if strategy.enabled:
                strategy.update(market_data)

    def generate_signal(self, market_data: MarketData) -> Optional[Signal]:
        """アンサンブルシグナル生成"""
        if not self.strategies:
            return None

        # 各戦略からシグナル収集
        signals: List[Signal] = []

        for strategy in self.strategies:
            if not strategy.enabled:
                continue

            signal = strategy.generate_signal(market_data)
            if signal:
                signals.append(signal)

        if not signals:
            return None

        # シグナル統合
        if self.voting_method == "soft":
            return self._soft_voting(signals, market_data)
        else:
            return self._hard_voting(signals, market_data)

    def _soft_voting(
        self,
        signals: List[Signal],
        market_data: MarketData
    ) -> Optional[Signal]:
        """ソフト投票（確率加重）"""
        buy_score = 0.0
        sell_score = 0.0
        total_weight = 0.0

        for signal in signals:
            weight = self.dynamic_weights.get(signal.strategy_name, 1.0)
            confidence = signal.confidence

            if signal.signal_type == SignalType.BUY:
                buy_score += weight * confidence
            elif signal.signal_type == SignalType.SELL:
                sell_score += weight * confidence

            total_weight += weight

        if total_weight == 0:
            return None

        # 正規化
        buy_score /= total_weight
        sell_score /= total_weight

        # 閾値チェック
        threshold = 0.3  # 最低限の信頼度

        if buy_score > sell_score and buy_score > threshold:
            signal_type = SignalType.BUY
            confidence = buy_score
        elif sell_score > buy_score and sell_score > threshold:
            signal_type = SignalType.SELL
            confidence = sell_score
        else:
            return None

        # ポジションサイズ計算
        avg_size = np.mean([s.size for s in signals])
        size = avg_size * confidence

        # メタデータ収集
        contributing_strategies = [
            {
                "name": s.strategy_name,
                "signal": s.signal_type.value,
                "confidence": s.confidence,
            }
            for s in signals
        ]

        self.signals_generated += 1

        return Signal(
            signal_type=signal_type,
            product_code=market_data.product_code,
            price=market_data.close,
            size=size,
            confidence=confidence,
            strategy_name=self.name,
            metadata={
                "voting_method": "soft",
                "buy_score": buy_score,
                "sell_score": sell_score,
                "contributing_strategies": contributing_strategies,
                "num_signals": len(signals),
            },
        )

    def _hard_voting(
        self,
        signals: List[Signal],
        market_data: MarketData
    ) -> Optional[Signal]:
        """ハード投票（多数決）"""
        buy_votes = 0
        sell_votes = 0
        buy_signals = []
        sell_signals = []

        for signal in signals:
            weight = self.dynamic_weights.get(signal.strategy_name, 1.0)

            if signal.signal_type == SignalType.BUY:
                buy_votes += weight
                buy_signals.append(signal)
            elif signal.signal_type == SignalType.SELL:
                sell_votes += weight
                sell_signals.append(signal)

        total_votes = buy_votes + sell_votes
        if total_votes == 0:
            return None

        # 多数派の決定
        if buy_votes > sell_votes:
            signal_type = SignalType.BUY
            winning_signals = buy_signals
            confidence = buy_votes / total_votes
        elif sell_votes > buy_votes:
            signal_type = SignalType.SELL
            winning_signals = sell_signals
            confidence = sell_votes / total_votes
        else:
            return None  # 同数の場合はシグナルなし

        # 閾値チェック（過半数以上）
        if confidence < 0.5:
            return None

        # ポジションサイズ
        avg_size = np.mean([s.size for s in winning_signals])
        size = avg_size * confidence

        self.signals_generated += 1

        return Signal(
            signal_type=signal_type,
            product_code=market_data.product_code,
            price=market_data.close,
            size=size,
            confidence=confidence,
            strategy_name=self.name,
            metadata={
                "voting_method": "hard",
                "buy_votes": buy_votes,
                "sell_votes": sell_votes,
                "num_signals": len(signals),
            },
        )

    def update_strategy_performance(
        self,
        strategy_name: str,
        pnl: float,
        was_profitable: bool
    ) -> None:
        """戦略パフォーマンス更新"""
        if strategy_name not in self.strategy_performance:
            return

        perf = self.strategy_performance[strategy_name]
        perf["total_signals"] += 1
        perf["total_pnl"] += pnl
        if was_profitable:
            perf["profitable_signals"] += 1

        # 動的重み更新
        self._update_dynamic_weights()

    def _update_dynamic_weights(self) -> None:
        """動的重み更新"""
        for strategy in self.strategies:
            name = strategy.name
            if name not in self.strategy_performance:
                continue

            perf = self.strategy_performance[name]

            if perf["total_signals"] < 10:
                # データ不足時は初期重みを維持
                continue

            # 勝率ベースの重み調整
            win_rate = perf["profitable_signals"] / perf["total_signals"]

            # 基本重みから勝率で調整
            base_weight = strategy.weight
            adjusted_weight = base_weight * (0.5 + win_rate)

            # 重みの範囲制限
            self.dynamic_weights[name] = max(0.1, min(2.0, adjusted_weight))

    def get_strategy_stats(self) -> Dict[str, Dict]:
        """全戦略の統計取得"""
        stats = {}

        for strategy in self.strategies:
            name = strategy.name
            perf = self.strategy_performance.get(name, {})

            total = perf.get("total_signals", 0)
            profitable = perf.get("profitable_signals", 0)
            pnl = perf.get("total_pnl", 0)

            stats[name] = {
                "enabled": strategy.enabled,
                "base_weight": strategy.weight,
                "dynamic_weight": self.dynamic_weights.get(name, strategy.weight),
                "total_signals": total,
                "win_rate": profitable / total if total > 0 else 0,
                "total_pnl": pnl,
                "avg_pnl": pnl / total if total > 0 else 0,
            }

        return stats

    def get_consensus_strength(self) -> float:
        """コンセンサス強度取得（-1〜1）"""
        if not self.signal_history:
            return 0.0

        recent = self.signal_history[-20:]

        buy_count = sum(1 for s in recent if s.get("type") == "BUY")
        sell_count = sum(1 for s in recent if s.get("type") == "SELL")

        total = buy_count + sell_count
        if total == 0:
            return 0.0

        return (buy_count - sell_count) / total

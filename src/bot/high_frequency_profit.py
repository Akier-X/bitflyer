"""
High Frequency Profit System
=============================
手数料を考慮した高頻度・高利益トレーディングシステム
小さな利益を高速で積み重ねる
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import deque
from loguru import logger


@dataclass
class FeeStructure:
    """手数料構造"""
    maker_fee: float = 0.0001      # 0.01% (Maker)
    taker_fee: float = 0.0015      # 0.15% (Taker)
    lightning_maker: float = 0.0   # Lightning Maker無料の場合
    lightning_taker: float = 0.0001  # 0.01%

    def get_fee(self, is_maker: bool, is_lightning: bool = False) -> float:
        """手数料率取得"""
        if is_lightning:
            return self.lightning_maker if is_maker else self.lightning_taker
        return self.maker_fee if is_maker else self.taker_fee

    def get_minimum_profit(self, is_maker: bool = True) -> float:
        """手数料を超える最小利益率"""
        fee = self.get_fee(is_maker, is_lightning=True)
        # 往復手数料 + 最小利益マージン
        return fee * 2 + 0.0002  # 0.02%の最小利益


@dataclass
class TradeOpportunity:
    """取引機会"""
    product_code: str
    action: int  # 0: BUY, 1: HOLD, 2: SELL
    confidence: float
    expected_profit: float  # 期待利益率
    expected_profit_after_fee: float  # 手数料後期待利益
    urgency: float  # 緊急度 (0-1)
    time_horizon: int  # 予想保有時間（秒）
    risk_reward_ratio: float


class DynamicThresholdManager:
    """
    動的閾値マネージャー

    市場状況と取引頻度に応じて閾値を自動調整
    """

    def __init__(
        self,
        base_confidence: float = 0.60,  # ベース確信度（低めに設定）
        min_confidence: float = 0.50,
        max_confidence: float = 0.85,
        target_trades_per_hour: int = 50,
    ):
        self.base_confidence = base_confidence
        self.min_confidence = min_confidence
        self.max_confidence = max_confidence
        self.target_trades_per_hour = target_trades_per_hour

        # 現在の閾値
        self.current_confidence = base_confidence

        # 取引履歴
        self.trade_times: deque = deque(maxlen=1000)
        self.recent_profits: deque = deque(maxlen=100)

        # 適応パラメータ
        self.win_streak = 0
        self.loss_streak = 0

    def get_threshold(self, market_volatility: float = 0.02) -> float:
        """
        現在の閾値を取得

        Args:
            market_volatility: 市場ボラティリティ

        Returns:
            確信度閾値
        """
        # ベース閾値
        threshold = self.base_confidence

        # 取引頻度による調整
        trades_last_hour = self._count_trades_last_hour()

        if trades_last_hour < self.target_trades_per_hour * 0.5:
            # 取引少なすぎ → 閾値下げる
            threshold -= 0.05
        elif trades_last_hour > self.target_trades_per_hour * 1.5:
            # 取引多すぎ → 閾値上げる
            threshold += 0.05

        # 勝率による調整
        if len(self.recent_profits) >= 10:
            recent_win_rate = sum(1 for p in self.recent_profits if p > 0) / len(self.recent_profits)
            if recent_win_rate > 0.7:
                # 勝率高い → 少し積極的に
                threshold -= 0.03
            elif recent_win_rate < 0.5:
                # 勝率低い → 慎重に
                threshold += 0.05

        # 連勝/連敗による調整
        if self.win_streak >= 5:
            threshold -= 0.02  # 連勝中は積極的
        if self.loss_streak >= 2:
            threshold += 0.05  # 連敗中は慎重

        # ボラティリティによる調整
        if market_volatility > 0.03:
            # 高ボラ時は利益チャンス大
            threshold -= 0.05
        elif market_volatility < 0.01:
            # 低ボラ時は厳しめ
            threshold += 0.03

        # 範囲内に収める
        return np.clip(threshold, self.min_confidence, self.max_confidence)

    def record_trade(self, profit: float) -> None:
        """取引結果記録"""
        self.trade_times.append(datetime.now())
        self.recent_profits.append(profit)

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


class ProfitCalculator:
    """
    利益計算機

    手数料を考慮した正確な利益計算
    """

    def __init__(self, fee_structure: FeeStructure = None):
        self.fees = fee_structure or FeeStructure()

    def calculate_expected_profit(
        self,
        entry_price: float,
        target_price: float,
        stop_price: float,
        confidence: float,
        is_maker: bool = True,
    ) -> Dict[str, float]:
        """
        期待利益計算

        Args:
            entry_price: エントリー価格
            target_price: 目標価格
            stop_price: 損切り価格
            confidence: 勝率予想
            is_maker: Maker注文かどうか

        Returns:
            各種利益指標
        """
        # 手数料
        fee = self.fees.get_fee(is_maker, is_lightning=True)
        round_trip_fee = fee * 2

        # 勝ち/負け時の利益率
        win_profit = (target_price - entry_price) / entry_price - round_trip_fee
        loss_profit = (stop_price - entry_price) / entry_price - round_trip_fee

        # 期待利益
        expected_profit = confidence * win_profit + (1 - confidence) * loss_profit

        # リスクリワード比
        risk = abs(entry_price - stop_price)
        reward = abs(target_price - entry_price)
        risk_reward = reward / risk if risk > 0 else 0

        # Kelly比率
        if loss_profit != 0:
            kelly = (confidence * risk_reward - (1 - confidence)) / risk_reward
        else:
            kelly = 0

        return {
            "expected_profit": expected_profit,
            "win_profit": win_profit,
            "loss_profit": loss_profit,
            "round_trip_fee": round_trip_fee,
            "risk_reward_ratio": risk_reward,
            "kelly_fraction": max(0, kelly),
            "breakeven_winrate": round_trip_fee / (win_profit - loss_profit + round_trip_fee) if (win_profit - loss_profit + round_trip_fee) != 0 else 0.5,
        }

    def is_profitable_trade(
        self,
        expected_move: float,
        confidence: float,
        is_maker: bool = True,
    ) -> Tuple[bool, float]:
        """
        利益が出る取引かどうか

        Args:
            expected_move: 予想価格変動率
            confidence: 確信度
            is_maker: Maker注文か

        Returns:
            (利益が出るか, 期待利益率)
        """
        fee = self.fees.get_fee(is_maker, is_lightning=True)
        min_profit = self.fees.get_minimum_profit(is_maker)

        # 手数料後の期待利益
        gross_profit = expected_move * confidence - expected_move * 0.3 * (1 - confidence)
        net_profit = gross_profit - fee * 2

        return net_profit > 0, net_profit


class ScalpingStrategy:
    """
    スキャルピング戦略

    小さな値幅を高頻度で取る
    """

    def __init__(
        self,
        min_spread_capture: float = 0.0003,  # 0.03%の値幅
        max_hold_time: int = 60,  # 最大60秒保有
        fee_structure: FeeStructure = None,
    ):
        self.min_spread_capture = min_spread_capture
        self.max_hold_time = max_hold_time
        self.fees = fee_structure or FeeStructure()
        self.profit_calc = ProfitCalculator(self.fees)

        # 価格履歴
        self.price_history: deque = deque(maxlen=1000)
        self.spread_history: deque = deque(maxlen=100)

    def analyze(
        self,
        current_price: float,
        best_bid: float,
        best_ask: float,
        order_book_imbalance: float,
    ) -> Optional[TradeOpportunity]:
        """
        スキャルピング機会分析

        Args:
            current_price: 現在価格
            best_bid: 最良買い気配
            best_ask: 最良売り気配
            order_book_imbalance: オーダーブック不均衡

        Returns:
            取引機会（なければNone）
        """
        # スプレッド計算
        spread = (best_ask - best_bid) / current_price
        self.spread_history.append(spread)

        # スプレッドが十分か
        min_required = self.fees.get_minimum_profit(is_maker=True)

        if spread < min_required:
            return None  # スプレッド不足

        # 方向予測（オーダーブック不均衡から）
        if abs(order_book_imbalance) < 0.1:
            return None  # 方向感なし

        # 確信度計算
        confidence = 0.5 + abs(order_book_imbalance) * 0.3

        # アクション決定
        if order_book_imbalance > 0.1:
            action = 0  # BUY (買い圧力)
            target_price = current_price * (1 + self.min_spread_capture)
            stop_price = current_price * (1 - self.min_spread_capture * 0.5)
        else:
            action = 2  # SELL (売り圧力)
            target_price = current_price * (1 - self.min_spread_capture)
            stop_price = current_price * (1 + self.min_spread_capture * 0.5)

        # 期待利益計算
        profit_info = self.profit_calc.calculate_expected_profit(
            entry_price=current_price,
            target_price=target_price,
            stop_price=stop_price,
            confidence=confidence,
            is_maker=True,
        )

        if profit_info["expected_profit"] <= 0:
            return None

        return TradeOpportunity(
            product_code="",
            action=action,
            confidence=confidence,
            expected_profit=profit_info["expected_profit"] + profit_info["round_trip_fee"],
            expected_profit_after_fee=profit_info["expected_profit"],
            urgency=min(1.0, abs(order_book_imbalance)),
            time_horizon=10,  # 10秒想定
            risk_reward_ratio=profit_info["risk_reward_ratio"],
        )


class SpreadCaptureStrategy:
    """
    スプレッド取得戦略

    Maker手数料を活用してスプレッドを取得
    """

    def __init__(self, min_spread_ratio: float = 0.0005):
        self.min_spread_ratio = min_spread_ratio
        self.active_orders: Dict[str, Dict] = {}

    def get_quotes(
        self,
        mid_price: float,
        volatility: float,
        inventory: float,
    ) -> Tuple[float, float]:
        """
        買い値・売り値のクオート取得

        Args:
            mid_price: 中央価格
            volatility: ボラティリティ
            inventory: 在庫（正=ロング、負=ショート）

        Returns:
            (買い価格, 売り価格)
        """
        # ベーススプレッド（ボラに応じて調整）
        base_spread = max(self.min_spread_ratio, volatility * 0.5)

        # 在庫スキュー（在庫を減らす方向に価格調整）
        inventory_skew = inventory * 0.0001

        bid_price = mid_price * (1 - base_spread / 2 - inventory_skew)
        ask_price = mid_price * (1 + base_spread / 2 - inventory_skew)

        return bid_price, ask_price

    def should_quote(
        self,
        current_spread: float,
        volatility: float,
    ) -> bool:
        """クオートすべきか"""
        # スプレッドが十分広く、ボラが適度な時
        return current_spread >= self.min_spread_ratio and volatility < 0.05


class MomentumScalper:
    """
    モメンタムスキャルパー

    短期モメンタムで高速取引
    """

    def __init__(
        self,
        lookback: int = 20,
        momentum_threshold: float = 0.001,
    ):
        self.lookback = lookback
        self.momentum_threshold = momentum_threshold
        self.prices: deque = deque(maxlen=lookback)
        self.volumes: deque = deque(maxlen=lookback)

    def update(self, price: float, volume: float) -> None:
        """データ更新"""
        self.prices.append(price)
        self.volumes.append(volume)

    def get_signal(self) -> Tuple[int, float]:
        """
        シグナル取得

        Returns:
            (action, confidence)
        """
        if len(self.prices) < self.lookback:
            return 1, 0.0  # HOLD

        prices = np.array(self.prices)
        volumes = np.array(self.volumes)

        # 短期モメンタム
        momentum_5 = (prices[-1] - prices[-5]) / prices[-5] if len(prices) >= 5 else 0
        momentum_10 = (prices[-1] - prices[-10]) / prices[-10] if len(prices) >= 10 else 0

        # 出来高加重モメンタム
        if volumes.sum() > 0:
            vwap = np.sum(prices * volumes) / volumes.sum()
            vwap_signal = (prices[-1] - vwap) / vwap
        else:
            vwap_signal = 0

        # 総合スコア
        score = momentum_5 * 0.5 + momentum_10 * 0.3 + vwap_signal * 0.2

        # シグナル生成
        if score > self.momentum_threshold:
            confidence = min(0.8, 0.5 + abs(score) * 10)
            return 0, confidence  # BUY
        elif score < -self.momentum_threshold:
            confidence = min(0.8, 0.5 + abs(score) * 10)
            return 2, confidence  # SELL
        else:
            return 1, 0.5  # HOLD


class HighFrequencyProfitSystem:
    """
    高頻度利益システム

    手数料を考慮しながら高頻度で利益を積み重ねる
    """

    def __init__(
        self,
        fee_structure: FeeStructure = None,
        target_trades_per_hour: int = 100,
        min_profit_per_trade: float = 0.0002,  # 0.02%
    ):
        self.fees = fee_structure or FeeStructure()
        self.target_trades_per_hour = target_trades_per_hour
        self.min_profit_per_trade = min_profit_per_trade

        # 戦略群
        self.scalping = ScalpingStrategy(fee_structure=self.fees)
        self.spread_capture = SpreadCaptureStrategy()
        self.momentum_scalper = MomentumScalper()

        # 閾値マネージャー
        self.threshold_manager = DynamicThresholdManager(
            base_confidence=0.55,  # 低めのベース閾値
            min_confidence=0.50,
            max_confidence=0.75,
            target_trades_per_hour=target_trades_per_hour,
        )

        # 利益計算機
        self.profit_calc = ProfitCalculator(self.fees)

        # 統計
        self.total_trades = 0
        self.total_profit = 0.0
        self.hourly_profits: deque = deque(maxlen=24)

    def analyze_opportunity(
        self,
        product_code: str,
        current_price: float,
        best_bid: float,
        best_ask: float,
        volume: float,
        order_book_imbalance: float,
        volatility: float,
    ) -> Optional[TradeOpportunity]:
        """
        取引機会分析

        Returns:
            最良の取引機会
        """
        opportunities = []

        # 1. スキャルピング機会
        scalp_opp = self.scalping.analyze(
            current_price, best_bid, best_ask, order_book_imbalance
        )
        if scalp_opp:
            scalp_opp.product_code = product_code
            opportunities.append(scalp_opp)

        # 2. モメンタムスキャルピング
        self.momentum_scalper.update(current_price, volume)
        mom_action, mom_conf = self.momentum_scalper.get_signal()

        if mom_action != 1 and mom_conf > 0.55:
            # 期待利益計算
            expected_move = volatility * mom_conf
            is_profitable, net_profit = self.profit_calc.is_profitable_trade(
                expected_move, mom_conf, is_maker=True
            )

            if is_profitable:
                opportunities.append(TradeOpportunity(
                    product_code=product_code,
                    action=mom_action,
                    confidence=mom_conf,
                    expected_profit=expected_move,
                    expected_profit_after_fee=net_profit,
                    urgency=0.7,
                    time_horizon=30,
                    risk_reward_ratio=2.0,
                ))

        # 3. スプレッド取得機会
        mid_price = (best_bid + best_ask) / 2
        current_spread = (best_ask - best_bid) / mid_price

        if self.spread_capture.should_quote(current_spread, volatility):
            opportunities.append(TradeOpportunity(
                product_code=product_code,
                action=0,  # 両サイドにクオート
                confidence=0.65,
                expected_profit=current_spread / 2,
                expected_profit_after_fee=current_spread / 2 - self.fees.maker_fee * 2,
                urgency=0.5,
                time_horizon=60,
                risk_reward_ratio=1.5,
            ))

        # 最良の機会を選択
        if not opportunities:
            return None

        # 期待利益でソート
        opportunities.sort(key=lambda x: x.expected_profit_after_fee, reverse=True)

        best = opportunities[0]

        # 動的閾値チェック
        threshold = self.threshold_manager.get_threshold(volatility)

        if best.confidence < threshold:
            return None

        if best.expected_profit_after_fee < self.min_profit_per_trade:
            return None

        return best

    def should_trade(
        self,
        opportunity: TradeOpportunity,
        current_position: float,
        max_position: float,
    ) -> Tuple[bool, str]:
        """
        取引すべきか判断

        Args:
            opportunity: 取引機会
            current_position: 現在ポジション
            max_position: 最大ポジション

        Returns:
            (取引すべきか, 理由)
        """
        # ポジション制限
        if opportunity.action == 0 and current_position >= max_position:
            return False, "Max long position reached"
        if opportunity.action == 2 and current_position <= -max_position:
            return False, "Max short position reached"

        # 期待利益チェック
        if opportunity.expected_profit_after_fee <= 0:
            return False, "Expected profit is negative after fees"

        # リスクリワードチェック
        if opportunity.risk_reward_ratio < 1.0:
            return False, "Risk/reward ratio too low"

        return True, "OK"

    def record_trade_result(self, profit: float) -> None:
        """取引結果記録"""
        self.total_trades += 1
        self.total_profit += profit
        self.threshold_manager.record_trade(profit)

    def get_stats(self) -> Dict[str, Any]:
        """統計取得"""
        trades_per_hour = self.threshold_manager._count_trades_last_hour()

        if len(self.threshold_manager.recent_profits) > 0:
            recent_profits = list(self.threshold_manager.recent_profits)
            win_rate = sum(1 for p in recent_profits if p > 0) / len(recent_profits)
            avg_profit = np.mean(recent_profits)
        else:
            win_rate = 0
            avg_profit = 0

        return {
            "total_trades": self.total_trades,
            "total_profit": self.total_profit,
            "trades_per_hour": trades_per_hour,
            "target_trades_per_hour": self.target_trades_per_hour,
            "current_threshold": self.threshold_manager.get_threshold(),
            "recent_win_rate": win_rate,
            "avg_profit_per_trade": avg_profit,
            "win_streak": self.threshold_manager.win_streak,
            "loss_streak": self.threshold_manager.loss_streak,
        }


class AdaptiveProfitOptimizer:
    """
    適応型利益最適化

    取引頻度と利益のバランスを自動最適化
    """

    def __init__(self):
        # 目標設定
        self.target_monthly_return = 0.30  # 30%
        self.target_daily_return = 0.30 / 30  # 約1%/日
        self.target_hourly_trades = 50

        # 現在のパフォーマンス追跡
        self.hourly_returns: deque = deque(maxlen=24 * 7)  # 1週間
        self.trade_counts: deque = deque(maxlen=24)

        # 最適化パラメータ
        self.profit_weight = 0.5  # 利益重視
        self.frequency_weight = 0.5  # 頻度重視

    def optimize_parameters(
        self,
        current_win_rate: float,
        current_trades_per_hour: int,
        current_avg_profit: float,
    ) -> Dict[str, float]:
        """
        パラメータ最適化

        Returns:
            推奨パラメータ
        """
        recommendations = {}

        # 取引頻度が低すぎる
        if current_trades_per_hour < self.target_hourly_trades * 0.5:
            recommendations["reduce_confidence_threshold"] = 0.05
            recommendations["increase_position_size"] = 1.2

        # 取引頻度が高すぎる
        elif current_trades_per_hour > self.target_hourly_trades * 1.5:
            recommendations["increase_confidence_threshold"] = 0.03
            recommendations["decrease_position_size"] = 0.9

        # 勝率が低い
        if current_win_rate < 0.55:
            recommendations["increase_confidence_threshold"] = 0.05
            recommendations["tighten_stop_loss"] = 0.8

        # 勝率が高いが利益が少ない
        if current_win_rate > 0.65 and current_avg_profit < 0.0003:
            recommendations["increase_take_profit"] = 1.3
            recommendations["loosen_stop_loss"] = 1.2

        return recommendations

    def calculate_optimal_trade_size(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        capital: float,
    ) -> float:
        """
        最適取引サイズ（Kelly基準ベース）
        """
        if avg_loss == 0:
            return capital * 0.01

        # Kelly比率
        win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 1
        kelly = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio

        # 安全のため半分Kelly
        safe_kelly = max(0.001, min(0.1, kelly * 0.5))

        return capital * safe_kelly

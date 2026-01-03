"""
Risk Manager
=============
リスク管理システム
"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from loguru import logger


class RiskLevel(Enum):
    """リスクレベル"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class RiskMetrics:
    """リスクメトリクス"""
    current_drawdown: float = 0.0
    max_drawdown: float = 0.0
    daily_pnl: float = 0.0
    daily_loss: float = 0.0
    position_exposure: float = 0.0
    volatility: float = 0.0
    var_95: float = 0.0  # Value at Risk 95%
    sharpe_ratio: float = 0.0
    risk_level: RiskLevel = RiskLevel.LOW


@dataclass
class RiskLimits:
    """リスク制限"""
    max_position_size: float = 0.5       # 総資産の50%
    max_single_trade: float = 0.1        # 1取引で10%
    max_daily_loss: float = 0.05         # 1日の最大損失5%
    max_drawdown: float = 0.15           # 最大ドローダウン15%
    stop_loss_pct: float = 0.02          # 損切り2%
    take_profit_pct: float = 0.03        # 利確3%
    trailing_stop_pct: float = 0.01      # トレーリングストップ1%
    max_trades_per_hour: int = 100       # 1時間最大取引数
    max_correlation: float = 0.7         # 最大相関


class RiskManager:
    """
    リスク管理システム

    特徴:
    - 動的ポジションサイジング（Kelly基準）
    - ドローダウン管理
    - VaR (Value at Risk) 計算
    - 相関リスク管理
    - 自動損切り・利確
    """

    def __init__(self, limits: RiskLimits = None, initial_capital: float = 1000000):
        """
        Args:
            limits: リスク制限設定
            initial_capital: 初期資本
        """
        self.limits = limits or RiskLimits()
        self.initial_capital = initial_capital
        self.current_capital = initial_capital

        # ピーク資本（ドローダウン計算用）
        self.peak_capital = initial_capital

        # 履歴
        self.pnl_history: List[Dict] = []
        self.trade_history: List[Dict] = []
        self.daily_pnl: Dict[str, float] = {}

        # 現在のメトリクス
        self.metrics = RiskMetrics()

        # リスク状態
        self.is_trading_allowed = True
        self.risk_alerts: List[str] = []

    def check_trade_allowed(
        self,
        product_code: str,
        side: str,
        size: float,
        price: float,
        current_position: float = 0,
    ) -> Tuple[bool, str]:
        """
        取引許可チェック

        Args:
            product_code: 取引ペア
            side: 売買方向
            size: サイズ
            price: 価格
            current_position: 現在のポジション

        Returns:
            (許可されるか, 理由)
        """
        # 取引停止中
        if not self.is_trading_allowed:
            return False, "Trading is currently disabled"

        # ドローダウンチェック
        if self.metrics.current_drawdown >= self.limits.max_drawdown:
            self._add_alert("Max drawdown reached")
            return False, f"Max drawdown limit reached: {self.metrics.current_drawdown:.2%}"

        # 日次損失チェック
        today = datetime.now().strftime("%Y-%m-%d")
        daily_loss = abs(min(0, self.daily_pnl.get(today, 0)))
        if daily_loss >= self.initial_capital * self.limits.max_daily_loss:
            self._add_alert("Max daily loss reached")
            return False, f"Max daily loss limit reached: {daily_loss:.0f} JPY"

        # ポジションサイズチェック
        trade_value = size * price
        if trade_value > self.current_capital * self.limits.max_single_trade:
            return False, f"Trade size exceeds limit: {trade_value:.0f} > {self.current_capital * self.limits.max_single_trade:.0f}"

        # 総エクスポージャーチェック
        new_position = current_position + size if side == "BUY" else current_position - size
        position_value = abs(new_position) * price
        if position_value > self.current_capital * self.limits.max_position_size:
            return False, f"Position exceeds limit: {position_value:.0f} > {self.current_capital * self.limits.max_position_size:.0f}"

        # 取引頻度チェック
        recent_trades = self._count_recent_trades(hours=1)
        if recent_trades >= self.limits.max_trades_per_hour:
            return False, f"Trade frequency limit reached: {recent_trades}/{self.limits.max_trades_per_hour}"

        return True, "OK"

    def calculate_position_size(
        self,
        product_code: str,
        side: str,
        price: float,
        confidence: float = 0.5,
        volatility: float = None,
        method: str = "kelly",
    ) -> float:
        """
        ポジションサイズ計算

        Args:
            product_code: 取引ペア
            side: 売買方向
            price: 現在価格
            confidence: シグナル信頼度
            volatility: 現在のボラティリティ
            method: 計算方法 (kelly, fixed, volatility_adjusted)

        Returns:
            推奨ポジションサイズ
        """
        if method == "kelly":
            return self._kelly_criterion(confidence, volatility)
        elif method == "volatility_adjusted":
            return self._volatility_adjusted_size(volatility, price)
        else:
            return self._fixed_size(price)

    def _kelly_criterion(self, confidence: float, volatility: float = None) -> float:
        """
        Kelly基準によるポジションサイズ計算

        f* = (bp - q) / b
        b = オッズ (期待リターン/リスク)
        p = 勝率
        q = 1 - p
        """
        # 勝率推定（信頼度から）
        win_probability = confidence

        # オッズ推定（リスク・リワード比）
        risk_reward_ratio = self.limits.take_profit_pct / self.limits.stop_loss_pct

        # Kelly計算
        kelly = (risk_reward_ratio * win_probability - (1 - win_probability)) / risk_reward_ratio

        # 安全のため半分Kelly
        half_kelly = max(0, kelly * 0.5)

        # 最大制限
        max_fraction = self.limits.max_single_trade

        return min(half_kelly, max_fraction) * self.current_capital

    def _volatility_adjusted_size(self, volatility: float, price: float) -> float:
        """ボラティリティ調整サイズ"""
        if volatility is None or volatility == 0:
            return self._fixed_size(price)

        # 目標ボラティリティ
        target_volatility = 0.02  # 2%

        # サイズ調整
        adjustment = target_volatility / (volatility + 1e-10)
        adjustment = min(adjustment, 2.0)  # 最大2倍まで

        base_size = self.current_capital * self.limits.max_single_trade * 0.5
        adjusted_size = base_size * adjustment

        return adjusted_size / price

    def _fixed_size(self, price: float) -> float:
        """固定サイズ"""
        return (self.current_capital * self.limits.max_single_trade * 0.5) / price

    def calculate_stop_loss(
        self,
        entry_price: float,
        side: str,
        atr: float = None,
    ) -> float:
        """
        損切り価格計算

        Args:
            entry_price: エントリー価格
            side: 売買方向
            atr: ATR（ボラティリティ）

        Returns:
            損切り価格
        """
        if atr:
            # ATRベースの動的ストップ
            stop_distance = atr * 2
        else:
            stop_distance = entry_price * self.limits.stop_loss_pct

        if side == "BUY":
            return entry_price - stop_distance
        else:
            return entry_price + stop_distance

    def calculate_take_profit(
        self,
        entry_price: float,
        side: str,
        atr: float = None,
    ) -> float:
        """
        利確価格計算

        Args:
            entry_price: エントリー価格
            side: 売買方向
            atr: ATR

        Returns:
            利確価格
        """
        if atr:
            profit_distance = atr * 3
        else:
            profit_distance = entry_price * self.limits.take_profit_pct

        if side == "BUY":
            return entry_price + profit_distance
        else:
            return entry_price - profit_distance

    def update_trailing_stop(
        self,
        current_price: float,
        entry_price: float,
        side: str,
        current_stop: float,
    ) -> float:
        """
        トレーリングストップ更新

        Args:
            current_price: 現在価格
            entry_price: エントリー価格
            side: 売買方向
            current_stop: 現在のストップ価格

        Returns:
            新しいストップ価格
        """
        trailing_distance = current_price * self.limits.trailing_stop_pct

        if side == "BUY":
            new_stop = current_price - trailing_distance
            return max(current_stop, new_stop)
        else:
            new_stop = current_price + trailing_distance
            return min(current_stop, new_stop)

    def update_pnl(self, pnl: float, trade_info: Dict = None) -> None:
        """
        P&L更新

        Args:
            pnl: 損益
            trade_info: 取引情報
        """
        self.current_capital += pnl

        # ピーク更新
        if self.current_capital > self.peak_capital:
            self.peak_capital = self.current_capital

        # 履歴に追加
        now = datetime.now()
        self.pnl_history.append({
            'timestamp': now,
            'pnl': pnl,
            'capital': self.current_capital,
        })

        # 日次P&L更新
        today = now.strftime("%Y-%m-%d")
        self.daily_pnl[today] = self.daily_pnl.get(today, 0) + pnl

        # 取引履歴
        if trade_info:
            trade_info['timestamp'] = now
            trade_info['pnl'] = pnl
            self.trade_history.append(trade_info)

        # メトリクス更新
        self._update_metrics()

    def _update_metrics(self) -> None:
        """メトリクス更新"""
        # ドローダウン計算
        self.metrics.current_drawdown = (self.peak_capital - self.current_capital) / self.peak_capital

        # 最大ドローダウン
        if self.metrics.current_drawdown > self.metrics.max_drawdown:
            self.metrics.max_drawdown = self.metrics.current_drawdown

        # 日次損益
        today = datetime.now().strftime("%Y-%m-%d")
        self.metrics.daily_pnl = self.daily_pnl.get(today, 0)
        self.metrics.daily_loss = abs(min(0, self.metrics.daily_pnl))

        # VaR計算
        if len(self.pnl_history) >= 20:
            recent_pnls = [h['pnl'] for h in self.pnl_history[-100:]]
            self.metrics.var_95 = np.percentile(recent_pnls, 5)

        # ボラティリティ計算
        if len(self.pnl_history) >= 20:
            returns = [h['pnl'] / self.initial_capital for h in self.pnl_history[-20:]]
            self.metrics.volatility = np.std(returns)

        # シャープレシオ計算
        if len(self.pnl_history) >= 20 and self.metrics.volatility > 0:
            avg_return = np.mean(returns)
            risk_free_rate = 0.001  # 0.1%
            self.metrics.sharpe_ratio = (avg_return - risk_free_rate) / self.metrics.volatility

        # リスクレベル判定
        self._update_risk_level()

    def _update_risk_level(self) -> None:
        """リスクレベル更新"""
        if self.metrics.current_drawdown >= self.limits.max_drawdown * 0.9:
            self.metrics.risk_level = RiskLevel.CRITICAL
            self.is_trading_allowed = False
        elif self.metrics.current_drawdown >= self.limits.max_drawdown * 0.7:
            self.metrics.risk_level = RiskLevel.HIGH
        elif self.metrics.current_drawdown >= self.limits.max_drawdown * 0.5:
            self.metrics.risk_level = RiskLevel.MEDIUM
        else:
            self.metrics.risk_level = RiskLevel.LOW
            self.is_trading_allowed = True

    def _count_recent_trades(self, hours: int = 1) -> int:
        """直近の取引数カウント"""
        cutoff = datetime.now() - timedelta(hours=hours)
        return sum(1 for t in self.trade_history if t['timestamp'] > cutoff)

    def _add_alert(self, message: str) -> None:
        """アラート追加"""
        self.risk_alerts.append({
            'timestamp': datetime.now(),
            'message': message,
        })
        logger.warning(f"Risk Alert: {message}")

    def get_risk_summary(self) -> Dict:
        """リスクサマリー取得"""
        return {
            'current_capital': self.current_capital,
            'initial_capital': self.initial_capital,
            'total_return': (self.current_capital - self.initial_capital) / self.initial_capital,
            'current_drawdown': self.metrics.current_drawdown,
            'max_drawdown': self.metrics.max_drawdown,
            'daily_pnl': self.metrics.daily_pnl,
            'var_95': self.metrics.var_95,
            'sharpe_ratio': self.metrics.sharpe_ratio,
            'risk_level': self.metrics.risk_level.value,
            'is_trading_allowed': self.is_trading_allowed,
            'recent_alerts': self.risk_alerts[-5:] if self.risk_alerts else [],
        }

    def reset_daily(self) -> None:
        """日次リセット"""
        today = datetime.now().strftime("%Y-%m-%d")
        self.daily_pnl[today] = 0

        # アラートクリア
        self.risk_alerts = []

        # 取引再開判定
        if self.metrics.current_drawdown < self.limits.max_drawdown * 0.8:
            self.is_trading_allowed = True

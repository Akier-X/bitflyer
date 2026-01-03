"""
Performance Tracker
====================
パフォーマンス追跡・分析システム
"""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import numpy as np
from loguru import logger


@dataclass
class TradeRecord:
    """取引記録"""
    id: str
    product_code: str
    side: str
    size: float
    entry_price: float
    exit_price: float
    pnl: float
    pnl_pct: float
    entry_time: datetime
    exit_time: datetime
    holding_time: float  # 秒
    strategy: str
    signal_confidence: float


class PerformanceTracker:
    """
    パフォーマンス追跡システム

    特徴:
    - 詳細な取引統計
    - 時系列パフォーマンス分析
    - リスク調整済みリターン計算
    - 戦略別パフォーマンス追跡
    """

    def __init__(self, initial_capital: float = 1000000):
        """
        Args:
            initial_capital: 初期資本
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital

        # 取引記録
        self.trades: List[TradeRecord] = []

        # 資本履歴
        self.equity_curve: List[Dict] = []

        # 日次・月次統計
        self.daily_stats: Dict[str, Dict] = {}
        self.monthly_stats: Dict[str, Dict] = {}

        # ベンチマーク
        self.benchmark_returns: List[float] = []

    def record_trade(
        self,
        trade_id: str,
        product_code: str,
        side: str,
        size: float,
        entry_price: float,
        exit_price: float,
        entry_time: datetime,
        exit_time: datetime,
        strategy: str = "",
        signal_confidence: float = 0.0,
    ) -> TradeRecord:
        """
        取引記録

        Args:
            trade_id: 取引ID
            product_code: 取引ペア
            side: 売買方向
            size: サイズ
            entry_price: エントリー価格
            exit_price: 決済価格
            entry_time: エントリー時間
            exit_time: 決済時間
            strategy: 戦略名
            signal_confidence: シグナル信頼度

        Returns:
            取引記録
        """
        # P&L計算
        if side == "BUY":
            pnl = (exit_price - entry_price) * size
        else:
            pnl = (entry_price - exit_price) * size

        pnl_pct = pnl / (entry_price * size) * 100
        holding_time = (exit_time - entry_time).total_seconds()

        record = TradeRecord(
            id=trade_id,
            product_code=product_code,
            side=side,
            size=size,
            entry_price=entry_price,
            exit_price=exit_price,
            pnl=pnl,
            pnl_pct=pnl_pct,
            entry_time=entry_time,
            exit_time=exit_time,
            holding_time=holding_time,
            strategy=strategy,
            signal_confidence=signal_confidence,
        )

        self.trades.append(record)
        self._update_capital(pnl)
        self._update_daily_stats(record)

        return record

    def _update_capital(self, pnl: float) -> None:
        """資本更新"""
        self.current_capital += pnl

        self.equity_curve.append({
            'timestamp': datetime.now(),
            'capital': self.current_capital,
            'pnl': pnl,
        })

    def _update_daily_stats(self, trade: TradeRecord) -> None:
        """日次統計更新"""
        date_key = trade.exit_time.strftime("%Y-%m-%d")

        if date_key not in self.daily_stats:
            self.daily_stats[date_key] = {
                'trades': 0,
                'winning_trades': 0,
                'pnl': 0,
                'volume': 0,
            }

        stats = self.daily_stats[date_key]
        stats['trades'] += 1
        stats['pnl'] += trade.pnl
        stats['volume'] += trade.size * trade.entry_price
        if trade.pnl > 0:
            stats['winning_trades'] += 1

    def calculate_metrics(self) -> Dict:
        """全体メトリクス計算"""
        if not self.trades:
            return {}

        pnls = [t.pnl for t in self.trades]
        returns = [t.pnl_pct / 100 for t in self.trades]

        winning_trades = [t for t in self.trades if t.pnl > 0]
        losing_trades = [t for t in self.trades if t.pnl < 0]

        # 基本統計
        total_pnl = sum(pnls)
        win_rate = len(winning_trades) / len(self.trades)
        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0

        # プロフィットファクター
        gross_profit = sum(t.pnl for t in winning_trades)
        gross_loss = abs(sum(t.pnl for t in losing_trades))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # リターン
        total_return = (self.current_capital - self.initial_capital) / self.initial_capital

        # シャープレシオ
        if len(returns) > 1:
            avg_return = np.mean(returns)
            std_return = np.std(returns)
            sharpe_ratio = (avg_return * np.sqrt(252 * 24)) / std_return if std_return > 0 else 0
        else:
            sharpe_ratio = 0

        # ソルティノレシオ（下方リスクのみ考慮）
        negative_returns = [r for r in returns if r < 0]
        if negative_returns:
            downside_std = np.std(negative_returns)
            sortino_ratio = (np.mean(returns) * np.sqrt(252 * 24)) / downside_std if downside_std > 0 else 0
        else:
            sortino_ratio = float('inf')

        # 最大ドローダウン
        max_drawdown = self._calculate_max_drawdown()

        # カルマーレシオ
        calmar_ratio = total_return / max_drawdown if max_drawdown > 0 else float('inf')

        # 平均保有時間
        avg_holding_time = np.mean([t.holding_time for t in self.trades])

        return {
            'total_trades': len(self.trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'total_return': total_return,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'avg_holding_time': avg_holding_time,
            'current_capital': self.current_capital,
        }

    def _calculate_max_drawdown(self) -> float:
        """最大ドローダウン計算"""
        if not self.equity_curve:
            return 0.0

        capitals = [e['capital'] for e in self.equity_curve]
        peak = capitals[0]
        max_dd = 0.0

        for capital in capitals:
            if capital > peak:
                peak = capital
            dd = (peak - capital) / peak
            if dd > max_dd:
                max_dd = dd

        return max_dd

    def calculate_monthly_returns(self) -> Dict[str, float]:
        """月次リターン計算"""
        monthly_returns = {}

        for trade in self.trades:
            month_key = trade.exit_time.strftime("%Y-%m")
            if month_key not in monthly_returns:
                monthly_returns[month_key] = 0
            monthly_returns[month_key] += trade.pnl

        # パーセンテージに変換
        for month in monthly_returns:
            monthly_returns[month] = monthly_returns[month] / self.initial_capital * 100

        return monthly_returns

    def get_strategy_performance(self) -> Dict[str, Dict]:
        """戦略別パフォーマンス"""
        strategy_stats = {}

        for trade in self.trades:
            strategy = trade.strategy or "Unknown"
            if strategy not in strategy_stats:
                strategy_stats[strategy] = {
                    'trades': 0,
                    'winning_trades': 0,
                    'total_pnl': 0,
                    'pnls': [],
                }

            stats = strategy_stats[strategy]
            stats['trades'] += 1
            stats['total_pnl'] += trade.pnl
            stats['pnls'].append(trade.pnl)
            if trade.pnl > 0:
                stats['winning_trades'] += 1

        # 集計
        result = {}
        for strategy, stats in strategy_stats.items():
            result[strategy] = {
                'trades': stats['trades'],
                'win_rate': stats['winning_trades'] / stats['trades'] if stats['trades'] > 0 else 0,
                'total_pnl': stats['total_pnl'],
                'avg_pnl': np.mean(stats['pnls']) if stats['pnls'] else 0,
                'pnl_std': np.std(stats['pnls']) if len(stats['pnls']) > 1 else 0,
            }

        return result

    def get_daily_returns(self, days: int = 30) -> List[Dict]:
        """日次リターン取得"""
        daily_returns = []
        start_date = datetime.now() - timedelta(days=days)

        for date_key, stats in sorted(self.daily_stats.items()):
            date = datetime.strptime(date_key, "%Y-%m-%d")
            if date >= start_date:
                daily_returns.append({
                    'date': date_key,
                    'trades': stats['trades'],
                    'pnl': stats['pnl'],
                    'win_rate': stats['winning_trades'] / stats['trades'] if stats['trades'] > 0 else 0,
                })

        return daily_returns

    def get_equity_curve(self) -> List[Dict]:
        """資本曲線取得"""
        return self.equity_curve

    def check_monthly_target(self, target_return: float = 0.30) -> Dict:
        """月次目標達成チェック"""
        monthly_returns = self.calculate_monthly_returns()

        if not monthly_returns:
            return {'achieved': False, 'current_return': 0, 'target': target_return}

        current_month = datetime.now().strftime("%Y-%m")
        current_return = monthly_returns.get(current_month, 0) / 100

        return {
            'achieved': current_return >= target_return,
            'current_return': current_return,
            'target': target_return,
            'remaining': target_return - current_return,
            'days_remaining': (datetime.now().replace(day=28) - datetime.now()).days,
        }

    def generate_report(self) -> str:
        """パフォーマンスレポート生成"""
        metrics = self.calculate_metrics()
        monthly = self.calculate_monthly_returns()
        strategy_perf = self.get_strategy_performance()
        target_check = self.check_monthly_target()

        report = f"""
╔══════════════════════════════════════════════════════════════════╗
║            📊 TRADING PERFORMANCE REPORT 📊                       ║
╠══════════════════════════════════════════════════════════════════╣
║ OVERALL METRICS                                                   ║
║ ──────────────────────────────────────────────────────────────────║
║ Total Trades: {metrics.get('total_trades', 0):>15,}                                  ║
║ Win Rate: {metrics.get('win_rate', 0):>19.2%}                                  ║
║ Profit Factor: {metrics.get('profit_factor', 0):>14.2f}                                  ║
║ Total Return: {metrics.get('total_return', 0):>15.2%}                                  ║
║ Sharpe Ratio: {metrics.get('sharpe_ratio', 0):>15.2f}                                  ║
║ Max Drawdown: {metrics.get('max_drawdown', 0):>15.2%}                                  ║
╠══════════════════════════════════════════════════════════════════╣
║ MONTHLY TARGET (30%)                                              ║
║ ──────────────────────────────────────────────────────────────────║
║ Current: {target_check['current_return']:>20.2%}                                  ║
║ Target: {target_check['target']:>21.2%}                                  ║
║ Status: {'✅ ACHIEVED' if target_check['achieved'] else '⏳ In Progress':>20}                                  ║
╠══════════════════════════════════════════════════════════════════╣
║ STRATEGY PERFORMANCE                                              ║
║ ──────────────────────────────────────────────────────────────────║"""

        for strategy, perf in strategy_perf.items():
            report += f"\n║ {strategy[:20]:<20}: WR={perf['win_rate']:.1%} PnL={perf['total_pnl']:>10,.0f}     ║"

        report += """
╠══════════════════════════════════════════════════════════════════╣
║ MONTHLY RETURNS                                                   ║
║ ──────────────────────────────────────────────────────────────────║"""

        for month, ret in list(monthly.items())[-6:]:
            bar = '█' * int(min(ret / 5, 10)) if ret > 0 else '░' * int(min(abs(ret) / 5, 10))
            report += f"\n║ {month}: {ret:>8.2f}% {bar:<10}                              ║"

        report += """
╚══════════════════════════════════════════════════════════════════╝
"""

        return report

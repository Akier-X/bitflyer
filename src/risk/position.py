"""
Position Manager
=================
ポジション管理システム
"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
from loguru import logger


class PositionSide(Enum):
    """ポジション方向"""
    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"


@dataclass
class Position:
    """ポジション情報"""
    product_code: str
    side: PositionSide
    size: float
    entry_price: float
    current_price: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    trailing_stop: float = 0.0
    entry_time: datetime = field(default_factory=datetime.now)
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    max_profit: float = 0.0
    max_drawdown: float = 0.0

    @property
    def is_open(self) -> bool:
        return self.size > 0

    @property
    def holding_time(self) -> float:
        """保有時間（秒）"""
        return (datetime.now() - self.entry_time).total_seconds()

    def update_pnl(self, current_price: float) -> None:
        """P&L更新"""
        self.current_price = current_price

        if self.side == PositionSide.LONG:
            self.unrealized_pnl = (current_price - self.entry_price) * self.size
        elif self.side == PositionSide.SHORT:
            self.unrealized_pnl = (self.entry_price - current_price) * self.size

        # 最大利益・ドローダウン追跡
        if self.unrealized_pnl > self.max_profit:
            self.max_profit = self.unrealized_pnl
        if self.unrealized_pnl < -self.max_drawdown:
            self.max_drawdown = abs(self.unrealized_pnl)


class PositionManager:
    """
    ポジション管理システム

    特徴:
    - マルチポジション管理
    - 自動損切り・利確
    - トレーリングストップ
    - ポジション統計
    """

    def __init__(self):
        """初期化"""
        self.positions: Dict[str, Position] = {}
        self.closed_positions: List[Position] = []
        self.total_realized_pnl: float = 0.0

    def open_position(
        self,
        product_code: str,
        side: str,
        size: float,
        entry_price: float,
        stop_loss: float = None,
        take_profit: float = None,
    ) -> Position:
        """
        ポジションオープン

        Args:
            product_code: 取引ペア
            side: 売買方向 ("BUY" or "SELL")
            size: サイズ
            entry_price: エントリー価格
            stop_loss: 損切り価格
            take_profit: 利確価格

        Returns:
            作成されたポジション
        """
        position_side = PositionSide.LONG if side == "BUY" else PositionSide.SHORT

        # 既存ポジションがあれば追加
        if product_code in self.positions:
            existing = self.positions[product_code]
            if existing.side == position_side:
                # 同方向 → 平均単価で追加
                total_value = existing.entry_price * existing.size + entry_price * size
                total_size = existing.size + size
                existing.entry_price = total_value / total_size
                existing.size = total_size
                logger.info(f"Position added: {product_code} {side} {size} @ {entry_price}")
                return existing
            else:
                # 逆方向 → 一部決済または反転
                if size >= existing.size:
                    # 完全反転
                    self.close_position(product_code, entry_price)
                    remaining_size = size - existing.size
                    if remaining_size > 0:
                        return self.open_position(
                            product_code, side, remaining_size, entry_price,
                            stop_loss, take_profit
                        )
                    return None
                else:
                    # 一部決済
                    existing.size -= size
                    return existing

        # 新規ポジション作成
        position = Position(
            product_code=product_code,
            side=position_side,
            size=size,
            entry_price=entry_price,
            current_price=entry_price,
            stop_loss=stop_loss or 0,
            take_profit=take_profit or 0,
        )

        self.positions[product_code] = position
        logger.info(f"Position opened: {product_code} {side} {size} @ {entry_price}")

        return position

    def close_position(
        self,
        product_code: str,
        exit_price: float,
        size: float = None,
    ) -> Tuple[float, Position]:
        """
        ポジションクローズ

        Args:
            product_code: 取引ペア
            exit_price: 決済価格
            size: 決済サイズ（Noneで全決済）

        Returns:
            (実現損益, クローズされたポジション)
        """
        if product_code not in self.positions:
            return 0.0, None

        position = self.positions[product_code]
        close_size = size if size and size < position.size else position.size

        # P&L計算
        if position.side == PositionSide.LONG:
            pnl = (exit_price - position.entry_price) * close_size
        else:
            pnl = (position.entry_price - exit_price) * close_size

        # ポジション更新
        if close_size >= position.size:
            # 全決済
            position.realized_pnl = pnl
            self.closed_positions.append(position)
            del self.positions[product_code]
        else:
            # 一部決済
            position.size -= close_size
            position.realized_pnl += pnl

        self.total_realized_pnl += pnl
        logger.info(f"Position closed: {product_code} {close_size} @ {exit_price}, PnL: {pnl:.0f}")

        return pnl, position

    def update_all_positions(self, prices: Dict[str, float]) -> None:
        """
        全ポジションの価格更新

        Args:
            prices: 取引ペア → 現在価格のマップ
        """
        for product_code, position in self.positions.items():
            if product_code in prices:
                position.update_pnl(prices[product_code])

    def check_stop_loss(self, product_code: str, current_price: float) -> bool:
        """
        損切りチェック

        Args:
            product_code: 取引ペア
            current_price: 現在価格

        Returns:
            損切りすべきか
        """
        if product_code not in self.positions:
            return False

        position = self.positions[product_code]
        if position.stop_loss <= 0:
            return False

        if position.side == PositionSide.LONG:
            return current_price <= position.stop_loss
        else:
            return current_price >= position.stop_loss

    def check_take_profit(self, product_code: str, current_price: float) -> bool:
        """
        利確チェック

        Args:
            product_code: 取引ペア
            current_price: 現在価格

        Returns:
            利確すべきか
        """
        if product_code not in self.positions:
            return False

        position = self.positions[product_code]
        if position.take_profit <= 0:
            return False

        if position.side == PositionSide.LONG:
            return current_price >= position.take_profit
        else:
            return current_price <= position.take_profit

    def update_trailing_stop(
        self,
        product_code: str,
        current_price: float,
        trailing_pct: float = 0.01,
    ) -> None:
        """
        トレーリングストップ更新

        Args:
            product_code: 取引ペア
            current_price: 現在価格
            trailing_pct: トレーリング幅（%）
        """
        if product_code not in self.positions:
            return

        position = self.positions[product_code]
        trailing_distance = current_price * trailing_pct

        if position.side == PositionSide.LONG:
            new_stop = current_price - trailing_distance
            if position.trailing_stop == 0 or new_stop > position.trailing_stop:
                position.trailing_stop = new_stop
                if new_stop > position.stop_loss:
                    position.stop_loss = new_stop
        else:
            new_stop = current_price + trailing_distance
            if position.trailing_stop == 0 or new_stop < position.trailing_stop:
                position.trailing_stop = new_stop
                if position.stop_loss == 0 or new_stop < position.stop_loss:
                    position.stop_loss = new_stop

    def get_position(self, product_code: str) -> Optional[Position]:
        """ポジション取得"""
        return self.positions.get(product_code)

    def get_all_positions(self) -> List[Position]:
        """全ポジション取得"""
        return list(self.positions.values())

    def get_total_exposure(self) -> float:
        """総エクスポージャー取得"""
        return sum(
            p.size * p.current_price
            for p in self.positions.values()
        )

    def get_total_unrealized_pnl(self) -> float:
        """総未実現損益取得"""
        return sum(p.unrealized_pnl for p in self.positions.values())

    def get_position_summary(self) -> Dict:
        """ポジションサマリー取得"""
        positions = list(self.positions.values())

        long_positions = [p for p in positions if p.side == PositionSide.LONG]
        short_positions = [p for p in positions if p.side == PositionSide.SHORT]

        return {
            'total_positions': len(positions),
            'long_positions': len(long_positions),
            'short_positions': len(short_positions),
            'total_exposure': self.get_total_exposure(),
            'total_unrealized_pnl': self.get_total_unrealized_pnl(),
            'total_realized_pnl': self.total_realized_pnl,
            'positions': [
                {
                    'product': p.product_code,
                    'side': p.side.value,
                    'size': p.size,
                    'entry_price': p.entry_price,
                    'current_price': p.current_price,
                    'unrealized_pnl': p.unrealized_pnl,
                    'holding_time': p.holding_time,
                }
                for p in positions
            ],
        }

    def get_performance_stats(self) -> Dict:
        """パフォーマンス統計取得"""
        if not self.closed_positions:
            return {}

        pnls = [p.realized_pnl for p in self.closed_positions]
        winning_trades = [pnl for pnl in pnls if pnl > 0]
        losing_trades = [pnl for pnl in pnls if pnl < 0]

        win_rate = len(winning_trades) / len(pnls) if pnls else 0
        avg_win = sum(winning_trades) / len(winning_trades) if winning_trades else 0
        avg_loss = sum(losing_trades) / len(losing_trades) if losing_trades else 0
        profit_factor = abs(sum(winning_trades) / sum(losing_trades)) if losing_trades else float('inf')

        return {
            'total_trades': len(pnls),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'total_pnl': sum(pnls),
            'max_win': max(pnls) if pnls else 0,
            'max_loss': min(pnls) if pnls else 0,
        }

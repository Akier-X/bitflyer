"""
Order Manager
==============
注文管理システム
"""

from typing import Dict, List, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import uuid
from loguru import logger


class OrderStatus(Enum):
    """注文ステータス"""
    PENDING = "PENDING"           # 送信前
    SUBMITTED = "SUBMITTED"       # 送信済み
    ACTIVE = "ACTIVE"            # アクティブ
    PARTIALLY_FILLED = "PARTIALLY_FILLED"  # 一部約定
    FILLED = "FILLED"            # 約定済み
    CANCELLED = "CANCELLED"      # キャンセル済み
    REJECTED = "REJECTED"        # 拒否
    EXPIRED = "EXPIRED"          # 期限切れ


class OrderType(Enum):
    """注文タイプ"""
    MARKET = "MARKET"
    LIMIT = "LIMIT"


class TimeInForce(Enum):
    """執行条件"""
    GTC = "GTC"  # Good Till Cancelled
    IOC = "IOC"  # Immediate or Cancel
    FOK = "FOK"  # Fill or Kill


@dataclass
class Order:
    """注文情報"""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    product_code: str = ""
    order_type: OrderType = OrderType.LIMIT
    side: str = ""  # BUY or SELL
    size: float = 0.0
    price: float = 0.0
    time_in_force: TimeInForce = TimeInForce.GTC

    # ステータス
    status: OrderStatus = OrderStatus.PENDING
    filled_size: float = 0.0
    average_price: float = 0.0

    # API情報
    child_order_acceptance_id: str = ""
    child_order_id: str = ""

    # タイミング
    created_at: datetime = field(default_factory=datetime.now)
    submitted_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None

    # メタデータ
    strategy_name: str = ""
    signal_confidence: float = 0.0

    @property
    def is_active(self) -> bool:
        return self.status in [OrderStatus.SUBMITTED, OrderStatus.ACTIVE, OrderStatus.PARTIALLY_FILLED]

    @property
    def is_completed(self) -> bool:
        return self.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED, OrderStatus.EXPIRED]

    @property
    def remaining_size(self) -> float:
        return self.size - self.filled_size

    @property
    def fill_percentage(self) -> float:
        return self.filled_size / self.size * 100 if self.size > 0 else 0


class OrderManager:
    """
    注文管理システム

    特徴:
    - 注文ライフサイクル管理
    - 注文追跡・監視
    - 自動キャンセル
    - 注文統計
    """

    def __init__(self, api_client=None):
        """
        Args:
            api_client: bitFlyer APIクライアント
        """
        self.api_client = api_client

        # 注文管理
        self.orders: Dict[str, Order] = {}
        self.active_orders: Dict[str, Order] = {}
        self.completed_orders: List[Order] = []

        # コールバック
        self.on_order_filled: Optional[Callable] = None
        self.on_order_cancelled: Optional[Callable] = None

        # 統計
        self.stats = {
            'total_orders': 0,
            'filled_orders': 0,
            'cancelled_orders': 0,
            'rejected_orders': 0,
            'total_volume': 0.0,
        }

    def create_order(
        self,
        product_code: str,
        side: str,
        size: float,
        price: float = None,
        order_type: OrderType = OrderType.LIMIT,
        time_in_force: TimeInForce = TimeInForce.GTC,
        strategy_name: str = "",
        signal_confidence: float = 0.0,
    ) -> Order:
        """
        注文作成

        Args:
            product_code: 取引ペア
            side: 売買方向
            size: サイズ
            price: 価格（成行の場合はNone）
            order_type: 注文タイプ
            time_in_force: 執行条件
            strategy_name: 戦略名
            signal_confidence: シグナル信頼度

        Returns:
            作成された注文
        """
        order = Order(
            product_code=product_code,
            order_type=order_type,
            side=side,
            size=size,
            price=price or 0,
            time_in_force=time_in_force,
            strategy_name=strategy_name,
            signal_confidence=signal_confidence,
        )

        self.orders[order.id] = order
        self.stats['total_orders'] += 1

        logger.info(f"Order created: {order.id} {side} {size} {product_code} @ {price}")

        return order

    def submit_order(self, order: Order) -> bool:
        """
        注文送信

        Args:
            order: 送信する注文

        Returns:
            送信成功したか
        """
        if not self.api_client:
            logger.warning("No API client configured")
            return False

        try:
            # API経由で注文送信
            result = self.api_client.send_child_order(
                product_code=order.product_code,
                child_order_type=order.order_type.value,
                side=order.side,
                size=order.size,
                price=order.price if order.order_type == OrderType.LIMIT else None,
                time_in_force=order.time_in_force.value,
            )

            # 注文ID更新
            order.child_order_acceptance_id = result.get('child_order_acceptance_id', '')
            order.status = OrderStatus.SUBMITTED
            order.submitted_at = datetime.now()

            # アクティブ注文に追加
            self.active_orders[order.id] = order

            logger.info(f"Order submitted: {order.id} -> {order.child_order_acceptance_id}")

            return True

        except Exception as e:
            logger.error(f"Order submission failed: {e}")
            order.status = OrderStatus.REJECTED
            self.stats['rejected_orders'] += 1
            return False

    def cancel_order(self, order_id: str) -> bool:
        """
        注文キャンセル

        Args:
            order_id: 注文ID

        Returns:
            キャンセル成功したか
        """
        if order_id not in self.orders:
            return False

        order = self.orders[order_id]

        if not order.is_active:
            return False

        try:
            if self.api_client and order.child_order_acceptance_id:
                self.api_client.cancel_child_order(
                    product_code=order.product_code,
                    child_order_acceptance_id=order.child_order_acceptance_id,
                )

            order.status = OrderStatus.CANCELLED
            self._move_to_completed(order)
            self.stats['cancelled_orders'] += 1

            if self.on_order_cancelled:
                self.on_order_cancelled(order)

            logger.info(f"Order cancelled: {order_id}")

            return True

        except Exception as e:
            logger.error(f"Order cancellation failed: {e}")
            return False

    def cancel_all_orders(self, product_code: str = None) -> int:
        """
        全注文キャンセル

        Args:
            product_code: 取引ペア（Noneで全ペア）

        Returns:
            キャンセルした注文数
        """
        cancelled_count = 0

        for order_id, order in list(self.active_orders.items()):
            if product_code is None or order.product_code == product_code:
                if self.cancel_order(order_id):
                    cancelled_count += 1

        return cancelled_count

    def update_order_status(self, order_id: str, new_status: OrderStatus) -> None:
        """
        注文ステータス更新

        Args:
            order_id: 注文ID
            new_status: 新しいステータス
        """
        if order_id not in self.orders:
            return

        order = self.orders[order_id]
        old_status = order.status
        order.status = new_status

        if new_status == OrderStatus.FILLED:
            order.filled_at = datetime.now()
            self._move_to_completed(order)
            self.stats['filled_orders'] += 1
            self.stats['total_volume'] += order.size

            if self.on_order_filled:
                self.on_order_filled(order)

        elif new_status in [OrderStatus.CANCELLED, OrderStatus.REJECTED, OrderStatus.EXPIRED]:
            self._move_to_completed(order)

        logger.debug(f"Order status updated: {order_id} {old_status.value} -> {new_status.value}")

    def update_order_fill(
        self,
        order_id: str,
        filled_size: float,
        average_price: float,
    ) -> None:
        """
        注文約定更新

        Args:
            order_id: 注文ID
            filled_size: 約定サイズ
            average_price: 平均約定価格
        """
        if order_id not in self.orders:
            return

        order = self.orders[order_id]
        order.filled_size = filled_size
        order.average_price = average_price

        if filled_size >= order.size:
            self.update_order_status(order_id, OrderStatus.FILLED)
        elif filled_size > 0:
            self.update_order_status(order_id, OrderStatus.PARTIALLY_FILLED)

    def _move_to_completed(self, order: Order) -> None:
        """注文を完了リストに移動"""
        if order.id in self.active_orders:
            del self.active_orders[order.id]
        self.completed_orders.append(order)

        # 履歴制限
        if len(self.completed_orders) > 1000:
            self.completed_orders = self.completed_orders[-1000:]

    def sync_with_exchange(self) -> None:
        """取引所と同期"""
        if not self.api_client:
            return

        for order_id, order in list(self.active_orders.items()):
            try:
                exchange_orders = self.api_client.get_child_orders(
                    product_code=order.product_code,
                    child_order_state="ACTIVE",
                )

                # 注文が取引所にない場合は約定済みの可能性
                found = False
                for ex_order in exchange_orders:
                    if ex_order.get('child_order_acceptance_id') == order.child_order_acceptance_id:
                        found = True
                        # ステータス更新
                        size = ex_order.get('executed_size', 0)
                        if size > 0:
                            self.update_order_fill(
                                order_id,
                                size,
                                ex_order.get('average_price', order.price)
                            )
                        break

                if not found:
                    # 約定履歴を確認
                    executions = self.api_client.get_my_executions(
                        product_code=order.product_code,
                        count=50
                    )
                    for execution in executions:
                        if execution.get('child_order_acceptance_id') == order.child_order_acceptance_id:
                            self.update_order_fill(
                                order_id,
                                execution.get('size', 0),
                                execution.get('price', order.price)
                            )
                            break

            except Exception as e:
                logger.error(f"Sync failed for order {order_id}: {e}")

    def check_order_expiry(self, max_age_seconds: int = 300) -> List[str]:
        """
        期限切れ注文チェック

        Args:
            max_age_seconds: 最大注文年齢（秒）

        Returns:
            期限切れ注文IDリスト
        """
        expired = []
        now = datetime.now()

        for order_id, order in list(self.active_orders.items()):
            if order.submitted_at:
                age = (now - order.submitted_at).total_seconds()
                if age > max_age_seconds:
                    expired.append(order_id)

        return expired

    def get_order(self, order_id: str) -> Optional[Order]:
        """注文取得"""
        return self.orders.get(order_id)

    def get_active_orders(self, product_code: str = None) -> List[Order]:
        """アクティブ注文取得"""
        orders = list(self.active_orders.values())
        if product_code:
            orders = [o for o in orders if o.product_code == product_code]
        return orders

    def get_order_stats(self) -> Dict:
        """注文統計取得"""
        return {
            **self.stats,
            'active_orders': len(self.active_orders),
            'fill_rate': self.stats['filled_orders'] / self.stats['total_orders']
                        if self.stats['total_orders'] > 0 else 0,
        }

    def get_recent_fills(self, limit: int = 10) -> List[Order]:
        """最近の約定取得"""
        filled = [o for o in self.completed_orders if o.status == OrderStatus.FILLED]
        return sorted(filled, key=lambda x: x.filled_at or datetime.min, reverse=True)[:limit]

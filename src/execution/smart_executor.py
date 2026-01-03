"""
Smart Executor
===============
高度な注文執行システム
"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import asyncio
import time
from loguru import logger

from .order_manager import OrderManager, Order, OrderType, OrderStatus, TimeInForce


@dataclass
class ExecutionResult:
    """執行結果"""
    success: bool
    order_id: str
    filled_size: float
    average_price: float
    total_cost: float
    slippage: float
    execution_time: float  # 秒
    num_fills: int
    message: str = ""


class SmartExecutor:
    """
    スマート執行システム

    特徴:
    - TWAP (Time Weighted Average Price)
    - VWAP (Volume Weighted Average Price)
    - Iceberg Orders（氷山注文）
    - スマートオーダールーティング
    - スリッページ最小化
    """

    def __init__(
        self,
        order_manager: OrderManager,
        api_client=None,
    ):
        """
        Args:
            order_manager: 注文マネージャー
            api_client: APIクライアント
        """
        self.order_manager = order_manager
        self.api_client = api_client

        # 執行設定
        self.default_slippage_tolerance = 0.001  # 0.1%
        self.min_order_size = 0.001  # 最小注文サイズ

    def execute_market(
        self,
        product_code: str,
        side: str,
        size: float,
        slippage_tolerance: float = None,
    ) -> ExecutionResult:
        """
        成行執行

        Args:
            product_code: 取引ペア
            side: 売買方向
            size: サイズ
            slippage_tolerance: スリッページ許容度

        Returns:
            執行結果
        """
        start_time = time.time()
        slippage_tolerance = slippage_tolerance or self.default_slippage_tolerance

        # 現在価格取得
        if self.api_client:
            ticker = self.api_client.get_ticker(product_code)
            reference_price = ticker.get('ltp', 0)
        else:
            reference_price = 0

        # 注文作成・送信
        order = self.order_manager.create_order(
            product_code=product_code,
            side=side,
            size=size,
            order_type=OrderType.MARKET,
            time_in_force=TimeInForce.IOC,
        )

        success = self.order_manager.submit_order(order)

        if not success:
            return ExecutionResult(
                success=False,
                order_id=order.id,
                filled_size=0,
                average_price=0,
                total_cost=0,
                slippage=0,
                execution_time=time.time() - start_time,
                num_fills=0,
                message="Order submission failed",
            )

        # 約定確認（簡易版）
        time.sleep(0.5)
        self.order_manager.sync_with_exchange()

        # 結果取得
        updated_order = self.order_manager.get_order(order.id)
        filled_size = updated_order.filled_size if updated_order else 0
        average_price = updated_order.average_price if updated_order else 0

        # スリッページ計算
        slippage = 0
        if reference_price > 0 and average_price > 0:
            if side == "BUY":
                slippage = (average_price - reference_price) / reference_price
            else:
                slippage = (reference_price - average_price) / reference_price

        return ExecutionResult(
            success=filled_size > 0,
            order_id=order.id,
            filled_size=filled_size,
            average_price=average_price,
            total_cost=filled_size * average_price,
            slippage=slippage,
            execution_time=time.time() - start_time,
            num_fills=1 if filled_size > 0 else 0,
        )

    def execute_limit(
        self,
        product_code: str,
        side: str,
        size: float,
        price: float,
        timeout_seconds: float = 60,
    ) -> ExecutionResult:
        """
        指値執行

        Args:
            product_code: 取引ペア
            side: 売買方向
            size: サイズ
            price: 価格
            timeout_seconds: タイムアウト

        Returns:
            執行結果
        """
        start_time = time.time()

        # 注文作成・送信
        order = self.order_manager.create_order(
            product_code=product_code,
            side=side,
            size=size,
            price=price,
            order_type=OrderType.LIMIT,
            time_in_force=TimeInForce.GTC,
        )

        success = self.order_manager.submit_order(order)

        if not success:
            return ExecutionResult(
                success=False,
                order_id=order.id,
                filled_size=0,
                average_price=0,
                total_cost=0,
                slippage=0,
                execution_time=time.time() - start_time,
                num_fills=0,
                message="Order submission failed",
            )

        # 約定待機
        while time.time() - start_time < timeout_seconds:
            self.order_manager.sync_with_exchange()
            updated_order = self.order_manager.get_order(order.id)

            if updated_order and updated_order.status == OrderStatus.FILLED:
                break

            time.sleep(0.5)

        # タイムアウト時はキャンセル
        updated_order = self.order_manager.get_order(order.id)
        if updated_order and updated_order.is_active:
            self.order_manager.cancel_order(order.id)

        filled_size = updated_order.filled_size if updated_order else 0
        average_price = updated_order.average_price if updated_order else price

        return ExecutionResult(
            success=filled_size > 0,
            order_id=order.id,
            filled_size=filled_size,
            average_price=average_price,
            total_cost=filled_size * average_price,
            slippage=0,  # 指値なのでスリッページなし
            execution_time=time.time() - start_time,
            num_fills=1 if filled_size > 0 else 0,
        )

    def execute_twap(
        self,
        product_code: str,
        side: str,
        total_size: float,
        duration_seconds: int,
        num_slices: int = 10,
    ) -> ExecutionResult:
        """
        TWAP (Time Weighted Average Price) 執行

        時間を均等に分割して注文を執行

        Args:
            product_code: 取引ペア
            side: 売買方向
            total_size: 総サイズ
            duration_seconds: 執行期間（秒）
            num_slices: 分割数

        Returns:
            執行結果
        """
        start_time = time.time()

        slice_size = total_size / num_slices
        interval = duration_seconds / num_slices

        total_filled = 0.0
        total_cost = 0.0
        num_fills = 0

        for i in range(num_slices):
            if slice_size < self.min_order_size:
                continue

            # 現在の最良価格取得
            if self.api_client:
                board = self.api_client.get_board(product_code)
                if side == "BUY":
                    price = board['asks'][0]['price'] if board.get('asks') else 0
                else:
                    price = board['bids'][0]['price'] if board.get('bids') else 0
            else:
                price = 0

            if price <= 0:
                continue

            # 注文執行
            result = self.execute_limit(
                product_code=product_code,
                side=side,
                size=slice_size,
                price=price,
                timeout_seconds=interval * 0.8,
            )

            if result.success:
                total_filled += result.filled_size
                total_cost += result.total_cost
                num_fills += 1

            # 残りサイズ調整
            remaining = total_size - total_filled
            remaining_slices = num_slices - i - 1
            if remaining_slices > 0:
                slice_size = remaining / remaining_slices

            # 次のスライスまで待機
            if i < num_slices - 1:
                elapsed = time.time() - start_time
                wait_until = (i + 1) * interval
                if wait_until > elapsed:
                    time.sleep(wait_until - elapsed)

        average_price = total_cost / total_filled if total_filled > 0 else 0

        return ExecutionResult(
            success=total_filled > 0,
            order_id="TWAP",
            filled_size=total_filled,
            average_price=average_price,
            total_cost=total_cost,
            slippage=0,
            execution_time=time.time() - start_time,
            num_fills=num_fills,
        )

    def execute_iceberg(
        self,
        product_code: str,
        side: str,
        total_size: float,
        visible_size: float,
        price: float,
        timeout_seconds: float = 300,
    ) -> ExecutionResult:
        """
        氷山注文執行

        大量注文を小さなチャンクに分割して執行

        Args:
            product_code: 取引ペア
            side: 売買方向
            total_size: 総サイズ
            visible_size: 表示サイズ（チャンクサイズ）
            price: 価格
            timeout_seconds: タイムアウト

        Returns:
            執行結果
        """
        start_time = time.time()

        remaining_size = total_size
        total_filled = 0.0
        total_cost = 0.0
        num_fills = 0

        while remaining_size > self.min_order_size:
            if time.time() - start_time > timeout_seconds:
                break

            chunk_size = min(visible_size, remaining_size)

            result = self.execute_limit(
                product_code=product_code,
                side=side,
                size=chunk_size,
                price=price,
                timeout_seconds=30,
            )

            if result.success:
                total_filled += result.filled_size
                total_cost += result.total_cost
                num_fills += 1
                remaining_size -= result.filled_size
            else:
                # 価格更新
                if self.api_client:
                    board = self.api_client.get_board(product_code)
                    if side == "BUY":
                        new_price = board['asks'][0]['price'] if board.get('asks') else price
                    else:
                        new_price = board['bids'][0]['price'] if board.get('bids') else price

                    # 価格が大きく変動した場合は中止
                    if abs(new_price - price) / price > 0.01:
                        break
                    price = new_price

            time.sleep(0.1)

        average_price = total_cost / total_filled if total_filled > 0 else 0

        return ExecutionResult(
            success=total_filled > 0,
            order_id="ICEBERG",
            filled_size=total_filled,
            average_price=average_price,
            total_cost=total_cost,
            slippage=0,
            execution_time=time.time() - start_time,
            num_fills=num_fills,
        )

    def execute_smart(
        self,
        product_code: str,
        side: str,
        size: float,
        urgency: str = "normal",  # low, normal, high
    ) -> ExecutionResult:
        """
        スマート執行

        状況に応じて最適な執行方法を自動選択

        Args:
            product_code: 取引ペア
            side: 売買方向
            size: サイズ
            urgency: 緊急度

        Returns:
            執行結果
        """
        # 市場状況分析
        if self.api_client:
            board = self.api_client.get_board(product_code)
            spread = self._calculate_spread(board)
            depth = self._calculate_depth(board, size)
        else:
            spread = 0.001
            depth = 1.0

        # 執行方法選択
        if urgency == "high":
            # 成行で即時執行
            return self.execute_market(product_code, side, size)

        elif size > depth * 0.1:
            # 大量注文 → TWAP
            return self.execute_twap(
                product_code, side, size,
                duration_seconds=60,
                num_slices=5,
            )

        elif spread > 0.002:
            # スプレッド広い → 指値で待機
            if self.api_client:
                board = self.api_client.get_board(product_code)
                if side == "BUY":
                    price = board['bids'][0]['price'] + 1 if board.get('bids') else 0
                else:
                    price = board['asks'][0]['price'] - 1 if board.get('asks') else 0
            else:
                price = 0

            if price > 0:
                return self.execute_limit(
                    product_code, side, size, price,
                    timeout_seconds=30,
                )

        # デフォルト: 成行
        return self.execute_market(product_code, side, size)

    def _calculate_spread(self, board: Dict) -> float:
        """スプレッド計算"""
        if not board.get('bids') or not board.get('asks'):
            return 0.01

        best_bid = board['bids'][0]['price']
        best_ask = board['asks'][0]['price']
        mid = (best_bid + best_ask) / 2

        return (best_ask - best_bid) / mid

    def _calculate_depth(self, board: Dict, side_size: float) -> float:
        """板の深さ計算"""
        if not board.get('bids') or not board.get('asks'):
            return 0

        bid_depth = sum(b['size'] for b in board['bids'][:10])
        ask_depth = sum(a['size'] for a in board['asks'][:10])

        return min(bid_depth, ask_depth)

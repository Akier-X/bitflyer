"""
bitFlyer Lightning API Client
==============================
bitFlyer取引所との通信クライアント
"""

import asyncio
import aiohttp
import hashlib
import hmac
import json
import time
import websockets
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from collections import deque
from loguru import logger


class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"


class TimeInForce(Enum):
    GTC = "GTC"  # Good Till Cancelled
    IOC = "IOC"  # Immediate Or Cancel
    FOK = "FOK"  # Fill Or Kill


@dataclass
class Ticker:
    """ティッカー情報"""
    product_code: str
    timestamp: datetime
    best_bid: float
    best_ask: float
    best_bid_size: float
    best_ask_size: float
    ltp: float  # Last Traded Price
    volume: float
    volume_by_product: float

    @property
    def mid_price(self) -> float:
        return (self.best_bid + self.best_ask) / 2

    @property
    def spread(self) -> float:
        return self.best_ask - self.best_bid

    @property
    def spread_pct(self) -> float:
        return self.spread / self.mid_price


@dataclass
class OrderBook:
    """板情報"""
    product_code: str
    timestamp: datetime
    bids: List[tuple]  # [(price, size), ...]
    asks: List[tuple]  # [(price, size), ...]

    @property
    def best_bid(self) -> float:
        return self.bids[0][0] if self.bids else 0

    @property
    def best_ask(self) -> float:
        return self.asks[0][0] if self.asks else 0

    @property
    def mid_price(self) -> float:
        return (self.best_bid + self.best_ask) / 2

    def get_imbalance(self, levels: int = 5) -> float:
        """板の不均衡を計算"""
        bid_volume = sum(b[1] for b in self.bids[:levels])
        ask_volume = sum(a[1] for a in self.asks[:levels])
        total = bid_volume + ask_volume
        if total == 0:
            return 0
        return (bid_volume - ask_volume) / total


@dataclass
class Execution:
    """約定情報"""
    id: int
    side: str
    price: float
    size: float
    timestamp: datetime
    buy_child_order_acceptance_id: str = ""
    sell_child_order_acceptance_id: str = ""


@dataclass
class Order:
    """注文情報"""
    id: str
    product_code: str
    side: str
    order_type: str
    price: float
    size: float
    status: str
    timestamp: datetime
    executed_size: float = 0
    average_price: float = 0


@dataclass
class Position:
    """ポジション情報"""
    product_code: str
    side: str
    size: float
    price: float
    commission: float
    pnl: float
    timestamp: datetime


class BitFlyerClient:
    """
    bitFlyer Lightning API クライアント

    REST API と WebSocket の両方をサポート
    """

    BASE_URL = "https://api.bitflyer.com"
    WS_URL = "wss://ws.lightstream.bitflyer.com/json-rpc"

    def __init__(
        self,
        api_key: str = "",
        api_secret: str = "",
        product_code: str = "BTC_JPY",
        testnet: bool = False,
    ):
        self.api_key = api_key
        self.api_secret = api_secret
        self.product_code = product_code
        self.testnet = testnet

        # WebSocket
        self.ws = None
        self.ws_connected = False
        self.ws_callbacks: Dict[str, List[Callable]] = {}

        # Data cache
        self.ticker: Optional[Ticker] = None
        self.order_book: Optional[OrderBook] = None
        self.executions: deque = deque(maxlen=1000)
        self.positions: List[Position] = []
        self.orders: Dict[str, Order] = {}

        # Rate limiting
        self.request_count = 0
        self.last_request_time = time.time()
        self.max_requests_per_minute = 500

        # Statistics
        self.total_requests = 0
        self.failed_requests = 0

        logger.info(f"BitFlyer client initialized for {product_code}")

    def _sign(self, method: str, path: str, body: str = "") -> Dict[str, str]:
        """API認証用署名を生成"""
        timestamp = str(int(time.time() * 1000))
        message = timestamp + method + path + body

        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            message.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()

        return {
            "ACCESS-KEY": self.api_key,
            "ACCESS-TIMESTAMP": timestamp,
            "ACCESS-SIGN": signature,
            "Content-Type": "application/json",
        }

    async def _request(
        self,
        method: str,
        path: str,
        params: Dict = None,
        data: Dict = None,
        auth: bool = False,
    ) -> Dict:
        """HTTPリクエストを送信"""
        # Rate limiting
        now = time.time()
        if now - self.last_request_time < 60:
            if self.request_count >= self.max_requests_per_minute:
                wait_time = 60 - (now - self.last_request_time)
                logger.warning(f"Rate limit reached, waiting {wait_time:.1f}s")
                await asyncio.sleep(wait_time)
                self.request_count = 0
                self.last_request_time = time.time()
        else:
            self.request_count = 0
            self.last_request_time = now

        url = self.BASE_URL + path

        headers = {}
        body = ""

        if data:
            body = json.dumps(data)

        if auth:
            headers = self._sign(method, path, body)

        self.request_count += 1
        self.total_requests += 1

        try:
            async with aiohttp.ClientSession() as session:
                async with session.request(
                    method,
                    url,
                    params=params,
                    data=body if data else None,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        self.failed_requests += 1
                        error_text = await response.text()
                        logger.error(f"API error {response.status}: {error_text}")
                        return {"error": error_text, "status": response.status}

        except Exception as e:
            self.failed_requests += 1
            logger.error(f"Request error: {e}")
            return {"error": str(e)}

    # ========== Public API ==========

    async def get_ticker(self, product_code: str = None) -> Optional[Ticker]:
        """ティッカー情報を取得"""
        product_code = product_code or self.product_code
        result = await self._request(
            "GET",
            "/v1/ticker",
            params={"product_code": product_code},
        )

        if "error" in result:
            return None

        ticker = Ticker(
            product_code=result.get("product_code", product_code),
            timestamp=datetime.fromisoformat(result.get("timestamp", "").replace("Z", "+00:00")),
            best_bid=result.get("best_bid", 0),
            best_ask=result.get("best_ask", 0),
            best_bid_size=result.get("best_bid_size", 0),
            best_ask_size=result.get("best_ask_size", 0),
            ltp=result.get("ltp", 0),
            volume=result.get("volume", 0),
            volume_by_product=result.get("volume_by_product", 0),
        )

        self.ticker = ticker
        return ticker

    async def get_order_book(self, product_code: str = None) -> Optional[OrderBook]:
        """板情報を取得"""
        product_code = product_code or self.product_code
        result = await self._request(
            "GET",
            "/v1/board",
            params={"product_code": product_code},
        )

        if "error" in result:
            return None

        order_book = OrderBook(
            product_code=product_code,
            timestamp=datetime.now(),
            bids=[(b["price"], b["size"]) for b in result.get("bids", [])],
            asks=[(a["price"], a["size"]) for a in result.get("asks", [])],
        )

        self.order_book = order_book
        return order_book

    async def get_executions(
        self,
        product_code: str = None,
        count: int = 100,
    ) -> List[Execution]:
        """約定履歴を取得"""
        product_code = product_code or self.product_code
        result = await self._request(
            "GET",
            "/v1/executions",
            params={"product_code": product_code, "count": count},
        )

        if "error" in result or not isinstance(result, list):
            return []

        executions = []
        for e in result:
            execution = Execution(
                id=e.get("id", 0),
                side=e.get("side", ""),
                price=e.get("price", 0),
                size=e.get("size", 0),
                timestamp=datetime.fromisoformat(e.get("exec_date", "").replace("Z", "+00:00")),
            )
            executions.append(execution)
            self.executions.append(execution)

        return executions

    # ========== Private API ==========

    async def get_balance(self) -> List[Dict]:
        """残高を取得"""
        result = await self._request(
            "GET",
            "/v1/me/getbalance",
            auth=True,
        )

        if "error" in result:
            return []

        return result

    async def get_positions(self, product_code: str = None) -> List[Position]:
        """ポジションを取得"""
        product_code = product_code or self.product_code
        result = await self._request(
            "GET",
            "/v1/me/getpositions",
            params={"product_code": product_code},
            auth=True,
        )

        if "error" in result or not isinstance(result, list):
            return []

        positions = []
        for p in result:
            position = Position(
                product_code=p.get("product_code", product_code),
                side=p.get("side", ""),
                size=p.get("size", 0),
                price=p.get("price", 0),
                commission=p.get("commission", 0),
                pnl=p.get("pnl", 0),
                timestamp=datetime.now(),
            )
            positions.append(position)

        self.positions = positions
        return positions

    async def send_order(
        self,
        side: OrderSide,
        size: float,
        order_type: OrderType = OrderType.MARKET,
        price: float = 0,
        time_in_force: TimeInForce = TimeInForce.GTC,
        product_code: str = None,
    ) -> Optional[str]:
        """注文を送信"""
        product_code = product_code or self.product_code

        data = {
            "product_code": product_code,
            "child_order_type": order_type.value,
            "side": side.value,
            "size": size,
            "time_in_force": time_in_force.value,
        }

        if order_type == OrderType.LIMIT:
            data["price"] = price

        result = await self._request(
            "POST",
            "/v1/me/sendchildorder",
            data=data,
            auth=True,
        )

        if "error" in result:
            logger.error(f"Order failed: {result}")
            return None

        order_id = result.get("child_order_acceptance_id")
        logger.info(f"Order sent: {side.value} {size} @ {price if price else 'MARKET'}, ID: {order_id}")

        return order_id

    async def cancel_order(
        self,
        order_id: str,
        product_code: str = None,
    ) -> bool:
        """注文をキャンセル"""
        product_code = product_code or self.product_code

        result = await self._request(
            "POST",
            "/v1/me/cancelchildorder",
            data={
                "product_code": product_code,
                "child_order_acceptance_id": order_id,
            },
            auth=True,
        )

        if "error" in result:
            logger.error(f"Cancel failed: {result}")
            return False

        logger.info(f"Order cancelled: {order_id}")
        return True

    async def cancel_all_orders(self, product_code: str = None) -> bool:
        """全注文をキャンセル"""
        product_code = product_code or self.product_code

        result = await self._request(
            "POST",
            "/v1/me/cancelallchildorders",
            data={"product_code": product_code},
            auth=True,
        )

        if "error" in result:
            logger.error(f"Cancel all failed: {result}")
            return False

        logger.info(f"All orders cancelled for {product_code}")
        return True

    async def get_orders(self, product_code: str = None) -> List[Order]:
        """注文一覧を取得"""
        product_code = product_code or self.product_code

        result = await self._request(
            "GET",
            "/v1/me/getchildorders",
            params={
                "product_code": product_code,
                "child_order_state": "ACTIVE",
            },
            auth=True,
        )

        if "error" in result or not isinstance(result, list):
            return []

        orders = []
        for o in result:
            order = Order(
                id=o.get("child_order_acceptance_id", ""),
                product_code=o.get("product_code", product_code),
                side=o.get("side", ""),
                order_type=o.get("child_order_type", ""),
                price=o.get("price", 0),
                size=o.get("size", 0),
                status=o.get("child_order_state", ""),
                timestamp=datetime.fromisoformat(o.get("child_order_date", "").replace("Z", "+00:00")),
                executed_size=o.get("executed_size", 0),
                average_price=o.get("average_price", 0),
            )
            orders.append(order)
            self.orders[order.id] = order

        return orders

    # ========== WebSocket ==========

    async def connect_websocket(self) -> None:
        """WebSocketに接続"""
        try:
            self.ws = await websockets.connect(self.WS_URL)
            self.ws_connected = True
            logger.info("WebSocket connected")

            # Subscribe to channels
            await self._subscribe_channels()

            # Start message handler
            asyncio.create_task(self._ws_message_handler())

        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")
            self.ws_connected = False

    async def _subscribe_channels(self) -> None:
        """チャンネルを購読"""
        channels = [
            f"lightning_ticker_{self.product_code}",
            f"lightning_board_snapshot_{self.product_code}",
            f"lightning_board_{self.product_code}",
            f"lightning_executions_{self.product_code}",
        ]

        for channel in channels:
            subscribe_msg = {
                "jsonrpc": "2.0",
                "method": "subscribe",
                "params": {"channel": channel},
                "id": None,
            }
            await self.ws.send(json.dumps(subscribe_msg))
            logger.debug(f"Subscribed to {channel}")

    async def _ws_message_handler(self) -> None:
        """WebSocketメッセージを処理"""
        try:
            async for message in self.ws:
                data = json.loads(message)

                if "params" in data:
                    channel = data["params"].get("channel", "")
                    msg = data["params"].get("message", {})

                    # Process different channels
                    if "ticker" in channel:
                        await self._handle_ticker(msg)
                    elif "board_snapshot" in channel:
                        await self._handle_board_snapshot(msg)
                    elif "board" in channel and "snapshot" not in channel:
                        await self._handle_board_diff(msg)
                    elif "executions" in channel:
                        await self._handle_executions(msg)

                    # Call registered callbacks
                    for callback in self.ws_callbacks.get(channel, []):
                        try:
                            await callback(msg)
                        except Exception as e:
                            logger.error(f"Callback error: {e}")

        except websockets.ConnectionClosed:
            logger.warning("WebSocket connection closed")
            self.ws_connected = False
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            self.ws_connected = False

    async def _handle_ticker(self, msg: Dict) -> None:
        """ティッカー更新を処理"""
        self.ticker = Ticker(
            product_code=msg.get("product_code", self.product_code),
            timestamp=datetime.fromisoformat(msg.get("timestamp", "").replace("Z", "+00:00")),
            best_bid=msg.get("best_bid", 0),
            best_ask=msg.get("best_ask", 0),
            best_bid_size=msg.get("best_bid_size", 0),
            best_ask_size=msg.get("best_ask_size", 0),
            ltp=msg.get("ltp", 0),
            volume=msg.get("volume", 0),
            volume_by_product=msg.get("volume_by_product", 0),
        )

    async def _handle_board_snapshot(self, msg: Dict) -> None:
        """板スナップショットを処理"""
        self.order_book = OrderBook(
            product_code=self.product_code,
            timestamp=datetime.now(),
            bids=[(b["price"], b["size"]) for b in msg.get("bids", [])],
            asks=[(a["price"], a["size"]) for a in msg.get("asks", [])],
        )

    async def _handle_board_diff(self, msg: Dict) -> None:
        """板差分を処理"""
        if self.order_book is None:
            return

        # Update bids
        for b in msg.get("bids", []):
            price, size = b["price"], b["size"]
            if size == 0:
                self.order_book.bids = [x for x in self.order_book.bids if x[0] != price]
            else:
                self.order_book.bids = [(price, size) if x[0] == price else x for x in self.order_book.bids]
                if not any(x[0] == price for x in self.order_book.bids):
                    self.order_book.bids.append((price, size))

        # Update asks
        for a in msg.get("asks", []):
            price, size = a["price"], a["size"]
            if size == 0:
                self.order_book.asks = [x for x in self.order_book.asks if x[0] != price]
            else:
                self.order_book.asks = [(price, size) if x[0] == price else x for x in self.order_book.asks]
                if not any(x[0] == price for x in self.order_book.asks):
                    self.order_book.asks.append((price, size))

        # Sort
        self.order_book.bids.sort(key=lambda x: x[0], reverse=True)
        self.order_book.asks.sort(key=lambda x: x[0])

        self.order_book.timestamp = datetime.now()

    async def _handle_executions(self, msg: List) -> None:
        """約定を処理"""
        for e in msg:
            execution = Execution(
                id=e.get("id", 0),
                side=e.get("side", ""),
                price=e.get("price", 0),
                size=e.get("size", 0),
                timestamp=datetime.fromisoformat(e.get("exec_date", "").replace("Z", "+00:00")),
                buy_child_order_acceptance_id=e.get("buy_child_order_acceptance_id", ""),
                sell_child_order_acceptance_id=e.get("sell_child_order_acceptance_id", ""),
            )
            self.executions.append(execution)

    def on_message(self, channel: str, callback: Callable) -> None:
        """メッセージコールバックを登録"""
        if channel not in self.ws_callbacks:
            self.ws_callbacks[channel] = []
        self.ws_callbacks[channel].append(callback)

    async def disconnect_websocket(self) -> None:
        """WebSocketを切断"""
        if self.ws:
            await self.ws.close()
            self.ws_connected = False
            logger.info("WebSocket disconnected")

    # ========== Utility ==========

    def get_stats(self) -> Dict:
        """統計情報を取得"""
        return {
            "total_requests": self.total_requests,
            "failed_requests": self.failed_requests,
            "success_rate": 1 - (self.failed_requests / max(1, self.total_requests)),
            "ws_connected": self.ws_connected,
            "cached_ticker": self.ticker is not None,
            "cached_executions": len(self.executions),
            "active_orders": len(self.orders),
        }


class MockBitFlyerClient(BitFlyerClient):
    """
    モックbitFlyerクライアント

    Paper Tradingやテスト用
    """

    # ペア別の現実的な価格（2024年相場）
    MOCK_BASE_PRICES = {
        "BTC_JPY": 5000000,    # ¥5,000,000
        "ETH_JPY": 350000,     # ¥350,000
        "XRP_JPY": 80,         # ¥80
        "XLM_JPY": 50,         # ¥50
        "MONA_JPY": 60,        # ¥60
        "BCH_JPY": 30000,      # ¥30,000
        "LTC_JPY": 10000,      # ¥10,000
    }

    def __init__(self, initial_balance: float = 1000000, **kwargs):
        super().__init__(**kwargs)

        self.mock_balance = {
            "JPY": {"amount": initial_balance, "available": initial_balance},
            "BTC": {"amount": 0, "available": 0},
        }
        self.mock_positions = []
        self.mock_orders = {}

        # ペアに応じた現実的な価格を設定
        self.mock_price = self.MOCK_BASE_PRICES.get(self.product_code, 5000000)

        logger.info(f"Mock BitFlyer client initialized for {self.product_code} @ ¥{self.mock_price:,.0f}")

    async def get_ticker(self, product_code: str = None) -> Optional[Ticker]:
        """モックティッカー"""
        import random

        # 正しいペアの価格を使用
        pc = product_code or self.product_code
        base_price = self.MOCK_BASE_PRICES.get(pc, self.mock_price)

        # Simulate price movement (±0.1%)
        price = base_price * (1 + random.uniform(-0.001, 0.001))

        self.ticker = Ticker(
            product_code=pc,
            timestamp=datetime.now(),
            best_bid=price * 0.9999,
            best_ask=price * 1.0001,
            best_bid_size=1.0,
            best_ask_size=1.0,
            ltp=price,
            volume=100,
            volume_by_product=100,
        )
        return self.ticker

    async def get_balance(self) -> List[Dict]:
        """モック残高"""
        return [
            {"currency_code": "JPY", **self.mock_balance["JPY"]},
            {"currency_code": "BTC", **self.mock_balance["BTC"]},
        ]

    async def send_order(
        self,
        side: OrderSide,
        size: float,
        order_type: OrderType = OrderType.MARKET,
        price: float = 0,
        **kwargs,
    ) -> Optional[str]:
        """モック注文"""
        import uuid

        order_id = str(uuid.uuid4())[:8]
        exec_price = self.mock_price

        if side == OrderSide.BUY:
            cost = exec_price * size
            if self.mock_balance["JPY"]["available"] >= cost:
                self.mock_balance["JPY"]["available"] -= cost
                self.mock_balance["JPY"]["amount"] -= cost
                self.mock_balance["BTC"]["available"] += size
                self.mock_balance["BTC"]["amount"] += size
                logger.info(f"[MOCK] BUY {size} BTC @ {exec_price}")
            else:
                logger.warning("[MOCK] Insufficient balance")
                return None
        else:
            if self.mock_balance["BTC"]["available"] >= size:
                self.mock_balance["BTC"]["available"] -= size
                self.mock_balance["BTC"]["amount"] -= size
                proceeds = exec_price * size
                self.mock_balance["JPY"]["available"] += proceeds
                self.mock_balance["JPY"]["amount"] += proceeds
                logger.info(f"[MOCK] SELL {size} BTC @ {exec_price}")
            else:
                logger.warning("[MOCK] Insufficient BTC")
                return None

        return order_id


logger.info("BitFlyer API client module loaded")

"""
bitFlyer WebSocket Client
==========================
リアルタイムデータ受信用WebSocketクライアント
高頻度取引に最適化
"""

import json
import threading
import time
from typing import Callable, Dict, List, Optional, Any
from datetime import datetime
from collections import deque
import websocket
from loguru import logger


class BitFlyerWebSocket:
    """bitFlyer Realtime API (WebSocket) クライアント"""

    WS_URL = "wss://ws.lightstream.bitflyer.com/json-rpc"

    def __init__(
        self,
        on_ticker: Optional[Callable] = None,
        on_executions: Optional[Callable] = None,
        on_board: Optional[Callable] = None,
        on_board_snapshot: Optional[Callable] = None,
    ):
        """
        Args:
            on_ticker: ティッカー更新時のコールバック
            on_executions: 約定時のコールバック
            on_board: 板更新時のコールバック
            on_board_snapshot: 板スナップショット時のコールバック
        """
        self.on_ticker = on_ticker
        self.on_executions = on_executions
        self.on_board = on_board
        self.on_board_snapshot = on_board_snapshot

        self.ws: Optional[websocket.WebSocketApp] = None
        self.ws_thread: Optional[threading.Thread] = None
        self.is_running = False
        self.subscribed_channels: List[str] = []

        # データキャッシュ
        self.ticker_cache: Dict[str, Dict] = {}
        self.execution_cache: Dict[str, deque] = {}
        self.board_cache: Dict[str, Dict] = {}

        # パフォーマンス統計
        self.message_count = 0
        self.last_message_time: Optional[datetime] = None

    def _on_open(self, ws):
        """WebSocket接続時"""
        logger.info("WebSocket connection established")
        self.is_running = True

        # チャンネル購読
        for channel in self.subscribed_channels:
            self._subscribe(channel)

    def _on_close(self, ws, close_status_code, close_msg):
        """WebSocket切断時"""
        logger.warning(f"WebSocket closed: {close_status_code} - {close_msg}")
        self.is_running = False

        # 自動再接続
        if self.subscribed_channels:
            logger.info("Attempting to reconnect...")
            time.sleep(1)
            self.connect()

    def _on_error(self, ws, error):
        """WebSocketエラー時"""
        logger.error(f"WebSocket error: {error}")

    def _on_message(self, ws, message):
        """メッセージ受信時"""
        self.message_count += 1
        self.last_message_time = datetime.now()

        try:
            data = json.loads(message)
            params = data.get("params", {})
            channel = params.get("channel", "")
            message_data = params.get("message", {})

            # チャンネルに応じて処理
            if "ticker" in channel:
                self._handle_ticker(channel, message_data)
            elif "executions" in channel:
                self._handle_executions(channel, message_data)
            elif "board_snapshot" in channel:
                self._handle_board_snapshot(channel, message_data)
            elif "board" in channel:
                self._handle_board(channel, message_data)

        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")

    def _handle_ticker(self, channel: str, data: Dict):
        """ティッカーデータ処理"""
        product_code = channel.replace("lightning_ticker_", "")
        self.ticker_cache[product_code] = {
            "timestamp": datetime.now(),
            "data": data,
        }

        if self.on_ticker:
            self.on_ticker(product_code, data)

    def _handle_executions(self, channel: str, data: List[Dict]):
        """約定データ処理"""
        product_code = channel.replace("lightning_executions_", "")

        if product_code not in self.execution_cache:
            self.execution_cache[product_code] = deque(maxlen=1000)

        for execution in data:
            self.execution_cache[product_code].append({
                "timestamp": datetime.now(),
                "data": execution,
            })

        if self.on_executions:
            self.on_executions(product_code, data)

    def _handle_board(self, channel: str, data: Dict):
        """板差分データ処理"""
        product_code = channel.replace("lightning_board_", "")

        if product_code in self.board_cache:
            # 差分更新
            current_board = self.board_cache[product_code]["data"]
            self._update_board(current_board, data)
            self.board_cache[product_code]["timestamp"] = datetime.now()

        if self.on_board:
            self.on_board(product_code, data)

    def _handle_board_snapshot(self, channel: str, data: Dict):
        """板スナップショット処理"""
        product_code = channel.replace("lightning_board_snapshot_", "")
        self.board_cache[product_code] = {
            "timestamp": datetime.now(),
            "data": data,
        }

        if self.on_board_snapshot:
            self.on_board_snapshot(product_code, data)

    def _update_board(self, board: Dict, diff: Dict):
        """板の差分更新"""
        # Bidsの更新
        for bid in diff.get("bids", []):
            price = bid["price"]
            size = bid["size"]
            if size == 0:
                board["bids"] = [b for b in board.get("bids", []) if b["price"] != price]
            else:
                updated = False
                for b in board.get("bids", []):
                    if b["price"] == price:
                        b["size"] = size
                        updated = True
                        break
                if not updated:
                    board.setdefault("bids", []).append({"price": price, "size": size})

        # Asksの更新
        for ask in diff.get("asks", []):
            price = ask["price"]
            size = ask["size"]
            if size == 0:
                board["asks"] = [a for a in board.get("asks", []) if a["price"] != price]
            else:
                updated = False
                for a in board.get("asks", []):
                    if a["price"] == price:
                        a["size"] = size
                        updated = True
                        break
                if not updated:
                    board.setdefault("asks", []).append({"price": price, "size": size})

        # ソート
        board["bids"] = sorted(board.get("bids", []), key=lambda x: x["price"], reverse=True)
        board["asks"] = sorted(board.get("asks", []), key=lambda x: x["price"])

    def _subscribe(self, channel: str):
        """チャンネル購読"""
        if self.ws:
            subscribe_msg = {
                "jsonrpc": "2.0",
                "method": "subscribe",
                "params": {"channel": channel},
                "id": None,
            }
            self.ws.send(json.dumps(subscribe_msg))
            logger.info(f"Subscribed to {channel}")

    def _unsubscribe(self, channel: str):
        """チャンネル購読解除"""
        if self.ws:
            unsubscribe_msg = {
                "jsonrpc": "2.0",
                "method": "unsubscribe",
                "params": {"channel": channel},
                "id": None,
            }
            self.ws.send(json.dumps(unsubscribe_msg))
            logger.info(f"Unsubscribed from {channel}")

    def subscribe_ticker(self, product_code: str = "BTC_JPY"):
        """ティッカー購読"""
        channel = f"lightning_ticker_{product_code}"
        self.subscribed_channels.append(channel)
        if self.is_running:
            self._subscribe(channel)

    def subscribe_executions(self, product_code: str = "BTC_JPY"):
        """約定購読"""
        channel = f"lightning_executions_{product_code}"
        self.subscribed_channels.append(channel)
        if self.is_running:
            self._subscribe(channel)

    def subscribe_board(self, product_code: str = "BTC_JPY"):
        """板購読"""
        channel = f"lightning_board_{product_code}"
        self.subscribed_channels.append(channel)
        if self.is_running:
            self._subscribe(channel)

    def subscribe_board_snapshot(self, product_code: str = "BTC_JPY"):
        """板スナップショット購読"""
        channel = f"lightning_board_snapshot_{product_code}"
        self.subscribed_channels.append(channel)
        if self.is_running:
            self._subscribe(channel)

    def subscribe_all(self, product_code: str = "BTC_JPY"):
        """全チャンネル購読"""
        self.subscribe_ticker(product_code)
        self.subscribe_executions(product_code)
        self.subscribe_board(product_code)
        self.subscribe_board_snapshot(product_code)

    def connect(self):
        """WebSocket接続開始"""
        websocket.enableTrace(False)

        self.ws = websocket.WebSocketApp(
            self.WS_URL,
            on_open=self._on_open,
            on_close=self._on_close,
            on_error=self._on_error,
            on_message=self._on_message,
        )

        self.ws_thread = threading.Thread(target=self.ws.run_forever)
        self.ws_thread.daemon = True
        self.ws_thread.start()

        logger.info("WebSocket connecting...")

    def disconnect(self):
        """WebSocket切断"""
        self.is_running = False
        if self.ws:
            self.ws.close()
        logger.info("WebSocket disconnected")

    def get_cached_ticker(self, product_code: str) -> Optional[Dict]:
        """キャッシュされたティッカー取得"""
        if product_code in self.ticker_cache:
            return self.ticker_cache[product_code]["data"]
        return None

    def get_cached_board(self, product_code: str) -> Optional[Dict]:
        """キャッシュされた板情報取得"""
        if product_code in self.board_cache:
            return self.board_cache[product_code]["data"]
        return None

    def get_recent_executions(self, product_code: str, count: int = 100) -> List[Dict]:
        """最近の約定取得"""
        if product_code in self.execution_cache:
            executions = list(self.execution_cache[product_code])[-count:]
            return [e["data"] for e in executions]
        return []

    def get_vwap(self, product_code: str, seconds: int = 60) -> Optional[float]:
        """VWAP (出来高加重平均価格) 計算"""
        if product_code not in self.execution_cache:
            return None

        now = datetime.now()
        total_value = 0.0
        total_volume = 0.0

        for execution in self.execution_cache[product_code]:
            age = (now - execution["timestamp"]).total_seconds()
            if age <= seconds:
                price = execution["data"]["price"]
                size = execution["data"]["size"]
                total_value += price * size
                total_volume += size

        if total_volume > 0:
            return total_value / total_volume
        return None

    def get_stats(self) -> Dict[str, Any]:
        """統計情報取得"""
        return {
            "is_running": self.is_running,
            "message_count": self.message_count,
            "last_message_time": self.last_message_time,
            "subscribed_channels": self.subscribed_channels,
            "cached_products": {
                "tickers": list(self.ticker_cache.keys()),
                "boards": list(self.board_cache.keys()),
                "executions": list(self.execution_cache.keys()),
            },
        }


class OrderBookAnalyzer:
    """オーダーブック分析ツール"""

    def __init__(self, ws_client: BitFlyerWebSocket):
        self.ws_client = ws_client

    def get_order_book_imbalance(self, product_code: str, depth: int = 10) -> float:
        """
        オーダーブック不均衡計算
        Returns: -1.0 (売り優勢) 〜 1.0 (買い優勢)
        """
        board = self.ws_client.get_cached_board(product_code)
        if not board:
            return 0.0

        bids = board.get("bids", [])[:depth]
        asks = board.get("asks", [])[:depth]

        bid_volume = sum(b["size"] for b in bids)
        ask_volume = sum(a["size"] for a in asks)

        total = bid_volume + ask_volume
        if total == 0:
            return 0.0

        return (bid_volume - ask_volume) / total

    def get_spread_info(self, product_code: str) -> Dict[str, float]:
        """スプレッド情報取得"""
        board = self.ws_client.get_cached_board(product_code)
        if not board or not board.get("bids") or not board.get("asks"):
            return {"spread": 0, "spread_percent": 0, "mid_price": 0}

        best_bid = board["bids"][0]["price"]
        best_ask = board["asks"][0]["price"]
        mid_price = (best_bid + best_ask) / 2
        spread = best_ask - best_bid

        return {
            "best_bid": best_bid,
            "best_ask": best_ask,
            "mid_price": mid_price,
            "spread": spread,
            "spread_percent": (spread / mid_price) * 100 if mid_price > 0 else 0,
        }

    def get_depth_levels(self, product_code: str, levels: int = 5) -> Dict[str, List]:
        """価格帯別の深さ取得"""
        board = self.ws_client.get_cached_board(product_code)
        if not board:
            return {"bids": [], "asks": []}

        return {
            "bids": board.get("bids", [])[:levels],
            "asks": board.get("asks", [])[:levels],
        }

    def get_market_pressure(self, product_code: str) -> Dict[str, float]:
        """市場圧力分析"""
        executions = self.ws_client.get_recent_executions(product_code, 100)
        if not executions:
            return {"buy_pressure": 0, "sell_pressure": 0, "net_pressure": 0}

        buy_volume = sum(e["size"] for e in executions if e["side"] == "BUY")
        sell_volume = sum(e["size"] for e in executions if e["side"] == "SELL")
        total = buy_volume + sell_volume

        if total == 0:
            return {"buy_pressure": 0, "sell_pressure": 0, "net_pressure": 0}

        return {
            "buy_pressure": buy_volume / total,
            "sell_pressure": sell_volume / total,
            "net_pressure": (buy_volume - sell_volume) / total,
        }

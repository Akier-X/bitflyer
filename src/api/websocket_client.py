#!/usr/bin/env python3
"""
================================================================================
    ⚡ WebSocket リアルタイム価格クライアント
================================================================================
    - bitFlyer Realtime API (JSON-RPC 2.0 over WebSocket)
    - 複数ペア同時購読
    - 自動再接続
================================================================================
"""

import asyncio
import json
from typing import Dict, Optional, Callable, List
from datetime import datetime
from dataclasses import dataclass
import websockets
from loguru import logger


BITFLYER_WS_URL = "wss://ws.lightstream.bitflyer.com/json-rpc"


@dataclass
class RealtimeTicker:
    """リアルタイムティッカー"""
    pair: str
    ltp: float  # Last Trade Price
    best_bid: float
    best_ask: float
    volume: float
    timestamp: datetime


class BitFlyerWebSocket:
    """bitFlyer WebSocketクライアント"""

    def __init__(self):
        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        self.is_connected: bool = False
        self.subscribed_pairs: List[str] = []

        self._callbacks: Dict[str, List[Callable]] = {}
        self._latest_tickers: Dict[str, RealtimeTicker] = {}
        self._reconnect_delay: int = 5
        self._should_run: bool = False

    async def connect(self):
        """接続"""
        try:
            self.ws = await websockets.connect(
                BITFLYER_WS_URL,
                ping_interval=30,
                ping_timeout=10
            )
            self.is_connected = True
            self._reconnect_delay = 5
            logger.info("  ⚡ WebSocket接続完了")
            return True
        except Exception as e:
            logger.error(f"  WebSocket接続エラー: {e}")
            self.is_connected = False
            return False

    async def subscribe(self, pairs: List[str]):
        """チャンネル購読"""
        if not self.ws or not self.is_connected:
            return False

        self.subscribed_pairs = pairs

        for pair in pairs:
            channel = f"lightning_ticker_{pair}"
            subscribe_msg = {
                "jsonrpc": "2.0",
                "method": "subscribe",
                "params": {"channel": channel}
            }

            try:
                await self.ws.send(json.dumps(subscribe_msg))
                logger.debug(f"  購読: {channel}")
            except Exception as e:
                logger.error(f"  購読エラー: {e}")
                return False

        return True

    def on_ticker(self, pair: str, callback: Callable[[RealtimeTicker], None]):
        """ティッカー受信コールバック登録"""
        if pair not in self._callbacks:
            self._callbacks[pair] = []
        self._callbacks[pair].append(callback)

    def get_latest_ticker(self, pair: str) -> Optional[RealtimeTicker]:
        """最新ティッカー取得"""
        return self._latest_tickers.get(pair)

    def get_latest_price(self, pair: str) -> Optional[float]:
        """最新価格取得"""
        ticker = self._latest_tickers.get(pair)
        return ticker.ltp if ticker else None

    async def _process_message(self, message: str):
        """メッセージ処理"""
        try:
            data = json.loads(message)

            if "params" in data:
                params = data["params"]
                channel = params.get("channel", "")
                msg_data = params.get("message", {})

                # lightning_ticker_XXX_JPY
                if channel.startswith("lightning_ticker_"):
                    pair = channel.replace("lightning_ticker_", "")

                    ticker = RealtimeTicker(
                        pair=pair,
                        ltp=float(msg_data.get("ltp", 0)),
                        best_bid=float(msg_data.get("best_bid", 0)),
                        best_ask=float(msg_data.get("best_ask", 0)),
                        volume=float(msg_data.get("volume", 0)),
                        timestamp=datetime.now()
                    )

                    self._latest_tickers[pair] = ticker

                    # コールバック実行
                    if pair in self._callbacks:
                        for callback in self._callbacks[pair]:
                            try:
                                await asyncio.create_task(
                                    asyncio.coroutine(lambda: callback(ticker))()
                                ) if asyncio.iscoroutinefunction(callback) else callback(ticker)
                            except:
                                callback(ticker)

        except json.JSONDecodeError:
            pass
        except Exception as e:
            logger.debug(f"  メッセージ処理エラー: {e}")

    async def _listen(self):
        """メッセージ受信ループ"""
        while self._should_run and self.ws:
            try:
                message = await asyncio.wait_for(
                    self.ws.recv(),
                    timeout=60
                )
                await self._process_message(message)

            except asyncio.TimeoutError:
                # タイムアウトは正常（データが来ない時間帯）
                continue

            except websockets.exceptions.ConnectionClosed:
                logger.warning("  WebSocket切断")
                self.is_connected = False
                break

            except Exception as e:
                logger.debug(f"  受信エラー: {e}")
                await asyncio.sleep(1)

    async def _auto_reconnect(self):
        """自動再接続ループ"""
        while self._should_run:
            if not self.is_connected:
                logger.info(f"  {self._reconnect_delay}秒後に再接続...")
                await asyncio.sleep(self._reconnect_delay)

                if await self.connect():
                    await self.subscribe(self.subscribed_pairs)
                else:
                    self._reconnect_delay = min(self._reconnect_delay * 2, 60)

            await asyncio.sleep(5)

    async def start(self, pairs: List[str]):
        """WebSocket開始"""
        self._should_run = True

        if not await self.connect():
            return False

        if not await self.subscribe(pairs):
            return False

        # バックグラウンドタスク開始
        asyncio.create_task(self._listen())
        asyncio.create_task(self._auto_reconnect())

        return True

    async def stop(self):
        """WebSocket停止"""
        self._should_run = False

        if self.ws:
            try:
                await self.ws.close()
            except:
                pass

        self.is_connected = False
        logger.info("  ⚡ WebSocket停止")


class MultiPairWebSocket:
    """複数ペア対応WebSocketマネージャー"""

    def __init__(self, pairs: List[str]):
        self.pairs = pairs
        self.ws = BitFlyerWebSocket()
        self._price_callbacks: Dict[str, List[Callable]] = {}

    def on_price_update(self, pair: str, callback: Callable[[float], None]):
        """価格更新コールバック登録"""
        if pair not in self._price_callbacks:
            self._price_callbacks[pair] = []
        self._price_callbacks[pair].append(callback)

    def _handle_ticker(self, ticker: RealtimeTicker):
        """ティッカー処理"""
        if ticker.pair in self._price_callbacks:
            for callback in self._price_callbacks[ticker.pair]:
                try:
                    callback(ticker.ltp)
                except Exception as e:
                    logger.debug(f"コールバックエラー: {e}")

    async def start(self):
        """開始"""
        # 全ペアにコールバック登録
        for pair in self.pairs:
            self.ws.on_ticker(pair, self._handle_ticker)

        return await self.ws.start(self.pairs)

    async def stop(self):
        """停止"""
        await self.ws.stop()

    def get_price(self, pair: str) -> Optional[float]:
        """現在価格取得"""
        return self.ws.get_latest_price(pair)

    def get_ticker(self, pair: str) -> Optional[RealtimeTicker]:
        """ティッカー取得"""
        return self.ws.get_latest_ticker(pair)

    @property
    def is_connected(self) -> bool:
        return self.ws.is_connected

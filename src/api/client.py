"""
bitFlyer REST API Client
=========================
高性能なAPIクライアント実装
"""

import hashlib
import hmac
import time
import json
from typing import Optional, Dict, Any, List
from datetime import datetime
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from loguru import logger


class BitFlyerClient:
    """bitFlyer REST APIクライアント"""

    BASE_URL = "https://api.bitflyer.com"

    def __init__(self, api_key: str, api_secret: str, timeout: int = 30):
        """
        Args:
            api_key: bitFlyer APIキー
            api_secret: bitFlyer APIシークレット
            timeout: リクエストタイムアウト秒数
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.timeout = timeout

        # 高性能HTTPセッション設定
        self.session = self._create_session()

        logger.info("BitFlyer API Client initialized")

    def _create_session(self) -> requests.Session:
        """リトライ機能付きセッション作成"""
        session = requests.Session()

        retry_strategy = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
        )

        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=100,
            pool_maxsize=100,
        )

        session.mount("https://", adapter)
        session.mount("http://", adapter)

        return session

    def _generate_signature(self, method: str, path: str, body: str = "") -> Dict[str, str]:
        """API認証シグネチャ生成"""
        timestamp = str(int(time.time() * 1000))
        text = timestamp + method + path + body

        signature = hmac.new(
            self.api_secret.encode("utf-8"),
            text.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()

        return {
            "ACCESS-KEY": self.api_key,
            "ACCESS-TIMESTAMP": timestamp,
            "ACCESS-SIGN": signature,
            "Content-Type": "application/json",
        }

    def _request(
        self,
        method: str,
        path: str,
        params: Optional[Dict] = None,
        data: Optional[Dict] = None,
        auth_required: bool = False
    ) -> Any:
        """HTTPリクエスト実行"""
        url = f"{self.BASE_URL}{path}"
        body = json.dumps(data) if data else ""

        headers = {}
        if auth_required:
            headers = self._generate_signature(method, path, body)

        try:
            response = self.session.request(
                method=method,
                url=url,
                params=params,
                data=body if data else None,
                headers=headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response.json() if response.text else None

        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            raise

    # ==================== Public API ====================

    def get_markets(self) -> List[Dict]:
        """取引可能なマーケット一覧取得"""
        return self._request("GET", "/v1/markets")

    def get_board(self, product_code: str = "BTC_JPY") -> Dict:
        """板情報取得"""
        return self._request("GET", "/v1/board", params={"product_code": product_code})

    def get_ticker(self, product_code: str = "BTC_JPY") -> Dict:
        """ティッカー情報取得"""
        return self._request("GET", "/v1/ticker", params={"product_code": product_code})

    def get_executions(
        self,
        product_code: str = "BTC_JPY",
        count: int = 100,
        before: Optional[int] = None,
        after: Optional[int] = None
    ) -> List[Dict]:
        """約定履歴取得"""
        params = {"product_code": product_code, "count": count}
        if before:
            params["before"] = before
        if after:
            params["after"] = after
        return self._request("GET", "/v1/executions", params=params)

    def get_board_state(self, product_code: str = "BTC_JPY") -> Dict:
        """板の状態取得"""
        return self._request("GET", "/v1/getboardstate", params={"product_code": product_code})

    def get_health(self, product_code: str = "BTC_JPY") -> Dict:
        """取引所の稼働状態取得"""
        return self._request("GET", "/v1/gethealth", params={"product_code": product_code})

    # ==================== Private API ====================

    def get_balance(self) -> List[Dict]:
        """残高取得"""
        return self._request("GET", "/v1/me/getbalance", auth_required=True)

    def get_collateral(self) -> Dict:
        """証拠金情報取得"""
        return self._request("GET", "/v1/me/getcollateral", auth_required=True)

    def get_positions(self, product_code: str = "FX_BTC_JPY") -> List[Dict]:
        """建玉情報取得"""
        return self._request(
            "GET",
            "/v1/me/getpositions",
            params={"product_code": product_code},
            auth_required=True
        )

    def get_trading_commission(self, product_code: str = "BTC_JPY") -> Dict:
        """取引手数料取得"""
        return self._request(
            "GET",
            "/v1/me/gettradingcommission",
            params={"product_code": product_code},
            auth_required=True
        )

    # ==================== Order API ====================

    def send_child_order(
        self,
        product_code: str,
        child_order_type: str,
        side: str,
        size: float,
        price: Optional[float] = None,
        minute_to_expire: int = 43200,
        time_in_force: str = "GTC"
    ) -> Dict:
        """
        注文送信

        Args:
            product_code: 取引ペア ("BTC_JPY", "FX_BTC_JPY"等)
            child_order_type: 注文タイプ ("LIMIT" or "MARKET")
            side: 売買方向 ("BUY" or "SELL")
            size: 注文数量
            price: 価格（成行の場合はNone）
            minute_to_expire: 有効期限（分）
            time_in_force: 執行条件 ("GTC", "IOC", "FOK")
        """
        data = {
            "product_code": product_code,
            "child_order_type": child_order_type,
            "side": side,
            "size": size,
            "minute_to_expire": minute_to_expire,
            "time_in_force": time_in_force,
        }

        if price is not None and child_order_type == "LIMIT":
            data["price"] = int(price)

        result = self._request("POST", "/v1/me/sendchildorder", data=data, auth_required=True)
        logger.info(f"Order sent: {side} {size} {product_code} @ {price}")
        return result

    def cancel_child_order(
        self,
        product_code: str,
        child_order_id: Optional[str] = None,
        child_order_acceptance_id: Optional[str] = None
    ) -> None:
        """注文キャンセル"""
        data = {"product_code": product_code}

        if child_order_id:
            data["child_order_id"] = child_order_id
        elif child_order_acceptance_id:
            data["child_order_acceptance_id"] = child_order_acceptance_id

        self._request("POST", "/v1/me/cancelchildorder", data=data, auth_required=True)
        logger.info(f"Order cancelled: {child_order_id or child_order_acceptance_id}")

    def cancel_all_child_orders(self, product_code: str) -> None:
        """全注文キャンセル"""
        self._request(
            "POST",
            "/v1/me/cancelallchildorders",
            data={"product_code": product_code},
            auth_required=True
        )
        logger.info(f"All orders cancelled for {product_code}")

    def get_child_orders(
        self,
        product_code: str,
        child_order_state: str = "ACTIVE",
        count: int = 100
    ) -> List[Dict]:
        """注文一覧取得"""
        return self._request(
            "GET",
            "/v1/me/getchildorders",
            params={
                "product_code": product_code,
                "child_order_state": child_order_state,
                "count": count,
            },
            auth_required=True
        )

    def get_my_executions(
        self,
        product_code: str,
        count: int = 100
    ) -> List[Dict]:
        """自分の約定履歴取得"""
        return self._request(
            "GET",
            "/v1/me/getexecutions",
            params={"product_code": product_code, "count": count},
            auth_required=True
        )

    # ==================== Special Orders ====================

    def send_parent_order(
        self,
        order_method: str,
        parameters: List[Dict],
        minute_to_expire: int = 43200,
        time_in_force: str = "GTC"
    ) -> Dict:
        """
        特殊注文送信（IFD, OCO, IFDOCO等）

        Args:
            order_method: "SIMPLE", "IFD", "OCO", "IFDOCO"
            parameters: 注文パラメータのリスト
        """
        data = {
            "order_method": order_method,
            "minute_to_expire": minute_to_expire,
            "time_in_force": time_in_force,
            "parameters": parameters,
        }
        return self._request("POST", "/v1/me/sendparentorder", data=data, auth_required=True)

    # ==================== Utility Methods ====================

    def get_total_balance_jpy(self) -> float:
        """JPY換算の総残高取得"""
        balances = self.get_balance()
        total = 0.0

        for balance in balances:
            currency = balance["currency_code"]
            amount = balance["amount"]

            if currency == "JPY":
                total += amount
            elif currency == "BTC":
                ticker = self.get_ticker("BTC_JPY")
                total += amount * ticker["ltp"]
            elif currency == "ETH":
                ticker = self.get_ticker("ETH_JPY")
                total += amount * ticker["ltp"]

        return total

    def get_order_book_imbalance(self, product_code: str = "BTC_JPY") -> float:
        """オーダーブック不均衡計算（-1〜1、正は買い優勢）"""
        board = self.get_board(product_code)

        bid_volume = sum(b["size"] for b in board.get("bids", [])[:10])
        ask_volume = sum(a["size"] for a in board.get("asks", [])[:10])

        total = bid_volume + ask_volume
        if total == 0:
            return 0.0

        return (bid_volume - ask_volume) / total

    def get_spread(self, product_code: str = "BTC_JPY") -> Dict[str, float]:
        """スプレッド情報取得"""
        board = self.get_board(product_code)

        best_bid = board["bids"][0]["price"] if board.get("bids") else 0
        best_ask = board["asks"][0]["price"] if board.get("asks") else 0

        mid_price = (best_bid + best_ask) / 2
        spread = best_ask - best_bid
        spread_percent = (spread / mid_price) * 100 if mid_price > 0 else 0

        return {
            "best_bid": best_bid,
            "best_ask": best_ask,
            "mid_price": mid_price,
            "spread": spread,
            "spread_percent": spread_percent,
        }

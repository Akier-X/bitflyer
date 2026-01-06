#!/usr/bin/env python3
"""
================================================================================
    🏆 ULTIMATE AI TRADER v13.0 - 世界最強・高利益AIトレーダー
================================================================================
    全機能統合版:
    - MACD / ボリンジャーバンド / RSI / 複数時間足
    - LSTM機械学習予測
    - WebSocketリアルタイム価格
    - LINE通知
    - パターン認識
    - 適応型パラメータ
    - 🛡️ BULLETPROOF ORDER SYSTEM (絶対残高不足エラーなし)
    - 🧠 MARKET INTELLIGENCE (マルチペア相関・ボラ適応・時間帯学習・感情分析)
    - 💰 PROFIT MAXIMIZER (損小利大・高勝率・トレンドフォロー)
================================================================================
"""

import asyncio
import math
import sys
import os
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass, field
from collections import deque

import numpy as np
from loguru import logger

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
env_path = PROJECT_ROOT / '.env'
if env_path.exists():
    load_dotenv(env_path)

from config.settings import get_config
from src.api.bitflyer_client import BitFlyerClient, OrderSide, OrderType
from src.analysis.indicators import TechnicalIndicators
from src.analysis.ml_predictor import EnsemblePredictor, Prediction
from src.analysis.market_intelligence import MarketIntelligence, MarketState, TradingSignal
from src.notifications.line_notify import SmartNotifier, TradeNotification, DailyReport

# WebSocket (オプション)
try:
    from src.api.websocket_client import MultiPairWebSocket
    WS_AVAILABLE = True
except ImportError:
    WS_AVAILABLE = False


# =============================================================================
# 💰 設定 - PROFIT MAXIMIZER パラメータ
# =============================================================================

TRADING_FEE = 0.0015  # 片道0.15%、往復0.3%

# =============================================================================
# 💰 損小利大パラメータ（これが利益の源泉）
# =============================================================================
#
# 【重要】リスクリワード比 = 1:2.5
# - 利確 1.5% - 手数料 0.3% = 純利益 +1.2%
# - 損切 0.3% + 手数料 0.3% = 純損失 -0.6%
# - 必要勝率: 33%以上で黒字（現実的に達成可能）
#
BASE_TAKE_PROFIT = 0.015     # 1.5%で利確（大きく取る）
BASE_STOP_LOSS = 0.003       # 0.3%で損切り（早く切る）
TRAILING_STOP = 0.004        # 0.4%のトレーリングストップ
MIN_HOLD_TIME = 60           # 最低60秒保持（利益を伸ばす）
MAX_HOLD_TIME = 1800         # 最大30分（塩漬け防止）

# =============================================================================
# 💰 取引頻度制限（オーバートレード防止）
# =============================================================================
TRADE_COOLDOWN = 120         # 2分間クールダウン（頻繁な取引を防ぐ）
FAIL_COOLDOWN = 180          # 失敗後3分待機
MAX_TRADES_PER_HOUR = 10     # 1時間あたり最大10取引

# API設定
PRICE_DELAY = 1.5
LOOP_DELAY = 3
STATUS_INTERVAL = 20

# =============================================================================
# 💰 高精度シグナル設定（勝率向上）
# =============================================================================
AI_CONFIDENCE_THRESHOLD = 0.65   # 65%以上の確信度のみ取引
ML_WEIGHT = 0.25                 # ML予測の重み
TECH_WEIGHT = 0.75               # テクニカル指標の重み（実績重視）
TREND_WEIGHT = 0.30              # トレンド方向の重み（順張り重視）

# トレンドフィルター
REQUIRE_TREND_ALIGNMENT = True   # トレンド方向との一致を必須に
MIN_TREND_STRENGTH = 0.3         # 最低トレンド強度

# =============================================================================
# 🛡️ BULLETPROOF ORDER SYSTEM - 絶対安全設定
# =============================================================================

# 安全マージン設定（Insufficient funds 完全防止）
SAFETY_MARGIN_BUY = 0.05     # 購入時5%の余裕（手数料+価格変動対策）
SAFETY_MARGIN_SELL = 0.02    # 売却時2%の余裕
JPY_RESERVE = 100            # 常に¥100を残す（APIエラー防止）
MAX_BUDGET_RATIO = 0.90      # 最大投資比率90%（10%は常にリザーブ）

# リトライ設定
MAX_ORDER_RETRIES = 3        # 最大リトライ回数
RETRY_SIZE_REDUCTION = 0.90  # リトライ時のサイズ縮小率（90%）

# 残高キャッシュ設定
BALANCE_CACHE_TTL = 5        # 残高キャッシュ有効期限（5秒に短縮）
FORCE_BALANCE_BEFORE_ORDER = True  # 注文前は必ず残高を強制取得


# =============================================================================
# ペア設定
# =============================================================================

@dataclass
class PairConfig:
    code: str
    min_size: float
    decimals: int
    priority: int
    volatility_mult: float = 1.0  # ボラティリティ倍率


PAIRS = {
    "MONA_JPY": PairConfig("MONA_JPY", 1.0, 0, 1, 1.2),
    "XLM_JPY": PairConfig("XLM_JPY", 1.0, 0, 2, 1.0),
    "XRP_JPY": PairConfig("XRP_JPY", 1.0, 0, 3, 0.9),
    "ETH_JPY": PairConfig("ETH_JPY", 0.01, 2, 4, 1.1),
    "BTC_JPY": PairConfig("BTC_JPY", 0.001, 3, 5, 0.8),
}


# =============================================================================
# ポジション管理
# =============================================================================

@dataclass
class Position:
    pair: str
    size: float
    entry_price: float
    entry_time: datetime
    highest: float = 0.0
    lowest: float = float('inf')
    ai_confidence: float = 0.0
    entry_reason: str = ""

    def pnl_pct(self, current: float) -> float:
        return (current - self.entry_price) / self.entry_price

    def hold_seconds(self) -> int:
        return int((datetime.now() - self.entry_time).total_seconds())


# =============================================================================
# 取引履歴
# =============================================================================

@dataclass
class TradeRecord:
    pair: str
    action: str
    size: float
    price: float
    pnl: float
    pnl_pct: float
    reason: str
    timestamp: datetime


# =============================================================================
# メイントレーダー
# =============================================================================

class UltimateTrader:
    """世界最強AIトレーダー v12.0"""

    def __init__(self):
        self.config = get_config()

        # API/データ
        self.clients: Dict[str, BitFlyerClient] = {}
        self.indicators: Dict[str, TechnicalIndicators] = {}
        self.predictors: Dict[str, EnsemblePredictor] = {}
        self.positions: Dict[str, Position] = {}

        # 🧠 Market Intelligence（統合市場分析）
        self.market_intel = MarketIntelligence()

        # WebSocket
        self.ws: Optional[MultiPairWebSocket] = None
        self.use_websocket = WS_AVAILABLE

        # 通知
        self.notifier = SmartNotifier()

        # 状態管理
        self.last_trade: Dict[str, datetime] = {}
        self.failed: Dict[str, datetime] = {}
        self._balances: Dict[str, float] = {}
        self._balance_time: Optional[datetime] = None

        # 統計
        self.trades = 0
        self.wins = 0
        self.pnl = 0.0
        self.start_value = 0.0
        self.start_time: Optional[datetime] = None
        self.trade_history: List[TradeRecord] = []

        # 適応パラメータ
        self.volatility: Dict[str, float] = {}
        self.win_streak = 0
        self.loss_streak = 0

        # 動的残高監視
        self._last_known_balances: Dict[str, float] = {}
        self._balance_check_interval = 30  # 30秒ごとに残高チェック
        self._last_balance_check: Optional[datetime] = None

    # =========================================================================
    # データ取得
    # =========================================================================

    async def get_balances(self, force: bool = False) -> Dict[str, float]:
        """
        🛡️ BULLETPROOF残高取得
        - キャッシュ有効期限を5秒に短縮
        - 注文前は必ずforce=Trueで最新を取得
        """
        if not force and self._balance_time:
            if (datetime.now() - self._balance_time).seconds < BALANCE_CACHE_TTL:
                return self._balances
        try:
            client = next(iter(self.clients.values()))
            data = await asyncio.wait_for(client.get_balance(), timeout=15)
            if data:
                self._balances = {}
                for b in data:
                    code = b.get("currency_code", "")
                    available = float(b.get("available", 0))
                    if available > 0:
                        self._balances[code] = available
                self._balance_time = datetime.now()
        except Exception as e:
            logger.debug(f"残高取得エラー: {e}")
        return self._balances

    async def get_fresh_balance(self, currency: str = "JPY") -> float:
        """
        🛡️ 絶対最新の残高を取得（注文直前用）
        キャッシュを無視して常にAPIから取得
        """
        try:
            client = next(iter(self.clients.values()))
            data = await asyncio.wait_for(client.get_balance(), timeout=10)
            if data:
                for b in data:
                    if b.get("currency_code") == currency:
                        return float(b.get("available", 0))
        except Exception as e:
            logger.debug(f"残高取得エラー: {e}")
        return 0.0

    def calculate_safe_buy_size(
        self, jpy: float, price: float, cfg: 'PairConfig', confidence: float
    ) -> Tuple[float, float]:
        """
        🛡️ 絶対安全な購入サイズを計算
        Returns: (size, estimated_cost)
        """
        # Step 1: 利用可能JPYから安全リザーブを引く
        available_jpy = jpy - JPY_RESERVE
        if available_jpy <= 0:
            return 0.0, 0.0

        # Step 2: 最大投資比率を適用（10%は常にリザーブ）
        max_budget = available_jpy * MAX_BUDGET_RATIO

        # Step 3: 信頼度に応じた投資額（30%～60%に制限）
        budget_ratio = 0.3 + (confidence * 0.3)  # 最大60%
        budget = min(max_budget, available_jpy * budget_ratio)

        # Step 4: 5%の安全マージンを適用（手数料+価格変動対策）
        safe_budget = budget * (1 - SAFETY_MARGIN_BUY)

        # Step 5: サイズ計算
        size = safe_budget / price
        size = math.floor(size * (10 ** cfg.decimals)) / (10 ** cfg.decimals)

        # Step 6: 最小サイズチェック（最小サイズの1.05倍以上必要）
        min_required = cfg.min_size * 1.05
        if size < min_required:
            # 最小サイズで購入可能かチェック
            min_cost = cfg.min_size * price * (1 + SAFETY_MARGIN_BUY)
            if min_cost <= available_jpy:
                size = cfg.min_size
            else:
                return 0.0, 0.0

        # Step 7: 最終コスト計算（5%マージン込み）
        estimated_cost = size * price * (1 + SAFETY_MARGIN_BUY)

        return size, estimated_cost

    def calculate_safe_sell_size(
        self, holding: float, cfg: 'PairConfig'
    ) -> float:
        """
        🛡️ 絶対安全な売却サイズを計算
        """
        # Step 1: 2%の安全マージンを引く
        safe_holding = holding * (1 - SAFETY_MARGIN_SELL)

        # Step 2: 小数点以下を切り捨て
        multiplier = 10 ** cfg.decimals
        size = math.floor(safe_holding * multiplier) / multiplier

        # Step 3: 最小取引量の1.01倍以上必要
        min_required = cfg.min_size * 1.01
        if size < min_required:
            return 0.0

        return size

    async def detect_balance_changes(self) -> Dict[str, Tuple[float, float, str]]:
        """
        残高変化を検出（入金・出金・手動売買対応）
        Returns: {currency: (old, new, change_type)}
        """
        changes = {}

        # 強制的に最新残高を取得
        current = await self.get_balances(force=True)

        for currency, new_amount in current.items():
            old_amount = self._last_known_balances.get(currency, 0)
            diff = new_amount - old_amount

            # 0.1%以上の変化を検出（微小な変動は無視）
            if old_amount > 0:
                pct_change = abs(diff) / old_amount
                threshold = 0.001  # 0.1%
            else:
                pct_change = 1.0 if diff > 0 else 0
                threshold = 0

            if pct_change > threshold and abs(diff) > 0.0001:
                if diff > 0:
                    change_type = "DEPOSIT" if currency == "JPY" else "RECEIVED"
                else:
                    change_type = "WITHDRAW" if currency == "JPY" else "SENT"
                changes[currency] = (old_amount, new_amount, change_type)

        # 消えた通貨もチェック（全額売却/出金）
        for currency, old_amount in self._last_known_balances.items():
            if currency not in current and old_amount > 0:
                change_type = "WITHDRAW" if currency == "JPY" else "SOLD_ALL"
                changes[currency] = (old_amount, 0, change_type)

        return changes

    async def handle_balance_changes(self, changes: Dict[str, Tuple[float, float, str]]):
        """残高変化に応じてポジションを再評価"""
        if not changes:
            return

        for currency, (old, new, change_type) in changes.items():
            diff = new - old

            if currency == "JPY":
                if change_type == "DEPOSIT":
                    logger.info(f"")
                    logger.info(f"  💰 【入金検出】¥{diff:+,.0f}")
                    logger.info(f"     残高: ¥{old:,.0f} → ¥{new:,.0f}")

                    # LINE通知
                    try:
                        await self.notifier.balance_change(change_type, currency, old, new)
                    except Exception as e:
                        logger.debug(f"  LINE通知スキップ: {e}")

                elif change_type == "WITHDRAW":
                    logger.info(f"")
                    logger.info(f"  📤 【出金検出】¥{diff:,.0f}")
                    logger.info(f"     残高: ¥{old:,.0f} → ¥{new:,.0f}")
            else:
                pair = f"{currency}_JPY"

                if change_type == "RECEIVED":
                    logger.info(f"")
                    logger.info(f"  📥 【受取検出】{currency}: {diff:+.8f}")
                    logger.info(f"     残高: {old:.8f} → {new:.8f}")

                    # 新しいポジションとして登録可能かチェック
                    await self._update_position_for_currency(pair, currency, new)

                elif change_type == "SENT" or change_type == "SOLD_ALL":
                    logger.info(f"")
                    logger.info(f"  📤 【送出検出】{currency}: {diff:.8f}")

                    # ポジションから削除
                    if pair in self.positions:
                        del self.positions[pair]
                        logger.info(f"     ポジション削除: {pair}")

            # 各通貨の処理後すぐに更新（繰り返し検出防止）
            self._last_known_balances[currency] = new

        # 最終的に全残高を同期
        current = await self.get_balances(force=True)
        self._last_known_balances = current.copy()

    async def _update_position_for_currency(self, pair: str, currency: str, amount: float):
        """通貨のポジションを更新/作成"""
        cfg = PAIRS.get(pair)
        if not cfg or pair not in self.clients:
            return

        price = await self.get_price(pair)
        if not price:
            return

        # 売却可能量計算
        multiplier = 10 ** cfg.decimals
        sellable = math.floor(amount * multiplier) / multiplier
        min_required = cfg.min_size * 1.005

        if amount >= min_required and sellable >= cfg.min_size:
            # 新しいポジションとして登録
            if pair not in self.positions:
                self.positions[pair] = Position(
                    pair=pair, size=sellable, entry_price=price,
                    entry_time=datetime.now(), highest=price, lowest=price
                )
                value = amount * price
                logger.info(f"  ✅ 新規ポジション登録: {currency} {amount:.8f} (¥{value:,.0f})")
            else:
                # 既存ポジションのサイズ更新
                self.positions[pair].size = sellable
                logger.info(f"  🔄 ポジション更新: {currency} {sellable}")
        else:
            logger.info(f"  ⚠️ {currency}: 取引不可（保有={amount:.8f}, 必要={min_required:.8f}）")

    async def refresh_all_positions(self):
        """全ポジションを再評価（入金後などに使用）"""
        logger.info("")
        logger.info("  🔄 【ポジション再評価中】")

        balances = await self.get_balances(force=True)
        jpy = balances.get("JPY", 0)
        logger.info(f"  💴 現金: ¥{jpy:,.0f}")

        total = jpy
        new_positions = {}

        for pair, cfg in PAIRS.items():
            if pair not in self.clients:
                continue

            currency = pair.replace("_JPY", "")
            amount = balances.get(currency, 0)

            price = await self.get_price(pair)
            if not price:
                continue

            value = amount * price
            total += value

            if amount > 0:
                multiplier = 10 ** cfg.decimals
                sellable = math.floor(amount * multiplier) / multiplier
                min_required = cfg.min_size * 1.005

                if amount >= min_required and sellable >= cfg.min_size:
                    # 既存のエントリー価格を保持
                    if pair in self.positions:
                        entry_price = self.positions[pair].entry_price
                        entry_time = self.positions[pair].entry_time
                    else:
                        entry_price = price
                        entry_time = datetime.now()

                    new_positions[pair] = Position(
                        pair=pair, size=sellable, entry_price=entry_price,
                        entry_time=entry_time, highest=price, lowest=price
                    )
                    logger.info(f"  💎 {currency}: {amount:.8f} (売却可能: {sellable}, ¥{value:,.0f})")
                else:
                    logger.info(f"  📌 {currency}: {amount:.8f} (¥{value:,.0f}) - 売却不可")

        self.positions = new_positions
        self._last_known_balances = balances.copy()

        logger.info(f"  📊 総資産: ¥{total:,.0f}")
        logger.info("")

        return total

    async def get_price(self, pair: str) -> Optional[float]:
        # WebSocket優先
        if self.ws and self.ws.is_connected:
            price = self.ws.get_price(pair)
            if price and price > 0:
                return price

        # REST APIフォールバック
        client = self.clients.get(pair)
        if not client:
            return None
        try:
            ticker = await asyncio.wait_for(client.get_ticker(), timeout=10)
            return ticker.ltp if ticker and ticker.ltp > 0 else None
        except:
            return None

    # =========================================================================
    # 動的パラメータ
    # =========================================================================

    def get_dynamic_params(self, pair: str) -> Tuple[float, float]:
        """ボラティリティに応じた動的利確/損切り"""
        cfg = PAIRS.get(pair)
        vol = self.volatility.get(pair, 1.0)

        # ボラティリティ調整
        vol_factor = max(0.5, min(2.0, vol))

        take_profit = BASE_TAKE_PROFIT * vol_factor * (cfg.volatility_mult if cfg else 1.0)
        stop_loss = BASE_STOP_LOSS * vol_factor * (cfg.volatility_mult if cfg else 1.0)

        # 連勝/連敗調整
        if self.win_streak >= 3:
            take_profit *= 1.2  # 調子がいい時は欲張る
        if self.loss_streak >= 2:
            stop_loss *= 0.8  # 損失時は早く切る
            take_profit *= 0.9

        return take_profit, stop_loss

    # =========================================================================
    # 🧠 AI分析 + Market Intelligence 統合
    # =========================================================================

    def analyze(self, pair: str, price: float) -> Tuple[str, float, str]:
        """
        🧠 AI統合分析 + Market Intelligence
        - テクニカル分析
        - ML予測
        - マルチペア相関
        - ボラティリティ適応
        - 時間帯パターン
        - 市場感情
        Returns: (シグナル, 信頼度, 理由)
        """
        indicators = self.indicators.get(pair)
        predictor = self.predictors.get(pair)

        if not indicators:
            return "NEUTRAL", 0.0, "データなし"

        # === 価格追加（全システムに） ===
        indicators.add_price(price)
        if predictor:
            predictor.add_price(price)
        self.market_intel.update(pair, price)

        # === 1. テクニカル分析 ===
        tech_signal, tech_conf, tech_reason = indicators.composite_signal()

        # === 2. ML予測 ===
        ml_signal = "NEUTRAL"
        ml_conf = 0.0
        ml_reason = ""

        if predictor:
            prediction = predictor.predict()
            if prediction:
                ml_signal = prediction.direction
                ml_conf = prediction.confidence
                ml_reason = f"AI予測:{prediction.predicted_change:+.2f}%"

        # === 3. Market Intelligence シグナル ===
        intel_signal = self.market_intel.get_trading_signal(pair, price)
        market_state = self.market_intel.get_market_state()

        # === 4. 統合スコア計算 ===
        # 各シグナルをスコア化
        def signal_to_score(sig: str) -> float:
            return {"BUY": 1.0, "UP": 1.0, "SELL": -1.0, "DOWN": -1.0}.get(sig, 0.0)

        tech_score = signal_to_score(tech_signal) * tech_conf * TECH_WEIGHT
        ml_score = signal_to_score(ml_signal) * ml_conf * ML_WEIGHT
        intel_score = signal_to_score(intel_signal.direction) * intel_signal.strength * 0.3

        combined_score = tech_score + ml_score + intel_score

        # === 5. 市場状態による調整 ===
        # 極端な感情時は慎重に
        if market_state.sentiment in ["EXTREME_FEAR", "EXTREME_GREED"]:
            combined_score *= 0.7

        # ボラティリティ高い時は信頼度下げる
        if market_state.volatility_regime == "EXTREME":
            combined_score *= 0.6
        elif market_state.volatility_regime == "HIGH":
            combined_score *= 0.8

        # 相関乖離時はチャンス
        if market_state.correlation_state == "DIVERGING":
            combined_score *= 1.2

        # === 6. 最終シグナル決定 ===
        reasons = []
        if tech_reason:
            reasons.append(tech_reason)
        if ml_reason:
            reasons.append(ml_reason)
        if intel_signal.reasons:
            reasons.extend(intel_signal.reasons[:2])

        combined_reason = " ".join(reasons)

        if combined_score > 0.3:
            return "BUY", min(abs(combined_score), 1.0), combined_reason
        elif combined_score < -0.3:
            return "SELL", min(abs(combined_score), 1.0), combined_reason
        else:
            return "NEUTRAL", 0.0, "シグナル弱い"

    def get_size_multiplier(self, pair: str, price: float) -> float:
        """Market Intelligenceからサイズ調整倍率を取得"""
        intel_signal = self.market_intel.get_trading_signal(pair, price)
        return intel_signal.suggested_size_mult

    # =========================================================================
    # 売買判断
    # =========================================================================

    def can_trade(self, pair: str) -> bool:
        now = datetime.now()
        if pair in self.last_trade:
            if (now - self.last_trade[pair]).seconds < TRADE_COOLDOWN:
                return False
        if pair in self.failed:
            if (now - self.failed[pair]).seconds < FAIL_COOLDOWN:
                return False
            del self.failed[pair]
        return True

    def should_buy(self, pair: str, price: float) -> Tuple[bool, float, str]:
        """
        💰 PROFIT MAXIMIZER買い判断
        - 高確信度シグナルのみ（65%以上）
        - トレンド方向と一致必須
        - オーバートレード防止
        """
        if not self.can_trade(pair):
            return False, 0.0, ""

        # オーバートレード防止（1時間あたり最大取引数）
        recent_trades = sum(
            1 for t in self.trade_history
            if (datetime.now() - t.timestamp).seconds < 3600
        )
        if recent_trades >= MAX_TRADES_PER_HOUR:
            return False, 0.0, ""

        signal, confidence, reason = self.analyze(pair, price)

        if signal != "BUY" or confidence < AI_CONFIDENCE_THRESHOLD:
            return False, 0.0, ""

        # === トレンドフィルター ===
        if REQUIRE_TREND_ALIGNMENT:
            market_state = self.market_intel.get_market_state()

            # 下落トレンド時は買わない
            if market_state.trend_direction in ["DOWN", "STRONG_DOWN"]:
                return False, 0.0, ""

            # 極度の恐怖時も買わない（パニック売りに巻き込まれる）
            if market_state.sentiment == "EXTREME_FEAR":
                return False, 0.0, ""

            # ボラティリティが極端に高い時は買わない
            if market_state.volatility_regime == "EXTREME":
                return False, 0.0, ""

            # トレンド方向と一致で信頼度ボーナス
            if market_state.trend_direction in ["UP", "STRONG_UP"]:
                confidence = min(confidence * 1.1, 1.0)
                reason += " トレンド↑"

        return True, confidence, reason

    def should_sell(self, pair: str, price: float) -> Tuple[bool, str]:
        """
        💰 PROFIT MAXIMIZER売り判断
        - 損は早く切る（0.3%）
        - 利益は大きく伸ばす（1.5%以上）
        - トレーリングストップで利益保護
        """
        pos = self.positions.get(pair)
        if not pos:
            return False, ""

        pnl = pos.pnl_pct(price)
        hold_time = pos.hold_seconds()
        take_profit, stop_loss = self.get_dynamic_params(pair)

        # 高値/安値更新
        if price > pos.highest:
            pos.highest = price
        if price < pos.lowest:
            pos.lowest = price

        # ==========================================
        # 💰 損切りルール（早く切る = 損失最小化）
        # ==========================================

        # 緊急損切り (-1.5%) - 絶対防衛ライン
        if pnl <= -0.015:
            return True, f"🚨緊急損切り {pnl*100:.2f}%"

        # 即時損切り（MIN_HOLD_TIME待たない）- 損は早く切る
        if pnl <= -stop_loss:
            return True, f"⚡損切り {pnl*100:.2f}%"

        # 塩漬け防止（30分以上保持で微損でも切る）
        if hold_time >= MAX_HOLD_TIME and pnl < 0.005:
            return True, f"⏰タイムアウト {pnl*100:.2f}%"

        # ==========================================
        # 💰 利確ルール（利益を伸ばす）
        # ==========================================

        # 大勝利確定 (+2.5%以上) - 確実に取る
        if pnl >= 0.025:
            return True, f"🎉大勝利確 +{pnl*100:.2f}%"

        # 通常利確（MIN_HOLD_TIME後）
        if hold_time >= MIN_HOLD_TIME and pnl >= take_profit:
            return True, f"💰利確 +{pnl*100:.2f}%"

        # ==========================================
        # 💰 トレーリングストップ（利益保護）
        # ==========================================

        # 高値から0.4%以上下落したら利確（利益が出ている場合のみ）
        if pos.highest > pos.entry_price:
            # 最高値からの下落率
            drop_from_high = (pos.highest - price) / pos.highest

            # 現在の含み益
            profit_from_entry = (pos.highest - pos.entry_price) / pos.entry_price

            # 含み益が0.8%以上あり、0.4%以上下落したら利確
            if profit_from_entry >= 0.008 and drop_from_high >= TRAILING_STOP:
                return True, f"📈トレール +{pnl*100:.2f}% (高値から-{drop_from_high*100:.1f}%)"

            # 含み益が1.2%以上あり、0.3%以上下落したら利確（より敏感）
            if profit_from_entry >= 0.012 and drop_from_high >= 0.003:
                return True, f"📈トレール +{pnl*100:.2f}%"

        # ==========================================
        # 💰 AI売りシグナル（確信度高い場合のみ）
        # ==========================================

        if hold_time >= MIN_HOLD_TIME and pnl > 0.003:
            signal, confidence, reason = self.analyze(pair, price)
            # 70%以上の確信度で売りシグナル
            if signal == "SELL" and confidence >= 0.70:
                return True, f"🤖AI売り {reason}"

        return False, ""

    # =========================================================================
    # 🛡️ BULLETPROOF ORDER SYSTEM - 絶対失敗しない注文実行
    # =========================================================================

    async def execute_buy(self, pair: str, confidence: float, reason: str) -> bool:
        """
        🛡️ BULLETPROOF購入 - Insufficient fundsエラー完全防止
        - トリプル残高チェック
        - 5%安全マージン
        - 失敗時自動リトライ（サイズ縮小）
        """
        cfg = PAIRS.get(pair)
        client = self.clients.get(pair)
        if not cfg or not client:
            return False

        currency = pair.replace("_JPY", "")

        # === STEP 1: 初回残高チェック ===
        balances = await self.get_balances(force=True)
        jpy = balances.get("JPY", 0)
        if jpy < JPY_RESERVE + 50:
            logger.debug(f"  {currency}: 残高不足 ¥{jpy:,.0f}")
            return False

        price = await self.get_price(pair)
        if not price:
            return False

        # === STEP 2: 安全サイズ計算（5%マージン込み） ===
        size, estimated_cost = self.calculate_safe_buy_size(jpy, price, cfg, confidence)
        if size <= 0:
            logger.debug(f"  {currency}: 安全サイズ計算失敗")
            return False

        # === STEP 2.5: Market Intelligence サイズ調整 ===
        size_mult = self.get_size_multiplier(pair, price)
        if size_mult < 1.0:
            size = size * size_mult
            size = math.floor(size * (10 ** cfg.decimals)) / (10 ** cfg.decimals)
            if size < cfg.min_size:
                logger.debug(f"  {currency}: MI調整後サイズ不足")
                return False

        # === STEP 3: リトライループ（最大3回） ===
        for attempt in range(MAX_ORDER_RETRIES):
            # 毎回最新の残高を取得（キャッシュ無視）
            fresh_jpy = await self.get_fresh_balance("JPY")

            # 最終コスト確認（5%マージン込み）
            final_cost = size * price * (1 + SAFETY_MARGIN_BUY)
            available = fresh_jpy - JPY_RESERVE

            if final_cost > available:
                # 残高不足の場合、サイズを縮小
                if attempt < MAX_ORDER_RETRIES - 1:
                    size = size * RETRY_SIZE_REDUCTION
                    size = math.floor(size * (10 ** cfg.decimals)) / (10 ** cfg.decimals)
                    if size < cfg.min_size:
                        logger.info(f"  ⚠️ {currency}: サイズ縮小限界に達しました")
                        return False
                    logger.info(f"  🔄 {currency}: サイズ縮小してリトライ ({attempt+1}/{MAX_ORDER_RETRIES})")
                    continue
                else:
                    logger.info(f"  ⚠️ {currency}: 残高不足（必要: ¥{final_cost:,.0f}, 利用可能: ¥{available:,.0f}）")
                    return False

            # ログ出力
            if attempt == 0:
                logger.info(f"")
                logger.info(f"  🛒 【購入】{currency}")
                logger.info(f"     数量: {size} @ ¥{price:,.0f}")
                logger.info(f"     AI信頼度: {confidence*100:.0f}%")
                logger.info(f"     安全コスト: ¥{final_cost:,.0f} (5%マージン込み)")
                logger.info(f"     利用可能残高: ¥{fresh_jpy:,.0f}")
                logger.info(f"     理由: {reason}")

            try:
                order_id = await asyncio.wait_for(
                    client.send_order(OrderSide.BUY, size, OrderType.MARKET),
                    timeout=20
                )
                if order_id:
                    self.last_trade[pair] = datetime.now()
                    self.trades += 1
                    actual_cost = size * price
                    self.positions[pair] = Position(
                        pair=pair, size=size, entry_price=price,
                        entry_time=datetime.now(), highest=price, lowest=price,
                        ai_confidence=confidence, entry_reason=reason
                    )
                    self._balances = {}

                    logger.info(f"  ✅ 購入成功！ 投資額: ¥{actual_cost:,.0f}")

                    await self.notifier.on_trade(TradeNotification(
                        action="BUY", pair=pair, size=size, price=price, reason=reason
                    ))

                    return True
                else:
                    # 注文失敗 - サイズを縮小してリトライ
                    if attempt < MAX_ORDER_RETRIES - 1:
                        size = size * RETRY_SIZE_REDUCTION
                        size = math.floor(size * (10 ** cfg.decimals)) / (10 ** cfg.decimals)
                        if size < cfg.min_size:
                            logger.info(f"  ⚠️ {currency}: サイズ縮小限界")
                            self.failed[pair] = datetime.now()
                            return False
                        logger.info(f"  🔄 {currency}: 注文失敗、サイズ縮小してリトライ ({attempt+1}/{MAX_ORDER_RETRIES})")
                        await asyncio.sleep(1)
                    else:
                        self.failed[pair] = datetime.now()
                        return False

            except Exception as e:
                if attempt < MAX_ORDER_RETRIES - 1:
                    size = size * RETRY_SIZE_REDUCTION
                    size = math.floor(size * (10 ** cfg.decimals)) / (10 ** cfg.decimals)
                    if size >= cfg.min_size:
                        logger.info(f"  🔄 {currency}: エラー発生、リトライ ({attempt+1}/{MAX_ORDER_RETRIES}): {e}")
                        await asyncio.sleep(1)
                        continue
                self.failed[pair] = datetime.now()
                logger.debug(f"  購入エラー: {e}")
                return False

        return False

    async def execute_sell(self, pair: str, reason: str) -> bool:
        """
        🛡️ BULLETPROOF売却 - Insufficient fundsエラー完全防止
        - 2%安全マージン
        - 失敗時自動リトライ（サイズ縮小）
        """
        cfg = PAIRS.get(pair)
        client = self.clients.get(pair)
        pos = self.positions.get(pair)
        if not cfg or not client:
            return False

        currency = pair.replace("_JPY", "")

        # 未決済キャンセル
        try:
            await client.cancel_all_orders(pair)
            await asyncio.sleep(0.3)
        except:
            pass

        # === STEP 1: 最新残高を取得 ===
        holding = await self.get_fresh_balance(currency)
        if holding <= 0:
            if pair in self.positions:
                del self.positions[pair]
            return False

        # === STEP 2: 安全サイズ計算（2%マージン込み） ===
        size = self.calculate_safe_sell_size(holding, cfg)
        if size <= 0:
            logger.info(f"  ⚠️ {currency}: 売却スキップ（保有={holding:.8f}, 安全サイズ計算失敗）")
            if pair in self.positions:
                del self.positions[pair]
            return False

        price = await self.get_price(pair)
        if not price:
            return False

        logger.info(f"")
        logger.info(f"  💰 【売却】{currency}")
        logger.info(f"     保有量: {holding:.8f}")
        logger.info(f"     売却サイズ: {size} (2%マージン適用)")
        logger.info(f"     価格: ¥{price:,.0f}")
        logger.info(f"     理由: {reason}")

        # === STEP 3: リトライループ（最大3回） ===
        for attempt in range(MAX_ORDER_RETRIES):
            # 毎回最新の残高を取得（キャッシュ無視）
            fresh_holding = await self.get_fresh_balance(currency)

            if size > fresh_holding:
                # 残高不足の場合、サイズを再計算
                size = self.calculate_safe_sell_size(fresh_holding, cfg)
                if size <= 0:
                    if attempt < MAX_ORDER_RETRIES - 1:
                        logger.info(f"  🔄 {currency}: 残高変動、リトライ ({attempt+1}/{MAX_ORDER_RETRIES})")
                        await asyncio.sleep(1)
                        continue
                    else:
                        logger.info(f"  ⚠️ {currency}: 売却可能量不足")
                        if pair in self.positions:
                            del self.positions[pair]
                        return False

            try:
                order_id = await asyncio.wait_for(
                    client.send_order(OrderSide.SELL, size, OrderType.MARKET),
                    timeout=20
                )
                if order_id:
                    self.last_trade[pair] = datetime.now()
                    self.trades += 1

                    # 損益計算
                    entry = pos.entry_price if pos else price
                    gross = (price - entry) * size
                    fee = price * size * TRADING_FEE * 2
                    net = gross - fee
                    pct = (price - entry) / entry * 100 if entry > 0 else 0

                    self.pnl += net

                    # 勝敗記録
                    if net > 0:
                        self.wins += 1
                        self.win_streak += 1
                        self.loss_streak = 0
                        emoji = "💰"
                    else:
                        self.loss_streak += 1
                        self.win_streak = 0
                        emoji = "📉"

                    # 履歴追加
                    self.trade_history.append(TradeRecord(
                        pair=pair, action="SELL", size=size, price=price,
                        pnl=net, pnl_pct=pct, reason=reason, timestamp=datetime.now()
                    ))

                    # 🧠 Market Intelligence 学習記録
                    hold_time = pos.hold_seconds() if pos else 0
                    self.market_intel.record_trade_result(pair, net, pct, hold_time)

                    if pair in self.positions:
                        del self.positions[pair]
                    self._balances = {}

                    logger.info(f"  {emoji} 売却完了！")
                    logger.info(f"     損益: ¥{net:+,.0f} ({pct:+.2f}%)")
                    logger.info(f"     累計: ¥{self.pnl:+,.0f}")

                    await self.notifier.on_trade(TradeNotification(
                        action="SELL", pair=pair, size=size, price=price,
                        pnl=net, pnl_pct=pct, reason=reason
                    ))

                    await self._check_milestones()

                    return True
                else:
                    # 注文失敗 - サイズを縮小してリトライ
                    if attempt < MAX_ORDER_RETRIES - 1:
                        size = size * RETRY_SIZE_REDUCTION
                        size = math.floor(size * (10 ** cfg.decimals)) / (10 ** cfg.decimals)
                        if size < cfg.min_size:
                            logger.info(f"  ⚠️ {currency}: サイズ縮小限界")
                            self.failed[pair] = datetime.now()
                            return False
                        logger.info(f"  🔄 {currency}: 注文失敗、サイズ縮小してリトライ ({attempt+1}/{MAX_ORDER_RETRIES})")
                        await asyncio.sleep(1)
                    else:
                        self.failed[pair] = datetime.now()
                        return False

            except Exception as e:
                if attempt < MAX_ORDER_RETRIES - 1:
                    size = size * RETRY_SIZE_REDUCTION
                    size = math.floor(size * (10 ** cfg.decimals)) / (10 ** cfg.decimals)
                    if size >= cfg.min_size:
                        logger.info(f"  🔄 {currency}: エラー発生、リトライ ({attempt+1}/{MAX_ORDER_RETRIES}): {e}")
                        await asyncio.sleep(1)
                        continue
                self.failed[pair] = datetime.now()
                logger.debug(f"  売却エラー: {e}")
                return False

        return False

    async def _check_milestones(self):
        """マイルストーンチェック"""
        balances = await self.get_balances(force=True)
        total = balances.get("JPY", 0)

        for pair in self.clients:
            currency = pair.replace("_JPY", "")
            amount = balances.get(currency, 0)
            if amount > 0:
                price = await self.get_price(pair)
                if price:
                    total += amount * price

        milestones = [
            (5000, "¥5,000達成！ETH取引可能に！"),
            (7500, "¥7,500達成！順調に成長中！"),
            (10000, "¥10,000達成！2倍到達！"),
            (15000, "¥15,000達成！目標達成！"),
            (20000, "¥20,000達成！4倍到達！"),
        ]

        for threshold, message in milestones:
            if self.start_value < threshold <= total:
                await self.notifier.notifier.notify_milestone(message, total, self.pnl)

    # =========================================================================
    # 初期化
    # =========================================================================

    async def init(self) -> bool:
        api_key = self.config.bitflyer.api_key
        api_secret = self.config.bitflyer.api_secret

        if not api_key or not api_secret:
            logger.error("❌ APIキー未設定")
            return False

        logger.info("")
        logger.info("╔══════════════════════════════════════════════════════════════╗")
        logger.info("║  🏆 ULTIMATE AI TRADER v13.0 - 世界最強・高利益トレーダー   ║")
        logger.info("╠══════════════════════════════════════════════════════════════╣")
        logger.info("║  💰 PROFIT MAXIMIZER - 損小利大(RR比1:2.5) トレンドフォロー ║")
        logger.info("║  🛡️ BULLETPROOF ORDER - 残高不足エラー完全防止              ║")
        logger.info("║  🧠 MARKET INTELLIGENCE - 相関/ボラ適応/時間帯/感情分析     ║")
        logger.info("╚══════════════════════════════════════════════════════════════╝")
        logger.info("")
        logger.info("  💰 利確: 1.5% | 損切: 0.3% | 必要勝率: 33%")

        active_pairs = []

        for pair, cfg in sorted(PAIRS.items(), key=lambda x: x[1].priority):
            try:
                client = BitFlyerClient(api_key, api_secret, pair)
                ticker = await asyncio.wait_for(client.get_ticker(), timeout=10)
                if ticker and ticker.ltp > 0:
                    self.clients[pair] = client
                    self.indicators[pair] = TechnicalIndicators()
                    self.predictors[pair] = EnsemblePredictor()
                    self.volatility[pair] = 1.0
                    active_pairs.append(pair)

                    min_jpy = cfg.min_size * ticker.ltp
                    logger.info(f"  ✓ {pair}: ¥{ticker.ltp:,.0f} (最小: ¥{min_jpy:,.0f})")
            except:
                pass

        if not self.clients:
            return False

        # WebSocket開始
        if self.use_websocket and WS_AVAILABLE:
            try:
                self.ws = MultiPairWebSocket(active_pairs)
                if await self.ws.start():
                    logger.info("  ⚡ WebSocket: 有効")
                else:
                    self.ws = None
                    logger.info("  ⚡ WebSocket: 無効 (REST API使用)")
            except:
                self.ws = None

        # 未決済キャンセル
        for pair in self.clients:
            try:
                await self.clients[pair].cancel_all_orders(pair)
            except:
                pass
        await asyncio.sleep(0.5)

        # 残高確認（APIから正確な情報を取得）
        logger.info("")
        balances = await self.get_balances(force=True)
        jpy = balances.get("JPY", 0)
        logger.info(f"  💴 現金: ¥{jpy:,.0f}")

        # 全残高を表示（API取得値）
        logger.info(f"  📋 API残高: {balances}")

        total = jpy

        # 各ペアの保有量をチェック
        for pair in self.clients:
            currency = pair.replace("_JPY", "")
            amount = balances.get(currency, 0)
            cfg = PAIRS.get(pair)

            # APIから現在価格を取得
            price = await self.get_price(pair)
            if not price or price <= 0:
                logger.debug(f"  {currency}: 価格取得失敗")
                continue

            value = amount * price
            total += value

            if amount > 0:
                # 売却可能量計算
                multiplier = 10 ** cfg.decimals
                sellable = math.floor(amount * multiplier) / multiplier

                # 最小取引量+0.5%の余裕が必要（API制限対策）
                min_required = cfg.min_size * 1.005

                logger.info(f"  📊 {currency}: 保有={amount:.8f}, 売却可能={sellable}, 最小={cfg.min_size}, 必要={min_required:.8f}")

                # 十分な余裕がある場合のみポジションとして登録
                if amount >= min_required and sellable >= cfg.min_size:
                    self.positions[pair] = Position(
                        pair=pair, size=sellable, entry_price=price,
                        entry_time=datetime.now(), highest=price, lowest=price
                    )
                    logger.info(f"  💎 {currency}: {amount} (売却可能: {sellable}, ¥{value:,.0f})")
                else:
                    logger.info(f"  📌 {currency}: {amount} (¥{value:,.0f}) - 売却不可（余裕不足）")

        self.start_value = total
        self._last_known_balances = balances.copy()
        self._last_balance_check = datetime.now()

        logger.info(f"  📊 総資産: ¥{total:,.0f}")
        logger.info("")

        # LINE通知
        await self.notifier.start(total, active_pairs)

        return True

    # =========================================================================
    # ステータス表示
    # =========================================================================

    def show_status(self):
        logger.info("")
        logger.info("  ══════════════════════════════════════════════════════════════")
        logger.info("  📊 【AI戦況】")

        # 🧠 Market Intelligence 状態表示
        try:
            state = self.market_intel.get_market_state()
            fgi = self.market_intel.sentiment.get_fear_greed_index()

            vol_emoji = {"LOW": "😴", "NORMAL": "📊", "HIGH": "⚡", "EXTREME": "🔥"}.get(state.volatility_regime, "📊")
            sent_emoji = {"EXTREME_FEAR": "😱", "FEAR": "😰", "NEUTRAL": "😐", "GREED": "🤑", "EXTREME_GREED": "🚀"}.get(state.sentiment, "😐")

            logger.info(f"  🧠 市場: {vol_emoji}{state.volatility_regime} | {sent_emoji}FGI:{fgi:.0f} | {state.time_regime}")
        except:
            pass

        logger.info("  ──────────────────────────────────────────────────────────────")

        unrealized = 0.0

        for pair, pos in self.positions.items():
            indicators = self.indicators.get(pair)
            if not indicators:
                continue

            price = indicators.current_price()
            if not price:
                continue

            currency = pair.replace("_JPY", "")
            pnl_pct = pos.pnl_pct(price) * 100
            pnl_jpy = (price - pos.entry_price) * pos.size
            unrealized += pnl_jpy
            hold = pos.hold_seconds()

            # AI分析
            signal, confidence, _ = self.analyze(pair, price)
            ai_str = f"AI:{signal[0]}{confidence*100:.0f}%"

            emoji = "📈" if pnl_pct >= 0 else "📉"
            logger.info(f"  {emoji} {currency}: {pnl_pct:+.2f}% (¥{pnl_jpy:+,.0f}) {hold}秒 {ai_str}")

        logger.info("  ──────────────────────────────────────────────────────────────")

        win_rate = (self.wins / self.trades * 100) if self.trades > 0 else 0
        logger.info(f"  💰 含み: ¥{unrealized:+,.0f} | 確定: ¥{self.pnl:+,.0f}")
        logger.info(f"  📊 取引: {self.trades}回 (勝率: {win_rate:.0f}%)")

        # 連勝/連敗表示
        if self.win_streak >= 2:
            logger.info(f"  🔥 {self.win_streak}連勝中！")
        if self.loss_streak >= 2:
            logger.info(f"  ⚠️ {self.loss_streak}連敗中...")

        jpy = self._balances.get("JPY", 0)
        if jpy >= 5000:
            logger.info(f"  🎉 ETH取引可能！")

        logger.info("  ══════════════════════════════════════════════════════════════")

    # =========================================================================
    # メインループ
    # =========================================================================

    async def run(self):
        if not await self.init():
            return

        self.start_time = datetime.now()
        logger.info("  🚀 世界最強AIトレーダー起動！ Ctrl+C で停止")
        logger.info("")

        tick = 0
        last_daily_report = datetime.now().date()

        while True:
            try:
                tick += 1

                # === 定期的な残高変化チェック（入金・出金・手動売買検出） ===
                now = datetime.now()
                if (self._last_balance_check is None or
                    (now - self._last_balance_check).seconds >= self._balance_check_interval):

                    changes = await self.detect_balance_changes()
                    if changes:
                        await self.handle_balance_changes(changes)
                        # 大きな変化があれば全ポジション再評価
                        jpy_change = changes.get("JPY", (0, 0, ""))[1] - changes.get("JPY", (0, 0, ""))[0]
                        if abs(jpy_change) >= 1000:  # ¥1,000以上の変化
                            await self.refresh_all_positions()

                    self._last_balance_check = now

                balances = await self.get_balances()
                jpy = balances.get("JPY", 0)

                for pair, cfg in sorted(PAIRS.items(), key=lambda x: x[1].priority):
                    if pair not in self.clients:
                        continue

                    price = await self.get_price(pair)
                    if not price:
                        await asyncio.sleep(PRICE_DELAY)
                        continue

                    # ボラティリティ更新
                    indicators = self.indicators[pair]
                    vol = indicators.volatility()
                    if vol:
                        self.volatility[pair] = vol

                    has_position = pair in self.positions

                    if has_position:
                        should, reason = self.should_sell(pair, price)
                        if should:
                            await self.execute_sell(pair, reason)
                    else:
                        min_cost = cfg.min_size * price * 1.01
                        if jpy - 50 >= min_cost:
                            should, confidence, reason = self.should_buy(pair, price)
                            if should:
                                await self.execute_buy(pair, confidence, reason)

                    await asyncio.sleep(PRICE_DELAY)

                # ステータス表示
                if tick % STATUS_INTERVAL == 0:
                    self.show_status()

                # 日次レポート
                today = datetime.now().date()
                if today != last_daily_report and datetime.now().hour >= 0:
                    await self._send_daily_report()
                    last_daily_report = today

                await asyncio.sleep(LOOP_DELAY)

            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"  エラー: {e}")
                await asyncio.sleep(10)

        # 終了処理
        logger.info("")
        logger.info("  🛑 停止")
        self.show_status()

        await self.notifier.stop(self.pnl, self.trades)

        if self.ws:
            await self.ws.stop()

    async def _send_daily_report(self):
        """日次レポート送信"""
        today_trades = [t for t in self.trade_history
                        if t.timestamp.date() == datetime.now().date()]

        if not today_trades:
            return

        wins = sum(1 for t in today_trades if t.pnl > 0)
        losses = len(today_trades) - wins
        total_pnl = sum(t.pnl for t in today_trades)

        best = max((t.pnl for t in today_trades), default=None)
        worst = min((t.pnl for t in today_trades), default=None)

        balances = await self.get_balances(force=True)
        end_balance = balances.get("JPY", 0)
        for pair in self.clients:
            currency = pair.replace("_JPY", "")
            amount = balances.get(currency, 0)
            price = await self.get_price(pair)
            if amount > 0 and price:
                end_balance += amount * price

        report = DailyReport(
            total_trades=len(today_trades),
            wins=wins,
            losses=losses,
            total_pnl=total_pnl,
            best_trade=best,
            worst_trade=worst,
            start_balance=self.start_value,
            end_balance=end_balance
        )

        await self.notifier.daily_report(report)


# =============================================================================
# エントリーポイント
# =============================================================================

async def main():
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | {message}",
        level="INFO",
        filter=lambda r: "Cancel all failed" not in r["message"]
                     and "Request error: 200" not in r["message"]
    )

    import logging
    logging.getLogger("src.api.bitflyer_client").setLevel(logging.CRITICAL)
    logging.getLogger("websockets").setLevel(logging.WARNING)

    trader = UltimateTrader()
    await trader.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass

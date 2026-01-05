#!/usr/bin/env python3
"""
================================================================================
    🧠 MARKET INTELLIGENCE SYSTEM - 世界最強の市場分析エンジン
================================================================================
    - マルチペア相関分析（BTC/ETH/XRP連動性）
    - ボラティリティ適応型戦略
    - 時間帯別パターン学習
    - 市場感情分析（急変検知）
    - 高速オンライン学習
================================================================================
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
from datetime import datetime, timedelta
import json
from pathlib import Path


# =============================================================================
# データ構造
# =============================================================================

@dataclass
class MarketState:
    """市場状態"""
    volatility_regime: str  # "LOW", "NORMAL", "HIGH", "EXTREME"
    trend_direction: str    # "STRONG_UP", "UP", "NEUTRAL", "DOWN", "STRONG_DOWN"
    correlation_state: str  # "SYNCED", "DIVERGING", "DECORRELATED"
    sentiment: str          # "FEAR", "NEUTRAL", "GREED", "EXTREME_FEAR", "EXTREME_GREED"
    time_regime: str        # "TOKYO", "LONDON", "NY", "QUIET"
    confidence: float       # 0.0 - 1.0


@dataclass
class TradingSignal:
    """取引シグナル"""
    pair: str
    direction: str          # "BUY", "SELL", "HOLD"
    strength: float         # 0.0 - 1.0
    reasons: List[str]
    risk_level: str         # "LOW", "MEDIUM", "HIGH"
    suggested_size_mult: float  # サイズ調整倍率 (0.5 - 2.0)


# =============================================================================
# マルチペア相関分析
# =============================================================================

class CorrelationAnalyzer:
    """マルチペア相関分析器"""

    def __init__(self, window: int = 60):
        self.window = window
        self.prices: Dict[str, deque] = {}
        self.correlations: Dict[str, float] = {}
        self._last_update = None

    def add_price(self, pair: str, price: float):
        """価格追加"""
        if pair not in self.prices:
            self.prices[pair] = deque(maxlen=self.window)
        self.prices[pair].append(price)

    def calculate_returns(self, pair: str) -> Optional[np.ndarray]:
        """リターン計算"""
        if pair not in self.prices or len(self.prices[pair]) < 10:
            return None
        prices = np.array(list(self.prices[pair]))
        returns = np.diff(prices) / prices[:-1]
        return returns

    def calculate_correlation(self, pair1: str, pair2: str) -> Optional[float]:
        """2ペア間の相関係数"""
        r1 = self.calculate_returns(pair1)
        r2 = self.calculate_returns(pair2)

        if r1 is None or r2 is None:
            return None

        min_len = min(len(r1), len(r2))
        if min_len < 5:
            return None

        r1 = r1[-min_len:]
        r2 = r2[-min_len:]

        # ピアソン相関係数
        corr = np.corrcoef(r1, r2)[0, 1]
        return corr if not np.isnan(corr) else 0.0

    def get_btc_correlation(self, pair: str) -> float:
        """BTC_JPYとの相関"""
        corr = self.calculate_correlation("BTC_JPY", pair)
        return corr if corr is not None else 0.5

    def get_correlation_matrix(self) -> Dict[str, Dict[str, float]]:
        """全ペア間の相関行列"""
        pairs = list(self.prices.keys())
        matrix = {}

        for p1 in pairs:
            matrix[p1] = {}
            for p2 in pairs:
                if p1 == p2:
                    matrix[p1][p2] = 1.0
                else:
                    corr = self.calculate_correlation(p1, p2)
                    matrix[p1][p2] = corr if corr is not None else 0.0

        return matrix

    def detect_divergence(self) -> List[Tuple[str, str, float]]:
        """
        相関乖離検出
        通常高相関のペアが乖離した場合 = 取引機会
        """
        divergences = []
        pairs = list(self.prices.keys())

        for i, p1 in enumerate(pairs):
            for p2 in pairs[i+1:]:
                corr = self.calculate_correlation(p1, p2)
                if corr is None:
                    continue

                # BTCとアルトの相関が通常0.6以上なのに0.3以下に低下
                if "BTC" in p1 or "BTC" in p2:
                    if corr < 0.3:
                        divergences.append((p1, p2, corr))

        return divergences

    def get_leader_follower(self) -> Optional[Tuple[str, str]]:
        """
        リーダー/フォロワー関係を検出
        （BTCが動いた後にアルトが追従するパターン）
        """
        if "BTC_JPY" not in self.prices or len(self.prices["BTC_JPY"]) < 30:
            return None

        btc_prices = list(self.prices["BTC_JPY"])
        btc_change = (btc_prices[-1] - btc_prices[-10]) / btc_prices[-10]

        # BTCが大きく動いた場合
        if abs(btc_change) > 0.005:  # 0.5%以上
            return ("BTC_JPY", "LEADER" if btc_change > 0 else "LEADER_DOWN")

        return None


# =============================================================================
# ボラティリティ分析
# =============================================================================

class VolatilityAnalyzer:
    """ボラティリティ分析器"""

    def __init__(self, window: int = 100):
        self.window = window
        self.prices: Dict[str, deque] = {}
        self.volatility_history: Dict[str, deque] = {}

        # ボラティリティ閾値
        self.thresholds = {
            "LOW": 0.002,      # 0.2%以下
            "NORMAL": 0.005,   # 0.5%以下
            "HIGH": 0.01,      # 1%以下
            "EXTREME": 0.02    # それ以上
        }

    def add_price(self, pair: str, price: float):
        """価格追加"""
        if pair not in self.prices:
            self.prices[pair] = deque(maxlen=self.window)
            self.volatility_history[pair] = deque(maxlen=50)
        self.prices[pair].append(price)

        # ボラティリティ計算して履歴に追加
        vol = self.calculate_volatility(pair)
        if vol is not None:
            self.volatility_history[pair].append(vol)

    def calculate_volatility(self, pair: str, period: int = 20) -> Optional[float]:
        """ボラティリティ計算（標準偏差/平均）"""
        if pair not in self.prices or len(self.prices[pair]) < period:
            return None

        prices = list(self.prices[pair])[-period:]
        returns = np.diff(prices) / np.array(prices[:-1])

        return float(np.std(returns))

    def get_regime(self, pair: str) -> str:
        """ボラティリティレジーム判定"""
        vol = self.calculate_volatility(pair)
        if vol is None:
            return "NORMAL"

        if vol < self.thresholds["LOW"]:
            return "LOW"
        elif vol < self.thresholds["NORMAL"]:
            return "NORMAL"
        elif vol < self.thresholds["HIGH"]:
            return "HIGH"
        else:
            return "EXTREME"

    def get_volatility_trend(self, pair: str) -> str:
        """ボラティリティのトレンド"""
        if pair not in self.volatility_history or len(self.volatility_history[pair]) < 10:
            return "STABLE"

        history = list(self.volatility_history[pair])
        recent = np.mean(history[-5:])
        older = np.mean(history[-10:-5])

        if recent > older * 1.3:
            return "INCREASING"
        elif recent < older * 0.7:
            return "DECREASING"
        else:
            return "STABLE"

    def get_recommended_params(self, pair: str) -> Dict[str, float]:
        """
        ボラティリティに基づく推奨パラメータ
        """
        regime = self.get_regime(pair)
        trend = self.get_volatility_trend(pair)

        params = {
            "take_profit_mult": 1.0,
            "stop_loss_mult": 1.0,
            "size_mult": 1.0,
            "hold_time_mult": 1.0
        }

        # レジーム別調整
        if regime == "LOW":
            # 低ボラティリティ = 小さな利確、長く持つ
            params["take_profit_mult"] = 0.7
            params["stop_loss_mult"] = 0.8
            params["size_mult"] = 1.2  # サイズ大きめ
            params["hold_time_mult"] = 1.5

        elif regime == "HIGH":
            # 高ボラティリティ = 大きな利確、早く切る
            params["take_profit_mult"] = 1.5
            params["stop_loss_mult"] = 1.2
            params["size_mult"] = 0.7  # サイズ小さめ
            params["hold_time_mult"] = 0.7

        elif regime == "EXTREME":
            # 極端なボラティリティ = 取引控えめ
            params["take_profit_mult"] = 2.0
            params["stop_loss_mult"] = 0.6  # 早期損切り
            params["size_mult"] = 0.5
            params["hold_time_mult"] = 0.5

        # トレンド調整
        if trend == "INCREASING":
            params["size_mult"] *= 0.8  # ボラ上昇中はサイズ縮小

        return params


# =============================================================================
# 時間帯別パターン学習
# =============================================================================

class TimePatternLearner:
    """時間帯別パターン学習"""

    def __init__(self):
        # 1時間ごとのパフォーマンス記録
        self.hourly_stats: Dict[int, Dict[str, List[float]]] = {
            h: {"returns": [], "volatility": [], "success_rate": []}
            for h in range(24)
        }

        # 取引結果記録
        self.trade_results: List[Dict] = []

        # 時間帯定義（日本時間）
        self.time_regimes = {
            "TOKYO": (9, 15),      # 東京市場
            "LONDON": (16, 21),    # ロンドン市場
            "NY": (22, 5),         # NY市場
            "QUIET": (5, 9)        # 閑散時間
        }

        # 学習済みパターン
        self.learned_patterns: Dict[int, Dict] = {}

        self._load_patterns()

    def get_current_regime(self) -> str:
        """現在の時間帯レジーム"""
        hour = datetime.now().hour

        if 9 <= hour < 15:
            return "TOKYO"
        elif 16 <= hour < 21:
            return "LONDON"
        elif hour >= 22 or hour < 5:
            return "NY"
        else:
            return "QUIET"

    def record_price(self, pair: str, price: float, returns: float):
        """価格データ記録"""
        hour = datetime.now().hour
        self.hourly_stats[hour]["returns"].append(returns)

        # 最大1000件保持
        if len(self.hourly_stats[hour]["returns"]) > 1000:
            self.hourly_stats[hour]["returns"] = self.hourly_stats[hour]["returns"][-1000:]

    def record_trade(self, pair: str, pnl: float, pnl_pct: float, hold_time: int):
        """取引結果記録"""
        hour = datetime.now().hour
        self.trade_results.append({
            "hour": hour,
            "pair": pair,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "hold_time": hold_time,
            "timestamp": datetime.now().isoformat()
        })

        # 成功率更新
        success = 1.0 if pnl > 0 else 0.0
        self.hourly_stats[hour]["success_rate"].append(success)

        # 最大500件保持
        if len(self.trade_results) > 500:
            self.trade_results = self.trade_results[-500:]

        # パターン再学習
        self._learn_patterns()
        self._save_patterns()

    def _learn_patterns(self):
        """時間帯パターン学習"""
        for hour in range(24):
            stats = self.hourly_stats[hour]

            if len(stats["returns"]) < 20:
                continue

            returns = np.array(stats["returns"][-100:])
            success_rates = stats["success_rate"][-50:] if stats["success_rate"] else [0.5]

            self.learned_patterns[hour] = {
                "avg_return": float(np.mean(returns)),
                "volatility": float(np.std(returns)),
                "success_rate": float(np.mean(success_rates)),
                "sample_count": len(returns),
                "is_favorable": np.mean(success_rates) > 0.55 if success_rates else False
            }

    def get_hour_recommendation(self) -> Dict:
        """現在時刻の推奨設定"""
        hour = datetime.now().hour
        regime = self.get_current_regime()

        if hour in self.learned_patterns:
            pattern = self.learned_patterns[hour]

            # 成功率に基づく推奨
            if pattern["success_rate"] > 0.6:
                trade_mult = 1.3  # 取引積極的
                size_mult = 1.2
            elif pattern["success_rate"] < 0.4:
                trade_mult = 0.5  # 取引控えめ
                size_mult = 0.7
            else:
                trade_mult = 1.0
                size_mult = 1.0

            return {
                "regime": regime,
                "hour": hour,
                "trade_multiplier": trade_mult,
                "size_multiplier": size_mult,
                "is_favorable": pattern["is_favorable"],
                "confidence": min(pattern["sample_count"] / 50, 1.0)
            }

        # デフォルト（学習データなし）
        return {
            "regime": regime,
            "hour": hour,
            "trade_multiplier": 1.0,
            "size_multiplier": 1.0,
            "is_favorable": regime in ["TOKYO", "LONDON"],
            "confidence": 0.3
        }

    def _save_patterns(self):
        """パターン保存"""
        try:
            path = Path(__file__).parent.parent.parent / "data" / "time_patterns.json"
            path.parent.mkdir(parents=True, exist_ok=True)

            data = {
                "patterns": {str(k): v for k, v in self.learned_patterns.items()},
                "trade_count": len(self.trade_results),
                "last_update": datetime.now().isoformat()
            }

            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
        except:
            pass

    def _load_patterns(self):
        """パターン読み込み"""
        try:
            path = Path(__file__).parent.parent.parent / "data" / "time_patterns.json"
            if path.exists():
                with open(path, 'r') as f:
                    data = json.load(f)
                    self.learned_patterns = {int(k): v for k, v in data.get("patterns", {}).items()}
        except:
            pass


# =============================================================================
# 市場感情分析
# =============================================================================

class SentimentAnalyzer:
    """市場感情分析（急変検知）"""

    def __init__(self, window: int = 100):
        self.window = window
        self.price_changes: Dict[str, deque] = {}
        self.volume_proxy: Dict[str, deque] = {}  # 価格変動の頻度で代用
        self.alerts: List[Dict] = []

    def add_price(self, pair: str, price: float):
        """価格追加"""
        if pair not in self.price_changes:
            self.price_changes[pair] = deque(maxlen=self.window)
            self.volume_proxy[pair] = deque(maxlen=self.window)

        if len(self.price_changes[pair]) > 0:
            last_prices = list(self.price_changes[pair])
            if last_prices:
                change = (price - last_prices[-1]) / last_prices[-1]
                self.volume_proxy[pair].append(abs(change))

        self.price_changes[pair].append(price)

        # 急変検知
        self._detect_sudden_move(pair, price)

    def _detect_sudden_move(self, pair: str, price: float):
        """急激な動きを検知"""
        if len(self.price_changes[pair]) < 10:
            return

        prices = list(self.price_changes[pair])
        recent_change = (price - prices[-5]) / prices[-5] if len(prices) >= 5 else 0
        avg_volatility = np.std(np.diff(prices) / np.array(prices[:-1])) if len(prices) > 1 else 0.01

        # 通常の3倍以上の動き
        if abs(recent_change) > avg_volatility * 3 and abs(recent_change) > 0.005:
            alert = {
                "pair": pair,
                "type": "SUDDEN_UP" if recent_change > 0 else "SUDDEN_DOWN",
                "magnitude": abs(recent_change),
                "timestamp": datetime.now()
            }
            self.alerts.append(alert)

            # 最新20件のみ保持
            if len(self.alerts) > 20:
                self.alerts = self.alerts[-20:]

    def get_sentiment(self, pair: str = None) -> str:
        """
        市場感情判定
        - EXTREME_FEAR: 急落検知
        - FEAR: 下落傾向
        - NEUTRAL: 横ばい
        - GREED: 上昇傾向
        - EXTREME_GREED: 急騰検知
        """
        # 直近のアラートチェック
        recent_alerts = [
            a for a in self.alerts
            if (datetime.now() - a["timestamp"]).seconds < 300  # 5分以内
            and (pair is None or a["pair"] == pair)
        ]

        if recent_alerts:
            latest = recent_alerts[-1]
            if latest["type"] == "SUDDEN_UP" and latest["magnitude"] > 0.01:
                return "EXTREME_GREED"
            elif latest["type"] == "SUDDEN_DOWN" and latest["magnitude"] > 0.01:
                return "EXTREME_FEAR"

        # 全体的なトレンド
        if pair and pair in self.price_changes and len(self.price_changes[pair]) >= 20:
            prices = list(self.price_changes[pair])
            short_avg = np.mean(prices[-5:])
            long_avg = np.mean(prices[-20:])

            ratio = short_avg / long_avg
            if ratio > 1.01:
                return "GREED"
            elif ratio < 0.99:
                return "FEAR"

        return "NEUTRAL"

    def get_fear_greed_index(self) -> float:
        """
        恐怖・貪欲指数 (0-100)
        0 = 極度の恐怖
        50 = 中立
        100 = 極度の貪欲
        """
        scores = []

        for pair, prices in self.price_changes.items():
            if len(prices) < 20:
                continue

            prices_list = list(prices)
            short_avg = np.mean(prices_list[-5:])
            long_avg = np.mean(prices_list[-20:])

            # 変化率を0-100にマップ
            change_pct = (short_avg / long_avg - 1) * 100
            score = 50 + change_pct * 10  # -5%で0、+5%で100
            score = max(0, min(100, score))
            scores.append(score)

        return float(np.mean(scores)) if scores else 50.0

    def should_trade(self) -> Tuple[bool, str]:
        """
        取引すべきかの判断
        """
        sentiment = self.get_sentiment()
        fgi = self.get_fear_greed_index()

        # 極端な感情時は取引控えめ
        if sentiment == "EXTREME_FEAR":
            return False, "急落検知 - 様子見推奨"
        if sentiment == "EXTREME_GREED":
            return False, "急騰検知 - 利確検討"

        # FGI極端値
        if fgi < 20:
            return True, "恐怖時 - 買い機会"
        if fgi > 80:
            return False, "貪欲時 - 取引控えめ"

        return True, "通常"


# =============================================================================
# 統合マーケットインテリジェンス
# =============================================================================

class MarketIntelligence:
    """
    🧠 統合マーケットインテリジェンス
    全ての分析を統合して最適な取引判断を提供
    """

    def __init__(self):
        self.correlation = CorrelationAnalyzer()
        self.volatility = VolatilityAnalyzer()
        self.time_pattern = TimePatternLearner()
        self.sentiment = SentimentAnalyzer()

        # キャッシュ
        self._last_state: Optional[MarketState] = None
        self._last_update: Optional[datetime] = None

    def update(self, pair: str, price: float):
        """価格更新"""
        self.correlation.add_price(pair, price)
        self.volatility.add_price(pair, price)
        self.sentiment.add_price(pair, price)

        # リターン計算
        if pair in self.correlation.prices and len(self.correlation.prices[pair]) >= 2:
            prices = list(self.correlation.prices[pair])
            returns = (prices[-1] - prices[-2]) / prices[-2]
            self.time_pattern.record_price(pair, price, returns)

    def record_trade_result(self, pair: str, pnl: float, pnl_pct: float, hold_time: int):
        """取引結果記録（学習用）"""
        self.time_pattern.record_trade(pair, pnl, pnl_pct, hold_time)

    def get_market_state(self) -> MarketState:
        """現在の市場状態を取得"""
        # キャッシュチェック（5秒）
        if self._last_state and self._last_update:
            if (datetime.now() - self._last_update).seconds < 5:
                return self._last_state

        # ボラティリティレジーム（全ペア平均）
        regimes = [self.volatility.get_regime(p) for p in self.volatility.prices.keys()]
        if regimes:
            regime_priority = {"EXTREME": 4, "HIGH": 3, "NORMAL": 2, "LOW": 1}
            avg_regime = sum(regime_priority.get(r, 2) for r in regimes) / len(regimes)
            if avg_regime >= 3.5:
                vol_regime = "EXTREME"
            elif avg_regime >= 2.5:
                vol_regime = "HIGH"
            elif avg_regime >= 1.5:
                vol_regime = "NORMAL"
            else:
                vol_regime = "LOW"
        else:
            vol_regime = "NORMAL"

        # トレンド方向
        fgi = self.sentiment.get_fear_greed_index()
        if fgi > 70:
            trend = "STRONG_UP"
        elif fgi > 55:
            trend = "UP"
        elif fgi < 30:
            trend = "STRONG_DOWN"
        elif fgi < 45:
            trend = "DOWN"
        else:
            trend = "NEUTRAL"

        # 相関状態
        divergences = self.correlation.detect_divergence()
        if len(divergences) >= 2:
            corr_state = "DECORRELATED"
        elif len(divergences) == 1:
            corr_state = "DIVERGING"
        else:
            corr_state = "SYNCED"

        # 感情
        sentiment = self.sentiment.get_sentiment()

        # 時間帯
        time_regime = self.time_pattern.get_current_regime()

        # 信頼度計算
        data_points = sum(len(p) for p in self.volatility.prices.values())
        confidence = min(data_points / 500, 1.0)

        state = MarketState(
            volatility_regime=vol_regime,
            trend_direction=trend,
            correlation_state=corr_state,
            sentiment=sentiment,
            time_regime=time_regime,
            confidence=confidence
        )

        self._last_state = state
        self._last_update = datetime.now()

        return state

    def get_trading_signal(self, pair: str, current_price: float) -> TradingSignal:
        """
        取引シグナル生成
        全ての分析を統合した最適な判断
        """
        state = self.get_market_state()
        reasons = []

        # === 方向判断 ===
        direction_score = 0.0

        # 1. トレンド
        trend_scores = {
            "STRONG_UP": 0.4,
            "UP": 0.2,
            "NEUTRAL": 0.0,
            "DOWN": -0.2,
            "STRONG_DOWN": -0.4
        }
        direction_score += trend_scores.get(state.trend_direction, 0)
        if state.trend_direction in ["STRONG_UP", "UP"]:
            reasons.append(f"トレンド↑")
        elif state.trend_direction in ["STRONG_DOWN", "DOWN"]:
            reasons.append(f"トレンド↓")

        # 2. BTC相関
        btc_corr = self.correlation.get_btc_correlation(pair)
        leader = self.correlation.get_leader_follower()
        if leader and leader[1] == "LEADER":
            direction_score += 0.2  # BTCが上昇中
            reasons.append("BTC先行↑")
        elif leader and leader[1] == "LEADER_DOWN":
            direction_score -= 0.2
            reasons.append("BTC先行↓")

        # 3. 感情
        if state.sentiment == "FEAR":
            direction_score += 0.1  # 恐怖時は買い
            reasons.append("恐怖買い")
        elif state.sentiment == "EXTREME_FEAR":
            direction_score += 0.2
            reasons.append("極度恐怖=買い機会")
        elif state.sentiment == "GREED":
            direction_score -= 0.1
            reasons.append("貪欲=注意")

        # 4. 時間帯
        time_rec = self.time_pattern.get_hour_recommendation()
        if time_rec["is_favorable"]:
            reasons.append(f"{time_rec['regime']}好調")

        # === 方向決定 ===
        if direction_score > 0.3:
            direction = "BUY"
            strength = min(direction_score, 1.0)
        elif direction_score < -0.3:
            direction = "SELL"
            strength = min(abs(direction_score), 1.0)
        else:
            direction = "HOLD"
            strength = 0.3

        # === リスクレベル ===
        risk_factors = 0
        if state.volatility_regime == "HIGH":
            risk_factors += 1
        if state.volatility_regime == "EXTREME":
            risk_factors += 2
        if state.sentiment in ["EXTREME_FEAR", "EXTREME_GREED"]:
            risk_factors += 1
        if state.correlation_state == "DECORRELATED":
            risk_factors += 1

        if risk_factors >= 3:
            risk_level = "HIGH"
        elif risk_factors >= 1:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        # === サイズ調整 ===
        vol_params = self.volatility.get_recommended_params(pair)
        size_mult = vol_params["size_mult"] * time_rec["size_multiplier"]

        # リスク高い場合は縮小
        if risk_level == "HIGH":
            size_mult *= 0.5
        elif risk_level == "MEDIUM":
            size_mult *= 0.8

        size_mult = max(0.3, min(2.0, size_mult))

        return TradingSignal(
            pair=pair,
            direction=direction,
            strength=strength,
            reasons=reasons,
            risk_level=risk_level,
            suggested_size_mult=size_mult
        )

    def get_all_signals(self, pairs: List[str]) -> Dict[str, TradingSignal]:
        """全ペアのシグナル取得"""
        signals = {}
        for pair in pairs:
            if pair in self.volatility.prices and len(self.volatility.prices[pair]) > 0:
                price = list(self.volatility.prices[pair])[-1]
                signals[pair] = self.get_trading_signal(pair, price)
        return signals

    def get_best_opportunity(self, pairs: List[str]) -> Optional[Tuple[str, TradingSignal]]:
        """最も良い取引機会を返す"""
        signals = self.get_all_signals(pairs)

        best = None
        best_score = 0

        for pair, signal in signals.items():
            if signal.direction == "HOLD":
                continue

            score = signal.strength
            if signal.risk_level == "LOW":
                score *= 1.2
            elif signal.risk_level == "HIGH":
                score *= 0.7

            if score > best_score:
                best_score = score
                best = (pair, signal)

        return best

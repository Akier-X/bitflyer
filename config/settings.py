"""
Configuration Management
=========================
環境変数と設定管理
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path
from loguru import logger


def get_env(key: str, default: Any = None, required: bool = False) -> Any:
    """環境変数を取得"""
    value = os.environ.get(key, default)
    if required and value is None:
        raise ValueError(f"Required environment variable {key} is not set")
    return value


def get_bool_env(key: str, default: bool = False) -> bool:
    """ブール環境変数を取得"""
    value = os.environ.get(key, "").lower()
    if value in ("true", "1", "yes", "on"):
        return True
    elif value in ("false", "0", "no", "off"):
        return False
    return default


def get_float_env(key: str, default: float = 0.0) -> float:
    """数値環境変数を取得"""
    try:
        return float(os.environ.get(key, default))
    except (ValueError, TypeError):
        return default


def get_int_env(key: str, default: int = 0) -> int:
    """整数環境変数を取得"""
    try:
        return int(os.environ.get(key, default))
    except (ValueError, TypeError):
        return default


@dataclass
class BitFlyerConfig:
    """bitFlyer API設定"""
    api_key: str = ""
    api_secret: str = ""
    product_code: str = "BTC_JPY"
    use_testnet: bool = False

    @classmethod
    def from_env(cls) -> "BitFlyerConfig":
        return cls(
            api_key=get_env("BITFLYER_API_KEY", ""),
            api_secret=get_env("BITFLYER_API_SECRET", ""),
            product_code=get_env("BITFLYER_PRODUCT_CODE", "BTC_JPY"),
            use_testnet=get_bool_env("BITFLYER_TESTNET", False),
        )


@dataclass
class LineConfig:
    """LINE通知設定（Messaging API対応）"""
    # LINE Messaging API (推奨 - LINE Notifyは2025年3月31日で終了)
    channel_access_token: str = ""  # Messaging APIチャネルアクセストークン
    user_id: str = ""  # 通知先ユーザーID（オプション、未設定時はブロードキャスト）

    # Legacy: LINE Notify (2025年3月31日終了予定)
    notify_token: str = ""  # 後方互換性のため保持

    # 通知設定
    enable_trade_notifications: bool = True
    enable_signal_notifications: bool = True
    enable_risk_alerts: bool = True
    min_confidence_to_notify: float = 0.6
    min_pnl_to_notify: float = 0.001

    @classmethod
    def from_env(cls) -> "LineConfig":
        return cls(
            # Messaging API (優先)
            channel_access_token=get_env("LINE_CHANNEL_ACCESS_TOKEN", ""),
            user_id=get_env("LINE_USER_ID", ""),
            # Legacy Notify
            notify_token=get_env("LINE_NOTIFY_TOKEN", ""),
            # Settings
            enable_trade_notifications=get_bool_env("LINE_ENABLE_TRADE", True),
            enable_signal_notifications=get_bool_env("LINE_ENABLE_SIGNAL", True),
            enable_risk_alerts=get_bool_env("LINE_ENABLE_RISK", True),
            min_confidence_to_notify=get_float_env("LINE_MIN_CONFIDENCE", 0.6),
            min_pnl_to_notify=get_float_env("LINE_MIN_PNL", 0.001),
        )

    @property
    def is_configured(self) -> bool:
        """LINE通知が設定されているか"""
        return bool(self.channel_access_token) or bool(self.notify_token)

    @property
    def use_messaging_api(self) -> bool:
        """Messaging APIを使用するか"""
        return bool(self.channel_access_token)


@dataclass
class TradingConfig:
    """取引設定"""
    # モード
    paper_trading: bool = True  # Paper trading mode
    enable_trading: bool = True

    # ポジションサイズ
    max_position_size: float = 0.1  # BTC
    min_order_size: float = 0.001
    max_leverage: float = 2.0

    # リスク管理
    max_drawdown: float = 0.1  # 10%
    daily_loss_limit: float = 0.05  # 5%
    stop_loss_pct: float = 0.02  # 2%
    take_profit_pct: float = 0.03  # 3%

    # 信頼度閾値
    min_confidence: float = 0.5
    max_confidence: float = 0.85
    base_confidence: float = 0.6

    # 取引頻度
    min_trade_interval: int = 30  # seconds
    max_trades_per_hour: int = 60

    @classmethod
    def from_env(cls) -> "TradingConfig":
        return cls(
            paper_trading=get_bool_env("PAPER_TRADING", True),
            enable_trading=get_bool_env("ENABLE_TRADING", True),
            max_position_size=get_float_env("MAX_POSITION_SIZE", 0.1),
            min_order_size=get_float_env("MIN_ORDER_SIZE", 0.001),
            max_leverage=get_float_env("MAX_LEVERAGE", 2.0),
            max_drawdown=get_float_env("MAX_DRAWDOWN", 0.1),
            daily_loss_limit=get_float_env("DAILY_LOSS_LIMIT", 0.05),
            stop_loss_pct=get_float_env("STOP_LOSS_PCT", 0.02),
            take_profit_pct=get_float_env("TAKE_PROFIT_PCT", 0.03),
            min_confidence=get_float_env("MIN_CONFIDENCE", 0.5),
            max_confidence=get_float_env("MAX_CONFIDENCE", 0.85),
            base_confidence=get_float_env("BASE_CONFIDENCE", 0.6),
            min_trade_interval=get_int_env("MIN_TRADE_INTERVAL", 30),
            max_trades_per_hour=get_int_env("MAX_TRADES_PER_HOUR", 60),
        )


@dataclass
class AIConfig:
    """AI設定"""
    feature_dim: int = 500
    seq_len: int = 60
    enable_sota_models: bool = True
    enable_rl: bool = True
    enable_world_model: bool = True
    enable_multi_agent: bool = True
    enable_pattern_matching: bool = True

    # モデルパス
    model_path: str = "models/ultimate_engine.pt"

    @classmethod
    def from_env(cls) -> "AIConfig":
        return cls(
            feature_dim=get_int_env("AI_FEATURE_DIM", 500),
            seq_len=get_int_env("AI_SEQ_LEN", 60),
            enable_sota_models=get_bool_env("AI_ENABLE_SOTA", True),
            enable_rl=get_bool_env("AI_ENABLE_RL", True),
            enable_world_model=get_bool_env("AI_ENABLE_WORLD_MODEL", True),
            enable_multi_agent=get_bool_env("AI_ENABLE_MULTI_AGENT", True),
            enable_pattern_matching=get_bool_env("AI_ENABLE_PATTERN", True),
            model_path=get_env("AI_MODEL_PATH", "models/ultimate_engine.pt"),
        )


@dataclass
class ServerConfig:
    """サーバー設定"""
    host: str = "0.0.0.0"
    port: int = 8080
    debug: bool = False

    # ヘルスチェック
    health_check_interval: int = 60

    @classmethod
    def from_env(cls) -> "ServerConfig":
        return cls(
            host=get_env("HOST", "0.0.0.0"),
            port=get_int_env("PORT", 8080),
            debug=get_bool_env("DEBUG", False),
            health_check_interval=get_int_env("HEALTH_CHECK_INTERVAL", 60),
        )


@dataclass
class Config:
    """統合設定"""
    bitflyer: BitFlyerConfig = field(default_factory=BitFlyerConfig)
    line: LineConfig = field(default_factory=LineConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    ai: AIConfig = field(default_factory=AIConfig)
    server: ServerConfig = field(default_factory=ServerConfig)

    @classmethod
    def from_env(cls) -> "Config":
        """環境変数から設定を読み込み"""
        return cls(
            bitflyer=BitFlyerConfig.from_env(),
            line=LineConfig.from_env(),
            trading=TradingConfig.from_env(),
            ai=AIConfig.from_env(),
            server=ServerConfig.from_env(),
        )

    def validate(self) -> List[str]:
        """設定を検証"""
        errors = []

        # bitFlyer API
        if not self.trading.paper_trading:
            if not self.bitflyer.api_key:
                errors.append("BITFLYER_API_KEY is required for live trading")
            if not self.bitflyer.api_secret:
                errors.append("BITFLYER_API_SECRET is required for live trading")

        # Trading config
        if self.trading.max_position_size <= 0:
            errors.append("MAX_POSITION_SIZE must be positive")
        if self.trading.max_drawdown <= 0 or self.trading.max_drawdown > 1:
            errors.append("MAX_DRAWDOWN must be between 0 and 1")

        return errors

    def to_dict(self) -> Dict:
        """設定を辞書に変換"""
        return {
            "bitflyer": {
                "product_code": self.bitflyer.product_code,
                "use_testnet": self.bitflyer.use_testnet,
                "api_configured": bool(self.bitflyer.api_key),
            },
            "line": {
                "configured": self.line.is_configured,
                "use_messaging_api": self.line.use_messaging_api,
                "enable_trade": self.line.enable_trade_notifications,
                "enable_signal": self.line.enable_signal_notifications,
                "enable_risk": self.line.enable_risk_alerts,
            },
            "trading": {
                "paper_trading": self.trading.paper_trading,
                "enable_trading": self.trading.enable_trading,
                "max_position_size": self.trading.max_position_size,
                "max_drawdown": self.trading.max_drawdown,
                "confidence_range": [self.trading.min_confidence, self.trading.max_confidence],
            },
            "ai": {
                "feature_dim": self.ai.feature_dim,
                "seq_len": self.ai.seq_len,
                "components": {
                    "sota_models": self.ai.enable_sota_models,
                    "rl": self.ai.enable_rl,
                    "world_model": self.ai.enable_world_model,
                    "multi_agent": self.ai.enable_multi_agent,
                    "pattern_matching": self.ai.enable_pattern_matching,
                },
            },
            "server": {
                "host": self.server.host,
                "port": self.server.port,
                "debug": self.server.debug,
            },
        }


# グローバル設定インスタンス
_config: Optional[Config] = None


def get_config() -> Config:
    """設定を取得"""
    global _config
    if _config is None:
        _config = Config.from_env()
        errors = _config.validate()
        if errors:
            for error in errors:
                logger.warning(f"Config warning: {error}")
    return _config


def reload_config() -> Config:
    """設定を再読み込み"""
    global _config
    _config = Config.from_env()
    return _config


# 環境変数テンプレート
ENV_TEMPLATE = """
# ===========================================
# Ultimate AI Trading System Configuration
# ===========================================

# ===== bitFlyer API =====
BITFLYER_API_KEY=your_api_key_here
BITFLYER_API_SECRET=your_api_secret_here
BITFLYER_PRODUCT_CODE=BTC_JPY
BITFLYER_TESTNET=false

# ===== LINE Messaging API (推奨) =====
# LINE Notifyは2025年3月31日で終了のため、Messaging APIを使用
# 設定方法: https://developers.line.biz/console/
# 1. プロバイダー作成 → Messaging APIチャネル作成
# 2. チャネルアクセストークンを発行
# 3. QRコードでBotを友だち追加
LINE_CHANNEL_ACCESS_TOKEN=your_channel_access_token_here
LINE_USER_ID=your_user_id_here

# ===== LINE Notify (2025年3月31日終了予定) =====
# 後方互換性のため残存。新規設定にはMessaging APIを使用してください
# LINE_NOTIFY_TOKEN=your_line_token_here

LINE_ENABLE_TRADE=true
LINE_ENABLE_SIGNAL=true
LINE_ENABLE_RISK=true
LINE_MIN_CONFIDENCE=0.6
LINE_MIN_PNL=0.001

# ===== Trading Settings =====
PAPER_TRADING=true
ENABLE_TRADING=true
MAX_POSITION_SIZE=0.1
MIN_ORDER_SIZE=0.001
MAX_LEVERAGE=2.0
MAX_DRAWDOWN=0.1
DAILY_LOSS_LIMIT=0.05
STOP_LOSS_PCT=0.02
TAKE_PROFIT_PCT=0.03
MIN_CONFIDENCE=0.5
MAX_CONFIDENCE=0.85
BASE_CONFIDENCE=0.6
MIN_TRADE_INTERVAL=30
MAX_TRADES_PER_HOUR=60

# ===== AI Settings =====
AI_FEATURE_DIM=500
AI_SEQ_LEN=60
AI_ENABLE_SOTA=true
AI_ENABLE_RL=true
AI_ENABLE_WORLD_MODEL=true
AI_ENABLE_MULTI_AGENT=true
AI_ENABLE_PATTERN=true
AI_MODEL_PATH=models/ultimate_engine.pt

# ===== Server Settings =====
HOST=0.0.0.0
PORT=8080
DEBUG=false
HEALTH_CHECK_INTERVAL=60
"""


def generate_env_template(path: str = ".env.template") -> None:
    """環境変数テンプレートを生成"""
    with open(path, "w") as f:
        f.write(ENV_TEMPLATE.strip())
    logger.info(f"Environment template generated: {path}")


logger.info("Configuration module loaded")

# 🏆 World's Strongest AI Trader

bitFlyer用の世界最強AIトレーディングシステム。月利30%以上を目指す高頻度取引ボット。

## 🎯 目標

- **月利30%以上**の達成
- **高頻度取引**による利益最大化
- **リスク管理**による資産保護
- **自動運用**による24時間取引

## 🚀 特徴

### 6つの戦略のアンサンブル

1. **マーケットメイキング** - スプレッド取得で安定収益
2. **モメンタム** - トレンドフォローで大きな利益
3. **平均回帰** - 過剰な価格変動からの回帰を狙う
4. **ブレイクアウト** - 重要な価格レベル突破を検出
5. **アービトラージ** - 価格差を利用したリスクフリー取引
6. **機械学習** - AI予測による高精度シグナル

### 高度なリスク管理

- **Kelly基準**によるポジションサイジング
- **動的ストップロス**・利確
- **トレーリングストップ**
- **最大ドローダウン制限**
- **VaR (Value at Risk)** 計算
- **相関リスク管理**

### スマートオーダー執行

- **TWAP** (Time Weighted Average Price)
- **VWAP** (Volume Weighted Average Price)
- **Iceberg Orders** (氷山注文)
- **スリッページ最小化**
- **Maker手数料活用**

## 📦 インストール

```bash
# リポジトリクローン
git clone <repository-url>
cd bitflyer

# 仮想環境作成
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# 依存関係インストール
pip install -r requirements.txt
```

## ⚙️ 設定

### 環境変数

```bash
export BITFLYER_API_KEY="your_api_key"
export BITFLYER_API_SECRET="your_api_secret"
```

### 設定ファイル

`config/settings.yaml` で詳細な設定が可能:

- 取引ペア
- 戦略の有効/無効・重み
- リスク管理パラメータ
- 高頻度取引設定
- 機械学習モデル設定

## 🏃 実行

```bash
# 本番モード
python main.py

# ドライランモード（実際の取引なし）
python main.py --dry-run

# カスタム設定ファイル
python main.py --config path/to/config.yaml
```

## 📊 アーキテクチャ

```
src/
├── api/                    # bitFlyer API
│   ├── client.py          # REST APIクライアント
│   └── websocket_client.py # WebSocketクライアント
├── strategies/             # 取引戦略
│   ├── market_making.py   # マーケットメイキング
│   ├── momentum.py        # モメンタム
│   ├── mean_reversion.py  # 平均回帰
│   ├── breakout.py        # ブレイクアウト
│   ├── arbitrage.py       # アービトラージ
│   ├── ml_strategy.py     # 機械学習
│   └── ensemble.py        # アンサンブル
├── ml/                     # 機械学習
│   ├── features.py        # 特徴量エンジニアリング
│   ├── models.py          # 予測モデル
│   └── training.py        # 訓練パイプライン
├── risk/                   # リスク管理
│   ├── manager.py         # リスクマネージャー
│   ├── position.py        # ポジション管理
│   └── portfolio.py       # ポートフォリオ最適化
├── execution/              # 注文執行
│   ├── order_manager.py   # 注文管理
│   └── smart_executor.py  # スマート執行
└── bot/                    # メインボット
    ├── trader.py          # AIトレーダー
    └── performance.py     # パフォーマンス追跡
```

## 📈 パフォーマンス目標

| 指標 | 目標値 |
|------|--------|
| 月利 | 30%以上 |
| シャープレシオ | 2.5以上 |
| 最大ドローダウン | 10%以内 |
| 勝率 | 55%以上 |
| プロフィットファクター | 1.8以上 |
| 1日の取引数 | 500回以上 |

## ⚠️ リスク警告

- 暗号資産取引には重大なリスクが伴います
- 投資した資金を失う可能性があります
- 過去のパフォーマンスは将来の結果を保証しません
- 自己責任で使用してください
- 失っても良い資金のみで運用してください

## 📝 ライセンス

MIT License

## 🔧 開発

```bash
# テスト実行
pytest tests/

# コードフォーマット
black src/

# 型チェック
mypy src/
```

## 🤝 貢献

プルリクエストを歓迎します！

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

---

**World's Strongest AI Trader** - 世界最強のAIトレーダーを目指して 🚀

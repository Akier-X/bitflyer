# 🚀 デプロイガイド - Ultimate AI Trading System

## 📋 目次
1. [事前準備](#事前準備)
2. [無料サーバーデプロイ](#無料サーバーデプロイ)
3. [LINE通知設定](#line通知設定)
4. [ローカル実行](#ローカル実行)

---

## 🔧 事前準備

### 1. LINE Notify トークン取得

1. https://notify-bot.line.me/ にアクセス
2. LINEアカウントでログイン
3. 「トークンを発行する」をクリック
4. トークン名を入力（例: "AI Trader"）
5. 通知を送るトークルームを選択
6. 「発行する」をクリック
7. 表示されたトークンをコピー（後で使用）

### 2. bitFlyer API キー取得（本番取引用）

1. https://bitflyer.com/ja-jp/ にログイン
2. 設定 > API
3. 新しいAPIキーを作成
4. 必要な権限を付与:
   - 資産残高を取得
   - 注文を行う
   - 注文をキャンセルする

---

## ☁️ 無料サーバーデプロイ

### Option 1: Render.com（推奨）

**特徴**: 750時間/月無料、簡単設定

```bash
# 1. GitHubリポジトリにpush
git add .
git commit -m "Deploy to Render"
git push origin main

# 2. https://dashboard.render.com にアクセス
# 3. New > Web Service
# 4. GitHubリポジトリを接続
# 5. 以下を設定:
#    - Environment: Python 3
#    - Build Command: pip install -r requirements.txt
#    - Start Command: python main.py --paper --port $PORT
#
# 6. 環境変数を設定:
#    - LINE_NOTIFY_TOKEN: [your-token]
#    - PAPER_TRADING: true
```

### Option 2: Railway.app

**特徴**: $5クレジット/月無料、高速デプロイ

```bash
# 1. Railway CLI インストール
npm i -g @railway/cli

# 2. ログイン & デプロイ
railway login
railway init
railway up

# 3. 環境変数設定
railway variables set LINE_NOTIFY_TOKEN=your-token
railway variables set PAPER_TRADING=true
```

### Option 3: Fly.io

**特徴**: 3VM無料、東京リージョン対応

```bash
# 1. Fly CLI インストール
curl -L https://fly.io/install.sh | sh

# 2. ログイン
flyctl auth login

# 3. デプロイ
flyctl launch
flyctl secrets set LINE_NOTIFY_TOKEN=your-token
flyctl deploy
```

### Option 4: Google Cloud Run（無料枠）

```bash
# 1. gcloud CLI セットアップ
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# 2. ビルド & デプロイ
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/ai-trader
gcloud run deploy ai-trader \
  --image gcr.io/YOUR_PROJECT_ID/ai-trader \
  --platform managed \
  --region asia-northeast1 \
  --allow-unauthenticated \
  --set-env-vars "LINE_NOTIFY_TOKEN=your-token,PAPER_TRADING=true"
```

---

## 📱 LINE通知設定

### 通知テスト

```bash
# 環境変数を設定してテスト
export LINE_NOTIFY_TOKEN="your-token-here"
python -c "
from src.notifications.line_notify import LINENotifier
n = LINENotifier('$LINE_NOTIFY_TOKEN')
n.send_sync('🚀 AI Trader テスト通知')
"
```

### 通知タイプ

| タイプ | 説明 | レート制限 |
|--------|------|-----------|
| 取引執行 | 注文が約定したとき | 5秒 |
| シグナル | 高信頼度シグナル検出時 | 30秒 |
| リスク警告 | ドローダウン等の警告 | 60秒 |
| 日次レポート | 毎日のサマリー | 1時間 |

---

## 💻 ローカル実行

### Docker使用

```bash
# 1. 環境変数ファイル作成
cp .env.template .env
# .env を編集してトークン等を設定

# 2. Docker Compose で起動
docker-compose up -d

# 3. ログ確認
docker-compose logs -f

# 4. ステータス確認
curl http://localhost:8080/health
```

### 直接実行

```bash
# 1. 仮想環境作成
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. 依存関係インストール
pip install -r requirements.txt

# 3. 環境変数設定
export LINE_NOTIFY_TOKEN="your-token"
export PAPER_TRADING=true

# 4. 実行
python main.py --paper
```

---

## 📊 ヘルスチェック

デプロイ後、以下のエンドポイントで確認:

```bash
# ヘルスチェック
curl https://your-app-url/health

# ステータス
curl https://your-app-url/status
```

---

## ⚠️ 注意事項

1. **Paper Tradingモード推奨**: 最初は必ず `PAPER_TRADING=true` で実行
2. **API制限**: 無料サーバーは休止することがある（Render: 15分無操作で休止）
3. **本番運用**: 本番取引は自己責任で行ってください
4. **セキュリティ**: APIキーは環境変数で管理し、コードにハードコードしない

---

## 🆘 トラブルシューティング

### "Module not found" エラー
```bash
pip install -r requirements.txt
```

### LINE通知が届かない
1. トークンが正しいか確認
2. LINEアプリで通知がオンか確認
3. ログを確認: `logs/trading_*.log`

### ポートエラー
```bash
# 別のポートを使用
python main.py --port 3000
```

---

## 📈 パフォーマンスモニタリング

### Render.com ダッシュボード
- メトリクス: CPU, メモリ使用率
- ログ: リアルタイムログ確認

### 自作ダッシュボード
`/status` エンドポイントでJSON形式の統計情報を取得可能

---

**Happy Trading! 🏆**

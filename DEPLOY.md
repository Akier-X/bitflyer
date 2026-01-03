# 🚀 デプロイガイド - Ultimate AI Trading System

## 📋 目次
1. [事前準備](#事前準備)
2. [無料サーバーデプロイ](#無料サーバーデプロイ)
3. [LINE通知設定](#line通知設定)
4. [ローカル実行](#ローカル実行)

---

## 🔧 事前準備

### 1. LINE Messaging API 設定（推奨）

> ⚠️ **重要**: LINE Notifyは2025年3月31日にサービス終了予定です。新規設定にはLINE Messaging APIを使用してください。

#### 手順

1. **LINE Developers Console** にアクセス
   - https://developers.line.biz/console/

2. **プロバイダーを作成**
   - 「新規プロバイダー作成」をクリック
   - プロバイダー名を入力（例: "My Trading Bot"）

3. **Messaging APIチャネルを作成**
   - 「新規チャネル作成」→「Messaging API」を選択
   - 必要情報を入力:
     - チャネル名: AI Trader Bot
     - チャネル説明: 自動売買通知Bot
     - 大業種/小業種: 適切なものを選択

4. **チャネルアクセストークンを発行**
   - チャネル設定 → 「Messaging API設定」タブ
   - 「チャネルアクセストークン（長期）」の「発行」をクリック
   - 表示されたトークンをコピー → `LINE_CHANNEL_ACCESS_TOKEN`

5. **Botを友だち追加**
   - 同じページにあるQRコードをスマホで読み取り
   - Botを友だち追加

6. **ユーザーIDを取得**（オプション）
   - Webhookを設定するか、LINE Official Account Managerで確認
   - 自分のユーザーIDをコピー → `LINE_USER_ID`

#### 無料枠について
- 月200通まで無料
- 追加メッセージは有料（従量課金）

#### （旧）LINE Notify（2025年3月31日終了予定）

既存のLINE Notifyトークンをお持ちの場合は一時的に使用可能ですが、
早めにMessaging APIへ移行してください。

1. https://notify-bot.line.me/ にアクセス
2. トークンを取得 → `LINE_NOTIFY_TOKEN`（非推奨）

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
#    - LINE_CHANNEL_ACCESS_TOKEN: [your-token]
#    - LINE_USER_ID: [your-user-id] (オプション)
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
railway variables set LINE_CHANNEL_ACCESS_TOKEN=your-token
railway variables set LINE_USER_ID=your-user-id
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
flyctl secrets set LINE_CHANNEL_ACCESS_TOKEN=your-token
flyctl secrets set LINE_USER_ID=your-user-id
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
  --set-env-vars "LINE_CHANNEL_ACCESS_TOKEN=your-token,LINE_USER_ID=your-user-id,PAPER_TRADING=true"
```

---

## 📱 LINE通知設定

### 通知テスト（Messaging API）

```bash
# 環境変数を設定してテスト
export LINE_CHANNEL_ACCESS_TOKEN="your-channel-access-token-here"
export LINE_USER_ID="your-user-id-here"  # オプション

python -c "
from src.notifications.line_messaging import LINEMessagingAPI
import os

notifier = LINEMessagingAPI(
    channel_access_token=os.environ['LINE_CHANNEL_ACCESS_TOKEN'],
    user_id=os.environ.get('LINE_USER_ID'),
)
notifier.send_text('🚀 AI Trader テスト通知')
print('通知送信完了！LINEを確認してください。')
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
1. チャネルアクセストークンが正しいか確認
2. Botを友だち追加しているか確認
3. LINEアプリで通知がオンか確認
4. 月間送信数が200通を超えていないか確認
5. ログを確認: `logs/trading_*.log`

### LINE Messaging API エラー
- **401エラー**: チャネルアクセストークンが無効です
- **400エラー**: ユーザーIDが不正、またはBotをブロックしています
- **429エラー**: レート制限に達しました（しばらく待ってください）

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

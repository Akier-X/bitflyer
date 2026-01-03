# 無料サーバー運用ガイド

## ⚠️ 重要: bitFlyer APIは日本国内IPからのみアクセス可能

海外サーバーからは403エラーになるため、**日本リージョン**のサーバーを選択してください。

---

## 1. Oracle Cloud Free Tier（推奨）

**永久無料**で日本リージョン（東京・大阪）が使えます。

### 手順:

1. **アカウント作成**
   - https://www.oracle.com/cloud/free/
   - 日本リージョン（ap-tokyo-1）を選択

2. **無料VMを作成**
   ```
   Shape: VM.Standard.E2.1.Micro (Always Free)
   OS: Oracle Linux 8 or Ubuntu 22.04
   ```

3. **SSHでログイン後、セットアップ**
   ```bash
   # Docker インストール
   sudo yum install -y docker
   sudo systemctl start docker
   sudo systemctl enable docker
   sudo usermod -aG docker $USER
   
   # Docker Compose インストール
   sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
   sudo chmod +x /usr/local/bin/docker-compose
   ```

4. **プロジェクトをクローン**
   ```bash
   git clone https://github.com/YOUR_REPO/bitflyer.git
   cd bitflyer
   ```

5. **環境変数を設定**
   ```bash
   cp .env.example .env
   nano .env
   # API keys を入力
   ```

6. **起動**
   ```bash
   docker-compose up -d
   ```

---

## 2. Google Cloud Free Tier

### 手順:

1. **Cloud Shell でVM作成**
   ```bash
   gcloud compute instances create ai-trader \
     --zone=asia-northeast1-a \
     --machine-type=e2-micro \
     --image-family=ubuntu-2204-lts \
     --image-project=ubuntu-os-cloud
   ```

2. **SSH接続してDocker設定**
   ```bash
   gcloud compute ssh ai-trader --zone=asia-northeast1-a
   
   # Docker インストール
   curl -fsSL https://get.docker.com | sh
   sudo usermod -aG docker $USER
   ```

3. 以降はOracle Cloudと同じ

---

## 3. ローカルDocker（自宅PC）

### 手順:

1. **Docker Desktop インストール**
   - Windows: https://docs.docker.com/desktop/install/windows-install/
   - Mac: https://docs.docker.com/desktop/install/mac-install/

2. **ビルド＆起動**
   ```bash
   # プロジェクトディレクトリで
   docker-compose up -d
   
   # ログ確認
   docker-compose logs -f
   
   # 停止
   docker-compose down
   ```

---

## 4. Railway.app（簡単デプロイ）

1. https://railway.app にGitHubでログイン
2. "New Project" → "Deploy from GitHub"
3. リポジトリを選択
4. 環境変数を設定:
   - `BITFLYER_API_KEY`
   - `BITFLYER_API_SECRET`
   - `LINE_CHANNEL_ACCESS_TOKEN`
   - `PAPER_TRADING=false`

⚠️ Railway のサーバーは海外のため、bitFlyer APIが使えない可能性があります。

---

## Docker コマンド一覧

```bash
# ビルド
docker-compose build

# 起動（バックグラウンド）
docker-compose up -d

# ログ確認
docker-compose logs -f

# ステータス確認
docker-compose ps

# 停止
docker-compose down

# 再起動
docker-compose restart

# ヘルスチェック
curl http://localhost:8080/health
```

---

## モード切替

```bash
# 攻撃的モード（デフォルト）
docker-compose up -d

# マルチアセットモード
docker run -d --env-file .env ai-trader python main.py --multi --live

# Paper Trading（テスト）
docker run -d --env-file .env ai-trader python main.py --aggressive
```

---

## トラブルシューティング

### API 403エラー
- 日本国内IPからアクセスしているか確認
- VPNを日本に設定、または日本リージョンのサーバーを使用

### コンテナが起動しない
```bash
docker-compose logs trader
```

### メモリ不足
```bash
# docker-compose.yml に追加
deploy:
  resources:
    limits:
      memory: 512M
```

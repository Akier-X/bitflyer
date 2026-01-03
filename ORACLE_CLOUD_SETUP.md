# Oracle Cloud Free Tier セットアップガイド（大阪リージョン）

## 永久無料で使える内容

- **VM.Standard.E2.1.Micro** x 2台
- **1GB RAM / 1 OCPU** 
- **ストレージ 200GB**
- **10TB/月 アウトバウンド通信**

---

## Step 1: アカウント作成

1. https://www.oracle.com/cloud/free/ にアクセス
2. 「無料で始める」をクリック
3. 以下を入力:
   - 国: Japan
   - **ホームリージョン: Japan Central (Osaka)** ← 重要！
   - メールアドレス、パスワード設定
4. クレジットカード認証（課金されません）
5. アカウント作成完了

---

## Step 2: VMインスタンス作成

### Oracle Cloud Console にログイン

1. https://cloud.oracle.com にアクセス
2. 「コンピュート」→「インスタンス」→「インスタンスの作成」

### インスタンス設定

```
名前: ai-trader

イメージ: Oracle Linux 8 または Ubuntu 22.04
  → 「イメージの変更」→「Ubuntu」→「Canonical Ubuntu 22.04」

シェイプ: VM.Standard.E2.1.Micro (Always Free)
  → 「シェイプの変更」→「Ampere」または「AMD」
  → 「VM.Standard.E2.1.Micro」を選択

ネットワーク: デフォルトVCN使用

SSHキー: 
  → 「キーペアを生成」でダウンロード
  → または既存の公開鍵をペースト
```

3. 「作成」をクリック
4. インスタンスが「実行中」になるまで待機（2-3分）

---

## Step 3: セキュリティ設定（ポート開放）

### イングレスルールを追加

1. 「ネットワーキング」→「仮想クラウドネットワーク」
2. 作成したVCNをクリック
3. 「セキュリティ・リスト」→「Default Security List」
4. 「イングレス・ルールの追加」

```
ソースCIDR: 0.0.0.0/0
IPプロトコル: TCP
宛先ポート範囲: 8080
説明: AI Trader Health Check
```

---

## Step 4: SSH接続

### Windows (PowerShell)

```powershell
ssh -i C:\path\to\ssh-key.key ubuntu@<パブリックIP>
```

### Mac/Linux

```bash
chmod 400 ~/Downloads/ssh-key.key
ssh -i ~/Downloads/ssh-key.key ubuntu@<パブリックIP>
```

※ パブリックIPはインスタンス詳細ページで確認

---

## Step 5: サーバーセットアップ

### SSH接続後、以下を実行:

```bash
# システム更新
sudo apt update && sudo apt upgrade -y

# Docker インストール
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER

# Docker Compose インストール
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# 一度ログアウトして再ログイン（Docker権限反映）
exit
```

再度SSH接続:
```bash
ssh -i ~/ssh-key.key ubuntu@<パブリックIP>

# Docker確認
docker --version
docker-compose --version
```

---

## Step 6: プロジェクトデプロイ

```bash
# プロジェクトクローン
git clone https://github.com/YOUR_USERNAME/bitflyer.git
cd bitflyer

# 環境変数ファイル作成
cat > .env << 'ENVEOF'
# bitFlyer API
BITFLYER_API_KEY=YOUR_API_KEY_HERE
BITFLYER_API_SECRET=YOUR_API_SECRET_HERE

# LINE Messaging API
LINE_CHANNEL_ACCESS_TOKEN=YOUR_LINE_TOKEN_HERE

# Trading Settings
PAPER_TRADING=false
TRADING_PAIRS=XRP_JPY,MONA_JPY,XLM_JPY
ENVEOF

# .envを編集（APIキーを入力）
nano .env
```

---

## Step 7: Bot起動

```bash
# Dockerイメージをビルド
docker-compose build

# バックグラウンドで起動
docker-compose up -d

# ログ確認
docker-compose logs -f

# ステータス確認
curl http://localhost:8080/health
```

---

## Step 8: 自動起動設定（再起動後も自動実行）

```bash
# systemdサービス作成
sudo tee /etc/systemd/system/ai-trader.service << 'SERVICEEOF'
[Unit]
Description=AI Trading Bot
After=docker.service
Requires=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/home/ubuntu/bitflyer
ExecStart=/usr/local/bin/docker-compose up -d
ExecStop=/usr/local/bin/docker-compose down
User=ubuntu

[Install]
WantedBy=multi-user.target
SERVICEEOF

# サービス有効化
sudo systemctl daemon-reload
sudo systemctl enable ai-trader
sudo systemctl start ai-trader
```

---

## 運用コマンド

```bash
# ステータス確認
docker-compose ps

# ログ確認（リアルタイム）
docker-compose logs -f

# 停止
docker-compose down

# 再起動
docker-compose restart

# 更新（新バージョン）
git pull
docker-compose build
docker-compose up -d
```

---

## トラブルシューティング

### SSH接続できない
```bash
# ファイアウォール確認
sudo iptables -L -n
```

### Dockerが動かない
```bash
sudo systemctl restart docker
```

### メモリ不足
```bash
# スワップ追加
sudo fallocate -l 1G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

### ログでエラー確認
```bash
docker-compose logs --tail=100
```

---

## 月額費用

**¥0（完全無料）**

Always Free枠内で運用可能:
- VM.Standard.E2.1.Micro: 無料
- ストレージ 50GB: 無料
- アウトバウンド通信 10TB: 無料

---

## セキュリティ注意事項

1. **SSHキーを安全に保管**
2. **.envファイルはGitにコミットしない**
3. **定期的にログを確認**
4. **不要なポートは開放しない**


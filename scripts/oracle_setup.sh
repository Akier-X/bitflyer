#!/bin/bash
# Oracle Cloud クイックセットアップスクリプト

set -e

echo "========================================"
echo "  AI Trading Bot - Oracle Cloud Setup"
echo "========================================"

# システム更新
echo "📦 Updating system..."
sudo apt update && sudo apt upgrade -y

# Docker インストール
echo "🐳 Installing Docker..."
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER

# Docker Compose インストール
echo "📦 Installing Docker Compose..."
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# スワップ追加（メモリ不足対策）
echo "💾 Adding swap space..."
sudo fallocate -l 1G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab

echo ""
echo "========================================"
echo "  Setup Complete!"
echo "========================================"
echo ""
echo "次のステップ:"
echo "1. 一度ログアウト: exit"
echo "2. 再度SSH接続"
echo "3. cd bitflyer"
echo "4. .envファイルを編集: nano .env"
echo "5. 起動: docker-compose up -d"
echo ""

# ULTIMATE AI TRADER v4.0

bitFlyer用の世界最強AIトレーディングシステム

## 目標

- **5000円 → 15000円+** (3倍リターン)
- 手数料を考慮した最適取引
- 24時間自動運用

## 特徴

- **ETH優先戦略**: 高ボラティリティで高収益
- **複合テクニカル分析**: RSI, EMA, ボリンジャーバンド, モメンタム
- **手数料考慮**: Lightning現物 0.15%を計算に含む
- **自動リスク管理**: 利確0.8%, 損切り0.5%
- **本番API + Mock自動フォールバック**

## 取引ペア

| ペア | 優先度 | 最小取引 |
|------|--------|----------|
| ETH_JPY | 1 | 0.01 ETH |
| XRP_JPY | 2 | 1 XRP |
| MONA_JPY | 3 | 1 MONA |
| BTC_JPY | 4 | 0.001 BTC |

## インストール

```bash
pip install -r requirements.txt
```

## 設定

`.env` ファイルを作成:

```env
BITFLYER_API_KEY=your_api_key
BITFLYER_API_SECRET=your_api_secret
PAPER_TRADING=false
```

## 実行

### Windows
```cmd
python -m src.bot.aggressive_trader
```

### Linux / Mac
```bash
python -m src.bot.aggressive_trader
```

## 停止

`Ctrl+C`

## 手数料情報 (bitFlyer 2025)

| サービス | 手数料 |
|----------|--------|
| Lightning 現物 | 0.01%〜0.15% |
| Crypto CFD | 0% + 0.04%/日 |
| 販売所 | 0.1%〜6.0% (使わない) |

## リスク警告

- 暗号資産取引には重大なリスクが伴います
- 投資した資金を失う可能性があります
- 自己責任で使用してください

---
**ULTIMATE AI TRADER v4.0** - 世界最強

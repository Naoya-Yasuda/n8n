# Google Sheets サービスアカウント設定ガイド

## サービスアカウントの利点
- ✅ ブラウザ認証不要
- ✅ 自動化に最適  
- ✅ 読み書き可能
- ✅ プライベートスプレッドシート対応

## 設定手順

### 1. Google Cloud Console設定
1. https://console.cloud.google.com にアクセス
2. プロジェクトを選択
3. **APIs & Services → Library** で以下を有効化:
   - Google Sheets API
   - Google Drive API

### 2. サービスアカウント作成
1. **APIs & Services → Credentials**
2. **+ CREATE CREDENTIALS → Service account**
3. 名前: `n8n-mlex-scraper`
4. **CREATE AND CONTINUE**
5. Role: `Editor` または `Owner`
6. **CONTINUE → DONE**

### 3. JSONキー作成
1. 作成したサービスアカウントをクリック
2. **Keys** タブ → **ADD KEY → Create new key**
3. **Key type: JSON** → **CREATE**
4. `service-account-key.json` がダウンロードされる

### 4. スプレッドシート共有設定
1. GoogleスプレッドシートでJSONファイル内の `client_email` をコピー
2. スプレッドシートで **Share** → このメールアドレスに **Editor** 権限で共有

### 5. n8nでの設定
1. n8n Web UI → **Settings → Credentials**
2. **Create New → Google Service Account** 
3. JSONファイルの内容を貼り付け

## ファイル例
```json
{
  "type": "service_account",
  "project_id": "your-project-id",
  "private_key_id": "key-id",
  "private_key": "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n",
  "client_email": "service-account-name@project-id.iam.gserviceaccount.com",
  "client_id": "...",
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://oauth2.googleapis.com/token"
}
```

## トラブルシューティング
- **403エラー**: サービスアカウントがスプレッドシートに共有されているか確認
- **API無効エラー**: Google Sheets API と Google Drive API が有効化されているか確認
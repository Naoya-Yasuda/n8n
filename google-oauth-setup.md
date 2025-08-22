# Google Sheets OAuth2 設定ガイド

## Google Cloud Console設定

1. **Google Cloud Console** (https://console.cloud.google.com) にアクセス
2. **新しいプロジェクト作成** または既存プロジェクトを選択
3. **APIs & Services → Library** に移動
4. **「Google Sheets API」を検索して有効化**
5. **「Google Drive API」も有効化** (必須)

## OAuth2 認証情報作成

1. **APIs & Services → Credentials** に移動
2. **「+ CREATE CREDENTIALS」→「OAuth client ID」をクリック**
3. **Application type:** `Web application`
4. **Name:** `n8n MLex Scraper`
5. **Authorized redirect URIs:** 
   ```
   http://localhost:5678/rest/oauth2-credential/callback
   ```
6. **「CREATE」をクリック**
7. **Client ID と Client Secret をコピー**

## n8nでの設定

1. n8n Web UI → Settings → Credentials
2. 「Create New Credential」→「Google Sheets OAuth2 API」
3. コピーしたClient IDとClient Secretを入力
4. 「OAuth Authorization」をクリックしてGoogleアカウント認証を完了

## 権限設定

認証時に以下の権限を許可してください:
- Google Sheets の表示と編集
- Google Drive の表示と編集

## トラブルシューティング

- **「This app isn't verified」エラー:** 「Advanced」→「Go to [app name] (unsafe)」をクリック
- **リダイレクトURIエラー:** 正確に `http://localhost:5678/rest/oauth2-credential/callback` を設定
- **スプレッドシート アクセス エラー:** Google Drive APIも有効化されているか確認
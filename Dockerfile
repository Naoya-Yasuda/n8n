FROM docker.n8n.io/n8nio/n8n:latest

# rootユーザーに切り替え
USER root

# 必要なパッケージをイメージに焼き込み
RUN npm install -g browser-use @aws-sdk/client-bedrock-runtime axios cheerio puppeteer

# 追加の依存関係（Alpine Linux用）
RUN apk add --no-cache \
    chromium \
    chromium-chromedriver \
    && rm -rf /var/cache/apk/*

# n8nユーザーに戻す
USER node

# 環境変数設定（Alpine Linux用）
ENV PUPPETEER_EXECUTABLE_PATH=/usr/bin/chromium-browser
ENV PUPPETEER_SKIP_CHROMIUM_DOWNLOAD=true
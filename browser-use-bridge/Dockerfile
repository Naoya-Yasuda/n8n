# browser-use-bridge/Dockerfile
FROM python:3.11-slim

WORKDIR /app

# システム依存関係（Chromium含む）
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    chromium \
    chromium-driver \
    && rm -rf /var/lib/apt/lists/*

# Python依存関係（browser-useライブラリ含む）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Playwright Chromium（browser-useが使用）
RUN playwright install chromium

# Bridge Server（app.py）
COPY . .

EXPOSE 8000
CMD ["python", "app.py"]
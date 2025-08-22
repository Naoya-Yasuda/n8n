# カスタムノード

このディレクトリにカスタムノードを配置してください。

## 使用方法

1. カスタムノードのパッケージをこのディレクトリに配置
2. 各ノードは独立したディレクトリとして配置
3. 各ディレクトリには`package.json`が必要

## 例

```
custom-nodes/
├── my-custom-node/
│   ├── package.json
│   ├── src/
│   └── README.md
└── another-node/
    ├── package.json
    └── src/
```

## 注意事項

- カスタムノードを追加した後は、n8nコンテナを再起動してください
- `docker-compose restart n8n` で再起動できます
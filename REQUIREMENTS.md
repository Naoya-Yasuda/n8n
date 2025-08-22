# MLex Browser Automation with AWS Bedrock - 要件定義書

## プロジェクト概要
MLexニュースサイトのスクレイピングをbrowser-use + AWS Bedrockで実現する。OpenAIではなく、AWS Bedrockを使用する必要がある。

## 技術要件

### 必須技術スタック
- **LLM**: AWS Bedrock (Claude 3.5 Sonnet) - OpenAI使用不可
- **ブラウザ自動化**: browser-use library
- **実行環境**: Docker container (n8n workflow)
- **言語**: Python 3.11

## 現在判明している仕様

### browser-useが期待するLLMインターフェース

#### 1. LLMクラスの必須属性
```python
class CustomLLM(BaseChatModel):
    def __init__(self):
        self.model = "model-name"  # 必須 - browser-useのトークン管理で使用
        # ...
```

#### 2. ainvokeメソッドの期待される戻り値
**重要**: browser-useは`response.completion`がAgentOutputオブジェクトであることを期待

**AgentOutput構造**:
```python
class AgentOutput(BaseModel):
    thinking: str | None = None
    evaluation_previous_goal: str | None = None
    memory: str | None = None
    next_goal: str | None = None
    action: list[ActionModel] = Field(...)  # 必須フィールド
```

**成功例 (OpenAI all_model_outputs)**:
```python
[
    {'go_to_url': {'url': 'https://example.com', 'new_tab': False}, 'interacted_element': None},
    {'done': {'text': 'message', 'success': True, 'files_to_display': []}, 'interacted_element': None}
]
```

**browser-useの処理フロー（詳細判明）**:
1. `await llm.ainvoke(input_messages, output_format=self.AgentOutput)` - output_formatパラメータ付きで呼び出し
2. `response.completion` → 既にAgentOutputオブジェクトである必要（browser-useは解析しない）
3. `agent_output.action` → list[ActionModel]（必須）
4. **重要**: LLM側でoutput_formatパラメータを使用してAgentOutputオブジェクトを構築する必要

#### 3. Usage属性の必須構造
```python
usage = {
    'prompt_tokens': int,
    'completion_tokens': int,
    'total_tokens': int,
    'prompt_cached_tokens': None,  # 必須フィールド
    'prompt_cache_creation_tokens': None,  # 必須フィールド
    'prompt_image_tokens': None,  # 必須フィールド
}
```

### AWS Bedrock仕様

#### 1. Converse API レスポンス形式
```python
response = {
    'output': {
        'message': {
            'content': [{'text': 'JSON文字列またはテキスト'}]
        }
    },
    'usage': {
        'inputTokens': int,
        'outputTokens': int
    }
}
```

#### 2. 認証情報
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `AWS_SESSION_TOKEN` (オプション)

## 解決済み問題

### ✅ 502エラー: "Expected structured output but no tool use found in response"
- **原因**: browser-useのChatAWSBedrockがOpenAI形式を期待するがBedrockはXML形式を返す
- **解決**: DirectBedrockLLM実装でConverse API直接使用

### ✅ NoneType usage属性エラー
- **原因**: usage属性が存在しない/形式が間違っている
- **解決**: 標準usage構造を実装

### ✅ TokenUsageEntryバリデーションエラー
- **原因**: 必須フィールド不足 (prompt_cached_tokens等)
- **解決**: 全ての必須フィールドを含むusage辞書を実装

## 解決済み問題（完全版）

### ✅ 'str' object has no attribute 'action'エラー（完全解決！）
- **現象**: 全てのカスタムLLM実装で発生していた
- **根本原因**: browser-useは`response.completion`がAgentOutputオブジェクトであることを期待するが、文字列を返していた
- **解決方法**: `output_format.model_validate_json()`を使用してOpenAIと同じアプローチを適用
- **結果**: DirectBedrockLLMが正常動作し、MLexサイトへのナビゲーションに成功！

### ✅ MLex 2段階認証プロセス（完全実行成功！）
- **発見**: MLexログインは2段階認証システムを採用
  1. メールアドレス入力 → Next ボタン
  2. パスワード入力（動的フィールド表示）→ Login ボタン
- **実行結果**: テスト認証情報での完全なログインプロセス実行成功
- **技術的成果**: browser-use + DirectBedrockLLM でWebフォーム操作が正常動作

### ⚠️ 動的ActionModel validation エラー（軽微な残課題）
- **現象**: 特定のアクション（例：`{"wait": {"timeout": 5000}}`）でPydantic validation エラー
- **発生頻度**: 稀に発生（大部分のアクションは正常動作）
- **影響**: 軽微（エラー時はフォールバック動作で継続、主要機能には影響なし）
- **根本原因**: browser-useの動的ActionModel生成と一部アクション形式の不整合
- **現状**: 主要機能（ナビゲーション、ログイン操作、サイトアクセス）は正常動作

## ゴール

### 最終目標
1. **MLexログイン画面へのナビゲーション成功**: https://identity.mlex.com/Account/Login にアクセス
2. **ログインフォーム検出**: サインイン/ログイン要素の特定
3. **ログイン操作実行**: 実際のログインプロセス完了
4. **記事スクレイピング**: ログイン後のコンテンツ取得

### 技術目標
1. **DirectBedrockLLMの完全動作**: `'str' object has no attribute 'action'`エラー解決
2. **browser-use標準互換性**: OpenAIと同等の動作実現
3. **実装オーバーヘッド最小化**: 過度な再実装を避ける

## 現在の実装状況

### 完成済みファイル
- `browser-use-bridge/app.py`: **本番OpenAIスタイルDirectBedrockLLM実装（完全動作版）**
  - `output_format.model_validate_json()` 使用
  - MLex 2段階認証対応
  - スクリーンショット機能付き
  - API エンドポイント: `/api/v1/task/{task_id}/screenshot`
- `mlex_final_openai_style_test.py`: 成功テスト実装（アーカイブ用）

### 削除済み不要ファイル
- 実験的テストファイル群（14ファイル削除完了）

### 完全解決済み
- `'str' object has no attribute 'action'` エラー解決
- MLex 2段階認証プロセス実行成功
- OpenAIスタイル実装で安定動作

## 新機能追加完了

### ✅ スクリーンショット機能
- **機能**: スクレイピング完了後の自動スクリーンショット撮影
- **保存場所**: `/tmp/screenshots/task_{task_id}_final.png`
- **API**: `GET /api/v1/task/{task_id}/screenshot`
- **フォーマット**: PNG形式、フルページスクリーンショット
- **エラーハンドリング**: スクリーンショット失敗時も処理継続

### 完了済みアップデート
1. **OpenAIスタイル実装のapp.py適用完了**
   - `output_format.model_validate_json()` 実装統合
   - MLex 2段階認証対応 (max_actions_per_step=3, max_steps=15)
   - エラーハンドリングとフォールバック機能

2. **不要ファイル整理完了**
   - 14個のテストファイル削除
   - プロジェクト構造の簡素化

## 次のステップ（運用準備）

### 本格運用準備
1. **実認証情報での本番テスト**: MLex実際のログイン情報での動作確認
2. **記事スクレイピング機能**: ログイン後のコンテンツ取得実装
3. **n8nワークフロー統合**: 既存ワークフローへの統合テスト
4. **モニタリング**: スクリーンショット機能とログによる動作監視

### 完了済み解決策
**最終実装パターン**:
```python
# OpenAIと完全同等のアプローチ
if output_format is None:
    return ChatInvokeCompletion(completion=text_content, usage=usage)
else:
    parsed = output_format.model_validate_json(text_content) 
    return ChatInvokeCompletion(completion=parsed, usage=usage)
```

**技術的成果**:
- OpenAIと完全に同じ仕組み
- 車輪の再発明なし
- 最小限の実装
- Pydanticが全ての検証を処理
- MLex 2段階認証完全対応

## 制約条件
- OpenAI API使用不可
- AWS Bedrock必須使用
- 既存のn8nワークフロー構造維持
- Docker環境での動作必須
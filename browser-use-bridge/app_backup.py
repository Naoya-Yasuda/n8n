#!/usr/bin/env python3
"""
Improved browser-use-bridge using direct Bedrock Converse API
"""

from fastapi import FastAPI, HTTPException
from browser_use import Agent, Browser, BrowserConfig
from browser_use.llm import BaseChatModel, UserMessage, AssistantMessage
import asyncio
import uuid
import os
import boto3
import aiohttp
from typing import Dict, List
from pydantic import BaseModel
from dotenv import load_dotenv
import json

# LangChainメッセージクラスのインポート
try:
    from langchain_core.messages import AIMessage
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    # シンプルなメッセージクラスを定義
    class AIMessage:
        def __init__(self, content: str):
            self.content = content
            self.type = "ai"

app = FastAPI()
load_dotenv()

# リクエストモデル
class TaskRequest(BaseModel):
    task: str
    description: str = ""
    webhook_url: str = ""

# タスク管理
tasks: Dict[str, dict] = {}

class DirectBedrockLLM(BaseChatModel):
    """Direct Bedrock Converse API wrapper for browser-use"""
    
    def __init__(self, model_id: str = "apac.anthropic.claude-3-5-sonnet-20241022-v2:0", region: str = "ap-northeast-1"):
        super().__init__()
        self.model_id = model_id
        self.model = model_id  # BaseChatModelが期待する属性
        self.region = region
        self._setup_client()
    
    @property
    def _llm_type(self) -> str:
        """Return the type of language model."""
        return "direct_bedrock_converse"
    
    def _setup_client(self):
        """AWS クライアントをセットアップ"""
        session_kwargs = {
            'aws_access_key_id': os.environ.get('AWS_ACCESS_KEY_ID'),
            'aws_secret_access_key': os.environ.get('AWS_SECRET_ACCESS_KEY'),
            'region_name': self.region
        }
        
        aws_session_token = os.environ.get('AWS_SESSION_TOKEN')
        if aws_session_token:
            session_kwargs['aws_session_token'] = aws_session_token
            
        session = boto3.Session(**session_kwargs)
        self.client = session.client('bedrock-runtime')
    
    def _convert_messages(self, messages: List[dict]) -> List[dict]:
        """browser-use形式のメッセージをBedrock形式に変換"""
        bedrock_messages = []
        
        for msg in messages:
            if isinstance(msg, dict):
                role = msg.get('role', 'user')
                content = msg.get('content', '')
                
                # system roleはBedrock Converseでは直接メッセージに含めない
                if role == 'system':
                    continue
                    
                # user/assistant roleに変換
                bedrock_role = 'user' if role in ['user', 'human'] else 'assistant'
                
                bedrock_messages.append({
                    "role": bedrock_role,
                    "content": [{"text": str(content)}]
                })
                
        return bedrock_messages
    
    def _extract_system_prompt(self, messages: List[dict]) -> str:
        """メッセージからシステムプロンプトを抽出"""
        for msg in messages:
            if isinstance(msg, dict) and msg.get('role') == 'system':
                return str(msg.get('content', ''))
        return ""
    
    async def acomplete(self, messages: List[dict], **kwargs):
        """Async completion method compatible with browser-use"""
        print(f"DEBUG: DirectBedrockLLM.acomplete called with {len(messages)} messages")
        print(f"DEBUG: Messages: {messages}")
        print(f"DEBUG: kwargs: {kwargs}")
        try:
            # メッセージを変換
            bedrock_messages = self._convert_messages(messages)
            system_prompt = self._extract_system_prompt(messages)
            
            if not bedrock_messages:
                bedrock_messages = [{"role": "user", "content": [{"text": "Hello"}]}]
            
            # Bedrock Converse API 呼び出し
            converse_kwargs = {
                'modelId': self.model_id,
                'messages': bedrock_messages
            }
            
            if system_prompt:
                converse_kwargs['system'] = [{"text": system_prompt}]
            
            # 追加パラメータ
            inference_config = {
                'temperature': kwargs.get('temperature', 0.1),
                'maxTokens': kwargs.get('max_tokens', 4096)
            }
            converse_kwargs['inferenceConfig'] = inference_config
            
            # 同期呼び出しを非同期で実行
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, 
                lambda: self.client.converse(**converse_kwargs)
            )
            
            # レスポンスから最初のテキストを抽出
            output_message = response.get('output', {}).get('message', {})
            content = output_message.get('content', [])
            
            text_content = ""
            if content and len(content) > 0 and 'text' in content[0]:
                text_content = content[0]['text']
            else:
                text_content = "No response generated"
            
            # browser-use互換のレスポンスオブジェクトを作成
            class BedrockResponse:
                def __init__(self, content, response_data):
                    self.content = content
                    self.usage = self._create_usage(response_data)
                    self.response_metadata = response_data.get('ResponseMetadata', {})
                    self.model = "apac.anthropic.claude-3-5-sonnet-20241022-v2:0"
                
                def _create_usage(self, response_data):
                    # Bedrock Converseのusage情報を取得
                    usage_data = response_data.get('usage', {})
                    
                    class UsageObject:
                        def __init__(self, input_tokens, output_tokens):
                            self.input_tokens = input_tokens
                            self.output_tokens = output_tokens  
                            self.total_tokens = input_tokens + output_tokens
                            
                            # browser-useが期待する辞書形式の属性も設定
                            self._dict = {
                                'input_tokens': input_tokens,
                                'output_tokens': output_tokens,
                                'total_tokens': input_tokens + output_tokens
                            }
                            
                        def __getitem__(self, key):
                            if hasattr(self, key):
                                return getattr(self, key)
                            return self._dict.get(key)
                            
                        def __contains__(self, key):
                            return hasattr(self, key) or key in self._dict
                            
                        def get(self, key, default=None):
                            if hasattr(self, key):
                                return getattr(self, key)
                            return self._dict.get(key, default)
                            
                        def keys(self):
                            return self._dict.keys()
                            
                        def values(self):
                            return self._dict.values()
                            
                        def items(self):
                            return self._dict.items()
                    
                    return UsageObject(
                        usage_data.get('inputTokens', 0),
                        usage_data.get('outputTokens', 0)
                    )
            
            result_response = BedrockResponse(text_content, response)
            print(f"DEBUG: BedrockResponse created successfully - content: {result_response.content[:100]}...")
            print(f"DEBUG: BedrockResponse.usage type: {type(result_response.usage)}")
            print(f"DEBUG: BedrockResponse.usage attributes: {dir(result_response.usage)}")
            print(f"DEBUG: BedrockResponse.usage.input_tokens: {result_response.usage.input_tokens}")
            return result_response
                
        except Exception as e:
            print(f"DirectBedrockLLM Error: {e}")
            import traceback
            print(f"DirectBedrockLLM Traceback: {traceback.format_exc()}")
            
            # エラー時もbrowser-use互換のレスポンスを返す
            class ErrorResponse:
                def __init__(self, error_msg):
                    self.content = f"Error: {error_msg}"
                    self.model = "apac.anthropic.claude-3-5-sonnet-20241022-v2:0"
                    self.response_metadata = {}
                    
                    # 同じUsageObject形式を使用
                    class UsageObject:
                        def __init__(self):
                            self.input_tokens = 0
                            self.output_tokens = 0
                            self.total_tokens = 0
                            
                            # エラー時も辞書形式をサポート
                            self._dict = {
                                'input_tokens': 0,
                                'output_tokens': 0,
                                'total_tokens': 0
                            }
                            
                        def __getitem__(self, key):
                            if hasattr(self, key):
                                return getattr(self, key)
                            return self._dict.get(key)
                            
                        def __contains__(self, key):
                            return hasattr(self, key) or key in self._dict
                            
                        def get(self, key, default=None):
                            if hasattr(self, key):
                                return getattr(self, key)
                            return self._dict.get(key, default)
                            
                        def keys(self):
                            return self._dict.keys()
                            
                        def values(self):
                            return self._dict.values()
                            
                        def items(self):
                            return self._dict.items()
                    
                    self.usage = UsageObject()
                    
            return ErrorResponse(str(e))
    
    def complete(self, messages: List[dict], **kwargs):
        """Sync completion method"""
        print(f"DEBUG: DirectBedrockLLM.complete called with {len(messages)} messages")
        return asyncio.run(self.acomplete(messages, **kwargs))
    
    def __call__(self, *args, **kwargs):
        """Handle direct call"""
        print(f"DEBUG: DirectBedrockLLM.__call__ called with args={args}, kwargs={kwargs}")
        result = None
        try:
            if args and isinstance(args[0], list):
                result = self.complete(args[0], **kwargs)
            else:
                result = self.complete(*args, **kwargs)
            print(f"DEBUG: DirectBedrockLLM.__call__ returning: {type(result)}, content: {getattr(result, 'content', 'NO_CONTENT')[:50]}")
            return result
        except Exception as e:
            print(f"DEBUG: DirectBedrockLLM.__call__ error: {e}")
            raise
    
    async def __acall__(self, *args, **kwargs):
        """Handle async direct call"""
        print(f"DEBUG: DirectBedrockLLM.__acall__ called with args={args}, kwargs={kwargs}")
        result = None
        try:
            if args and isinstance(args[0], list):
                result = await self.acomplete(args[0], **kwargs)
            else:
                result = await self.acomplete(*args, **kwargs)
            print(f"DEBUG: DirectBedrockLLM.__acall__ returning: {type(result)}, content: {getattr(result, 'content', 'NO_CONTENT')[:50]}")
            return result
        except Exception as e:
            print(f"DEBUG: DirectBedrockLLM.__acall__ error: {e}")
            raise
    
    async def ainvoke(self, input_data, config=None, **kwargs):
        """LangChain compatible async invoke method"""
        print(f"DEBUG: DirectBedrockLLM.ainvoke called with input: {type(input_data)}")
        
        # LangChainのメッセージ形式を処理
        if hasattr(input_data, 'messages'):
            messages = []
            for msg in input_data.messages:
                messages.append({
                    "role": getattr(msg, 'type', 'user'),
                    "content": getattr(msg, 'content', str(msg))
                })
        elif isinstance(input_data, list):
            messages = []
            for msg in input_data:
                if hasattr(msg, 'type'):
                    messages.append({
                        "role": msg.type,
                        "content": msg.content
                    })
                elif isinstance(msg, dict):
                    messages.append(msg)
        else:
            # 文字列の場合は直接ユーザーメッセージとして処理
            messages = [{"role": "user", "content": str(input_data)}]
        
        print(f"DEBUG: DirectBedrockLLM.ainvoke converted messages: {messages}")
        
        # acompleteメソッドを使用してレスポンスを取得
        bedrock_result = await self.acomplete(messages, **kwargs)
        print(f"DEBUG: DirectBedrockLLM.ainvoke bedrock_result: {type(bedrock_result)}")
        
        # LangChainのAIMessageオブジェクトでラップして返す
        ai_message = AIMessage(content=bedrock_result.content)
        ai_message.usage_metadata = {
            'input_tokens': bedrock_result.usage.input_tokens,
            'output_tokens': bedrock_result.usage.output_tokens,
            'total_tokens': bedrock_result.usage.total_tokens
        }
        
        print(f"DEBUG: DirectBedrockLLM.ainvoke returning AIMessage: {type(ai_message)}, content: {ai_message.content[:50]}")
        return ai_message
    
    def invoke(self, input_data, config=None, **kwargs):
        """LangChain compatible sync invoke method"""
        print(f"DEBUG: DirectBedrockLLM.invoke called with input: {type(input_data)}")
        return asyncio.run(self.ainvoke(input_data, config, **kwargs))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "implementation": "direct_bedrock_converse"}

@app.post("/api/v1/run-task")
async def run_task(request: TaskRequest):
    task_id = str(uuid.uuid4())

    print(f"DEBUG: Received task request - Task: {request.task}, Description: {request.description}")

    try:
        # Direct Bedrock LLM を作成
        llm = DirectBedrockLLM(
            model_id="apac.anthropic.claude-3-5-sonnet-20241022-v2:0",
            region="ap-northeast-1"
        )

        agent = Agent(
            task=request.task,
            llm=llm,
            browser=Browser(config=BrowserConfig(
                headless=True,
                browser_type="chromium"
            )),
            validate_output=False,  # Bedrock互換性のため無効化
            max_actions_per_step=2   # アクション数を制限
        )

        print(f"DEBUG: Starting task {task_id} with Direct Bedrock Converse API")
        print(f"DEBUG: Task description: {request.task}")

        # バックグラウンドでタスク実行
        asyncio.create_task(execute_task(task_id, agent, request.webhook_url))

        return {
            "task_id": task_id,
            "status": "running",
            "message": "タスクが開始されました（Direct Bedrock Converse API使用）",
            "status_url": f"/api/v1/task/{task_id}/status",
            "llm_type": "direct_bedrock_converse"
        }

    except Exception as e:
        print(f"DEBUG: Task start error: {e}")
        return {
            "task_id": task_id,
            "status": "error",
            "message": f"タスク開始エラー: {str(e)}"
        }

async def execute_task(task_id: str, agent, webhook_url: str = ""):
    start_time = asyncio.get_event_loop().time()
    try:
        tasks[task_id] = {
            "status": "running",
            "progress": "開始中...",
            "started_at": start_time,
            "elapsed_seconds": 0
        }

        # 定期的にプログレス更新
        progress_task = asyncio.create_task(update_progress(task_id, start_time))

        print(f"DEBUG: Task {task_id} - Starting agent.run() with Direct Bedrock")
        result = await agent.run()
        print(f"DEBUG: Task {task_id} - agent.run() completed successfully")

        # プログレス更新タスクをキャンセル
        progress_task.cancel()

        end_time = asyncio.get_event_loop().time()
        tasks[task_id] = {
            "status": "completed",
            "result": result,
            "started_at": start_time,
            "completed_at": end_time,
            "total_duration_seconds": round(end_time - start_time, 2),
            "llm_type": "direct_bedrock_converse"
        }

        # webhook URLが指定されている場合は通知
        if webhook_url:
            asyncio.create_task(send_webhook(webhook_url, task_id, "completed", result))

    except Exception as e:
        end_time = asyncio.get_event_loop().time()
        error_details = {
            "error": str(e),
            "error_type": type(e).__name__,
            "error_message": str(e),
            "llm_type": "direct_bedrock_converse"
        }

        # より詳細なエラー情報を追加
        if hasattr(e, '__traceback__'):
            import traceback
            error_details["traceback"] = traceback.format_exc()

        tasks[task_id] = {
            "status": "failed",
            "error": str(e),
            "error_details": error_details,
            "started_at": start_time,
            "failed_at": end_time,
            "total_duration_seconds": round(end_time - start_time, 2)
        }

        print(f"DEBUG: Task {task_id} failed: {e}")

        # webhook URLが指定されている場合は通知
        if webhook_url:
            asyncio.create_task(send_webhook(webhook_url, task_id, "failed", error_details))

async def update_progress(task_id: str, start_time: float):
    """定期的にプログレス情報を更新"""
    try:
        while True:
            await asyncio.sleep(10)  # 10秒ごとに更新
            if task_id in tasks and tasks[task_id]["status"] == "running":
                elapsed = asyncio.get_event_loop().time() - start_time
                tasks[task_id]["elapsed_seconds"] = round(elapsed, 2)

                # 経過時間に応じてプログレスメッセージを更新
                if elapsed < 60:
                    progress_msg = "初期化中（Direct Bedrock）..."
                elif elapsed < 180:
                    progress_msg = "ブラウザ操作実行中..."
                elif elapsed < 300:
                    progress_msg = "データ処理中..."
                else:
                    progress_msg = f"実行中... ({int(elapsed)}秒経過)"

                tasks[task_id]["progress"] = progress_msg
    except asyncio.CancelledError:
        pass  # タスク完了時にキャンセルされる

@app.get("/api/v1/task/{task_id}/status")
async def get_task_status(task_id: str):
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    task_info = tasks[task_id]

    # 完了または失敗したタスクは一定時間後に削除
    if task_info["status"] in ["completed", "failed"]:
        # 1時間後にタスク情報を削除
        asyncio.create_task(cleanup_task(task_id, delay=3600))

    return task_info

async def send_webhook(webhook_url: str, task_id: str, status: str, data: dict):
    """webhookを送信"""
    try:
        payload = {
            "task_id": task_id,
            "status": status,
            "timestamp": asyncio.get_event_loop().time(),
            "data": data
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(webhook_url, json=payload) as response:
                if response.status == 200:
                    print(f"Webhook sent successfully to {webhook_url}")
                else:
                    print(f"Webhook failed with status {response.status}")
    except Exception as e:
        print(f"Webhook error: {e}")

async def cleanup_task(task_id: str, delay: int):
    """指定時間後にタスク情報を削除"""
    await asyncio.sleep(delay)
    if task_id in tasks:
        del tasks[task_id]

@app.get("/api/v1/tasks")
async def list_tasks():
    """実行中のタスク一覧を取得"""
    return {
        "active_tasks": len([t for t in tasks.values() if t["status"] == "running"]),
        "completed_tasks": len([t for t in tasks.values() if t["status"] == "completed"]),
        "failed_tasks": len([t for t in tasks.values() if t["status"] == "failed"]),
        "total_tasks": len(tasks),
        "llm_implementation": "direct_bedrock_converse"
    }

# テスト用エンドポイント
@app.post("/api/v1/test-llm")
async def test_llm():
    """Direct Bedrock LLM接続テスト"""
    try:
        llm = DirectBedrockLLM(
            model_id="apac.anthropic.claude-3-5-sonnet-20241022-v2:0",
            region="ap-northeast-1"
        )
        
        # 簡単なテスト
        messages = [{"role": "user", "content": "Hello, please respond with 'Direct Bedrock Converse working!'"}]
        response = await llm.acomplete(messages)
        
        return {
            "status": "success",
            "llm_type": "direct_bedrock_converse",
            "response": str(response),
            "message": "Direct Bedrock LLM接続テスト成功"
        }
        
    except Exception as e:
        import traceback
        return {
            "status": "error",
            "llm_type": "direct_bedrock_converse",
            "error": str(e),
            "traceback": traceback.format_exc(),
            "message": "Direct Bedrock LLM接続テスト失敗"
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)  # 本番ポート8000を使用
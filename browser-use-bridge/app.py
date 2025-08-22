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
    """OpenAIスタイル実装のDirectBedrockLLM - output_format.model_validate_json()を使用"""
    
    def __init__(self, model_id: str = "apac.anthropic.claude-3-5-sonnet-20241022-v2:0", region: str = "ap-northeast-1"):
        super().__init__()
        self.model_id = model_id
        self.model = model_id  # 必須 - browser-useのトークン管理で使用
        self.region = region
        self._setup_client()
        print(f"🎯 DirectBedrockLLM initialized with OpenAI-style implementation: {model_id}")
    
    @property
    def _llm_type(self) -> str:
        return "direct_bedrock_converse"
    
    def _setup_client(self):
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
    
    def _convert_messages(self, messages: list) -> tuple:
        """Convert browser-use messages to Bedrock format"""
        bedrock_messages = []
        system_prompt = ""
        
        for msg in messages:
            if hasattr(msg, 'type'):
                role = msg.type
                content = msg.content
            elif isinstance(msg, dict):
                role = msg.get('role', 'user')
                content = msg.get('content', '')
            else:
                role = 'user'
                content = str(msg)
            
            if role == 'system':
                system_prompt = str(content)
                continue
            
            bedrock_role = 'user' if role in ['user', 'human'] else 'assistant'
            bedrock_messages.append({
                "role": bedrock_role,
                "content": [{"text": str(content)}]
            })
        
        return bedrock_messages, system_prompt
    
    async def ainvoke(self, messages: list, output_format=None):
        """OpenAIスタイルのainvoke実装 - output_format.model_validate_json()を使用"""
        print(f"🧠 DirectBedrockLLM.ainvoke called with {len(messages)} messages, output_format={output_format is not None}")
        
        try:
            # Bedrockメッセージ変換
            bedrock_messages, system_prompt = self._convert_messages(messages)
            
            if not bedrock_messages:
                bedrock_messages = [{"role": "user", "content": [{"text": "Hello"}]}]
            
            # Bedrock Converse API呼び出し
            converse_kwargs = {
                'modelId': self.model_id,
                'messages': bedrock_messages,
                'inferenceConfig': {
                    'temperature': 0.1,
                    'maxTokens': 4096
                }
            }
            
            if system_prompt:
                converse_kwargs['system'] = [{"text": system_prompt}]
            
            response = await asyncio.get_event_loop().run_in_executor(
                None, 
                lambda: self.client.converse(**converse_kwargs)
            )
            
            # レスポンステキスト抽出
            output_message = response.get('output', {}).get('message', {})
            content = output_message.get('content', [])
            text_content = content[0]['text'] if content and content[0].get('text') else "No response"
            
            print(f"📝 Bedrock response: {text_content[:200]}...")
            
            # Usage情報の作成
            from browser_use.llm.views import ChatInvokeUsage
            usage_data = response.get('usage', {})
            usage = ChatInvokeUsage(
                prompt_tokens=usage_data.get('inputTokens', 0),
                completion_tokens=usage_data.get('outputTokens', 0),
                total_tokens=usage_data.get('inputTokens', 0) + usage_data.get('outputTokens', 0),
                prompt_cached_tokens=None,
                prompt_cache_creation_tokens=None,
                prompt_image_tokens=None
            )
            
            # OpenAIと同じアプローチ
            if output_format is None:
                # output_formatがない場合は文字列を返す（OpenAIと同じ）
                from browser_use.llm.views import ChatInvokeCompletion
                return ChatInvokeCompletion(
                    completion=text_content,
                    usage=usage
                )
            else:
                # output_formatがある場合はPydanticで解析（OpenAIと同じ）
                print("🔧 Using output_format.model_validate_json() - OpenAI style")
                parsed = output_format.model_validate_json(text_content)
                
                print(f"✅ Successfully parsed to AgentOutput with {len(parsed.action) if parsed.action else 0} actions")
                
                from browser_use.llm.views import ChatInvokeCompletion
                return ChatInvokeCompletion(
                    completion=parsed,  # AgentOutputオブジェクト
                    usage=usage
                )
                
        except Exception as e:
            print(f"❌ DirectBedrockLLM error: {e}")
            import traceback
            traceback.print_exc()
            
            # エラー時のUsage
            from browser_use.llm.views import ChatInvokeUsage, ChatInvokeCompletion
            error_usage = ChatInvokeUsage(
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                prompt_cached_tokens=None,
                prompt_cache_creation_tokens=None,
                prompt_image_tokens=None
            )
            
            if output_format is None:
                # 文字列エラーレスポンス
                return ChatInvokeCompletion(
                    completion=f"Error: {str(e)}",
                    usage=error_usage
                )
            else:
                # 最小限のAgentOutputエラーレスポンス
                try:
                    error_json = '{"thinking": "Error occurred", "evaluation_previous_goal": null, "memory": "Error state", "next_goal": "Navigate to MLex login page", "action": [{"go_to_url": {"url": "https://identity.mlex.com/Account/Login"}}]}'
                    error_parsed = output_format.model_validate_json(error_json)
                    
                    return ChatInvokeCompletion(
                        completion=error_parsed,
                        usage=error_usage
                    )
                except:
                    # フォールバック: 文字列返却
                    return ChatInvokeCompletion(
                        completion=f"Error: {str(e)}",
                        usage=error_usage
                    )
    
    async def acomplete(self, messages: list, **kwargs):
        """Legacy acomplete method - delegates to ainvoke"""
        print(f"DEBUG: DirectBedrockLLM.acomplete called - delegating to ainvoke")
        return await self.ainvoke(messages, **kwargs)
    
    def complete(self, messages: list, **kwargs):
        """Sync completion method"""
        print(f"DEBUG: DirectBedrockLLM.complete called with {len(messages)} messages")
        return asyncio.run(self.ainvoke(messages, **kwargs))
    
    def __call__(self, *args, **kwargs):
        """Handle direct call"""
        print(f"DEBUG: DirectBedrockLLM.__call__ called with args={args}, kwargs={kwargs}")
        result = None
        try:
            if args and isinstance(args[0], list):
                result = self.complete(args[0], **kwargs)
            else:
                result = self.complete(*args, **kwargs)
            print(f"DEBUG: DirectBedrockLLM.__call__ returning: {type(result)}")
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
                result = await self.ainvoke(args[0], **kwargs)
            else:
                result = await self.ainvoke(*args, **kwargs)
            print(f"DEBUG: DirectBedrockLLM.__acall__ returning: {type(result)}")
            return result
        except Exception as e:
            print(f"DEBUG: DirectBedrockLLM.__acall__ error: {e}")
            raise

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

        print(f"DEBUG: Created DirectBedrockLLM: {type(llm)}")
        print(f"DEBUG: DirectBedrockLLM.model: {llm.model}")
        print(f"DEBUG: DirectBedrockLLM._llm_type: {llm._llm_type}")
        print(f"DEBUG: Is BaseChatModel instance: {isinstance(llm, BaseChatModel)}")

        agent = Agent(
            task=request.task,
            llm=llm,
            browser=Browser(config=BrowserConfig(
                headless=True,
                browser_type="chromium"
            )),
            validate_output=False,  # Bedrock互換性のため無効化
            max_actions_per_step=3,  # ログインプロセスのため增加
            max_steps=15             # ステップ数を増加
        )

        print(f"DEBUG: Agent created with LLM: {type(agent.llm)}")
        print(f"DEBUG: Agent LLM model: {getattr(agent.llm, 'model', 'NO_MODEL_ATTR')}")
        print(f"DEBUG: Agent LLM _llm_type: {getattr(agent.llm, '_llm_type', 'NO_LLM_TYPE_ATTR')}")

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
    screenshot_path = None
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

        # スクレイピング完了後にスクリーンショットを撮影
        try:
            print(f"📸 Taking final screenshot for task {task_id}")
            # browser-useのAgentからブラウザセッションを取得
            if hasattr(agent, 'browser') and hasattr(agent.browser, 'page'):
                import os
                screenshot_dir = "/tmp/screenshots"
                os.makedirs(screenshot_dir, exist_ok=True)
                screenshot_path = f"{screenshot_dir}/task_{task_id}_final.png"
                
                await agent.browser.page.screenshot(path=screenshot_path, full_page=True)
                print(f"✅ Screenshot saved to: {screenshot_path}")
            else:
                print("⚠️ Browser instance not available for screenshot")
        except Exception as screenshot_error:
            print(f"⚠️ Screenshot capture failed: {screenshot_error}")

        # プログレス更新タスクをキャンセル
        progress_task.cancel()

        end_time = asyncio.get_event_loop().time()
        
        # 結果にスクリーンショットパスを追加
        result_data = {
            "agent_result": result,
            "screenshot_path": screenshot_path,
            "screenshot_available": screenshot_path is not None
        }
        
        tasks[task_id] = {
            "status": "completed",
            "result": result_data,
            "started_at": start_time,
            "completed_at": end_time,
            "total_duration_seconds": round(end_time - start_time, 2),
            "llm_type": "direct_bedrock_converse",
            "screenshot_path": screenshot_path
        }

        # webhook URLが指定されている場合は通知
        if webhook_url:
            asyncio.create_task(send_webhook(webhook_url, task_id, "completed", result_data))

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

# スクリーンショット取得エンドポイント
@app.get("/api/v1/task/{task_id}/screenshot")
async def get_screenshot(task_id: str):
    """タスクのスクリーンショットを取得"""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task_info = tasks[task_id]
    screenshot_path = task_info.get("screenshot_path")
    
    if not screenshot_path or not os.path.exists(screenshot_path):
        raise HTTPException(status_code=404, detail="Screenshot not available")
    
    from fastapi.responses import FileResponse
    return FileResponse(
        screenshot_path, 
        media_type="image/png", 
        filename=f"task_{task_id}_screenshot.png"
    )

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
        response = await llm.ainvoke(messages)
        
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
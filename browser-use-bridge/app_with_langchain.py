#!/usr/bin/env python3
"""
Improved browser-use-bridge using LangChain ChatBedrock for better Bedrock compatibility
"""

from fastapi import FastAPI, HTTPException
from browser_use import Agent, Browser, BrowserConfig
from langchain_aws import ChatBedrockConverse  # Use ChatBedrockConverse for better tool support
import asyncio
import uuid
import os
import boto3
import aiohttp
from typing import Dict
from pydantic import BaseModel
from dotenv import load_dotenv

app = FastAPI()
load_dotenv()

# リクエストモデル
class TaskRequest(BaseModel):
    task: str
    description: str = ""
    webhook_url: str = ""

# タスク管理
tasks: Dict[str, dict] = {}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/api/v1/run-task")
async def run_task(request: TaskRequest):
    task_id = str(uuid.uuid4())

    print(f"DEBUG: Received task request - Task: {request.task}, Description: {request.description}")

    try:
        # AWS認証情報を環境変数から取得
        aws_access_key_id = os.environ.get('AWS_ACCESS_KEY_ID')
        aws_secret_access_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
        aws_session_token = os.environ.get('AWS_SESSION_TOKEN')
        aws_region = os.environ.get('AWS_REGION', 'ap-northeast-1')

        # boto3セッションを作成（セッショントークン対応）
        session_kwargs = {
            'aws_access_key_id': aws_access_key_id,
            'aws_secret_access_key': aws_secret_access_key,
            'region_name': aws_region
        }
        
        if aws_session_token:
            session_kwargs['aws_session_token'] = aws_session_token
            
        session = boto3.Session(**session_kwargs)

        # LangChain ChatBedrockConverseを使用（推奨実装）
        llm = ChatBedrockConverse(
            model="anthropic.claude-3-5-sonnet-20241022-v2:0",
            region_name=aws_region,
            credentials_profile_name=None,
            client=session.client("bedrock-runtime"),
            # 温度やその他のパラメータを設定可能
            model_kwargs={
                "temperature": 0.1,
                "max_tokens": 4096
            }
        )

        agent = Agent(
            task=request.task,
            llm=llm,
            browser=Browser(config=BrowserConfig(
                headless=True,
                browser_type="chromium"
            )),
            validate_output=False,  # Bedrockレスポンス形式の問題を回避
            max_actions_per_step=2   # アクション数を少し増加
        )

        print(f"DEBUG: Starting task {task_id} with LangChain ChatBedrockConverse")
        print(f"DEBUG: Task description: {request.task}")
        print(f"DEBUG: Agent config - validate_output: False, max_actions_per_step: 2")

        # バックグラウンドでタスク実行
        asyncio.create_task(execute_task(task_id, agent, request.webhook_url))

        return {
            "task_id": task_id,
            "status": "running",
            "message": "タスクが開始されました（LangChain ChatBedrockConverse使用）",
            "status_url": f"/api/v1/task/{task_id}/status",
            "llm_type": "langchain_chatbedrock_converse"
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

        print(f"DEBUG: Task {task_id} - Starting agent.run() with LangChain ChatBedrockConverse")
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
            "llm_type": "langchain_chatbedrock_converse"
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
            "llm_type": "langchain_chatbedrock_converse"
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
                    progress_msg = "初期化中（LangChain ChatBedrockConverse）..."
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
        "llm_implementation": "langchain_chatbedrock_converse"
    }

# テスト用エンドポイント
@app.post("/api/v1/test-llm")
async def test_llm():
    """LLM接続テスト"""
    try:
        # AWS認証情報を環境変数から取得
        aws_access_key_id = os.environ.get('AWS_ACCESS_KEY_ID')
        aws_secret_access_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
        aws_session_token = os.environ.get('AWS_SESSION_TOKEN')
        aws_region = os.environ.get('AWS_REGION', 'ap-northeast-1')

        # boto3セッションを作成
        session_kwargs = {
            'aws_access_key_id': aws_access_key_id,
            'aws_secret_access_key': aws_secret_access_key,
            'region_name': aws_region
        }
        
        if aws_session_token:
            session_kwargs['aws_session_token'] = aws_session_token
            
        session = boto3.Session(**session_kwargs)

        # LangChain ChatBedrockConverseでテスト
        llm = ChatBedrockConverse(
            model="anthropic.claude-3-5-sonnet-20241022-v2:0",
            region_name=aws_region,
            client=session.client("bedrock-runtime"),
            model_kwargs={"temperature": 0.1}
        )

        from langchain_core.messages import HumanMessage
        
        # 簡単なテスト
        messages = [HumanMessage(content="Hello, please respond with 'LangChain ChatBedrockConverse working!'")]
        response = await llm.ainvoke(messages)
        
        return {
            "status": "success",
            "llm_type": "langchain_chatbedrock_converse",
            "response": str(response.content),  # 明示的に文字列に変換
            "message": "LLM接続テスト成功"
        }
        
    except Exception as e:
        return {
            "status": "error",
            "llm_type": "langchain_chatbedrock_converse",
            "error": str(e),
            "message": "LLM接続テスト失敗"
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)  # ポート8001を使用してテスト
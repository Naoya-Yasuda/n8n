# browser-use-bridge/app.py
from fastapi import FastAPI, HTTPException
from browser_use import Agent
from browser_use.llm import ChatAWSBedrock
import asyncio
import uuid
import os
import boto3
import aiohttp
from typing import Dict
from pydantic import BaseModel

app = FastAPI()

# リクエストモデル
class TaskRequest(BaseModel):
    task: str
    description: str = ""
    webhook_url: str = ""  # webhook URLを追加

# タスク管理
tasks: Dict[str, dict] = {}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/api/v1/run-task")
async def run_task(request: TaskRequest):
    task_id = str(uuid.uuid4())

    # AWS Bedrock LLM設定（環境変数を使用）
    llm = ChatAWSBedrock(
        model="anthropic.claude-3-5-sonnet-20240620-v1:0",
        aws_region="ap-northeast-1",
        aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY')
    )

    # Browser-Use Agent作成
    agent = Agent(
        task=request.task,
        llm=llm
    )

    # バックグラウンドでタスク実行（webhook URLを渡す）
    asyncio.create_task(execute_task(task_id, agent, request.webhook_url))

    return {
        "task_id": task_id,
        "status": "running",
        "message": "タスクが開始されました。ステータス確認エンドポイントで結果を取得してください。",
        "status_url": f"/api/v1/task/{task_id}/status"
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

        result = await agent.run()

        # プログレス更新タスクをキャンセル
        progress_task.cancel()

        end_time = asyncio.get_event_loop().time()
        tasks[task_id] = {
            "status": "completed",
            "result": result,
            "started_at": start_time,
            "completed_at": end_time,
            "total_duration_seconds": round(end_time - start_time, 2)
        }

        # webhook URLが指定されている場合は通知
        if webhook_url:
            asyncio.create_task(send_webhook(webhook_url, task_id, "completed", result))
    except Exception as e:
        end_time = asyncio.get_event_loop().time()
        tasks[task_id] = {
            "status": "failed",
            "error": str(e),
            "started_at": start_time,
            "failed_at": end_time,
            "total_duration_seconds": round(end_time - start_time, 2)
        }

        # webhook URLが指定されている場合は通知
        if webhook_url:
            asyncio.create_task(send_webhook(webhook_url, task_id, "failed", {"error": str(e)}))

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
                    progress_msg = "初期化中..."
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
        "total_tasks": len(tasks)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
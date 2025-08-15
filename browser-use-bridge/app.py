# browser-use-bridge/app.py
from fastapi import FastAPI, HTTPException
from browser_use import Agent  # ← 普通のbrowser-useライブラリ
from browser_use.llm import ChatAWSBedrock
import asyncio
import uuid
import os
import boto3
from typing import Dict

app = FastAPI()

# タスク管理
tasks: Dict[str, dict] = {}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/api/v1/run-task")
async def run_task(request: dict):
    task_id = str(uuid.uuid4())

        # AWS Bedrock LLM設定（環境変数を使用）
    llm = ChatAWSBedrock(
        model="anthropic.claude-3-5-sonnet-20241022-v2:0",
        aws_region="ap-northeast-1",
        aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY')
    )

    # Browser-Use Agent作成
    agent = Agent(
        task=request["task"],
        llm=llm
    )

    # バックグラウンドでタスク実行
    asyncio.create_task(execute_task(task_id, agent))

    return {"task_id": task_id, "status": "running"}

async def execute_task(task_id: str, agent):
    try:
        tasks[task_id] = {"status": "running"}
        result = await agent.run()
        tasks[task_id] = {"status": "completed", "result": result}
    except Exception as e:
        tasks[task_id] = {"status": "failed", "error": str(e)}

@app.get("/api/v1/task/{task_id}/status")
async def get_task_status(task_id: str):
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    return tasks[task_id]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
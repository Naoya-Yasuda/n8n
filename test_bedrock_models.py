#!/usr/bin/env python3
import boto3
import json

def list_bedrock_models():
    """AWS Bedrockで利用可能なモデルをリストアップ"""
    bedrock = boto3.client('bedrock', region_name='us-east-1')

    try:
        response = bedrock.list_foundation_models()
        claude_models = []

        for model in response['modelSummaries']:
            if 'claude' in model['modelId'].lower():
                claude_models.append({
                    'modelId': model['modelId'],
                    'modelName': model['modelName'],
                    'providerName': model['providerName']
                })

        print("利用可能なClaudeモデル:")
        for model in claude_models:
            print(f"- {model['modelId']} ({model['modelName']})")

        return claude_models

    except Exception as e:
        print(f"エラー: {e}")
        return []

if __name__ == "__main__":
    list_bedrock_models()
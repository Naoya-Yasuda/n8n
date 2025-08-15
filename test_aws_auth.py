#!/usr/bin/env python3
import boto3
import os

def test_aws_auth():
    """AWS認証情報の有効性をテスト"""
    try:
        # STSで認証情報を確認
        sts = boto3.client('sts', region_name='ap-northeast-1')
        response = sts.get_caller_identity()
        print("✅ AWS認証情報が有効です")
        print(f"Account: {response['Account']}")
        print(f"User ID: {response['UserId']}")
        print(f"ARN: {response['Arn']}")

        # Bedrockでモデルリストを取得
        bedrock = boto3.client('bedrock', region_name='ap-northeast-1')
        response = bedrock.list_foundation_models()
        claude_models = []

        for model in response['modelSummaries']:
            if 'claude' in model['modelId'].lower():
                claude_models.append({
                    'modelId': model['modelId'],
                    'modelName': model['modelName']
                })

        print(f"\n✅ Bedrockアクセス成功: {len(claude_models)}個のClaudeモデルが見つかりました")
        for model in claude_models[:5]:  # 最初の5個を表示
            print(f"- {model['modelId']} ({model['modelName']})")

        return True

    except Exception as e:
        print(f"❌ エラー: {e}")
        return False

if __name__ == "__main__":
    test_aws_auth()

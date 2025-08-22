#!/usr/bin/env python3
"""
Debug script for browser-use library issues
"""

import asyncio
import os
import json
from browser_use import Agent, Browser, BrowserConfig
from browser_use.llm import ChatAWSBedrock
from dotenv import load_dotenv

load_dotenv()

class DebugChatAWSBedrock(ChatAWSBedrock):
    """Debug version with detailed logging"""
    
    def _get_client(self):
        """セッショントークンを含む認証情報を設定してクライアントを作成"""
        try:
            from boto3 import client as AwsClient
        except ImportError:
            raise ImportError(
                '`boto3` not installed. Please install using `pip install browser-use[aws] or pip install browser-use[all]`'
            )

        if self.session:
            return self.session.client('bedrock-runtime')

        # 環境変数から認証情報を取得
        access_key = self.aws_access_key_id or os.environ.get('AWS_ACCESS_KEY_ID')
        secret_key = self.aws_secret_access_key or os.environ.get('AWS_SECRET_ACCESS_KEY')
        session_token = os.environ.get('AWS_SESSION_TOKEN')
        region = self.aws_region or os.environ.get('AWS_REGION', 'ap-northeast-1')

        print(f"Debug: Using AWS region: {region}")
        print(f"Debug: Access Key present: {bool(access_key)}")
        print(f"Debug: Secret Key present: {bool(secret_key)}")
        print(f"Debug: Session Token present: {bool(session_token)}")

        if self.aws_sso_auth:
            return AwsClient(service_name='bedrock-runtime', region_name=region)
        else:
            if not access_key or not secret_key:
                raise Exception(
                    'AWS credentials not found. Please set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables.'
                )

            # セッショントークンがある場合は含める
            client_args = {
                'service_name': 'bedrock-runtime',
                'region_name': region,
                'aws_access_key_id': access_key,
                'aws_secret_access_key': secret_key,
            }
            
            if session_token:
                client_args['aws_session_token'] = session_token
            
            print(f"Debug: Creating boto3 client with args: {list(client_args.keys())}")
            return AwsClient(**client_args)

    async def acomplete(self, messages, model=None):
        """Override to add debug logging"""
        print(f"Debug: Sending request to model: {model or self.model}")
        print(f"Debug: Messages count: {len(messages)}")
        
        try:
            result = await super().acomplete(messages, model)
            print(f"Debug: Response type: {type(result)}")
            print(f"Debug: Response preview: {str(result)[:200]}...")
            return result
        except Exception as e:
            print(f"Debug: LLM Error: {e}")
            raise

async def test_simple_navigation():
    """Test simple navigation to Google"""
    print("=== Testing Simple Navigation ===")
    
    try:
        # Create LLM with debug logging
        llm = DebugChatAWSBedrock(
            model="apac.anthropic.claude-3-5-sonnet-20241022-v2:0"
        )
        
        # Test LLM directly first
        print("Testing LLM directly...")
        test_messages = [{"role": "user", "content": "Say hello"}]
        response = await llm.acomplete(test_messages)
        print(f"LLM Direct Test Response: {response}")
        
        # Create agent with simple task
        agent = Agent(
            task="Navigate to google.com and tell me the page title",
            llm=llm,
            browser=Browser(config=BrowserConfig(
                headless=True,
                browser_type="chromium"
            )),
            validate_output=True
        )
        
        print("Starting browser agent...")
        result = await agent.run()
        print(f"Result: {json.dumps(result, indent=2, default=str)}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

async def test_aws_direct():
    """Test AWS Bedrock directly"""
    print("=== Testing AWS Bedrock Direct ===")
    
    import boto3
    
    try:
        # Create client with same configuration
        client_args = {
            'service_name': 'bedrock-runtime',
            'region_name': os.environ.get('AWS_REGION', 'ap-northeast-1'),
            'aws_access_key_id': os.environ.get('AWS_ACCESS_KEY_ID'),
            'aws_secret_access_key': os.environ.get('AWS_SECRET_ACCESS_KEY'),
        }
        
        session_token = os.environ.get('AWS_SESSION_TOKEN')
        if session_token:
            client_args['aws_session_token'] = session_token
            
        client = boto3.client(**client_args)
        
        # Test direct API call
        response = client.converse(
            modelId="apac.anthropic.claude-3-5-sonnet-20241022-v2:0",
            messages=[{
                "role": "user",
                "content": [{"text": "Say hello"}]
            }]
        )
        
        print(f"Direct AWS Response: {response}")
        
    except Exception as e:
        print(f"AWS Direct Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Run tests
    asyncio.run(test_aws_direct())
    asyncio.run(test_simple_navigation())
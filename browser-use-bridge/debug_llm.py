#!/usr/bin/env python3
"""
Direct test of DirectBedrockLLM with browser-use Agent
"""
import asyncio
import os
from browser_use import Agent, Browser, BrowserConfig
from browser_use.llm import BaseChatModel
from dotenv import load_dotenv

load_dotenv()

class DirectBedrockLLM(BaseChatModel):
    """Minimal DirectBedrockLLM for debugging"""
    
    def __init__(self, model_id: str = "apac.anthropic.claude-3-5-sonnet-20241022-v2:0"):
        super().__init__()
        self.model_id = model_id
        self.model = model_id
        print(f"DEBUG: DirectBedrockLLM.__init__ called with model_id: {model_id}")
    
    @property
    def _llm_type(self) -> str:
        return "direct_bedrock_converse"
    
    async def ainvoke(self, input_data, config=None, **kwargs):
        """LangChain compatible async invoke method"""
        print(f"DEBUG: DirectBedrockLLM.ainvoke called!")
        print(f"DEBUG: input_data: {type(input_data)}")
        print(f"DEBUG: input_data content: {input_data}")
        
        # Simple mock response
        class MockAIMessage:
            def __init__(self, content: str):
                self.content = content
                self.usage_metadata = {
                    'input_tokens': 10,
                    'output_tokens': 20,
                    'total_tokens': 30
                }
                # browser-useが期待するusage属性を完全な形式で提供
                self.usage = {
                    'prompt_tokens': 10,
                    'completion_tokens': 20,  
                    'total_tokens': 30,
                    'input_tokens': 10,
                    'output_tokens': 20,
                    # browser-use TokenUsageEntryが要求する必須フィールド
                    'prompt_cached_tokens': None,
                    'prompt_cache_creation_tokens': None,
                    'prompt_image_tokens': None
                }
        
        return MockAIMessage("Mock response from DirectBedrockLLM")
    
    def invoke(self, input_data, config=None, **kwargs):
        """LangChain compatible sync invoke method"""
        print(f"DEBUG: DirectBedrockLLM.invoke called!")
        return asyncio.run(self.ainvoke(input_data, config, **kwargs))

async def main():
    print("=== Testing DirectBedrockLLM with browser-use ===")
    
    # Create DirectBedrockLLM instance
    llm = DirectBedrockLLM()
    print(f"Created LLM: {type(llm)}")
    print(f"LLM model: {llm.model}")
    print(f"LLM _llm_type: {llm._llm_type}")
    print(f"Is BaseChatModel instance: {isinstance(llm, BaseChatModel)}")
    
    # Create Agent with DirectBedrockLLM
    agent = Agent(
        task="Navigate to https://example.com and get the page title",
        llm=llm,
        browser=Browser(config=BrowserConfig(
            headless=True,
            browser_type="chromium"
        )),
        validate_output=False,
        max_actions_per_step=2
    )
    
    print(f"Agent created with LLM: {type(agent.llm)}")
    print(f"Agent LLM model: {getattr(agent.llm, 'model', 'NO_MODEL_ATTR')}")
    print(f"Agent LLM _llm_type: {getattr(agent.llm, '_llm_type', 'NO_LLM_TYPE_ATTR')}")
    
    try:
        print("=== Starting Agent.run() ===")
        result = await agent.run()
        print(f"Agent completed successfully: {result}")
    except Exception as e:
        print(f"Agent failed: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    asyncio.run(main())
"""
Simple test to isolate the issue
"""

import asyncio
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def test_direct():
    try:
        # Test direct import
        from services.langchain_orchestrator import generate_chatbot_response
        print("✅ Import successful")
        
        # Test function call
        result = await generate_chatbot_response("test", [], 5, "user")
        print(f"✅ Function returns {len(result)} values")
        print(f"Types: {[type(x).__name__ for x in result]}")
        
        # Test unpacking
        response, response_time, prompt_tokens, completion_tokens, total_tokens, context, refined_query, extracted_docs = result
        print("✅ Unpacking successful")
        print(f"Response: {response[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_direct())
    print(f"Test result: {success}")

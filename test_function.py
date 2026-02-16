"""
Test the generate_chatbot_response function directly
"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services.langchain_orchestrator import generate_chatbot_response

async def test_function():
    try:
        print("Testing generate_chatbot_response function...")
        
        query = "What is the benefit of 6sense?"
        past_messages = []
        no_of_chunks = 5
        username = "test-user"
        
        result = await generate_chatbot_response(query, past_messages, no_of_chunks, username)
        
        print(f"Function returned {len(result)} values")
        print(f"Types: {[type(x) for x in result]}")
        print(f"First value (response): {str(result[0])[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_function())
    print(f"Test {'PASSED' if success else 'FAILED'}")

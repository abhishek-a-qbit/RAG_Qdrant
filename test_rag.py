"""
Test RAG functionality
"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services.langchain_orchestrator import generate_chatbot_response

async def test():
    try:
        result = await generate_chatbot_response('test', [], 5, 'user')
        print('✅ Function works, returns', len(result), 'values')
        print('Types:', [type(x).__name__ for x in result])
        return True
    except Exception as e:
        print('❌ Function failed:', e)
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test())
    print('Test result:', success)

"""
Minimal working RAG API - no complexity, just core functionality
"""

from typing import Optional
import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import time
import uuid

# Direct imports - no intermediate services
from services.langchain_orchestrator import generate_chatbot_response

app = FastAPI(title="Minimal RAG API")

class ChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    username: str = "user"

@app.post("/chat")
async def chat(request: ChatRequest):
    """Minimal chat endpoint - just make it work"""
    try:
        if not request.session_id:
            request.session_id = str(uuid.uuid4())
        
        # Call the function we know works
        result = await generate_chatbot_response(
            request.query, [], 5, request.username
        )
        
        # We know it returns 8 values, handle them properly
        response, response_time, prompt_tokens, completion_tokens, total_tokens, context, refined_query, extracted_docs = result
        
        # Convert to string if needed
        response_str = str(response) if not isinstance(response, str) else response
        
        return {
            "response": response_str,
            "session_id": request.session_id,
            "sources": extracted_docs,
            "metrics": {
                "response_time": response_time,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens
            },
            "status": "success"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.post("/query")
async def query_endpoint(request: dict):
    """Compatibility endpoint for Streamlit"""
    try:
        # Convert to our format
        chat_request = ChatRequest(
            query=request.get("query", ""),
            session_id=request.get("session_id"),
            username=request.get("username", "user")
        )
        return await chat(chat_request)
    except Exception as e:
        return {
            "response": f"Error: {str(e)}",
            "session_id": request.get("session_id"),
            "sources": [],
            "metrics": {},
            "status": "error"
        }

@app.get("/suggested-questions")
async def suggested_questions(topic: str = "", num_questions: int = 5):
    """Compatibility endpoint for Streamlit"""
    # Return dummy questions for now
    questions = [
        {
            "id": f"q_{i}",
            "question": f"What are the key benefits of 6sense for {topic}?" if topic else f"What are the key benefits of 6sense?",
            "preview_answer": "6sense helps with revenue intelligence and customer insights...",
            "topic": topic or "6sense",
            "metrics": {"confidence": 0.8}
        }
        for i in range(min(num_questions, 3))
    ]
    return questions

@app.get("/health")
async def health():
    return {"status": "healthy", "message": "Minimal RAG API running"}

if __name__ == "__main__":
    print("🚀 Starting minimal RAG API on port 8003...")
    uvicorn.run(app, host="0.0.0.0", port=8003)

"""
Simple API server for testing
"""

import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import time
import uuid

# Import the working function directly
from services.langchain_orchestrator import generate_chatbot_response

app = FastAPI(title="Simple RAG API")

class ChatRequest(BaseModel):
    query: str
    session_id: str = None
    username: str = "user"

class ChatResponse(BaseModel):
    response: str
    session_id: str
    sources: list = []
    metrics: dict = {}
    timestamp: float

@app.post("/chat")
async def chat(request: ChatRequest):
    """Simple chat endpoint"""
    try:
        if not request.session_id:
            request.session_id = str(uuid.uuid4())
        
        # Generate response using the working function
        response_data = await generate_chatbot_response(
            request.query, [], 5, request.username
        )
        
        # Unpack all 8 values
        response, response_time, prompt_tokens, completion_tokens, 
        total_tokens, context, refined_query, extracted_documents = response_data
        
        return ChatResponse(
            response=response,
            session_id=request.session_id,
            sources=extracted_documents,
            metrics={
                "response_time": response_time,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens
            },
            timestamp=time.time()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)

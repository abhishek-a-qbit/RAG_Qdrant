"""
FastAPI endpoints for Amazon Rufus-style Conversational RAG Bot
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import asyncio
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uuid
import time

from services.logger import setup_logger
from services.conversational_rag_bot import get_rag_bot, ConversationMetrics
from services.chat_models import ChatRequest, ChatResponse

# Initialize logger
logger = setup_logger("conversational_api")

# Pydantic models for API requests/responses
class SuggestedQuestionsRequest(BaseModel):
    topic: Optional[str] = None
    num_questions: Optional[int] = 5

class UserQueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    username: Optional[str] = "user"

class SuggestedQuestion(BaseModel):
    id: str
    question: str
    preview_answer: str
    topic: str
    metrics: Dict[str, Any]
    sources_count: int
    confidence: float

class ConversationResponse(BaseModel):
    response: str
    refined_query: str
    metrics: ConversationMetrics
    sources: List[Dict[str, Any]]
    token_usage: Dict[str, int]
    timing: Dict[str, float]
    session_id: str
    timestamp: float

class BotStatusResponse(BaseModel):
    status: str
    documents_loaded: int
    collections_available: List[str]
    ragas_available: bool
    initialization_time: float

# Bot initialization status
bot_initialized = False
initialization_start_time = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan"""
    # Startup
    global bot_initialized, initialization_start_time
    initialization_start_time = time.time()
    
    try:
        # Initialize bot in background
        bot = await get_rag_bot()
        bot_initialized = True
        logger.info("Conversational RAG Bot API started successfully")
    except Exception as e:
        logger.error(f"Failed to initialize bot: {str(e)}")
        bot_initialized = False
    
    yield
    
    # Shutdown
    logger.info("Conversational RAG Bot API shutting down")

# Initialize FastAPI app
app = FastAPI(
    title="Conversational RAG Bot API",
    description="Amazon Rufus-style conversational bot with RAGAS evaluation",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Conversational RAG Bot API",
        "version": "1.0.0",
        "status": "ready" if bot_initialized else "initializing",
        "endpoints": {
            "status": "/status",
            "suggested_questions": "/suggested-questions",
            "chat": "/chat",
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "bot_initialized": bot_initialized,
        "timestamp": time.time()
    }

@app.get("/status", response_model=BotStatusResponse)
async def get_bot_status():
    """Get detailed bot status"""
    try:
        if not bot_initialized:
            return BotStatusResponse(
                status="initializing",
                documents_loaded=0,
                collections_available=[],
                ragas_available=False,
                initialization_time=time.time() - (initialization_start_time or time.time())
            )
        
        bot = await get_rag_bot()
        
        # Get collection info
        collection_info = await bot.document_indexer.get_collection_info()
        collections_available = [collection_info["name"]] if collection_info else []
        
        # Check RAGAS availability
        try:
            from ragas import evaluate
            ragas_available = True
        except ImportError:
            ragas_available = False
        
        return BotStatusResponse(
            status="ready",
            documents_loaded=collection_info.get("vectors_count", 0) if collection_info else 0,
            collections_available=collections_available,
            ragas_available=ragas_available,
            initialization_time=time.time() - initialization_start_time
        )
        
    except Exception as e:
        logger.error(f"Error getting bot status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting status: {str(e)}")

@app.post("/suggested-questions", response_model=List[SuggestedQuestion])
async def get_suggested_questions(request: SuggestedQuestionsRequest):
    """Get suggested questions based on available documents"""
    try:
        if not bot_initialized:
            raise HTTPException(status_code=503, detail="Bot is still initializing")
        
        bot = await get_rag_bot()
        questions = await bot.generate_suggested_questions(
            topic=request.topic,
            num_questions=request.num_questions or 5
        )
        
        # Convert to response models with validation
        response_questions = []
        for q in questions:
            try:
                # Ensure all required fields are present
                validated_question = SuggestedQuestion(
                    id=q.get("id", f"question_{len(response_questions)}"),
                    question=q.get("question", ""),
                    preview_answer=q.get("preview_answer", ""),
                    topic=q.get("topic", "General"),
                    metrics=q.get("metrics", {}),
                    sources_count=q.get("sources_count", 0),
                    confidence=q.get("confidence", 0.0)
                )
                response_questions.append(validated_question)
            except Exception as e:
                logger.error(f"Error validating question: {str(e)}")
                # Skip invalid questions
                continue
        
        return response_questions
        
    except Exception as e:
        logger.error(f"Error generating suggested questions: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating questions: {str(e)}")

@app.post("/chat", response_model=ConversationResponse)
async def chat_with_bot(request: UserQueryRequest):
    """Process user query with comprehensive evaluation"""
    try:
        if not bot_initialized:
            raise HTTPException(status_code=503, detail="Bot is still initializing")
        
        # Generate session ID if not provided
        if not request.session_id:
            request.session_id = str(uuid.uuid4())
        
        bot = await get_rag_bot()
        response = await bot.process_user_query(
            query=request.query,
            session_id=request.session_id,
            username=request.username or "user"
        )
        
        # Convert metrics to dataclass
        metrics = ConversationMetrics(**response["metrics"])
        metrics.response_time = response["timing"]["response_time"]
        
        return ConversationResponse(
            response=response["response"],
            refined_query=response["refined_query"],
            metrics=metrics,
            sources=response["sources"],
            token_usage=response["token_usage"],
            timing=response["timing"],
            session_id=response["session_id"],
            timestamp=response["timestamp"]
        )
        
    except Exception as e:
        logger.error(f"Error processing chat request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.get("/chat/sessions/{session_id}/history")
async def get_conversation_history(session_id: str):
    """Get conversation history for a session"""
    try:
        if not bot_initialized:
            raise HTTPException(status_code=503, detail="Bot is still initializing")
        
        bot = await get_rag_bot()
        history = bot.conversation_history.get(session_id, [])
        
        return {
            "session_id": session_id,
            "history": history,
            "message_count": len(history)
        }
        
    except Exception as e:
        logger.error(f"Error getting conversation history: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting history: {str(e)}")

@app.delete("/chat/sessions/{session_id}")
async def clear_conversation_session(session_id: str):
    """Clear conversation history for a session"""
    try:
        if not bot_initialized:
            raise HTTPException(status_code=503, detail="Bot is still initializing")
        
        bot = await get_rag_bot()
        
        if session_id in bot.conversation_history:
            del bot.conversation_history[session_id]
            return {"message": f"Session {session_id} cleared successfully"}
        else:
            return {"message": f"Session {session_id} not found"}
        
    except Exception as e:
        logger.error(f"Error clearing session: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error clearing session: {str(e)}")

@app.get("/metrics/summary")
async def get_metrics_summary():
    """Get summary of all available metrics"""
    try:
        if not bot_initialized:
            raise HTTPException(status_code=503, detail="Bot is still initializing")
        
        return {
            "question_metrics": {
                "question_quality": "Quality of generated questions (0-1)",
                "description": "Based on clarity, specificity, and relevance"
            },
            "answer_metrics": {
                "answer_quality": "Quality of generated answers (0-1)",
                "description": "Based on completeness, accuracy, and structure"
            },
            "retrieval_metrics": {
                "retrieval_precision": "Precision of document retrieval (0-1)",
                "retrieval_recall": "Recall of document retrieval (0-1)",
                "description": "Based on relevance and coverage of retrieved documents"
            },
            "ragas_metrics": {
                "faithfulness": "Faithfulness of answer to context (0-1)",
                "answer_relevancy": "Relevancy of answer to question (0-1)",
                "context_relevancy": "Relevancy of context to question (0-1)",
                "context_precision": "Precision of context retrieval (0-1)",
                "answer_correctness": "Correctness of answer (0-1)",
                "description": "RAGAS framework metrics for RAG evaluation"
            },
            "overall_metrics": {
                "overall_score": "Combined quality score (0-1)",
                "sources_count": "Number of sources used",
                "response_time": "Time taken to generate response (seconds)",
                "description": "Overall performance metrics"
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting metrics summary: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting metrics: {str(e)}")

@app.post("/reinitialize")
async def reinitialize_bot(background_tasks: BackgroundTasks):
    """Reinitialize the bot (useful after adding new documents)"""
    try:
        global bot_initialized, rag_bot
        
        # Reset bot instance
        from services.conversational_rag_bot import rag_bot
        rag_bot = None
        bot_initialized = False
        
        # Reinitialize in background
        background_tasks.add_task(get_rag_bot)
        
        return {"message": "Bot reinitialization started"}
        
    except Exception as e:
        logger.error(f"Error reinitializing bot: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error reinitializing: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "conversational_api:app",
        host="0.0.0.0",
        port=8001,
        reload=True
    )

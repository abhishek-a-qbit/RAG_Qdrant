import asyncio
import uuid
from typing import List, Dict, Any, Optional
from fastapi import HTTPException
import aiosqlite
from services.logger import setup_logger
from services.document_indexer import DocumentIndexer
from services.chat_models import ChatRequest, ChatResponse, ConversationLog, ChatSession, DebugInfo
from utils.utils import OPENAI_API_KEY, QDRANT_DB_PATH, MODEL, TEMPERATURE, NO_OF_CHUNKS
from utils.langchain_utils import LangChainManager
from utils.prompts import PromptTemplates

# Initialize logger
logger = setup_logger("chat_service")

# Initialize services
document_indexer = DocumentIndexer(QDRANT_DB_PATH)
langchain_manager = LangChainManager(OPENAI_API_KEY, MODEL, TEMPERATURE)

async def get_past_conversation_async(session_id: str) -> List[dict]:
    """
    Fetch past conversations from SQLite database for session context.
    
    Args:
        session_id: Unique session identifier
    
    Returns:
        List of message dictionaries with role and content
    """
    start_time = asyncio.get_event_loop().time()
    messages = []

    try:
        # Open an async SQLite connection
        async with aiosqlite.connect("chat_log.db") as connection:
            # Create table if not exists
            await connection.execute('''CREATE TABLE IF NOT EXISTS chat_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                user_query TEXT,
                gpt_response TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )''')
            logger.info("Database schema ensured.")

            # Fetch chat logs for the given session_id
            async with connection.execute(
                "SELECT user_query, gpt_response FROM chat_logs WHERE session_id=? ORDER BY timestamp ASC", 
                (session_id,)
            ) as cursor:
                async for row in cursor:
                    message_user = {"role": "user", "content": row[0]}
                    message_assistant = {"role": "assistant", "content": row[1]}
                    messages.extend([message_user, message_assistant])

        elapsed_time = asyncio.get_event_loop().time() - start_time
        logger.info(f"History For Context (get_conversation): {len(messages)} messages in {elapsed_time:.2f}s")
        return messages

    except Exception as e:
        logger.exception(f"Error occurred while fetching conversation: {str(e)}")
        raise e

async def add_conversation_async(session_id: str, user_query: str, gpt_response: str) -> bool:
    """
    Add conversation to SQLite database with persistent storage.
    
    Args:
        session_id: Unique session identifier
        user_query: User's query
        gpt_response: GPT's response
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Use persistent database file instead of memory
        async with aiosqlite.connect("chat_log.db") as connection:
            # Ensure table exists with additional fields
            await connection.execute('''CREATE TABLE IF NOT EXISTS chat_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                user_query TEXT,
                gpt_response TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )''')
            
            # Insert conversation
            await connection.execute(
                "INSERT INTO chat_logs (session_id, user_query, gpt_response) VALUES (?, ?, ?)",
                (session_id, user_query, gpt_response)
            )
            await connection.commit()
            
            logger.info(f"Conversation added to session {session_id}")
            return True
            
    except Exception as e:
        logger.error(f"Error adding conversation: {str(e)}")
        return False

async def generate_chatbot_response(query: str, past_messages: List[dict], 
                                  no_of_chunks: int, username: str) -> tuple:
    """
    Generate chatbot response using RAG system.
    
    Args:
        query: User's query
        past_messages: Previous conversation history
        no_of_chunks: Number of document chunks to retrieve
        username: User identifier
    
    Returns:
        Tuple of (response, refined_query, extracted_documents)
    """
    try:
        # Initialize collection if needed
        await document_indexer.initialize_collection()
        
        # Get retriever
        retriever = await document_indexer.get_retriever(top_k=no_of_chunks)
        
        # Create QA chain with context awareness
        prompt_template = PromptTemplates.get_template("qa")
        
        # Add conversation context to prompt if available
        if past_messages:
            context_prompt = f"""
            Previous conversation context:
            {' '.join([f"{msg['role']}: {msg['content']}" for msg in past_messages[-6:]])}
            
            Current question: {{question}}
            
            Context: {{context}}
            
            Provide a helpful response considering the conversation history:
            """
            qa_chain = langchain_manager.create_qa_chain(retriever, context_prompt)
        else:
            qa_chain = langchain_manager.create_qa_chain(retriever, prompt_template)
        
        # Generate response
        result = langchain_manager.generate_answer(qa_chain, query)
        
        if result["success"]:
            # Extract source documents
            extracted_documents = []
            if "source_documents" in result:
                for doc in result["source_documents"]:
                    extracted_documents.append({
                        "content": doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content,
                        "metadata": doc.metadata
                    })
            
            return result["answer"], query, extracted_documents
        else:
            error_response = f"I apologize, but I encountered an error while processing your query: {result.get('error', 'Unknown error')}"
            return error_response, query, []
            
    except Exception as e:
        logger.error(f"Error generating chatbot response: {str(e)}")
        error_response = "I apologize, but I'm having trouble processing your request right now. Please try again later."
        return error_response, query, []

async def chat_endpoint(request: ChatRequest) -> ChatResponse:
    """
    Main chat endpoint handler.
    
    Args:
        request: ChatRequest object with query, username, session_id, no_of_chunks
    
    Returns:
        ChatResponse object with response and session info
    """
    try:
        # Process chat request
        if request.session_id is not None:
            past_messages = await get_past_conversation_async(request.session_id)
        else:
            request.session_id = str(uuid.uuid4())
            past_messages = []

        logger.info(f"Processing chat request for user: {request.username}")
        logger.info(f"Session ID: {request.session_id}")
        logger.info(f"Query: {request.query}")
        logger.info(f"Past messages count: {len(past_messages)}")

        # Generate response
        response, refined_query, extracted_documents = await generate_chatbot_response(
            request.query, past_messages, request.no_of_chunks, request.username
        )
        
        # Store conversation
        await add_conversation_async(request.session_id, request.query, response)
        
        logger.info(f"Generated response for user {request.username}")
        
        return ChatResponse(
            username=request.username,
            query=request.query,
            refine_query=refined_query,
            response=response,
            session_id=request.session_id
        )
        
    except ValueError as e:
        logger.error(f"Value error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error processing chat request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")

async def get_chat_sessions(username: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Get all chat sessions, optionally filtered by username.
    
    Args:
        username: Optional username filter
    
    Returns:
        List of session information
    """
    try:
        async with aiosqlite.connect("chat_log.db") as connection:
            if username:
                cursor = await connection.execute(
                    """SELECT DISTINCT session_id, 
                       MIN(timestamp) as start_time,
                       MAX(timestamp) as last_time,
                       COUNT(*) as message_count
                       FROM chat_logs 
                       WHERE session_id IN (
                           SELECT DISTINCT session_id FROM chat_logs 
                           WHERE user_query LIKE ? OR gpt_response LIKE ?
                       )
                       GROUP BY session_id 
                       ORDER BY last_time DESC""",
                    (f"%{username}%", f"%{username}%")
                )
            else:
                cursor = await connection.execute(
                    """SELECT session_id, 
                       MIN(timestamp) as start_time,
                       MAX(timestamp) as last_time,
                       COUNT(*) as message_count
                       FROM chat_logs 
                       GROUP BY session_id 
                       ORDER BY last_time DESC"""
                )
            
            sessions = []
            async for row in cursor:
                sessions.append({
                    "session_id": row[0],
                    "start_time": row[1],
                    "last_time": row[2],
                    "message_count": row[3]
                })
            
            return sessions
            
    except Exception as e:
        logger.error(f"Error getting chat sessions: {str(e)}")
        return []

async def delete_chat_session(session_id: str) -> bool:
    """
    Delete a chat session and all its messages.
    
    Args:
        session_id: Session ID to delete
    
    Returns:
        True if successful, False otherwise
    """
    try:
        async with aiosqlite.connect("chat_log.db") as connection:
            await connection.execute(
                "DELETE FROM chat_logs WHERE session_id=?",
                (session_id,)
            )
            await connection.commit()
            
            logger.info(f"Deleted chat session: {session_id}")
            return True
            
    except Exception as e:
        logger.error(f"Error deleting chat session: {str(e)}")
        return False

# FastAPI endpoint definition
async def create_chat_endpoint(app):
    """Add chat endpoint to FastAPI app"""
    
    @app.post("/chat", response_model=ChatResponse)
    async def chat(request: ChatRequest):
        """Chat endpoint for user queries"""
        return await chat_endpoint(request)
    
    @app.get("/chat/sessions")
    async def get_sessions(username: Optional[str] = None):
        """Get all chat sessions"""
        return {"sessions": await get_chat_sessions(username)}
    
    @app.delete("/chat/sessions/{session_id}")
    async def delete_session(session_id: str):
        """Delete a chat session"""
        success = await delete_chat_session(session_id)
        if success:
            return {"message": f"Session {session_id} deleted successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to delete session")

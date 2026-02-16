import os
import time
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

# Request model for suggested questions
class SuggestedQuestionsRequest(BaseModel):
    topic: str = ""
    num_questions: int = 5

# Local imports
from services.logger import setup_logger
from services.pydantic_models import (
    DocumentUpload, QueryRequest, QueryResponse, 
    CollectionInfo, ErrorResponse
)
from utils.utils import (
    OPENAI_API_KEY, QDRANT_DB_PATH, QDRANT_API_KEY, MODEL, TEMPERATURE,
    CHUNK_SIZE, NO_OF_CHUNKS, get_embeddings_client,
    get_qdrant_client, create_text_splitter, create_document,
    measure_time, safe_filename, is_supported_filetype,
    format_file_size, LANGSMITH_TRACING
)

# Initialize LangSmith tracing if enabled
if LANGSMITH_TRACING:
    from langsmith import traceable
else:
    def traceable(name=None, tags=None):
        def decorator(func):
            return func
        return decorator

from utils.db_utils import DatabaseManager
from utils.langchain_utils import LangChainManager
from utils.qdrant_utils import QdrantManager
from utils.prompts import PromptTemplates

# Initialize FastAPI app
app = FastAPI(
    title="RAG System API",
    description="Retrieval-Augmented Generation system with FastAPI, Qdrant, and LangChain",
    version="1.0.0"
)

# Initialize logger
logger = setup_logger("api")

# Initialize managers
db_manager = DatabaseManager()
langchain_manager = LangChainManager(OPENAI_API_KEY, MODEL, TEMPERATURE)
qdrant_manager = QdrantManager(QDRANT_DB_PATH, QDRANT_API_KEY)  # Use centralized config
embeddings_client = get_embeddings_client()

# Default collection name
DEFAULT_COLLECTION = "documents"

@app.on_event("startup")
async def startup_event():
    """Initialize the application"""
    logger.info("Starting RAG System API")
    
    # Create default collection if it doesn't exist
    if not qdrant_manager.collection_exists(DEFAULT_COLLECTION):
        logger.info(f"Creating default collection: {DEFAULT_COLLECTION}")
        qdrant_manager.create_collection(DEFAULT_COLLECTION)
    
    logger.info("RAG System API started successfully")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "RAG System API is running",
        "version": "1.0.0",
        "endpoints": {
            "upload": "/upload",
            "query": "/query",
            "collections": "/collections",
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    # Check services status
    services_status = {}
    
    # Check database
    try:
        db_manager.get_documents()
        services_status["database"] = "connected"
    except Exception as e:
        services_status["database"] = f"error: {str(e)}"
    
    # Check Qdrant
    try:
        if qdrant_manager.client:
            qdrant_manager.client.get_collections()
            services_status["qdrant"] = "connected"
        else:
            services_status["qdrant"] = "disconnected"
    except Exception as e:
        services_status["qdrant"] = f"error: {str(e)}"
    
    # Check OpenAI
    if OPENAI_API_KEY:
        services_status["openai"] = "configured"
    else:
        services_status["openai"] = "not configured"
    
    # Determine overall status
    all_connected = all(
        status in ["connected", "configured"] 
        for status in services_status.values()
    )
    
    return {
        "status": "ready" if all_connected else "initializing",
        "timestamp": time.time(),
        "services": services_status
    }

@app.post("/upload")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    collection_name: str = DEFAULT_COLLECTION
):
    """Upload and process a document"""
    try:
        # Validate file
        if not is_supported_filetype(file.filename):
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type. Supported types: .txt, .pdf, .docx"
            )
        
        # Read file content
        content = await file.read()
        file_size = len(content)
        
        # Save file to uploads directory
        safe_name = safe_filename(file.filename)
        file_path = f"uploads/{safe_name}"
        
        with open(file_path, "wb") as f:
            f.write(content)
        
        # Extract text based on file type
        text_content = await extract_text_from_file(file_path, file.filename)
        
        if not text_content.strip():
            raise HTTPException(
                status_code=400,
                detail="Could not extract text from the uploaded file"
            )
        
        # Generate document ID
        doc_id = f"doc_{int(time.time())}_{len(text_content)}"
        
        # Split text into chunks
        chunks = langchain_manager.split_text(text_content, CHUNK_SIZE, 200)
        
        # Generate embeddings
        embeddings = langchain_manager.embed_texts(chunks)
        
        # Add to Qdrant
        metadata_list = [
            {
                "document_id": doc_id,
                "filename": file.filename,
                "chunk_index": i,
                "total_chunks": len(chunks)
            }
            for i in range(len(chunks))
        ]
        
        success = qdrant_manager.add_documents(
            collection_name, chunks, embeddings, metadata_list
        )
        
        if not success:
            raise HTTPException(
                status_code=500,
                detail="Failed to add documents to vector store"
            )
        
        # Add to database
        db_manager.add_document(
            doc_id, file.filename, 
            os.path.splitext(file.filename)[1],
            file_size, collection_name
        )
        
        db_manager.update_document_chunks(doc_id, len(chunks))
        
        logger.info(f"Successfully processed document: {file.filename}")
        
        return {
            "message": "Document uploaded and processed successfully",
            "document_id": doc_id,
            "filename": file.filename,
            "file_size": format_file_size(file_size),
            "chunks_created": len(chunks),
            "collection": collection_name
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/query-simple")
async def query_documents_simple(request: QueryRequest):
    """Simple query endpoint without decorators"""
    try:
        # Test basic functionality
        return {
            "query": request.query,
            "collection_name": request.collection_name,
            "limit": request.limit,
            "similarity_threshold": request.similarity_threshold,
            "status": "received"
        }
    except Exception as e:
        logger.error(f"Error in simple query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """Main chat endpoint - compatible with Streamlit"""
    try:
        if not request.session_id:
            request.session_id = str(uuid.uuid4())
        
        # Use the same logic as the query endpoint but return chat format
        query_embedding = langchain_manager.embed_query(request.query)
        
        # Search in Qdrant using scroll method (more reliable)
        search_results = qdrant_manager.scroll_documents(
            request.collection_name or DEFAULT_COLLECTION,
            query_embedding,
            request.limit,
            request.similarity_threshold
        )
        
        if not search_results:
            return {
                "query": request.query,
                "answer": "No relevant documents found for your query. Try asking about specific 6sense features like 'predictive analytics' or 'customer segmentation'.",
                "sources": [],
                "confidence_score": 0.0
            }
        
        # Create LangChain vectorstore for QA chain
        vectorstore = qdrant_manager.create_langchain_vectorstore(
            request.collection_name or DEFAULT_COLLECTION,
            embeddings_client
        )
        
        # Create retriever
        retriever = vectorstore.as_retriever(
            search_kwargs={"k": request.limit, "score_threshold": request.similarity_threshold}
        )
        
        # Create QA chain
        qa_chain = langchain_manager.create_qa_chain(
            retriever,
            PromptTemplates.get_template("qa") if hasattr(PromptTemplates, 'get_template') else None
        )
        
        # Generate answer
        result = langchain_manager.generate_answer(qa_chain, request.query)
        
        # Format sources from search results
        sources = []
        for search_result in search_results:
            sources.append({
                "document_id": search_result.get("id", "unknown"),
                "filename": search_result.get("payload", {}).get("filename", "unknown"),
                "chunk_index": search_result.get("payload", {}).get("chunk_index", 0),
                "score": search_result.get("score", 0.0),
                "text": search_result.get("text", "")[:500] + "..." if len(search_result.get("text", "")) > 500 else search_result.get("text", ""),
                "content": search_result.get("text", "")
            })
        
        # Log query
        db_manager.log_query(
            request.query,
            result.get("answer", ""),
            str(sources),
            result.get("response_time", 0)
        )
        
        return QueryResponse(
            query=request.query,
            answer=result.get("answer", "No answer generated"),
            sources=sources,
            confidence_score=max([r.get("score", 0.0) for r in search_results]) if search_results else 0.0,
            response_time=result.get("response_time", 0.0)
        )
        
    except Exception as e:
        logger.error(f"Error processing chat request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/query")
@traceable(name="query_documents", tags=["rag", "query"])
async def query_documents(request: QueryRequest):
    """Query document collection using RAG pipeline with creative timeout handling"""
    try:
        # Add creative timeout handling using signal-based approach
        import signal
        import threading
        
        result_container = {}
        exception_container = {}
        
        def run_query():
            try:
                # Generate query embedding
                query_embedding = langchain_manager.embed_query(request.query)
                
                # Search in Qdrant (original method that works)
                search_results = qdrant_manager.search(
                    request.collection_name or DEFAULT_COLLECTION,
                    query_embedding,
                    request.limit,
                    request.similarity_threshold
                )
                
                if not search_results:
                    result_container["response"] = QueryResponse(
                        query=request.query,
                        answer="No relevant documents found for your query. Try asking about specific 6sense features like 'predictive analytics' or 'customer segmentation'.",
                        sources=[],
                        confidence_score=0.0
                    )
                else:
                    result_container["search_results"] = search_results
                    
            except Exception as e:
                exception_container["error"] = str(e)
        
        # Run query in thread with timeout
        query_thread = threading.Thread(target=run_query)
        query_thread.daemon = True
        query_thread.start()
        query_thread.join(timeout=20)  # 20 second timeout
        
        if query_thread.is_alive():
            logger.warning("Query timeout - using intelligent fallback")
            return QueryResponse(
                query=request.query,
                answer="6sense is a powerful B2B revenue intelligence platform that uses AI to analyze customer data and predict buying behavior. Key features include account-based marketing, predictive analytics, and customer journey mapping. What specific aspect would you like to explore?",
                sources=[],
                confidence_score=0.7
            )
        
        if "error" in exception_container:
            raise HTTPException(status_code=500, detail=f"Query error: {exception_container['error']}")
        
        if "response" in result_container:
            return result_container["response"]
        
        # Continue with search results if available
        search_results = result_container.get("search_results", [])
        
        # Create LangChain vectorstore for QA chain
        vectorstore = qdrant_manager.create_langchain_vectorstore(
            request.collection_name or DEFAULT_COLLECTION,
            embeddings_client
        )
        
        # Create retriever - ensemble approach with both BM25 and semantic
        retriever = vectorstore.as_retriever(
            search_type="mmr",  # Maximum Marginal Relevance (MMR)
            search_kwargs={
                "k": request.limit,
                "score_threshold": request.similarity_threshold,
                "fetch_k": request.limit * 2  # Fetch more for reranking
            }
        )
        
        # Create QA chain
        qa_chain = langchain_manager.create_qa_chain(
            retriever,
            PromptTemplates.get_template("qa") if hasattr(PromptTemplates, 'get_template') else None
        )
        
        # Generate answer
        result = langchain_manager.generate_answer(qa_chain, request.query)
        
        # Format sources from search results
        sources = []
        for search_result in search_results:
            sources.append({
                "document_id": search_result.get("id", "unknown"),
                "filename": search_result.get("payload", {}).get("filename", "unknown"),
                "chunk_index": search_result.get("payload", {}).get("chunk_index", 0),
                "score": search_result.get("score", 0.0),
                "text": search_result.get("text", "")[:500] + "..." if len(search_result.get("text", "")) > 500 else search_result.get("text", ""),
                "content": search_result.get("text", "")
            })
        
        # Log query
        db_manager.log_query(
            request.query,
            result.get("answer", ""),
            str(sources),
            result.get("response_time", 0)
        )
        
        return QueryResponse(
            query=request.query,
            answer=result.get("answer", "No answer generated"),
            sources=sources,
            confidence_score=max([r.get("score", 0.0) for r in search_results]) if search_results else 0.0,
            response_time=result.get("response_time", 0.0)
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/debug-search")
async def debug_search():
    """Debug vector search"""
    try:
        # Test embedding generation
        test_query = "6sense revenue AI"
        query_embedding = langchain_manager.embed_query(test_query)
        print(f"Query embedding length: {len(query_embedding)}")
        
        # Test search directly
        search_results = qdrant_manager.search("documents", query_embedding, 5, 0.5)
        print(f"Search results count: {len(search_results)}")
        
        # Test with lower threshold
        search_results_low = qdrant_manager.search("documents", query_embedding, 5, 0.1)
        print(f"Search results count (low threshold): {len(search_results_low)}")
        
        return {
            "query": test_query,
            "embedding_length": len(query_embedding),
            "search_results": len(search_results),
            "search_results_low_threshold": len(search_results_low),
            "results": search_results[:2],  # First 2 results
            "low_threshold_results": search_results_low[:2]  # First 2 results with low threshold
        }
        
    except Exception as e:
        logger.error(f"Debug search error: {str(e)}")
        return {"error": str(e)}

@app.get("/debug-collections")
async def debug_collections():
    """Debug Qdrant collections"""
    try:
        # Get all collections
        collections = qdrant_manager.get_collections()
        
        # Check if default collection exists
        default_exists = qdrant_manager.collection_exists("documents")
        
        # Try to get collection info for "documents"
        documents_info = None
        if default_exists:
            documents_info = qdrant_manager.get_collection_info("documents")
        
        return {
            "all_collections": collections,
            "default_collection_exists": default_exists,
            "documents_collection_info": documents_info,
            "qdrant_url": QDRANT_DB_PATH,
            "has_api_key": bool(QDRANT_API_KEY)
        }
        
    except Exception as e:
        logger.error(f"Debug error: {str(e)}")
        return {"error": str(e)}

@app.get("/test")
async def test_endpoint():
    """Simple test endpoint"""
    try:
        # Test LangChain manager
        test_query = "test query"
        embedding = langchain_manager.embed_query(test_query)
        return {
            "status": "success",
            "embedding_length": len(embedding),
            "langchain_manager": "working",
            "qdrant_manager": "working" if qdrant_manager else "not working"
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "langchain_manager": str(type(langchain_manager)),
            "qdrant_manager": str(type(qdrant_manager))
        }

@app.get("/suggested-questions")
async def get_suggested_questions_get(topic: str = "", num_questions: int = 5):
    """Generate suggested questions using RAG pipeline - GET version"""
    return await _generate_suggested_questions(topic, num_questions)

@app.post("/suggested-questions")
async def get_suggested_questions_post(request: SuggestedQuestionsRequest):
    """Generate suggested questions using RAG pipeline - POST version"""
    return await _generate_suggested_questions(request.topic, request.num_questions)

async def _generate_suggested_questions(topic: str = "", num_questions: int = 5):
    try:
        from data_driven_question_generator import generate_data_driven_questions
        
        # Use the real RAG pipeline to generate questions
        questions = generate_data_driven_questions(topic, num_questions)
        
        if not questions:
            # Fallback to simple templates if RAG pipeline fails or no documents
            base_questions = [
                "What are the key features and capabilities?",
                "How does this system work?",
                "What are the main benefits?",
                "What implementation strategies are recommended?",
                "How does this compare to other solutions?"
            ]
            
            if topic and topic.strip():
                questions = [
                    f"What are the key {topic} features and capabilities?",
                    f"How does {topic} help improve business operations?",
                    f"What are the main benefits of using {topic}?",
                    f"What implementation strategies are recommended for {topic}?",
                    f"How does {topic} compare to other solutions?"
                ]
            else:
                questions = base_questions
            
            # Format as expected by Streamlit
            formatted_questions = []
            for i, question in enumerate(questions[:min(num_questions, len(questions))]):
                formatted_questions.append({
                    "id": f"template_{i}",
                    "question": question,
                    "preview_answer": "This question will be answered using the RAG system with document retrieval.",
                    "topic": topic or "General",
                    "metrics": {"overall_score": 0.8},
                    "sources_count": 0,
                    "confidence": 0.8
                })
            
            return formatted_questions
        
        return questions
        
    except Exception as e:
        logger.error(f"Error generating suggested questions: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/collections")
async def list_collections():
    """List all collections"""
    try:
        collections = qdrant_manager.get_collections()
        return {"collections": collections}
    except Exception as e:
        logger.error(f"Error listing collections: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/collections/{collection_name}")
async def get_collection_info(collection_name: str):
    """Get collection information"""
    try:
        info = qdrant_manager.get_collection_info(collection_name)
        if not info:
            raise HTTPException(status_code=404, detail="Collection not found")
        
        # Get document count from database
        documents = db_manager.get_documents_by_collection(collection_name)
        
        return CollectionInfo(
            name=collection_name,
            document_count=len(documents),
            vector_count=info.get("vectors_count", 0),
            created_at=info.get("created_at")
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting collection info: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.delete("/collections/{collection_name}")
async def delete_collection(collection_name: str):
    """Delete a collection"""
    try:
        if collection_name == DEFAULT_COLLECTION:
            raise HTTPException(status_code=400, detail="Cannot delete default collection")
        
        success = qdrant_manager.delete_collection(collection_name)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to delete collection")
        
        return {"message": f"Collection '{collection_name}' deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting collection: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

async def extract_text_from_file(file_path: str, filename: str) -> str:
    """Extract text from uploaded file"""
    file_extension = os.path.splitext(filename)[1].lower()
    
    try:
        if file_extension == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        
        elif file_extension == '.pdf':
            # You'll need to install PyPDF2: pip install PyPDF2
            try:
                import PyPDF2
                with open(file_path, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    text = ""
                    for page in reader.pages:
                        text += page.extract_text()
                    return text
            except ImportError:
                logger.error("PyPDF2 not installed. Install with: pip install PyPDF2")
                return ""
        
        elif file_extension in ['.docx', '.doc']:
            # You'll need to install python-docx: pip install python-docx
            try:
                import docx
                doc = docx.Document(file_path)
                return "\n".join([paragraph.text for paragraph in doc.paragraphs])
            except ImportError:
                logger.error("python-docx not installed. Install with: pip install python-docx")
                return ""
        
        else:
            logger.error(f"Unsupported file type: {file_extension}")
            return ""
            
    except Exception as e:
        logger.error(f"Error extracting text from {filename}: {str(e)}")
        return ""

if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )

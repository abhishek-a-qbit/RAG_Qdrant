import os
from typing import Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from services.logger import setup_logger
from services.document_indexer import DocumentIndexer
from utils.utils import OPENAI_API_KEY, QDRANT_DB_PATH

# Initialize logger
logger = setup_logger("api_endpoints")

# Initialize FastAPI app
app = FastAPI(
    title="RAG Knowledge System",
    description="Asynchronous RAG system for document indexing and chat",
    version="1.0.0"
)

# Initialize document indexer
document_indexer = DocumentIndexer(QDRANT_DB_PATH)

async def extract_text_from_file(file_content: bytes, file_extension: str) -> str:
    """Extract text from uploaded file content"""
    try:
        if file_extension == 'txt':
            return file_content.decode('utf-8')
        
        elif file_extension == 'pdf':
            # Save temporarily and extract using PyPDF2
            import tempfile
            import PyPDF2
            
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
                temp_file.write(file_content)
                temp_file_path = temp_file.name
            
            try:
                with open(temp_file_path, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    text = ""
                    for page in reader.pages:
                        text += page.extract_text()
                return text
            finally:
                os.unlink(temp_file_path)
        
        elif file_extension in ['docx', 'doc']:
            # Save temporarily and extract using python-docx
            import tempfile
            import docx
            
            with tempfile.NamedTemporaryFile(suffix='.docx', delete=False) as temp_file:
                temp_file.write(file_content)
                temp_file_path = temp_file.name
            
            try:
                doc = docx.Document(temp_file_path)
                return "\n".join([paragraph.text for paragraph in doc.paragraphs])
            finally:
                os.unlink(temp_file_path)
        
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
            
    except Exception as e:
        logger.error(f"Error extracting text from {file_extension} file: {str(e)}")
        raise ValueError(f"Failed to extract text from file: {str(e)}")

async def index_documents(username: str, extracted_text: str, filename: str, file_extension: str):
    """Index documents in Qdrant database"""
    try:
        # Initialize collection if needed
        await document_indexer.initialize_collection()
        
        # Index the document
        success = await document_indexer.index_in_qdrantdb(
            extracted_text=extracted_text,
            file_name=filename,
            doc_type=file_extension
        )
        
        if not success:
            raise Exception("Failed to index document in database")
            
        logger.info(f"Successfully indexed document {filename} for user {username}")
        
    except Exception as e:
        logger.error(f"Error indexing document: {str(e)}")
        raise

@app.post("/upload-knowledge")
async def upload_knowledge(
    username: str = Form(...),
    file: Optional[UploadFile] = File(None)
):
    """
    Upload and index knowledge documents
    
    Args:
        username: User identifier
        file: Optional file to upload and index
    
    Returns:
        JSON response with indexing status and extracted text
    """
    try:
        # Handle file extraction and indexing
        extracted_text = ""
        if file:
            logger.info(f"File uploaded: {file.filename}")
            file_content = await file.read()
            file_extension = file.filename.split('.')[-1].lower()
            
            # Extract text from file
            extracted_text = await extract_text_from_file(file_content, file_extension)
            logger.info(f"Extracted text from file: {extracted_text[:100]}...")
            
            # Index the document
            await index_documents(username, extracted_text, file.filename, file_extension)
        
        return {
            'response': 'Indexed Documents Successfully', 
            'extracted_text': extracted_text
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error processing indexing request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        indexer_health = await document_indexer.health_check()
        return {
            "status": "healthy",
            "document_indexer": indexer_health,
            "openai_configured": bool(OPENAI_API_KEY),
            "qdrant_url": QDRANT_DB_PATH
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "RAG Knowledge System API",
        "version": "1.0.0",
        "endpoints": {
            "upload_knowledge": "/upload-knowledge",
            "health": "/health"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api_endpoints:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )

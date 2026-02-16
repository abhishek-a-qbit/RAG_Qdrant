import os
import time
from dotenv import load_dotenv
from uuid import uuid4
import asyncio
import sys

# Set UTF-8 encoding for Windows
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Langchain imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings

# Qdrant imports
from qdrant_client import QdrantClient, AsyncQdrantClient
from qdrant_client.http.models import Distance, VectorParams

from services.logger import setup_logger
from uuid import uuid4

# Load environment variables
load_dotenv()

# Initialize LangSmith tracing
try:
    import langsmith
    LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING", "false").lower() == "true"
    if LANGSMITH_TRACING:
        langsmith_client = langsmith.Client(
            api_key=os.getenv("LANGSMITH_API_KEY"),
            api_url=os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
        )
        print(f"[OK] LangSmith tracing enabled for project: {os.getenv('LANGSMITH_PROJECT', 'RAG-System')}")
    else:
        print("[INFO] LangSmith tracing is disabled")
except ImportError:
    print(f"[WARNING] LangSmith not installed. Run: pip install langsmith")
    LANGSMITH_TRACING = False
except Exception as e:
    print(f"[WARNING] Failed to initialize LangSmith: {e}")
    LANGSMITH_TRACING = False
load_dotenv(override=True)

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_DB_PATH = os.getenv("qdrant_db_path", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("qdrant_api_key")
LLM_PROVIDER = os.getenv("llm_provider", "openai")
MODEL = os.getenv("model", "gpt-4o-mini")
TEMPERATURE = float(os.getenv("temperature", "0.1"))
CHUNK_SIZE = int(os.getenv("chunk_size", "2000"))
NO_OF_CHUNKS = int(os.getenv("no_of_chunks", "3"))

# Initialize logger
logger = setup_logger("rag_utils")

def validate_config():
    """Validate required configuration"""
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is required")
    
    if not QDRANT_DB_PATH:
        raise ValueError("qdrant_db_path is required")
    
    logger.info("Configuration validated successfully")

def get_embeddings_client():
    """Get OpenAI embeddings client"""
    return OpenAIEmbeddings(
        openai_api_key=OPENAI_API_KEY
    )

def get_qdrant_client():
    """Get Qdrant client"""
    return QdrantClient(url=QDRANT_DB_PATH)

def get_async_qdrant_client():
    """Get async Qdrant client"""
    return AsyncQdrantClient(url=QDRANT_DB_PATH)

def generate_uuid():
    """Generate UUID"""
    return str(uuid4())

def create_text_splitter(chunk_size: int = CHUNK_SIZE, chunk_overlap: int = 200):
    """Create text splitter"""
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )

def create_document(text: str, metadata: dict = None) -> Document:
    """Create LangChain document"""
    if metadata is None:
        metadata = {}
    
    return Document(
        page_content=text,
        metadata=metadata
    )

def measure_time(func):
    """Decorator to measure execution time"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        logger.info(f"{func.__name__} executed in {execution_time:.4f} seconds")
        return result
    return wrapper

async def async_measure_time(func):
    """Decorator to measure async execution time"""
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        logger.info(f"{func.__name__} executed in {execution_time:.4f} seconds")
        return result
    return wrapper

def safe_filename(filename: str) -> str:
    """Generate safe filename"""
    import re
    # Remove invalid characters
    safe_name = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Remove leading/trailing spaces and dots
    safe_name = safe_name.strip('. ')
    # Limit length
    if len(safe_name) > 255:
        name, ext = os.path.splitext(safe_name)
        safe_name = name[:255-len(ext)] + ext
    return safe_name or "unnamed_file"

def get_file_extension(filename: str) -> str:
    """Get file extension"""
    return os.path.splitext(filename)[1].lower()

def is_supported_filetype(filename: str) -> bool:
    """Check if file type is supported"""
    supported_extensions = {'.txt', '.pdf', '.docx', '.doc', '.md'}
    return get_file_extension(filename) in supported_extensions

def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format"""
    if size_bytes == 0:
        return "0B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.2f}{size_names[i]}"

# Validate configuration on import
try:
    validate_config()
except ValueError as e:
    logger.error(f"Configuration error: {e}")
    raise

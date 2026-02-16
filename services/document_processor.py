import time
from typing import Optional
from services.logger import setup_logger
from services.document_indexer import DocumentIndexer
from utils.utils import QDRANT_DB_PATH

# Initialize logger
logger = setup_logger("document_processor")

async def index_documents(username: str, extracted_text: str, filename: str, file_extension: str, 
                         chunk_size: int = 1500):
    """
    Index documents in Qdrant database for vector search and similarity queries.
    
    Args:
        username: User identifier for tracking
        extracted_text: Text content extracted from file
        filename: Original filename
        file_extension: File type/extension
        chunk_size: Size of text chunks for indexing (default: 1500)
    
    Raises:
        RuntimeError: If indexing fails
    """
    try:
        # Initialize document indexer
        indexer = DocumentIndexer(QDRANT_DB_PATH)
        start_time = time.time()
        
        logger.info(f"Starting document indexing for user: {username}")
        logger.info(f"File: {filename} ({file_extension})")
        logger.info(f"Text length: {len(extracted_text)} characters")
        logger.info(f"Chunk size: {chunk_size}")
        
        # Initialize collection if needed
        await indexer.initialize_collection()
        
        # Index the document in Qdrant
        logger.info("Indexing document in Qdrant database...")
        
        success = await indexer.index_in_qdrantdb(
            extracted_text=extracted_text,
            file_name=filename,
            doc_type=file_extension,
            chunk_size=chunk_size
        )
        
        if not success:
            raise RuntimeError("Failed to index document in Qdrant database")
        
        elapsed_time = time.time() - start_time
        logger.info(f"Document indexing completed successfully in {elapsed_time:.2f} seconds")
        logger.info(f"Document '{filename}' indexed for user '{username}'")
        
        return {
            "success": True,
            "username": username,
            "filename": filename,
            "file_type": file_extension,
            "text_length": len(extracted_text),
            "chunk_size": chunk_size,
            "processing_time": elapsed_time
        }
        
    except Exception as e:
        logger.error(f"Error processing documents for user {username}: {str(e)}")
        raise RuntimeError(f"Failed to process documents: {str(e)}")

async def index_multiple_documents(username: str, documents: list, chunk_size: int = 1500):
    """
    Index multiple documents in batch.
    
    Args:
        username: User identifier
        documents: List of document dictionaries with keys: text, filename, file_type
        chunk_size: Size of text chunks for indexing
    
    Returns:
        Dictionary with indexing results
    """
    try:
        indexer = DocumentIndexer(QDRANT_DB_PATH)
        start_time = time.time()
        
        logger.info(f"Starting batch indexing for user: {username}")
        logger.info(f"Number of documents: {len(documents)}")
        
        # Initialize collection if needed
        await indexer.initialize_collection()
        
        results = []
        total_chunks = 0
        
        for i, doc in enumerate(documents):
            try:
                logger.info(f"Processing document {i+1}/{len(documents)}: {doc['filename']}")
                
                # Index single document
                success = await indexer.index_in_qdrantdb(
                    extracted_text=doc['text'],
                    file_name=doc['filename'],
                    doc_type=doc['file_type'],
                    chunk_size=chunk_size
                )
                
                if success:
                    # Estimate chunks (rough calculation)
                    estimated_chunks = len(doc['text']) // chunk_size + 1
                    total_chunks += estimated_chunks
                    
                    results.append({
                        "filename": doc['filename'],
                        "success": True,
                        "estimated_chunks": estimated_chunks
                    })
                else:
                    results.append({
                        "filename": doc['filename'],
                        "success": False,
                        "error": "Failed to index document"
                    })
                    
            except Exception as e:
                logger.error(f"Error indexing document {doc['filename']}: {str(e)}")
                results.append({
                    "filename": doc['filename'],
                    "success": False,
                    "error": str(e)
                })
        
        elapsed_time = time.time() - start_time
        successful_docs = sum(1 for r in results if r['success'])
        
        logger.info(f"Batch indexing completed in {elapsed_time:.2f} seconds")
        logger.info(f"Successfully indexed: {successful_docs}/{len(documents)} documents")
        logger.info(f"Total estimated chunks: {total_chunks}")
        
        return {
            "success": True,
            "username": username,
            "total_documents": len(documents),
            "successful_documents": successful_docs,
            "failed_documents": len(documents) - successful_docs,
            "total_chunks": total_chunks,
            "chunk_size": chunk_size,
            "processing_time": elapsed_time,
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Error in batch document processing: {str(e)}")
        raise RuntimeError(f"Failed to process batch documents: {str(e)}")

async def get_indexing_stats(username: Optional[str] = None) -> dict:
    """
    Get indexing statistics.
    
    Args:
        username: Optional username filter
    
    Returns:
        Dictionary with indexing statistics
    """
    try:
        indexer = DocumentIndexer(QDRANT_DB_PATH)
        
        # Get collection info
        collection_info = await indexer.get_collection_info()
        document_count = await indexer.get_document_count()
        
        stats = {
            "collection_name": "rag_demo_collection",
            "total_documents": document_count,
            "collection_info": collection_info,
            "indexer_health": await indexer.health_check()
        }
        
        if username:
            stats["username"] = username
        
        return stats
        
    except Exception as e:
        logger.error(f"Error getting indexing stats: {str(e)}")
        return {
            "error": str(e),
            "collection_name": "rag_demo_collection",
            "total_documents": 0
        }

async def clear_user_documents(username: str) -> dict:
    """
    Clear all documents for a specific user (placeholder implementation).
    
    Args:
        username: User identifier
    
    Returns:
        Dictionary with operation results
    """
    try:
        # Note: This is a placeholder implementation
        # In a real system, you'd need to track user-document relationships
        
        logger.warning(f"Clear user documents not fully implemented for user: {username}")
        
        return {
            "success": False,
            "message": "User-specific document clearing not implemented",
            "username": username
        }
        
    except Exception as e:
        logger.error(f"Error clearing user documents: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "username": username
        }

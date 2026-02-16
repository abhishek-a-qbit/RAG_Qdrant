import os
import time
from typing import List, Dict, Any, Tuple, Optional
from langchain.schema import Document
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain.schema import MessagesPlaceholder
from langchain_community.callbacks import get_openai_callback
from services.logger import setup_logger
from services.document_indexer import DocumentIndexer
from utils.utils import QDRANT_DB_PATH, NO_OF_CHUNKS

# Initialize logger
logger = setup_logger("document_retrieval")

async def retrieve_similar_documents(refined_query: str, num_of_chunks: int, username: str) -> Tuple[str, List[Document]]:
    """
    Retrieve similar documents from Qdrant using the refined query.
    
    Args:
        refined_query: The refined query for document retrieval
        num_of_chunks: Number of document chunks to retrieve
        username: User identifier for logging/tracking
    
    Returns:
        Tuple of (extracted_text_data, extracted_documents)
    """
    try:
        indexer = DocumentIndexer(QDRANT_DB_PATH)
        start_time = time.time()
        logger.info(f"Searching for similar documents in Qdrant for user: {username}")
        logger.info(f"Refined query: {refined_query}")
        logger.info(f"Number of chunks requested: {num_of_chunks}")

        # Validate num_of_chunks
        if num_of_chunks is None:
            num_of_chunks = int(NO_OF_CHUNKS)
        
        if not isinstance(num_of_chunks, int) or num_of_chunks <= 0:
            raise ValueError(f"Invalid number of chunks: {num_of_chunks}")
        
        # Initialize collection if needed
        await indexer.initialize_collection()
        
        # Get retriever
        retriever = await indexer.get_retriever(top_k=num_of_chunks)
        if not retriever:
            raise ValueError("Failed to initialize document retriever")
        
        # Retrieve documents
        extracted_documents = await retriever.ainvoke(refined_query)
        
        # Format documents
        if not extracted_documents:
            extracted_text_data = ""
            logger.info("No documents retrieved for the query")
        else:
            extracted_text_data = await format_docs(extracted_documents)
            logger.info(f"Retrieved and formatted {len(extracted_documents)} documents")
        
        elapsed_time = time.time() - start_time
        logger.info(f"Document retrieval and formatting completed in {elapsed_time:.2f} seconds")
        
        return extracted_text_data, extracted_documents

    except Exception as e:
        logger.error(f"Error processing documents: {str(e)}")
        raise RuntimeError(f"Failed to process documents: {str(e)}")

async def format_docs(docs: List[Document]) -> str:
    """
    Format retrieved documents into a single text string.
    
    Args:
        docs: List of LangChain Document objects
    
    Returns:
        Formatted text string with document content and metadata
    """
    try:
        formatted_text = ""
        
        for i, doc in enumerate(docs, 1):
            # Add document content
            formatted_text += f"Document {i}:\n"
            formatted_text += doc.page_content.strip() + "\n\n"
            
            # Add metadata if available
            if doc.metadata:
                formatted_text += "Metadata:\n"
                for key, value in doc.metadata.items():
                    if key not in ['chunk_index', 'total_chunks']:  # Skip technical metadata
                        formatted_text += f"  {key}: {value}\n"
                formatted_text += "\n"
            
            formatted_text += "---\n\n"
        
        return formatted_text.strip()
        
    except Exception as e:
        logger.error(f"Error formatting documents: {str(e)}")
        return ""

async def retrieve_with_scores(refined_query: str, num_of_chunks: int, 
                             username: str, score_threshold: float = 0.7) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Retrieve similar documents with similarity scores and filtering.
    
    Args:
        refined_query: The refined query for document retrieval
        num_of_chunks: Number of document chunks to retrieve
        username: User identifier
        score_threshold: Minimum similarity score threshold
    
    Returns:
        Tuple of (extracted_text_data, extracted_documents_with_scores)
    """
    try:
        indexer = DocumentIndexer(QDRANT_DB_PATH)
        start_time = time.time()
        logger.info(f"Retrieving documents with score threshold: {score_threshold}")

        # Initialize collection if needed
        await indexer.initialize_collection()
        
        # Search with scores
        results = await indexer.search_with_scores(refined_query, k=num_of_chunks)
        
        # Filter by score threshold
        filtered_results = []
        for doc, score in results:
            if score >= score_threshold:
                filtered_results.append({
                    "document": doc,
                    "score": score,
                    "content": doc.page_content,
                    "metadata": doc.metadata
                })
        
        # Format filtered documents
        if not filtered_results:
            extracted_text_data = ""
            logger.info(f"No documents met score threshold {score_threshold}")
        else:
            # Create Document objects from filtered results
            docs = [result["document"] for result in filtered_results]
            extracted_text_data = await format_docs(docs)
            logger.info(f"Retrieved {len(filtered_results)} documents above threshold")
        
        elapsed_time = time.time() - start_time
        logger.info(f"Score-based retrieval completed in {elapsed_time:.2f} seconds")
        
        return extracted_text_data, filtered_results

    except Exception as e:
        logger.error(f"Error in score-based retrieval: {str(e)}")
        raise RuntimeError(f"Failed to retrieve documents with scores: {str(e)}")

def get_main_prompt():
    """
    Create the main prompt template for RAG response generation.
    
    Returns:
        ChatPromptTemplate configured for RAG tasks
    """
    system_prompt = """You are an assistant for question-answering tasks.
    Use the following pieces of retrieved context and user information to answer the question.
    If you don't find the answer of the query, then just say I don't have that information at hand. Please provide more details or check your sources.
    Provide a comprehensive and accurate response based on the given context.
    """

    # Add context placeholder
    prompt = system_prompt + "\n\nContext:\n{context}"

    final_prompt = ChatPromptTemplate.from_messages([
        ("system", prompt),
        MessagesPlaceholder(variable_name="messages"),
        ("human", "{user_query}")
    ])
    
    return final_prompt

def get_contextual_prompt():
    """
    Create a contextual prompt that considers conversation history.
    
    Returns:
        ChatPromptTemplate with conversation history support
    """
    system_prompt = """You are a helpful assistant with access to retrieved context.
    Consider the conversation history and the retrieved context to answer the user's question.
    If the context doesn't contain the answer, say so politely.
    Provide accurate, helpful responses based on the available information."""

    final_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        ("system", "Retrieved Context:\n{context}"),
        ("human", "{user_query}")
    ])
    
    return final_prompt

async def invoke_chain(query: str, context: str, history, llm, use_contextual_prompt: bool = False):
    """
    Handles the RAG chain invocation asynchronously with callback tracking.
    
    Args:
        query: User's query
        context: Retrieved document context
        history: Conversation history object
        llm: Initialized language model
        use_contextual_prompt: Whether to use contextual prompt template
    
    Returns:
        Tuple of (final_response, callback_object)
    """
    try:
        logger.info("Initializing RAG chain...")
        
        # Choose prompt template
        if use_contextual_prompt:
            final_prompt = get_contextual_prompt()
        else:
            final_prompt = get_main_prompt()
        
        # Create chain
        final_chain = final_prompt | llm | StrOutputParser()
        logger.info("Chain initialized successfully.")
        
        # Prepare input data
        input_data = {
            "user_query": query,
            "context": context,
            "messages": history.messages if hasattr(history, 'messages') else history
        }
        
        # Invoke chain with callback tracking
        with get_openai_callback() as cb:
            final_response = await final_chain.ainvoke(input_data)
        
        logger.info(f"Chain invocation completed. Tokens used: {cb.total_tokens}")
        return final_response, cb
        
    except Exception as e:
        logger.error(f"Error in chain invocation: {str(e)}")
        # Return error response
        error_response = "I apologize, but I encountered an error while generating a response."
        return error_response, None

async def batch_retrieve(queries: List[str], num_of_chunks: int, 
                         username: str) -> List[Tuple[str, List[Document]]]:
    """
    Retrieve documents for multiple queries in batch.
    
    Args:
        queries: List of queries to process
        num_of_chunks: Number of chunks per query
        username: User identifier
    
    Returns:
        List of tuples containing (extracted_text_data, extracted_documents) for each query
    """
    try:
        logger.info(f"Processing batch retrieval for {len(queries)} queries")
        start_time = time.time()
        
        results = []
        for i, query in enumerate(queries):
            logger.info(f"Processing query {i+1}/{len(queries)}: {query[:50]}...")
            
            text_data, documents = await retrieve_similar_documents(
                query, num_of_chunks, username
            )
            results.append((text_data, documents))
        
        elapsed_time = time.time() - start_time
        logger.info(f"Batch retrieval completed in {elapsed_time:.2f} seconds")
        
        return results
        
    except Exception as e:
        logger.error(f"Error in batch retrieval: {str(e)}")
        raise RuntimeError(f"Failed to process batch retrieval: {str(e)}")

def get_retrieval_stats(retrieved_docs: List[Document], query: str) -> Dict[str, Any]:
    """
    Generate statistics about the retrieval process.
    
    Args:
        retrieved_docs: List of retrieved documents
        query: Original query
    
    Returns:
        Dictionary with retrieval statistics
    """
    try:
        total_chars = sum(len(doc.page_content) for doc in retrieved_docs)
        avg_doc_length = total_chars / len(retrieved_docs) if retrieved_docs else 0
        
        # Extract unique sources
        sources = set()
        for doc in retrieved_docs:
            if 'source' in doc.metadata:
                sources.add(doc.metadata['source'])
            elif 'file_name' in doc.metadata:
                sources.add(doc.metadata['file_name'])
        
        return {
            "query_length": len(query),
            "documents_retrieved": len(retrieved_docs),
            "total_characters": total_chars,
            "average_document_length": avg_doc_length,
            "unique_sources": len(sources),
            "sources": list(sources)
        }
        
    except Exception as e:
        logger.error(f"Error generating retrieval stats: {str(e)}")
        return {}

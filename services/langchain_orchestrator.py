import time
from typing import List, Dict, Any, Optional, Tuple
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import BaseMessage
from langchain_core.callbacks import BaseCallbackHandler
from services.logger import setup_logger
from services.document_indexer import DocumentIndexer
from utils.utils import OPENAI_API_KEY, MODEL, TEMPERATURE
from utils.prompts import PromptTemplates

# Initialize logger
logger = setup_logger("langchain_orchestrator")

class TokenCallback(BaseCallbackHandler):
    """Callback handler to track token usage"""
    def __init__(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
    
    def on_llm_end(self, response, **kwargs):
        """Called when LLM ends"""
        if hasattr(response, 'llm_output') and response.llm_output:
            usage = response.llm_output.get('token_usage', {})
            self.prompt_tokens = usage.get('prompt_tokens', 0)
            self.completion_tokens = usage.get('completion_tokens', 0)
            self.total_tokens = usage.get('total_tokens', 0)

class ChatHistory:
    """Manages chat history for LangChain"""
    def __init__(self):
        self.messages = []
    
    def add_user_message(self, content: str):
        """Add user message to history"""
        self.messages.append(HumanMessage(content=content))
    
    def add_ai_message(self, content: str):
        """Add AI message to history"""
        self.messages.append(AIMessage(content=content))
    
    def add_system_message(self, content: str):
        """Add system message to history"""
        self.messages.append(SystemMessage(content=content))
    
    def get_messages(self) -> List[BaseMessage]:
        """Get all messages"""
        return self.messages
    
    def clear(self):
        """Clear all messages"""
        self.messages = []

def get_query_refiner_prompt():
    """Get prompt for query refinement"""
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as it is."
    )

    final_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        ("human", "{query}"),
        ("system", "Chat History:\n{chat_history}")
    ])
    
    return final_prompt

def get_rag_prompt():
    """Get RAG prompt for response generation"""
    rag_system_prompt = (
        "You are a helpful assistant. Use the following context to answer the user's question. "
        "If you don't know the answer based on the context, say so politely. "
        "Provide a comprehensive and accurate response based on the given information."
    )

    final_prompt = ChatPromptTemplate.from_messages([
        ("system", rag_system_prompt),
        ("system", "Context:\n{context}"),
        ("human", "{query}")
    ])
    
    return final_prompt

async def refine_user_query(query: str, messages: List[Dict[str, str]]) -> str:
    """
    Refines the user query asynchronously using chat history.
    
    Args:
        query: Original user query
        messages: List of previous messages with role and content
    
    Returns:
        Refined standalone query
    """
    try:
        logger.info(f"Refining user query: {query}")
        
        # Initialize LLM for query refinement
        llm = ChatOpenAI(
            temperature=0,
            model_name="gpt-4o-mini",
            openai_api_key=OPENAI_API_KEY
        )
        
        # Create chat history from messages
        history = create_history(messages)
        
        # Get prompt and create chain
        prompt = get_query_refiner_prompt()
        refined_query_chain = prompt | llm | StrOutputParser()
        
        # Format chat history for prompt
        chat_history_str = "\n".join([
            f"{msg['role']}: {msg['content']}" 
            for msg in messages[-6:]  # Last 6 messages for context
        ])
        
        # Invoke chain asynchronously
        refined_query = await refined_query_chain.ainvoke({
            "query": query,
            "chat_history": chat_history_str
        })
        
        logger.info(f"Generated refined query: {refined_query}")
        return refined_query.strip()
        
    except Exception as e:
        logger.error(f"Error refining query: {str(e)}")
        # Return original query if refinement fails
        return query

def create_history(messages: List[Dict[str, str]]) -> ChatHistory:
    """
    Create LangChain ChatHistory from message list.
    
    Args:
        messages: List of message dictionaries with role and content
    
    Returns:
        ChatHistory object
    """
    history = ChatHistory()
    
    for msg in messages:
        if msg['role'] == 'user':
            history.add_user_message(msg['content'])
        elif msg['role'] == 'assistant':
            history.add_ai_message(msg['content'])
        elif msg['role'] == 'system':
            history.add_system_message(msg['content'])
    
    return history

def initialize_llm(temperature: float = TEMPERATURE, model: str = MODEL) -> ChatOpenAI:
    """
    Initialize LLM for response generation.
    
    Args:
        temperature: Sampling temperature
        model: Model name
    
    Returns:
        Initialized ChatOpenAI instance
    """
    return ChatOpenAI(
        temperature=temperature,
        model_name=model,
        openai_api_key=OPENAI_API_KEY,
        streaming=False
    )

async def retrieve_similar_documents(query: str, no_of_chunks: int, username: str) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Retrieve similar documents from vector store.
    
    Args:
        query: Search query
        no_of_chunks: Number of chunks to retrieve
        username: User identifier
    
    Returns:
        Tuple of (extracted_text_data, extracted_documents)
    """
    try:
        logger.info(f"Retrieving {no_of_chunks} similar documents for query: {query}")
        
        # Initialize document indexer
        indexer = DocumentIndexer("http://localhost:6333")
        await indexer.initialize_collection()
        
        # Get retriever
        retriever = await indexer.get_retriever(top_k=no_of_chunks)
        
        # Search for similar documents
        results = await indexer.search_with_scores(query, k=no_of_chunks)
        
        # Extract text and metadata
        extracted_text_data = ""
        extracted_documents = []
        
        for doc, score in results:
            if score > 0.7:  # Similarity threshold
                extracted_text_data += doc.page_content + "\n\n"
                extracted_documents.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": score
                })
        
        logger.info(f"Retrieved {len(extracted_documents)} relevant documents")
        return extracted_text_data.strip(), extracted_documents
        
    except Exception as e:
        logger.error(f"Error retrieving documents: {str(e)}")
        return "", []

async def invoke_chain(query: str, context: str, history: ChatHistory, llm: ChatOpenAI) -> Tuple[str, TokenCallback]:
    """
    Invoke the RAG chain to generate response.
    
    Args:
        query: User query
        context: Retrieved context
        history: Chat history
        llm: Initialized LLM
    
    Returns:
        Tuple of (response, callback_handler)
    """
    try:
        logger.info("Invoking RAG chain for response generation")
        
        # Initialize callback handler
        callback_handler = TokenCallback()
        
        # Get RAG prompt
        prompt = get_rag_prompt()
        
        # Create chain
        chain = prompt | llm | StrOutputParser()
        
        # Invoke chain with callbacks
        response = await chain.ainvoke({
            "query": query,
            "context": context
        }, callbacks=[callback_handler])
        
        logger.info("Generated response from RAG chain")
        return response, callback_handler
        
    except Exception as e:
        logger.error(f"Error invoking chain: {str(e)}")
        error_response = "I apologize, but I encountered an error while generating a response."
        return error_response, TokenCallback()

async def generate_chatbot_response(query: str, past_messages: List[Dict[str, str]], 
                                  no_of_chunks: int, username: str) -> Tuple[str, float, int, int, int, str, str, List[Dict[str, Any]]]:
    """
    Main function to generate chatbot responses asynchronously with LangChain orchestration.
    
    Args:
        query: User's query
        past_messages: Previous conversation history
        no_of_chunks: Number of document chunks to retrieve
        username: User identifier
    
    Returns:
        Tuple of (final_response, response_time, prompt_tokens, completion_tokens, total_tokens, 
                 extracted_text_data, refined_query, extracted_documents)
    """
    try:
        logger.info("Starting chatbot response generation")
        start_time = time.time()
        
        # Step 1: Refine user query
        logger.info("Refining user query")
        refined_query = await refine_user_query(query, past_messages)
        logger.info(f"Generated refined query: {refined_query}")

        # Step 2: Retrieve similar documents
        logger.info("Retrieving similar documents")
        extracted_text_data, extracted_documents = await retrieve_similar_documents(
            refined_query, int(no_of_chunks), username
        )
        logger.info(f"Retrieved {len(extracted_documents)} documents")

        # Step 3: Initialize LLM and create history
        llm = initialize_llm()
        history = create_history(past_messages)
        logger.info(f"Created history with {len(history.get_messages())} messages")

        # Step 4: Generate response
        logger.info("Generating response")
        response_start = time.time()
        final_response, cb = await invoke_chain(query, extracted_text_data, history, llm)
        response_time = time.time() - response_start

        total_time = time.time() - start_time
        
        logger.info(f"Generated response in {response_time:.2f}s (total: {total_time:.2f}s)")
        logger.info(f"Token usage - Prompt: {cb.prompt_tokens}, Completion: {cb.completion_tokens}, Total: {cb.total_tokens}")

        return (final_response, response_time, cb.prompt_tokens, cb.completion_tokens, 
                cb.total_tokens, extracted_text_data, refined_query, extracted_documents)

    except Exception as e:
        logger.error(f"Error in generate_chatbot_response: {str(e)}")
        error_response = "I apologize, but I encountered an error while processing your request."
        return (error_response, 0.0, 0, 0, 0, "", query, [])

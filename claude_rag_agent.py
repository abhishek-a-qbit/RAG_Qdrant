"""
Claude Agent SDK integration with RAG pipeline and LangSmith tracing
"""

import asyncio
import os
from typing import Any, Dict, List
from claude_agent_sdk import (
    ClaudeAgentOptions,
    ClaudeSDKClient,
    tool,
    create_sdk_mcp_server,
)
from langsmith.integrations.claude_agent_sdk import configure_claude_agent_sdk

# Import existing RAG components
from utils.langchain_utils import LangChainManager
from utils.qdrant_utils import QdrantManager
from utils.utils import OPENAI_API_KEY, QDRANT_DB_PATH, QDRANT_API_KEY, MODEL, TEMPERATURE

# Setup claude_agent_sdk with langsmith tracing
configure_claude_agent_sdk()

# Initialize RAG components
langchain_manager = LangChainManager(OPENAI_API_KEY, MODEL, TEMPERATURE)
qdrant_manager = QdrantManager(QDRANT_DB_PATH, QDRANT_API_KEY)

@tool(
    "search_documents",
    "Searches for relevant documents in the RAG knowledge base",
    {
        "query": str,
        "limit": int,
        "similarity_threshold": float,
    },
)
async def search_documents(args: dict[str, Any]) -> dict[str, Any]:
    """Search documents using RAG pipeline"""
    try:
        query = args["query"]
        limit = args.get("limit", 5)
        similarity_threshold = args.get("similarity_threshold", 0.5)
        
        # Generate query embedding
        query_embedding = langchain_manager.embed_query(query)
        
        # Search in Qdrant
        search_results = qdrant_manager.search(
            "documents", query_embedding, limit, similarity_threshold
        )
        
        if not search_results:
            return {
                "content": [{
                    "type": "text", 
                    "text": f"No relevant documents found for query: '{query}'"
                }]
            }
        
        # Format search results
        formatted_results = []
        for i, result in enumerate(search_results):
            chunk_text = result.get('text', '')[:500] + "..." if len(result.get('text', '')) > 500 else result.get('text', '')
            metadata = result.get('metadata', {})
            
            formatted_results.append({
                "type": "text",
                "text": f"Document {i+1}:\nSource: {metadata.get('filename', 'Unknown')}\nContent: {chunk_text}\nScore: {result.get('score', 0):.3f}"
            })
        
        return {"content": formatted_results}
        
    except Exception as e:
        return {
            "content": [{
                "type": "text", 
                "text": f"Error searching documents: {str(e)}"
            }]
        }

@tool(
    "answer_question",
    "Answers a question using the RAG pipeline with comprehensive context",
    {
        "question": str,
        "include_sources": bool,
        "include_metrics": bool,
    },
)
async def answer_question(args: dict[str, Any]) -> dict[str, Any]:
    """Answer question using full RAG pipeline"""
    try:
        question = args["question"]
        include_sources = args.get("include_sources", True)
        include_metrics = args.get("include_metrics", True)
        
        # Generate query embedding
        query_embedding = langchain_manager.embed_query(question)
        
        # Search for relevant documents
        search_results = qdrant_manager.search("documents", query_embedding, 5, 0.5)
        
        if not search_results:
            return {
                "content": [{
                    "type": "text", 
                    "text": f"No relevant documents found to answer: '{question}'"
                }]
            }
        
        # Create context from search results
        context = "\n\n".join([
            f"Source: {result.get('metadata', {}).get('filename', 'Unknown')}\n"
            f"Content: {result.get('text', '')}\n"
            f"Relevance Score: {result.get('score', 0):.3f}"
            for result in search_results
        ])
        
        # Generate answer using LangChain
        prompt = f"""Use the following context to answer the question comprehensively.

Context:
{context}

Question: {question}

Provide a detailed answer based on the context. If the context doesn't contain enough information, say so explicitly.

Answer:"""
        
        # Create a simple chain for answering
        chain = langchain_manager.llm
        answer = chain.invoke(prompt)
        
        # Format response
        response_parts = [{"type": "text", "text": f"Answer: {answer}"}]
        
        if include_sources:
            sources_text = "\n\nSources used:\n" + "\n".join([
                f"- {result.get('metadata', {}).get('filename', 'Unknown')} (Score: {result.get('score', 0):.3f})"
                for result in search_results
            ])
            response_parts.append({"type": "text", "text": sources_text})
        
        if include_metrics:
            metrics_text = f"\n\nRAG Metrics:\n- Documents retrieved: {len(search_results)}\n- Context length: {len(context)} characters\n- Query similarity threshold: 0.5"
            response_parts.append({"type": "text", "text": metrics_text})
        
        return {"content": response_parts}
        
    except Exception as e:
        return {
            "content": [{
                "type": "text", 
                "text": f"Error answering question: {str(e)}"
            }]
        }

@tool(
    "get_document_stats",
    "Gets statistics about the document collection",
    {},
)
async def get_document_stats(args: dict[str, Any]) -> dict[str, Any]:
    """Get collection statistics"""
    try:
        # Get collection info
        collection_info = qdrant_manager.get_collection_info("documents")
        
        if collection_info:
            stats_text = f"""Document Collection Statistics:
- Collection: documents
- Total vectors: {collection_info.get('vectors_count', 'Unknown')}
- Indexed vectors: {collection_info.get('indexed_vectors_count', 'Unknown')}
- Points count: {collection_info.get('points_count', 'Unknown')}
- Status: {collection_info.get('status', 'Unknown')}"""
        else:
            stats_text = "Could not retrieve collection statistics."
        
        return {"content": [{"type": "text", "text": stats_text}]}
        
    except Exception as e:
        return {
            "content": [{
                "type": "text", 
                "text": f"Error getting stats: {str(e)}"
            }]
        }

async def main():
    """Main function to run Claude Agent SDK with RAG tools"""
    
    # Create SDK MCP server with RAG tools
    rag_server = create_sdk_mcp_server(
        name="rag_system",
        version="1.0.0",
        tools=[search_documents, answer_question, get_document_stats],
    )

    options = ClaudeAgentOptions(
        model="claude-sonnet-4-5-20250929",
        system_prompt="""You are an intelligent RAG (Retrieval-Augmented Generation) assistant with access to a comprehensive document knowledge base.

Your capabilities:
1. Search documents for relevant information
2. Answer questions using retrieved context with sources
3. Provide statistics about the document collection
4. Always cite sources when providing information
5. Include RAG metrics when answering questions

When answering questions:
- Use the answer_question tool for comprehensive responses
- Always include sources and metrics when available
- Be transparent about limitations in the available data
- Provide detailed, contextual answers based on retrieved documents

The document collection contains information about 6sense, revenue AI, B2B marketing platforms, case studies, and industry reports.""",
        mcp_servers={"rag_system": rag_server},
        allowed_tools=[
            "mcp__rag_system__search_documents",
            "mcp__rag_system__answer_question", 
            "mcp__rag_system__get_document_stats"
        ],
    )

    async with ClaudeSDKClient(options=options) as client:
        await client.query("What can you tell me about 6sense revenue AI features and capabilities?")

        async for message in client.receive_response():
            print(message)

if __name__ == "__main__":
    asyncio.run(main())

import os
from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document as LangChainDocument
from langchain_core.prompts import PromptTemplate
import time
from utils.utils import LANGSMITH_TRACING

# Initialize LangSmith tracing if enabled
if LANGSMITH_TRACING:
    from langsmith import traceable
else:
    # Create a dummy decorator if tracing is disabled
    def traceable(name=None, tags=None):
        def decorator(func):
            return func
        return decorator

class LangChainManager:
    """Manager for LangChain operations including LLM and embeddings"""
    
    def __init__(self, openai_api_key: str, model: str = "gpt-4o-mini", temperature: float = 0.1):
        self.openai_api_key = openai_api_key
        self.model = model
        self.temperature = temperature
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model_name=model,
            temperature=temperature,
            openai_api_key=openai_api_key
        )
        
        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=openai_api_key
        )
        
        # Text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200,
            length_function=len
        )
    
    def split_text(self, text: str, chunk_size: int = 2000, chunk_overlap: int = 200) -> List[str]:
        """Split text into chunks"""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
        return splitter.split_text(text)
    
    def create_documents(self, texts: List[str], metadata: Optional[List[Dict[str, Any]]] = None) -> List[LangChainDocument]:
        """Create LangChain documents from texts"""
        if metadata is None:
            metadata = [{}] * len(texts)
        
        documents = []
        for i, text in enumerate(texts):
            doc = LangChainDocument(
                page_content=text,
                metadata=metadata[i] if i < len(metadata) else {}
            )
            documents.append(doc)
        
        return documents
    
    @traceable(name="embed_texts", tags=["embedding"])
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for texts"""
        return self.embeddings.embed_documents(texts)
    
    @traceable(name="embed_query", tags=["embedding"])
    def embed_query(self, query: str) -> List[float]:
        """Generate embedding for query"""
        return self.embeddings.embed_query(query)
    
    @traceable(name="create_qa_chain", tags=["chain", "qa"])
    def create_qa_chain(self, retriever, prompt_template: Optional[str] = None):
        """Create a QA chain with retriever using LCEL"""
        if prompt_template:
            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )
        else:
            prompt = PromptTemplate(
                template="""Use the following pieces of context to answer the question at the end.
                If you don't know the answer, just say that you don't know, don't try to make up an answer.
                Keep the answer concise and relevant to the question.
                
                Context: {context}
                
                Question: {question}
                
                Helpful Answer:""",
                input_variables=["context", "question"]
            )
        
        # Create LCEL chain
        rag_chain = (
            RunnableParallel({
                "context": retriever | (lambda docs: "\n\n".join([doc.page_content for doc in docs])),
                "question": RunnablePassthrough()
            })
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        return rag_chain
    
    @traceable(name="generate_answer", tags=["generation", "qa"])
    def generate_answer(self, qa_chain, query: str) -> Dict[str, Any]:
        """Generate answer using QA chain"""
        start_time = time.time()
        
        try:
            # Invoke LCEL chain
            result = qa_chain.invoke(query)
            response_time = time.time() - start_time
            
            return {
                "answer": result,
                "source_documents": [],  # Will be populated separately if needed
                "response_time": response_time,
                "success": True
            }
        except Exception as e:
            return {
                "answer": f"Error generating answer: {str(e)}",
                "source_documents": [],
                "response_time": time.time() - start_time,
                "success": False,
                "error": str(e)
            }
    
    def summarize_text(self, text: str, max_length: int = 500) -> str:
        """Summarize text using LLM"""
        prompt = f"""
        Please summarize the following text in no more than {max_length} characters:
        
        Text: {text}
        
        Summary:"""
        
        try:
            response = self.llm.invoke(prompt)
            return response.content.strip()
        except Exception as e:
            return f"Error summarizing text: {str(e)}"
    
    def extract_keywords(self, text: str, num_keywords: int = 10) -> List[str]:
        """Extract keywords from text"""
        prompt = f"""
        Extract the {num_keywords} most important keywords from the following text.
        Return only the keywords as a comma-separated list, no additional text.
        
        Text: {text}
        
        Keywords:"""
        
        try:
            response = self.llm.invoke(prompt)
            keywords = response.content.strip().split(',')
            return [keyword.strip() for keyword in keywords if keyword.strip()]
        except Exception as e:
            return []

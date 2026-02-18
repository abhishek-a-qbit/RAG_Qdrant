from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

class DocumentUpload(BaseModel):
    """Model for document upload requests"""
    filename: str = Field(..., description="Name of the uploaded file")
    file_type: str = Field(..., description="Type of file (pdf, txt, docx)")
    file_size: int = Field(..., description="Size of file in bytes")

class DocumentChunk(BaseModel):
    """Model for document chunks"""
    chunk_id: str = Field(..., description="Unique identifier for chunk")
    document_id: str = Field(..., description="Document identifier")
    chunk_text: str = Field(..., description="Text content of chunk")
    chunk_index: int = Field(..., description="Index of chunk in document")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

class QueryRequest(BaseModel):
    """Model for query requests"""
    query: str = Field(..., description="User query text")
    collection_name: Optional[str] = Field(default="documents", description="Qdrant collection name")
    limit: Optional[int] = Field(default=5, description="Number of results to retrieve")
    similarity_threshold: Optional[float] = Field(default=0.7, description="Minimum similarity score")

class SourceDocument(BaseModel):
    """Model for source documents with links and metrics"""
    document_id: str = Field(..., description="Unique identifier for document")
    filename: str = Field(..., description="Name of source file")
    chunk_index: int = Field(..., description="Index of chunk in document")
    score: float = Field(..., description="Similarity score")
    text: str = Field(..., description="Text content preview")
    content: str = Field(..., description="Full text content")
    link: str = Field(..., description="Direct link to document")
    
class QueryMetrics(BaseModel):
    """Model for query quality metrics"""
    coverage: float = Field(..., description="Coverage score (0-1)")
    specificity: float = Field(..., description="Specificity score (0-1)")
    insightfulness: float = Field(..., description="Insightfulness score (0-1)")
    groundedness: float = Field(..., description="Groundedness score (0-1)")
    overall_score: float = Field(..., description="Overall quality score (0-1)")

class QueryResponse(BaseModel):
    """Model for query responses"""
    query: str = Field(..., description="Original query")
    answer: str = Field(..., description="Generated answer")
    sources: List[SourceDocument] = Field(default_factory=list, description="Source documents used")
    confidence_score: Optional[float] = Field(None, description="Confidence score of answer")
    response_time: Optional[float] = Field(None, description="Time taken to generate response")
    metrics: Optional[QueryMetrics] = Field(None, description="Quality metrics for the response")

class CollectionInfo(BaseModel):
    """Model for collection information"""
    name: str = Field(..., description="Collection name")
    document_count: int = Field(..., description="Number of documents in collection")
    vector_count: int = Field(..., description="Number of vectors in collection")
    created_at: Optional[datetime] = Field(None, description="Collection creation time")

class ErrorResponse(BaseModel):
    """Model for error responses"""
    error: str = Field(..., description="Error message")
    error_code: Optional[str] = Field(None, description="Error code")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")

class ChatRequest(BaseModel):
    """Model for chat requests - compatible with Streamlit"""
    query: str = Field(..., description="User query text")
    collection_name: Optional[str] = Field(default="documents", description="Qdrant collection name")
    limit: Optional[int] = Field(default=5, description="Number of results to retrieve")
    similarity_threshold: Optional[float] = Field(default=0.7, description="Minimum similarity score")
    session_id: Optional[str] = Field(None, description="Session identifier for conversation tracking")

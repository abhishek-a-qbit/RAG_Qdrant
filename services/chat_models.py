from pydantic import BaseModel, field_validator
from typing import Optional, List, Dict, Any
import re

class ChatRequest(BaseModel):
    """Defines the structure of incoming requests to the chat endpoint"""
    username: str
    query: str
    session_id: Optional[str] = None
    no_of_chunks: Optional[int] = 3
    
    @field_validator('username')
    @classmethod
    def validate_username(cls, v):
        """Validate username field"""
        if not v or not v.strip():
            raise ValueError('Username cannot be empty')
        if len(v.strip()) < 2:
            raise ValueError('Username must be at least 2 characters long')
        if len(v.strip()) > 50:
            raise ValueError('Username cannot exceed 50 characters')
        # Allow alphanumeric, spaces, hyphens, and underscores
        if not re.match(r'^[a-zA-Z0-9\s\-_]+$', v.strip()):
            raise ValueError('Username can only contain letters, numbers, spaces, hyphens, and underscores')
        return v.strip()
    
    @field_validator('query')
    @classmethod
    def validate_query(cls, v):
        """Validate query field"""
        if not v or not v.strip():
            raise ValueError('Query cannot be empty')
        if len(v.strip()) < 1:
            raise ValueError('Query must contain at least 1 character')
        if len(v.strip()) > 2000:
            raise ValueError('Query cannot exceed 2000 characters')
        return v.strip()
    
    @field_validator('session_id')
    @classmethod
    def validate_session_id(cls, v):
        """Validate session_id field"""
        if v is not None:
            if not isinstance(v, str):
                raise ValueError('Session ID must be a string')
            if len(v) < 1:
                raise ValueError('Session ID cannot be empty')
            if len(v) > 100:
                raise ValueError('Session ID cannot exceed 100 characters')
        return v
    
    @field_validator('no_of_chunks')
    @classmethod
    def validate_no_of_chunks(cls, v):
        """Validate no_of_chunks field"""
        if v is not None:
            if not isinstance(v, int):
                raise ValueError('Number of chunks must be an integer')
            if v < 1:
                raise ValueError('Number of chunks must be at least 1')
            if v > 20:
                raise ValueError('Number of chunks cannot exceed 20')
        return v or 3

class ChatResponse(BaseModel):
    """Specifies the format of the API response"""
    username: str
    query: str
    refine_query: str
    response: str
    session_id: str
    debug_info: Optional[Dict[str, Any]] = None
    
    @field_validator('username')
    @classmethod
    def validate_username(cls, v):
        """Validate username in response"""
        if not v or not v.strip():
            raise ValueError('Username cannot be empty in response')
        return v.strip()
    
    @field_validator('query')
    @classmethod
    def validate_query(cls, v):
        """Validate query in response"""
        if not v or not v.strip():
            raise ValueError('Query cannot be empty in response')
        return v.strip()
    
    @field_validator('response')
    @classmethod
    def validate_response(cls, v):
        """Validate response field"""
        if not isinstance(v, str):
            raise ValueError('Response must be a string')
        if len(v) > 10000:
            raise ValueError('Response cannot exceed 10000 characters')
        return v
    
    @field_validator('session_id')
    @classmethod
    def validate_session_id(cls, v):
        """Validate session_id in response"""
        if not v or not isinstance(v, str):
            raise ValueError('Session ID must be a non-empty string in response')
        return v

class ConversationLog(BaseModel):
    """Model for individual conversation log entries"""
    id: Optional[int] = None
    session_id: str
    user_query: str
    gpt_response: str
    timestamp: Optional[str] = None
    
    @field_validator('session_id')
    @classmethod
    def validate_session_id(cls, v):
        """Validate session_id"""
        if not v or not v.strip():
            raise ValueError('Session ID cannot be empty')
        return v.strip()
    
    @field_validator('user_query')
    @classmethod
    def validate_user_query(cls, v):
        """Validate user query"""
        if not v or not v.strip():
            raise ValueError('User query cannot be empty')
        return v.strip()
    
    @field_validator('gpt_response')
    @classmethod
    def validate_gpt_response(cls, v):
        """Validate GPT response"""
        if not isinstance(v, str):
            raise ValueError('GPT response must be a string')
        return v

class ChatSession(BaseModel):
    """Model for chat session information"""
    session_id: str
    username: str
    start_time: Optional[str] = None
    last_time: Optional[str] = None
    message_count: int = 0
    is_active: bool = True
    
    @field_validator('session_id')
    @classmethod
    def validate_session_id(cls, v):
        """Validate session_id"""
        if not v or not v.strip():
            raise ValueError('Session ID cannot be empty')
        return v.strip()
    
    @field_validator('username')
    @classmethod
    def validate_username(cls, v):
        """Validate username"""
        if not v or not v.strip():
            raise ValueError('Username cannot be empty')
        return v.strip()
    
    @field_validator('message_count')
    @classmethod
    def validate_message_count(cls, v):
        """Validate message count"""
        if not isinstance(v, int) or v < 0:
            raise ValueError('Message count must be a non-negative integer')
        return v

class ChatHistoryResponse(BaseModel):
    """Model for chat history response"""
    sessions: List[ChatSession]
    total_sessions: int
    total_messages: int
    
    @field_validator('sessions')
    @classmethod
    def validate_sessions(cls, v):
        """Validate sessions list"""
        if not isinstance(v, list):
            raise ValueError('Sessions must be a list')
        return v
    
    @field_validator('total_sessions')
    @classmethod
    def validate_total_sessions(cls, v):
        """Validate total sessions"""
        if not isinstance(v, int) or v < 0:
            raise ValueError('Total sessions must be a non-negative integer')
        return v
    
    @field_validator('total_messages')
    @classmethod
    def validate_total_messages(cls, v):
        """Validate total messages"""
        if not isinstance(v, int) or v < 0:
            raise ValueError('Total messages must be a non-negative integer')
        return v

class DebugInfo(BaseModel):
    """Model for debug information in responses"""
    retrieval_time: Optional[float] = None
    generation_time: Optional[float] = None
    total_time: Optional[float] = None
    chunks_retrieved: Optional[int] = None
    sources_used: Optional[List[str]] = None
    model_used: Optional[str] = None
    error_details: Optional[str] = None

class ErrorResponse(BaseModel):
    """Model for error responses"""
    error: str
    error_code: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    timestamp: Optional[str] = None
    
    @field_validator('error')
    @classmethod
    def validate_error(cls, v):
        """Validate error message"""
        if not v or not v.strip():
            raise ValueError('Error message cannot be empty')
        return v.strip()

# Example usage and validation functions
def validate_chat_request(data: dict) -> ChatRequest:
    """Validate and create ChatRequest from dictionary"""
    try:
        return ChatRequest(**data)
    except Exception as e:
        raise ValueError(f"Invalid chat request: {str(e)}")

def validate_chat_response(data: dict) -> ChatResponse:
    """Validate and create ChatResponse from dictionary"""
    try:
        return ChatResponse(**data)
    except Exception as e:
        raise ValueError(f"Invalid chat response: {str(e)}")

# Constants for validation
MAX_QUERY_LENGTH = 2000
MAX_RESPONSE_LENGTH = 10000
MAX_USERNAME_LENGTH = 50
MIN_USERNAME_LENGTH = 2
MAX_CHUNKS = 20
MIN_CHUNKS = 1
MAX_SESSION_ID_LENGTH = 100

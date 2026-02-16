"""
Amazon Rufus-style Conversational RAG Bot with RAGAS Evaluation
Generates suggested questions and provides answers with comprehensive metrics
"""

import os
import json
import time
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from fastapi import HTTPException
import pandas as pd

# RAGAS for evaluation
try:
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness, answer_relevancy, context_relevancy, 
        context_precision, answer_correctness
    )
    from datasets import Dataset
    RAGAS_AVAILABLE = True
    print("[RAG Bot] RAGAS evaluation framework loaded successfully")
except ImportError as e:
    print(f"[RAG Bot] Warning: RAGAS not available: {e}")
    print("[RAG Bot] Using custom evaluation metrics instead")
    RAGAS_AVAILABLE = False

# Local imports
from services.logger import setup_logger
from services.document_indexer import DocumentIndexer
from services.chat_models import ChatRequest, ChatResponse
from services.langchain_orchestrator import generate_chatbot_response
from utils.utils import OPENAI_API_KEY, QDRANT_DB_PATH, MODEL, TEMPERATURE
from utils.file_extractor import extract_text_with_metadata, get_file_type_from_filename
from utils.langchain_utils import LangChainManager

# Initialize logger
logger = setup_logger("conversational_rag_bot")

@dataclass
class ConversationMetrics:
    """Metrics for conversation quality evaluation"""
    question_quality: float
    answer_quality: float
    retrieval_precision: float
    retrieval_recall: float
    faithfulness: float
    answer_relevancy: float
    context_relevancy: float
    context_precision: float
    answer_correctness: float
    overall_score: float
    sources_count: int
    response_time: float

class EnhancedDataLoader:
    """Enhanced data loader for various document types"""
    
    def __init__(self):
        self.supported_formats = {
            '.json', '.csv', '.txt', '.pdf', '.docx', '.xlsx', '.md'
        }
    
    async def load_document(self, file_path: str) -> Dict[str, Any]:
        """Load and process document from various formats"""
        try:
            file_ext = get_file_type_from_filename(file_path)
            
            if file_ext == '.json':
                return await self._load_json(file_path)
            elif file_ext == '.csv':
                return await self._load_csv(file_path)
            elif file_ext in ['.txt', '.md']:
                return await self._load_text(file_path)
            elif file_ext in ['.pdf', '.docx']:
                return await self._load_document_file(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_ext}")
                
        except Exception as e:
            logger.error(f"Error loading document {file_path}: {str(e)}")
            return {"content": "", "metadata": {"error": str(e)}}
    
    async def _load_json(self, file_path: str) -> Dict[str, Any]:
        """Load JSON file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Convert JSON to text content
        if isinstance(data, list):
            content = "\n".join([json.dumps(item, indent=2) for item in data])
        else:
            content = json.dumps(data, indent=2)
        
        return {
            "content": content,
            "metadata": {
                "source": file_path,
                "type": "json",
                "items_count": len(data) if isinstance(data, list) else 1
            }
        }
    
    async def _load_csv(self, file_path: str) -> Dict[str, Any]:
        """Load CSV file"""
        df = pd.read_csv(file_path)
        content = df.to_string(index=False)
        
        return {
            "content": content,
            "metadata": {
                "source": file_path,
                "type": "csv",
                "rows": len(df),
                "columns": len(df.columns)
            }
        }
    
    async def _load_text(self, file_path: str) -> Dict[str, Any]:
        """Load text file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return {
            "content": content,
            "metadata": {
                "source": file_path,
                "type": "text",
                "character_count": len(content)
            }
        }
    
    async def _load_document_file(self, file_path: str) -> Dict[str, Any]:
        """Load PDF/DOCX file using file extractor"""
        with open(file_path, 'rb') as f:
            file_content = f.read()
        
        result = await extract_text_with_metadata(file_content, os.path.basename(file_path))
        
        return {
            "content": result["text"],
            "metadata": {
                "source": file_path,
                "type": result["file_type"],
                "character_count": result["character_count"],
                "word_count": result["word_count"]
            }
        }

class ConversationalRAGBot:
    """Amazon Rufus-style conversational RAG bot with evaluation"""
    
    def __init__(self):
        self.document_indexer = DocumentIndexer(QDRANT_DB_PATH)
        self.langchain_manager = LangChainManager(OPENAI_API_KEY, MODEL, TEMPERATURE)
        self.data_loader = EnhancedDataLoader()
        self.conversation_history = {}
        
    async def initialize(self):
        """Initialize the bot and load documents"""
        try:
            # Initialize Qdrant collection
            await self.document_indexer.initialize_collection()
            
            # Load all documents from uploads folder
            await self._load_all_documents()
            
            logger.info("Conversational RAG Bot initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing bot: {str(e)}")
            raise
    
    async def _load_all_documents(self):
        """Load all documents from uploads directory"""
        uploads_dir = "uploads"
        
        if not os.path.exists(uploads_dir):
            logger.warning(f"Uploads directory {uploads_dir} not found")
            return
        
        # Process all subdirectories
        for root, dirs, files in os.walk(uploads_dir):
            for file in files:
                file_path = os.path.join(root, file)
                
                # Skip hidden files
                if file.startswith('.'):
                    continue
                
                try:
                    # Load document
                    doc_data = await self.data_loader.load_document(file_path)
                    
                    if doc_data["content"].strip():
                        # Index document
                        success = await self.document_indexer.index_in_qdrantdb(
                            extracted_text=doc_data["content"],
                            file_name=file,
                            doc_type=doc_data["metadata"]["type"]
                        )
                        
                        if success:
                            logger.info(f"Loaded and indexed: {file}")
                        else:
                            logger.warning(f"Failed to index: {file}")
                    else:
                        logger.warning(f"Empty content in: {file}")
                        
                except Exception as e:
                    logger.error(f"Error processing {file}: {str(e)}")
    
    async def generate_suggested_questions(self, topic: str = None, num_questions: int = 5) -> List[Dict[str, Any]]:
        """Generate suggested questions based on available documents"""
        try:
            # Use the existing data-driven question generator
            from data_driven_question_generator import DataDrivenQuestionGenerator
            
            generator = DataDrivenQuestionGenerator()
            questions = generator.generate_question_from_data(topic, num_questions)
            
            # Enhance with additional metadata
            enhanced_questions = []
            for i, q in enumerate(questions):
                enhanced_questions.append({
                    "id": f"suggested_{i}",
                    "question": q["question"],
                    "preview_answer": q.get("answer", "")[:200] + "..." if len(q.get("answer", "")) > 200 else q.get("answer", ""),
                    "topic": q.get("topic_name", "General"),
                    "metrics": q.get("metrics", {}),
                    "sources_count": q.get("retrieved_sources", 0),
                    "confidence": q.get("metrics", {}).get("overall_score", 0.0)
                })
            
            return enhanced_questions
            
        except Exception as e:
            logger.error(f"Error generating suggested questions: {str(e)}")
            return []
    
    async def process_user_query(self, query: str, session_id: str, username: str = "user") -> Dict[str, Any]:
        """Process user query with comprehensive evaluation"""
        # Initialize default values
        final_response = "I apologize, but I couldn't process your request."
        refined_query = query
        response_time = 0.0
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
        context = ""
        extracted_docs = []
        
        try:
            start_time = time.time()
            
            # Get conversation history
            past_messages = self.conversation_history.get(session_id, [])
            
            # Generate response using existing orchestrator
            response_data = await generate_chatbot_response(
                query, past_messages, 5, username
            )
            
            final_response, response_time, prompt_tokens, completion_tokens, 
            total_tokens, context, extracted_docs = response_data
            refined_query = query  # Use original query as fallback
            
            # Calculate metrics
            metrics = await self._calculate_comprehensive_metrics(
                query, final_response, context, extracted_docs
            )
            
            # Update conversation history
            self.conversation_history.setdefault(session_id, []).extend([
                {"role": "user", "content": query},
                {"role": "assistant", "content": final_response}
            ])
            
            # Keep only last 10 messages
            if len(self.conversation_history[session_id]) > 20:
                self.conversation_history[session_id] = self.conversation_history[session_id][-20:]
            
            total_time = time.time() - start_time
            
            return {
                "response": final_response,
                "refined_query": refined_query,
                "metrics": metrics.__dict__,
                "sources": self._format_sources(extracted_docs),
                "token_usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens
                },
                "timing": {
                    "response_time": response_time,
                    "total_time": total_time
                },
                "session_id": session_id,
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")
    
    async def _calculate_comprehensive_metrics(self, question: str, answer: str, 
                                             context: str, retrieved_docs: List) -> ConversationMetrics:
        """Calculate comprehensive metrics using RAGAS and custom metrics"""
        try:
            # Basic metrics
            question_quality = self._calculate_question_quality(question)
            answer_quality = self._calculate_answer_quality(answer, context)
            retrieval_precision = self._calculate_retrieval_precision(question, retrieved_docs)
            retrieval_recall = self._calculate_retrieval_recall(question, retrieved_docs)
            
            # RAGAS metrics if available
            faithfulness = 0.0
            answer_relevancy = 0.0
            context_relevancy = 0.0
            context_precision = 0.0
            answer_correctness = 0.0
            
            if RAGAS_AVAILABLE and retrieved_docs:
                try:
                    ragas_metrics = await self._calculate_ragas_metrics(question, answer, context, retrieved_docs)
                    faithfulness = ragas_metrics.get("faithfulness", 0.0)
                    answer_relevancy = ragas_metrics.get("answer_relevancy", 0.0)
                    context_relevancy = ragas_metrics.get("context_relevancy", 0.0)
                    context_precision = ragas_metrics.get("context_precision", 0.0)
                    answer_correctness = ragas_metrics.get("answer_correctness", 0.0)
                except Exception as e:
                    logger.warning(f"RAGAS evaluation failed: {e}")
            
            # Overall score
            overall_score = (
                question_quality * 0.15 +
                answer_quality * 0.20 +
                retrieval_precision * 0.15 +
                retrieval_recall * 0.10 +
                faithfulness * 0.10 +
                answer_relevancy * 0.15 +
                context_relevancy * 0.10 +
                context_precision * 0.05
            )
            
            return ConversationMetrics(
                question_quality=question_quality,
                answer_quality=answer_quality,
                retrieval_precision=retrieval_precision,
                retrieval_recall=retrieval_recall,
                faithfulness=faithfulness,
                answer_relevancy=answer_relevancy,
                context_relevancy=context_relevancy,
                context_precision=context_precision,
                answer_correctness=answer_correctness,
                overall_score=overall_score
            )
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            # Return default metrics
            return ConversationMetrics(
                question_quality=0.5,
                answer_quality=0.5,
                retrieval_precision=0.5,
                retrieval_recall=0.5,
                faithfulness=0.5,
                answer_relevancy=0.5,
                context_relevancy=0.5,
                context_precision=0.5,
                answer_correctness=0.5,
                overall_score=0.5
            )
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            # Return default metrics
            return ConversationMetrics(
                question_quality=0.5, answer_quality=0.5, retrieval_precision=0.5,
                retrieval_recall=0.5, faithfulness=0.5, answer_relevancy=0.5,
                context_relevancy=0.5, context_precision=0.5, answer_correctness=0.5,
                overall_score=0.5, sources_count=0, response_time=0.0
            )
    
    async def _calculate_ragas_metrics(self, question: str, answer: str, 
                                    context: str, retrieved_docs: List) -> Dict[str, float]:
        """Calculate RAGAS metrics"""
        try:
            # Prepare dataset for RAGAS
            dataset_dict = {
                "question": [question],
                "answer": [answer],
                "contexts": [[doc.page_content for doc in retrieved_docs]],
                "ground_truths": [""]  # No ground truth available
            }
            
            dataset = Dataset.from_dict(dataset_dict)
            
            # Evaluate with RAGAS
            result = evaluate(
                dataset,
                metrics=[
                    faithfulness,
                    answer_relevancy,
                    context_relevancy,
                    context_precision
                ]
            )
            
            return {
                "faithfulness": result["faithfulness"],
                "answer_relevancy": result["answer_relevancy"],
                "context_relevancy": result["context_relevancy"],
                "context_precision": result["context_precision"]
            }
            
        except Exception as e:
            logger.error(f"Error calculating RAGAS metrics: {str(e)}")
            return {}
    
    def _calculate_question_quality(self, question: str) -> float:
        """Calculate question quality score"""
        # Simple heuristic-based scoring
        score = 0.5  # Base score
        
        # Length factor
        if 10 <= len(question.split()) <= 20:
            score += 0.2
        
        # Question words
        question_words = ["what", "how", "why", "when", "where", "which", "who"]
        if any(qw in question.lower().split() for qw in question_words):
            score += 0.2
        
        # Specificity indicators
        if any(indicator in question.lower() for indicator in ["specific", "example", "best", "compare"]):
            score += 0.1
        
        return min(1.0, score)
    
    def _calculate_answer_quality(self, answer: str, context: str) -> float:
        """Calculate answer quality score"""
        if not answer:
            return 0.0
        
        score = 0.3  # Base score
        
        # Length factor
        if 50 <= len(answer) <= 500:
            score += 0.2
        
        # Structure indicators
        if any(indicator in answer for indicator in [".", "\n", "•", "-"]):
            score += 0.2
        
        # Context alignment
        if context:
            context_words = set(context.lower().split())
            answer_words = set(answer.lower().split())
            overlap = len(context_words & answer_words) / len(answer_words) if answer_words else 0
            score += overlap * 0.3
        
        return min(1.0, score)
    
    def _calculate_retrieval_precision(self, question: str, retrieved_docs: List) -> float:
        """Calculate retrieval precision"""
        if not retrieved_docs:
            return 0.0
        
        # Simple heuristic: check if documents contain question keywords
        question_words = set(question.lower().split())
        relevant_docs = 0
        
        for doc in retrieved_docs:
            doc_words = set(doc.page_content.lower().split())
            overlap = len(question_words & doc_words) / len(question_words) if question_words else 0
            if overlap > 0.1:  # 10% overlap threshold
                relevant_docs += 1
        
        return relevant_docs / len(retrieved_docs)
    
    def _calculate_retrieval_recall(self, question: str, retrieved_docs: List) -> float:
        """Calculate retrieval recall"""
        # Simplified recall calculation
        if not retrieved_docs:
            return 0.0
        
        # Assume we need at least 3 relevant documents for good recall
        return min(1.0, len(retrieved_docs) / 3.0)
    
    def _calculate_faithfulness(self, answer: str, context: str) -> float:
        """Calculate faithfulness score"""
        if not context:
            return 0.5
        
        # Simple heuristic: check answer words against context
        context_words = set(context.lower().split())
        answer_words = set(answer.lower().split())
        
        # Calculate overlap
        overlap = len(context_words & answer_words) / len(answer_words) if answer_words else 0
        
        return min(1.0, overlap + 0.2)  # Boost base score
    
    def _calculate_answer_relevancy(self, question: str, answer: str) -> float:
        """Calculate answer relevancy"""
        if not answer:
            return 0.0
        
        # Check if answer addresses question keywords
        question_words = set(question.lower().split())
        answer_words = set(answer.lower().split())
        
        overlap = len(question_words & answer_words) / len(question_words) if question_words else 0
        
        return min(1.0, overlap + 0.3)  # Boost base score
    
    def _calculate_context_relevancy(self, question: str, context: str) -> float:
        """Calculate context relevancy"""
        if not context:
            return 0.0
        
        question_words = set(question.lower().split())
        context_words = set(context.lower().split())
        
        overlap = len(question_words & context_words) / len(question_words) if question_words else 0
        
        return min(1.0, overlap + 0.2)
    
    def _calculate_context_precision(self, retrieved_docs: List) -> float:
        """Calculate context precision"""
        if not retrieved_docs:
            return 0.0
        
        # Simple heuristic: check if documents are diverse and relevant
        # For now, return a score based on document count
        return min(1.0, len(retrieved_docs) / 5.0)
    
    def _format_sources(self, retrieved_docs: List) -> List[Dict[str, Any]]:
        """Format retrieved sources for display"""
        sources = []
        
        for i, doc in enumerate(retrieved_docs):
            source = {
                "id": f"source_{i}",
                "content": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
                "metadata": doc.metadata,
                "score": getattr(doc, 'score', 0.0)
            }
            sources.append(source)
        
        return sources

# Global bot instance
rag_bot = None

async def get_rag_bot() -> ConversationalRAGBot:
    """Get or initialize the global RAG bot instance"""
    global rag_bot
    if rag_bot is None:
        rag_bot = ConversationalRAGBot()
        await rag_bot.initialize()
    return rag_bot

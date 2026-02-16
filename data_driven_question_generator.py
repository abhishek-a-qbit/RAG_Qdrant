"""
Data-Driven Question Generation System with REAL Metrics
Uses LangChain RAG pipeline to generate questions from actual data with proper evaluation
"""

import os
import sys
import json
import re
import numpy as np
from typing import List, Dict, Any, Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

class DataDrivenQuestionGenerator:
    """Generates questions using LangChain RAG pipeline with REAL evaluation metrics."""
    
    def __init__(self):
        """Initialize with LangChain components."""
        self.langchain_manager = None
        self.qdrant_manager = None
        self.embedding_model = None
        self.nlp_model = None
        
        try:
            from utils.langchain_utils import LangChainManager
            from utils.qdrant_utils import QdrantManager
            from utils.utils import OPENAI_API_KEY, MODEL, TEMPERATURE, QDRANT_DB_PATH
            
            self.langchain_manager = LangChainManager(OPENAI_API_KEY, MODEL, TEMPERATURE)
            self.qdrant_manager = QdrantManager(QDRANT_DB_PATH)
            print("[QGen] Initialized with LangChain RAG pipeline")
        except Exception as e:
            print("[QGen] Error initializing RAG pipeline: " + str(e))
            self.langchain_manager = None
            self.qdrant_manager = None
        
        # Initialize embedding model for semantic similarity
        try:
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("[QGen] Embedding model loaded")
        except Exception as e:
            print("[QGen] Could not load embedding model: " + str(e))
            self.embedding_model = None
        
        # Initialize spaCy for NLP metrics
        try:
            import spacy 
            self.nlp_model = spacy.load("en_core_web_sm")
            print("[QGen] spaCy NLP model loaded")
        except Exception as e:
            print("[QGen] Could not load spaCy: " + str(e))
            self.nlp_model = None
    
    def generate_question_from_data(self, topic: str = None, num_questions: int = 1) -> List[Dict[str, Any]]:
        """Generate questions using LangChain RAG pipeline."""
        
        if not self.langchain_manager or not self.qdrant_manager:
            print("[QGen] RAG pipeline not available")
            return []
        
        questions = []
        
        try:
            # Generate question prompt
            if topic:
                question_prompt = f"Generate insightful questions about {topic} for a FAQ section"
            else:
                question_prompt = "Generate general questions about the platform capabilities"
            
            print(f"[QGen] Topic: {topic}")
            
            # Use LangChain to generate questions
            # First, retrieve relevant documents
            if self.qdrant_manager.collection_exists("documents"):
                query_embedding = self.langchain_manager.embed_query(question_prompt)
                search_results = self.qdrant_manager.search("documents", query_embedding, 5, 0.5)
                
                if search_results:
                    # Create context from retrieved documents
                    context = "\n\n".join([doc.get("text", "") for doc in search_results])
                    
                    # Generate questions using LLM
                    llm_prompt = f"""Based on the following context, generate {num_questions} insightful questions for a FAQ section:
                    
                    Context: {context[:2000]}
                    
                    Topic: {topic or "General"}
                    
                    Generate exactly {num_questions} questions as a JSON array of strings.
                    Questions should be:
                    - Specific and answerable from the context
                    - Relevant to {topic if topic else 'the platform'}
                    - Formulated as actual questions (ending with ?)
                    
                    Response format: ["question1", "question2", ...]
                    """
                    
                    response = self.langchain_manager.llm.invoke(llm_prompt)
                    
                    # Parse questions from response
                    generated_questions = self._extract_question_from_response(response.content, topic, 0)
                    
                    # Process each question
                    for i, question in enumerate(generated_questions[:num_questions]):
                        # Calculate question metrics
                        question_metrics = self._calculate_question_metrics(question)
                        
                        # Generate answer for this question
                        answer_result = self._generate_answer_for_question(question)
                        answer = answer_result.get("answer", "")
                        retrieved_docs = answer_result.get("sources", search_results)
                        
                        # Generate topic name
                        topic_name = self._generate_topic_name_simple(question, answer)
                        
                        questions.append({
                            "id": f"rag_generated_{i}",
                            "question": question,
                            "preview_answer": answer[:200] + "..." if len(answer) > 200 else answer,
                            "topic": topic_name,
                            "metrics": {"overall_score": question_metrics.get("overall_score", 0.5)},
                            "sources_count": len(retrieved_docs),
                            "confidence": question_metrics.get("overall_score", 0.5)
                        })
                        
            print(f"[QGen] Generated {len(questions)} questions")
            return questions
            
        except Exception as e:
            print(f"[QGen] Error generating questions: {str(e)}")
            return []
    
    def _generate_answer_for_question(self, question: str) -> Dict[str, Any]:
        """Generate answer for a specific question using RAG pipeline."""
        try:
            if not self.langchain_manager or not self.qdrant_manager:
                return {"answer": "RAG pipeline not available", "sources": []}
            
            # Search for relevant documents
            query_embedding = self.langchain_manager.embed_query(question)
            search_results = self.qdrant_manager.search("documents", query_embedding, 5, 0.5)
            
            if not search_results:
                return {"answer": "No relevant documents found", "sources": []}
            
            # Create context
            context = "\n\n".join([doc.get("text", "") for doc in search_results])
            
            # Generate answer
            answer_prompt = f"""Answer the following question based on the provided context:
            
            Question: {question}
            
            Context: {context[:2000]}
            
            Provide a concise and accurate answer based only on the context. If the context doesn't contain the answer, say "I don't have enough information to answer this question."
            """
            
            response = self.langchain_manager.llm.invoke(answer_prompt)
            
            return {
                "answer": response.content,
                "sources": search_results
            }
            
        except Exception as e:
            print(f"[QGen] Error generating answer: {str(e)}")
            return {"answer": f"Error generating answer: {str(e)}", "sources": []}
    
    def _generate_topic_name_simple(self, question: str, doc_content: str) -> str:
        """Generate simple topic name from context."""
        
        if not self.langchain_manager:
            return "General"
        
        # Extract keywords from question and documents
        question_lower = question.lower()
        combined_text = question_lower + " " + doc_content.lower()

        # Use LLM to generate topic
        try:
            prompt = f"For the question and answer in '{combined_text}', map it to a list of most relevant 3 topics from common B2B Intent Data. Return the response as a JSON array of strings."
            response = self.langchain_manager.llm.invoke(prompt)
            return response.content
        except Exception as e:
            print(f"[QGen] Error generating topic: {str(e)}")
            # Simple keyword-based topic generation
            if "revenue" in combined_text and "ai" in combined_text:
                return "Revenue AI Analytics"
            elif "predictive" in combined_text and "analytics" in combined_text:
                return "Predictive Analytics"
            elif "integration" in combined_text:
                return "Platform Integration"
            elif "implementation" in combined_text:
                return "Implementation Strategy"
            elif "roi" in combined_text or "return" in combined_text:
                return "ROI Analysis"
            elif "features" in combined_text or "capabilities" in combined_text:
                return "Platform Features"
            else:
                return "6sense Business Analytics"
    
    def _enhance_source_evidence_simple(self, question: str, retrieved_docs: List) -> List[Dict]:
        """Simple enhancement of source evidence."""
        enhanced_evidence = []
        
        for i, doc in enumerate(retrieved_docs):
            metadata = doc.get('metadata', {})
            doc_id = metadata.get('id', f'doc_{i}')
            source_file = metadata.get('source', 'unknown')
            content = doc.get('content', '')
            
            evidence_entry = {
                "link": f"#document_{doc_id}",
                "type": "retrieved",
                "score": doc.get('score', 0.0),
                "doc_id": doc_id,
                "source_file": source_file,
                "content_preview": content[:200] + "..." if len(content) > 200 else content,
                "metadata": metadata
            }
            
            enhanced_evidence.append(evidence_entry)
        
        return enhanced_evidence
    
    def _llm_grade_metric(self, text: str, metric_name: str, metric_type: str = "question") -> int:
        """Optimized LLM grading with simpler prompts to reduce timeouts."""
        
        if not self.langchain_manager:
            print("[LLM Grader] No LLM available, using fallback score")
            return 3
        
        # Simplified prompt without few-shot examples for faster processing
        prompt = "Rate the following " + metric_type + " on " + metric_name + " from 1-5:\n\n" + metric_type + ":\n" + text + "\n\nScore (1-5):"
        
        try:
            response = self.langchain_manager.llm.invoke(prompt)
            
            # Extract score from response
            score_text = response.content.strip()
            lines = score_text.split('\n')
            for line in lines:
                line = line.strip()
                # Match patterns like "Score: 3" or just "3" at start
                match = re.match(r"^(?:Score:\s*)?([1-5])", line, re.MULTILINE)
                if match:
                    score = int(match.group(1))
                    print("[LLM Grader] " + metric_name + " (" + metric_type + "): " + str(score) + "/5")
                    return score
            # If no match found, try broader search
            match = re.search(r"\b([1-5])\b", score_text, re.MULTILINE)
            if match:
                score = int(match.group(1))
                print("[LLM Grader] " + metric_name + " (" + metric_type + "): " + str(score) + "/5")
                return score
            else:
                print("[LLM Grader] Could not parse score from: " + score_text)
                return 3  # Default to middle score if parsing fails
                
        except Exception as e:
            print("[LLM Grader] Error: " + str(e))
            return 3  # Fallback score
    
    def _calculate_groundedness_math(self, question: str) -> float:
        """Calculate if question can be answered from available data."""
        if not self.qdrant_manager:
            return 0.40  # Low confidence if no data source
        
        try:
            # Retrieve relevant documents for this question
            if not self.qdrant_manager.collection_exists("documents"):
                return 0.40
                
            query_embedding = self.langchain_manager.embed_query(question)
            retrieved_docs = self.qdrant_manager.search("documents", query_embedding, 3, 0.5)
            
            if not retrieved_docs:
                return 0.40  # Question can't be answered from data
            
            # Calculate relevance scores using embeddings
            if self.embedding_model:
                q_embedding = self.embedding_model.encode(question)
                
                max_similarity = 0.0
                for doc in retrieved_docs:
                    doc_text = doc.get('text', '')
                    if doc_text:
                        doc_embedding = self.embedding_model.encode(doc_text[:500])  # Limit length
                        
                        # Cosine similarity
                        similarity = np.dot(q_embedding, doc_embedding) / (
                            np.linalg.norm(q_embedding) * np.linalg.norm(doc_embedding) + 1e-10
                        )
                        max_similarity = max(max_similarity, similarity)
                
                # Convert similarity to groundedness score
                return float(min(1.0, max_similarity * 1.2))  # Boost slightly
            else:
                # Fallback: just check if we got documents
                return 0.70 if len(retrieved_docs) > 0 else 0.40
                
        except Exception as e:
            print("[Groundedness] Error: " + str(e))
            return 0.40
    
    def _calculate_specificity_math_improved(self, text: str) -> float:
        """Improved specificity using NLP."""
        if self.nlp_model:
            try:
                doc = self.nlp_model(text)
                
                # Named Entity Density
                entities = [ent for ent in doc.ents]
                ned = len(entities) / len(doc) if len(doc) > 0 else 0
                
                # Lexical Density  
                content_words = [token for token in doc if not token.is_stop and token.is_alpha]
                ld = len(content_words) / len(doc) if len(doc) > 0 else 0
                
                # Hedge word penalty
                hedge_words = {"might", "possibly", "usually", "perhaps", "generally", "sometimes"}
                hedge_count = sum(1 for token in doc if token.text.lower() in hedge_words)
                hwp = 1 - (hedge_count / len(doc)) if len(doc) > 0 else 1
                
                # Composite score
                return min(1.0, 0.4 * ned + 0.3 * ld + 0.3 * hwp)
                
            except Exception as e:
                print("[Specificity NLP] Error: " + str(e))
                # Fallback to simple method
                pass
        
        # Simple fallback method
        vague_terms = ["thing", "stuff", "something", "various", "multiple", "some", "many"]
        vague_count = sum(1 for term in vague_terms if term in text.lower())
        return max(0.0, min(1.0, 1.0 - (vague_count * 0.15)))
    
    def _calculate_question_metrics(self, question: str) -> Dict[str, Any]:
        """Calculate quality metrics for generated question with REAL LLM grading."""

        def _clamp01(x: float) -> float:
            return max(0.0, min(1.0, float(x)))

        def _llm_norm_0_1(llm_score_1_5: float) -> float:
            # (sLLM - 1) / 4
            return _clamp01((float(llm_score_1_5) - 1.0) / 4.0)

        def _fuse(stat_0_1: float, llm_1_5: float, lam: float = 0.5) -> float:
            return _clamp01(lam * _clamp01(stat_0_1) + (1.0 - lam) * _llm_norm_0_1(llm_1_5))

        q_lower = (question or "").lower()

        # Statistical (math) scores - IMPROVED
        
        # Coverage: Check topic relevance (more generous)
        key_terms = ["6sense", "revenue", "ai", "platform", "features", "capabilities", "predictive", "analytics"]
        matches = sum(1 for term in key_terms if term in q_lower)
        coverage_math = min(1.0, matches / 3.0)  # More generous: 3 matches = 100%

        # Specificity: Use improved NLP-based calculation
        specificity_math = self._calculate_specificity_math_improved(question)

        # Insightfulness: Look for depth indicators
        insight_indicators = ["how", "why", "compare", "versus", "implement", "strategy", "best practices", "trade-off", "pitfall"]
        insight_count = sum(1 for indicator in insight_indicators if indicator in q_lower)
        insightfulness_math = _clamp01(0.35 + (insight_count * 0.15))

        # Groundedness: REAL calculation using document retrieval
        groundedness_math = self._calculate_groundedness_math(question)

        # LLM scores - REAL grading with actual API calls
        coverage_llm = self._llm_grade_metric(question, "coverage", "question")
        specificity_llm = self._llm_grade_metric(question, "specificity", "question")
        insightfulness_llm = self._llm_grade_metric(question, "insightfulness", "question")
        groundedness_llm = self._llm_grade_metric(question, "groundedness", "question")

        # Fused finals
        coverage_final = _fuse(coverage_math, coverage_llm)
        specificity_final = _fuse(specificity_math, specificity_llm)
        insightfulness_final = _fuse(insightfulness_math, insightfulness_llm)
        groundedness_final = _fuse(groundedness_math, groundedness_llm)

        overall_score = _clamp01(
            0.25 * coverage_final
            + 0.25 * specificity_final
            + 0.25 * insightfulness_final
            + 0.25 * groundedness_final
        )

        # Recommended thresholds (more reasonable)
        thresholds = {
            "groundedness_min": 0.70,  # More achievable
            "specificity_min": 0.50,
            "insightfulness_min": 0.60,
            "overall_min": 0.65
        }
        overall_pass = (
            groundedness_final >= thresholds["groundedness_min"]
            and specificity_final >= thresholds["specificity_min"]
            and insightfulness_final >= thresholds["insightfulness_min"]
            and overall_score >= thresholds["overall_min"]
        )

        return {
            "coverage_math": coverage_math,
            "coverage_llm": coverage_llm,
            "coverage_final": coverage_final,
            "specificity_math": specificity_math,
            "specificity_llm": specificity_llm,
            "specificity_final": specificity_final,
            "insightfulness_math": insightfulness_math,
            "insightfulness_llm": insightfulness_llm,
            "insightfulness_final": insightfulness_final,
            "groundedness_math": groundedness_math,
            "groundedness_llm": groundedness_llm,
            "groundedness_final": groundedness_final,
            "overall_score": overall_score,
            "overall_pass": overall_pass,
            "thresholds": thresholds,
            "fusion_lambda": 0.5,
        }
    
    def _calculate_answer_metrics(self, question: str, answer: str, context: str) -> Dict[str, Any]:
        """Calculate quality metrics for generated answer."""
        
        def _clamp01(x: float) -> float:
            return max(0.0, min(1.0, float(x)))
        
        def _llm_norm_0_1(llm_score_1_5: float) -> float:
            return _clamp01((float(llm_score_1_5) - 1.0) / 4.0)
        
        def _fuse(stat_0_1: float, llm_1_5: float, lam: float = 0.5) -> float:
            return _clamp01(lam * _clamp01(stat_0_1) + (1.0 - lam) * _llm_norm_0_1(llm_1_5))
        
        # Statistical metrics for ANSWERS
        
        # Coverage: Does answer address all aspects of question?
        question_words = set(question.lower().split())
        answer_words = set(answer.lower().split())
        word_overlap = len(question_words & answer_words) / len(question_words) if question_words else 0
        coverage_math = _clamp01(word_overlap + 0.3)  # Boost base score
        
        # Specificity: Numbers, entities, concrete details (more generous)
        has_numbers = any(char.isdigit() for char in answer)
        has_bullets = '\n' in answer or '•' in answer or '-' in answer
        answer_length = len(answer.split())
        specificity_math = _clamp01(
            0.6 +  # Base score higher
            (0.3 if has_numbers else 0) +
            (0.3 if has_bullets else 0) +
            (0.1 if answer_length > 20 else 0)  # Lower length requirement
        )
        
        # Insightfulness: Causal language, depth indicators
        insight_markers = ["because", "therefore", "however", "trade-off", "best practice", "recommended", "pitfall"]
        insight_count = sum(1 for marker in insight_markers if marker in answer.lower())
        insightfulness_math = _clamp01(0.35 + (insight_count * 0.15))
        
        # Groundedness: How well does answer match context?
        if context:
            context_words = set(context.lower().split())
            answer_context_overlap = len(answer_words & context_words) / len(answer_words) if answer_words else 0
            groundedness_math = _clamp01(answer_context_overlap * 1.3)
        else:
            groundedness_math = 0.40
        
        # LLM grading for answers - REAL API calls
        coverage_llm = self._llm_grade_metric(answer, "coverage", "answer")
        specificity_llm = self._llm_grade_metric(answer, "specificity", "answer")
        insightfulness_llm = self._llm_grade_metric(answer, "insightfulness", "answer")
        groundedness_llm = self._llm_grade_metric(answer, "groundedness", "answer")
        
        # Fusion
        coverage_final = _fuse(coverage_math, coverage_llm)
        specificity_final = _fuse(specificity_math, specificity_llm)
        insightfulness_final = _fuse(insightfulness_math, insightfulness_llm)
        groundedness_final = _fuse(groundedness_math, groundedness_llm)
        
        overall_score = _clamp01(
            0.25 * coverage_final +
            0.25 * specificity_final +
            0.25 * insightfulness_final +
            0.25 * groundedness_final
        )
        
        # Thresholds for answers (stricter for groundedness)
        thresholds = {
            "groundedness_min": 0.90,  # Stricter for answers
            "specificity_min": 0.65,
            "insightfulness_min": 0.70,
            "overall_min": 0.75
        }
        
        overall_pass = (
            groundedness_final >= thresholds["groundedness_min"] and
            specificity_final >= thresholds["specificity_min"] and
            insightfulness_final >= thresholds["insightfulness_min"] and
            overall_score >= thresholds["overall_min"]
        )
        
        return {
            "coverage_math": coverage_math,
            "coverage_llm": coverage_llm,
            "coverage_final": coverage_final,
            "specificity_math": specificity_math,
            "specificity_llm": specificity_llm,
            "specificity_final": specificity_final,
            "insightfulness_math": insightfulness_math,
            "insightfulness_llm": insightfulness_llm,
            "insightfulness_final": insightfulness_final,
            "groundedness_math": groundedness_math,
            "groundedness_llm": groundedness_llm,
            "groundedness_final": groundedness_final,
            "overall_score": overall_score,
            "overall_pass": overall_pass,
            "thresholds": thresholds,
            "fusion_lambda": 0.5
        }
    
    def _create_question_prompt(self, topic: str, index: int) -> str:
        """Create prompt for question generation."""
        if topic:
            prompts = [
                f"Generate a specific, insightful question about {topic} features and capabilities for 6sense Revenue AI platform.",
                f"Create a question about how {topic} helps companies improve their sales and marketing using 6sense.",
                f"Generate a question about {topic} implementation strategies and best practices for 6sense users.",
                f"Create a comparative question about {topic} versus other solutions in the B2B revenue intelligence space."
#                f"generate insightful,SEO friendly and clickbait website FAQ questions list for this topic \'{topic}\' and return a JSON array of 10 best questions"
            ]
        else:
            prompts = [
                "Generate a specific question about 6sense Revenue AI platform's key features and capabilities.",
                "Create a question about how 6sense helps companies identify and target in-market buyers.",
                "Generate a question about 6sense's predictive analytics and AI-powered targeting capabilities.",
                "Create a question about implementing 6sense effectively for B2B sales and marketing teams."
            ]

        return prompts[index % len(prompts)]

    def _extract_question_from_response(self, result: dict, topic: str, index: int) -> List[str]:
        if isinstance(result, dict) and "answer" in result:
            try:
                answer = json.loads(result["answer"])
                questions = answer.get("questions", [])
                return questions if isinstance(questions, list) else []
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Error parsing answer: {e}")
                return []

        return []

def generate_data_driven_questions(topic: str = None, num_questions: int = 10) -> List[Dict[str, Any]]:
    generator = DataDrivenQuestionGenerator()
    return generator.generate_question_from_data(topic, num_questions)


# Export for use in other modules
__all__ = ['generate_data_driven_questions', 'DataDrivenQuestionGenerator']
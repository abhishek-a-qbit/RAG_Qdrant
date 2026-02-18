"""
API-Based Full Pipeline Test
Tests the complete RAG system using actual API endpoints
"""

import os
import sys
import json
import time
import requests
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

class APIPipelineTester:
    """Test the full RAG pipeline using API endpoints."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """Initialize API tester."""
        self.base_url = base_url
        self.session = requests.Session()
        self.test_results = {
            "timestamp": datetime.now().isoformat(),
            "api_tests": {},
            "qa_results": [],
            "metrics": {},
            "summary": {}
        }
        
        print("=" * 80)
        print("🚀 API-BASED FULL PIPELINE TESTER")
        print("=" * 80)
        print(f"Base URL: {self.base_url}")
    
    def test_api_health(self) -> Dict[str, Any]:
        """Test API health endpoint."""
        print("\n1️⃣ Testing API Health...")
        
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=10)
            
            if response.status_code == 200:
                health_data = response.json()
                print(f"   ✅ API Status: {health_data.get('status', 'unknown')}")
                print(f"   Services: {health_data.get('services', {})}")
                
                self.test_results["api_tests"]["health"] = {
                    "status": "success",
                    "data": health_data
                }
                return health_data
            else:
                print(f"   ❌ Health check failed: {response.status_code}")
                self.test_results["api_tests"]["health"] = {
                    "status": "failed",
                    "error": f"HTTP {response.status_code}"
                }
                return {}
                
        except requests.exceptions.RequestException as e:
            print(f"   ❌ API connection error: {str(e)}")
            self.test_results["api_tests"]["health"] = {
                "status": "error",
                "error": str(e)
            }
            return {}
    
    def test_collections_endpoint(self) -> Dict[str, Any]:
        """Test collections endpoint."""
        print("\n2️⃣ Testing Collections...")
        
        try:
            response = self.session.get(f"{self.base_url}/collections", timeout=10)
            
            if response.status_code == 200:
                collections_data = response.json()
                print(f"   ✅ Collections: {len(collections_data.get('collections', []))} found")
                
                for collection in collections_data.get('collections', []):
                    print(f"      - {collection.get('name', 'unknown')}: {collection.get('points_count', 0)} points")
                
                self.test_results["api_tests"]["collections"] = {
                    "status": "success",
                    "data": collections_data
                }
                return collections_data
            else:
                print(f"   ❌ Collections check failed: {response.status_code}")
                self.test_results["api_tests"]["collections"] = {
                    "status": "failed",
                    "error": f"HTTP {response.status_code}"
                }
                return {}
                
        except requests.exceptions.RequestException as e:
            print(f"   ❌ Collections error: {str(e)}")
            self.test_results["api_tests"]["collections"] = {
                "status": "error",
                "error": str(e)
            }
            return {}
    
    def generate_test_questions(self, topics: List[str]) -> List[Dict[str, Any]]:
        """Generate test questions using RAG pipeline from actual document content."""
        questions = []
        
        print(f"🔍 Generating RAG-based questions for {len(topics)} topics...")
        
        for topic in topics:
            try:
                # Use RAG pipeline to generate context-aware questions
                print(f"   Processing topic: {topic}")
                
                # Query for relevant documents about this topic
                query_data = {
                    "query": f"key features and information about {topic}",
                    "collection_name": "documents",
                    "limit": 5,
                    "similarity_threshold": 0.5
                }
                
                start_time = time.time()
                response = self.session.post(
                    f"{self.base_url}/query",
                    json=query_data,
                    timeout=30
                )
                response_time = time.time() - start_time
                
                if response.status_code == 200:
                    result = response.json()
                    context = result.get("answer", "")
                    sources = result.get("sources", [])
                    
                    # Generate questions based on retrieved context
                    if context and len(context) > 50:  # Only if we got meaningful context
                        # Use the data-driven question generator
                        try:
                            from data_driven_question_generator import DataDrivenQuestionGenerator
                            qgen = DataDrivenQuestionGenerator()
                            generated_questions = qgen.generate_question_from_data(topic, 2)
                            
                            for q_data in generated_questions:
                                questions.append({
                                    "topic": topic,
                                    "question": q_data["question"],
                                    "context_used": context[:200] + "..." if len(context) > 200 else context,
                                    "sources_count": len(sources),
                                    "generation_method": "RAG-based",
                                    "response_time": response_time
                                })
                                print(f"   ✅ Generated: {q_data['question'][:50]}...")
                                
                        except Exception as e:
                            print(f"   ⚠️ Fallback to template for {topic}: {str(e)}")
                            # Fallback to template if RAG generation fails
                            fallback_questions = [
                                f"What are the key features of {topic}?",
                                f"How does {topic} benefit businesses?"
                            ]
                            
                            for question in fallback_questions:
                                questions.append({
                                    "topic": topic,
                                    "question": question,
                                    "context_used": context[:200] + "..." if len(context) > 200 else context,
                                    "sources_count": len(sources),
                                    "generation_method": "template-fallback",
                                    "response_time": response_time
                                })
                    else:
                        # No context found, use template
                        template_questions = [
                            f"What are the key features of {topic}?",
                            f"How does {topic} benefit businesses?"
                        ]
                        
                        for question in template_questions:
                            questions.append({
                                "topic": topic,
                                "question": question,
                                "context_used": "No context available",
                                "sources_count": 0,
                                "generation_method": "template-only",
                                "response_time": response_time
                            })
                        print(f"   ⚠️ No context for {topic}, using templates")
                        
                else:
                    print(f"   ❌ Failed to get context for {topic}: HTTP {response.status_code}")
                    # Add template question as fallback
                    questions.append({
                        "topic": topic,
                        "question": f"What are the key features of {topic}?",
                        "context_used": "Query failed",
                        "sources_count": 0,
                        "generation_method": "template-fallback",
                        "response_time": response_time
                    })
                    
            except Exception as e:
                print(f"   ❌ Error processing {topic}: {str(e)}")
                # Add template question as ultimate fallback
                questions.append({
                    "topic": topic,
                    "question": f"What are the key features of {topic}?",
                    "context_used": f"Error: {str(e)}",
                    "sources_count": 0,
                    "generation_method": "error-fallback",
                    "response_time": 0.0
                })
        
        print(f"✅ Generated {len(questions)} RAG-based questions")
        return questions
    
    def test_query_pipeline(self, questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Test the query pipeline with generated questions."""
        print(f"\n3️⃣ Testing Query Pipeline ({len(questions)} questions)...")
        
        qa_results = []
        
        for i, q_data in enumerate(questions):
            print(f"   Processing Q{i+1}: {q_data['question'][:50]}...")
            
            # Prepare query request
            query_data = {
                "query": q_data["question"],
                "collection_name": "documents",
                "limit": 5,
                "similarity_threshold": 0.5
            }
            
            try:
                start_time = time.time()
                response = self.session.post(
                    f"{self.base_url}/query",
                    json=query_data,
                    timeout=30
                )
                response_time = time.time() - start_time
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Extract sources and metrics from new format
                    sources = result.get("sources", [])
                    metrics = result.get("metrics", {})
                    
                    qa_result = {
                        "id": i,
                        "topic": q_data["topic"],
                        "question": q_data["question"],
                        "answer": result.get("answer", ""),
                        "sources": sources,
                        "confidence_score": result.get("confidence_score", 0.0),
                        "response_time": response_time,
                        "sources_count": len(sources),
                        "api_response_time": result.get("response_time", 0.0),
                        "status": "success",
                        # New fields for enhanced testing
                        "source_links": [s.get("link", "") for s in sources if s.get("link")],
                        "metrics": metrics,
                        "coverage": metrics.get("coverage", 0.0),
                        "specificity": metrics.get("specificity", 0.0),
                        "insightfulness": metrics.get("insightfulness", 0.0),
                        "groundedness": metrics.get("groundedness", 0.0),
                        "overall_score": metrics.get("overall_score", 0.0)
                    }
                    
                    # Calculate basic metrics
                    qa_result["answer_length"] = len(qa_result["answer"])
                    qa_result["question_length"] = len(qa_result["question"])
                    qa_result["has_sources"] = len(qa_result["sources"]) > 0
                    
                    # Quality indicators
                    qa_result["answer_quality"] = self._assess_answer_quality(qa_result)
                    
                    qa_results.append(qa_result)
                    
                    print(f"      ✅ Answer: {qa_result['answer'][:50]}...")
                    print(f"      📊 Sources: {qa_result['sources_count']}, Confidence: {qa_result['confidence_score']:.3f}")
                    if qa_result.get("overall_score"):
                        print(f"      🎯 Quality Score: {qa_result['overall_score']:.3f} (Coverage: {qa_result['coverage']:.2f}, Specificity: {qa_result['specificity']:.2f})")
                    if qa_result.get("source_links"):
                        print(f"      🔗 Source Links: {len(qa_result['source_links'])} available")
                    
                else:
                    print(f"      ❌ Query failed: HTTP {response.status_code}")
                    qa_results.append({
                        "id": i,
                        "topic": q_data["topic"],
                        "question": q_data["question"],
                        "status": "failed",
                        "error": f"HTTP {response.status_code}",
                        "response_time": response_time
                    })
                    
            except requests.exceptions.RequestException as e:
                print(f"      ❌ Query error: {str(e)}")
                qa_results.append({
                    "id": i,
                    "topic": q_data["topic"],
                    "question": q_data["question"],
                    "status": "error",
                    "error": str(e),
                    "response_time": time.time() - start_time
                })
        
        self.test_results["qa_results"] = qa_results
        return qa_results
    
    def test_chat_pipeline(self, questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Test the chat pipeline."""
        print(f"\n4️⃣ Testing Chat Pipeline ({len(questions)} questions)...")
        
        chat_results = []
        
        for i, q_data in enumerate(questions[:3]):  # Test fewer for chat
            print(f"   Chat Q{i+1}: {q_data['question'][:50]}...")
            
            # Prepare chat request
            chat_data = {
                "query": q_data["question"],
                "username": "test_user",
                "session_id": f"test_session_{int(time.time())}",
                "collection_name": "documents",
                "limit": 5,
                "similarity_threshold": 0.5
            }
            
            try:
                start_time = time.time()
                response = self.session.post(
                    f"{self.base_url}/chat",
                    json=chat_data,
                    timeout=30
                )
                response_time = time.time() - start_time
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Extract sources and metrics from new format
                    sources = result.get("sources", [])
                    metrics = result.get("metrics", {})
                    
                    chat_result = {
                        "id": i,
                        "topic": q_data["topic"],
                        "question": q_data["question"],
                        "answer": result.get("response", result.get("answer", "")),
                        "session_id": result.get("session_id", ""),
                        "response_time": response_time,
                        "status": "success",
                        # New fields for enhanced testing
                        "sources": sources,
                        "sources_count": len(sources),
                        "source_links": [s.get("link", "") for s in sources if s.get("link")],
                        "metrics": metrics,
                        "coverage": metrics.get("coverage", 0.0),
                        "specificity": metrics.get("specificity", 0.0),
                        "insightfulness": metrics.get("insightfulness", 0.0),
                        "groundedness": metrics.get("groundedness", 0.0),
                        "overall_score": metrics.get("overall_score", 0.0)
                    }
                    
                    chat_results.append(chat_result)
                    
                    print(f"      ✅ Chat Response: {chat_result['answer'][:50]}...")
                    print(f"      🆔 Session: {chat_result['session_id']}")
                    if chat_result.get("overall_score"):
                        print(f"      🎯 Quality Score: {chat_result['overall_score']:.3f}")
                    if chat_result.get("source_links"):
                        print(f"      🔗 Source Links: {len(chat_result['source_links'])} available")
                    
                else:
                    print(f"      ❌ Chat failed: HTTP {response.status_code}")
                    chat_results.append({
                        "id": i,
                        "topic": q_data["topic"],
                        "question": q_data["question"],
                        "status": "failed",
                        "error": f"HTTP {response.status_code}"
                    })
                    
            except requests.exceptions.RequestException as e:
                print(f"      ❌ Chat error: {str(e)}")
                chat_results.append({
                    "id": i,
                    "topic": q_data["topic"],
                    "question": q_data["question"],
                    "status": "error",
                    "error": str(e)
                })
        
        self.test_results["api_tests"]["chat"] = {
            "status": "completed",
            "results": chat_results
        }
        
        return chat_results
    
    def _assess_answer_quality(self, qa_result: Dict[str, Any]) -> Dict[str, Any]:
        """Assess basic answer quality metrics."""
        answer = qa_result.get("answer", "")
        question = qa_result.get("question", "")
        
        quality = {
            "length_score": min(1.0, len(answer) / 200),  # Prefer longer answers
            "has_numbers": 1.0 if any(char.isdigit() for char in answer) else 0.0,
            "question_relevance": 0.0,  # Simple word overlap
            "source_based": 1.0 if qa_result.get("has_sources", False) else 0.0,
            "confidence_bonus": qa_result.get("confidence_score", 0.0)
        }
        
        # Simple question relevance (word overlap)
        question_words = set(question.lower().split())
        answer_words = set(answer.lower().split())
        if question_words:
            overlap = len(question_words & answer_words) / len(question_words)
            quality["question_relevance"] = min(1.0, overlap * 2)  # Boost relevance
        
        # Calculate overall quality score
        quality["overall_score"] = (
            quality["length_score"] * 0.2 +
            quality["has_numbers"] * 0.1 +
            quality["question_relevance"] * 0.3 +
            quality["source_based"] * 0.2 +
            quality["confidence_bonus"] * 0.2
        )
        
        return quality
    
    def calculate_comprehensive_metrics(self, qa_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate comprehensive metrics from QA results."""
        print("\n5️⃣ Calculating Comprehensive Metrics...")
        
        if not qa_results:
            return {"error": "No QA results to analyze"}
        
        successful_results = [r for r in qa_results if r.get("status") == "success"]
        
        if not successful_results:
            return {"error": "No successful QA results"}
        
        # Basic metrics
        metrics = {
            "total_questions": len(qa_results),
            "successful_queries": len(successful_results),
            "success_rate": len(successful_results) / len(qa_results),
            "avg_response_time": np.mean([r["response_time"] for r in successful_results]),
            "avg_answer_length": np.mean([r["answer_length"] for r in successful_results]),
            "avg_sources_count": np.mean([r["sources_count"] for r in successful_results]),
            "avg_confidence_score": np.mean([r["confidence_score"] for r in successful_results])
        }
        
        # Quality metrics
        quality_scores = [r["answer_quality"]["overall_score"] for r in successful_results]
        metrics.update({
            "avg_quality_score": np.mean(quality_scores),
            "min_quality_score": np.min(quality_scores),
            "max_quality_score": np.max(quality_scores),
            "high_quality_count": sum(1 for score in quality_scores if score > 0.7),
            "medium_quality_count": sum(1 for score in quality_scores if 0.4 <= score <= 0.7),
            "low_quality_count": sum(1 for score in quality_scores if score < 0.4)
        })
        
        # Topic-based metrics
        topics = list(set(r["topic"] for r in successful_results))
        topic_metrics = {}
        for topic in topics:
            topic_results = [r for r in successful_results if r["topic"] == topic]
            if topic_results:
                topic_metrics[topic] = {
                    "count": len(topic_results),
                    "avg_quality": np.mean([r["answer_quality"]["overall_score"] for r in topic_results]),
                    "avg_confidence": np.mean([r["confidence_score"] for r in topic_results])
                }
        
        metrics["topic_metrics"] = topic_metrics
        
        # Performance distribution
        response_times = [r["response_time"] for r in successful_results]
        metrics.update({
            "fast_responses": sum(1 for t in response_times if t < 2.0),
            "medium_responses": sum(1 for t in response_times if 2.0 <= t <= 5.0),
            "slow_responses": sum(1 for t in response_times if t > 5.0)
        })
        
        print(f"   ✅ Success Rate: {metrics['success_rate']:.1%}")
        print(f"   ✅ Avg Quality Score: {metrics['avg_quality_score']:.3f}")
        print(f"   ✅ Avg Response Time: {metrics['avg_response_time']:.2f}s")
        print(f"   ✅ High Quality Answers: {metrics['high_quality_count']}")
        
        self.test_results["metrics"] = metrics
        return metrics
    
    def generate_final_summary(self) -> Dict[str, Any]:
        """Generate final summary of all tests."""
        print("\n6️⃣ Generating Final Summary...")
        
        summary = {
            "test_timestamp": self.test_results["timestamp"],
            "api_status": "unknown",
            "pipeline_health": "unknown",
            "recommendations": []
        }
        
        # API Health Check
        health_test = self.test_results["api_tests"].get("health", {})
        if health_test.get("status") == "success":
            summary["api_status"] = "🟢 Healthy"
            services = health_test.get("data", {}).get("services", {})
            if all("connected" in str(status) or "configured" in str(status) for status in services.values()):
                summary["api_status"] = "🟢 All Systems Ready"
            else:
                summary["api_status"] = "🟡 Partially Ready"
        else:
            summary["api_status"] = "🔴 Unhealthy"
        
        # Pipeline Health
        metrics = self.test_results.get("metrics", {})
        if metrics:
            success_rate = metrics.get("success_rate", 0)
            avg_quality = metrics.get("avg_quality_score", 0)
            
            if success_rate > 0.9 and avg_quality > 0.7:
                summary["pipeline_health"] = "🟢 Excellent"
            elif success_rate > 0.7 and avg_quality > 0.5:
                summary["pipeline_health"] = "🟡 Good"
            elif success_rate > 0.5:
                summary["pipeline_health"] = "🟠 Fair"
            else:
                summary["pipeline_health"] = "🔴 Poor"
        
        # Recommendations
        if metrics.get("success_rate", 0) < 0.8:
            summary["recommendations"].append("Improve query success rate - check document indexing")
        
        if metrics.get("avg_quality_score", 0) < 0.6:
            summary["recommendations"].append("Improve answer quality - enhance prompts or add more documents")
        
        if metrics.get("avg_response_time", 0) > 5.0:
            summary["recommendations"].append("Optimize response time - check API performance")
        
        if not summary["recommendations"]:
            summary["recommendations"].append("System is performing well!")
        
        self.test_results["summary"] = summary
        return summary
    
    def display_results(self):
        """Display comprehensive test results."""
        print("\n" + "=" * 80)
        print("🎉 API PIPELINE TEST RESULTS")
        print("=" * 80)
        
        # API Status
        summary = self.test_results.get("summary", {})
        print(f"API Status: {summary.get('api_status', 'Unknown')}")
        print(f"Pipeline Health: {summary.get('pipeline_health', 'Unknown')}")
        
        # Test Results
        api_tests = self.test_results.get("api_tests", {})
        print(f"\n📊 API Tests:")
        for test_name, result in api_tests.items():
            status_icon = "✅" if result.get("status") == "success" else "❌"
            print(f"  {status_icon} {test_name}: {result.get('status', 'unknown')}")
        
        # QA Metrics
        metrics = self.test_results.get("metrics", {})
        if metrics:
            print(f"\n🎯 Performance Metrics:")
            print(f"  Success Rate: {metrics.get('success_rate', 0):.1%}")
            print(f"  Avg Quality Score: {metrics.get('avg_quality_score', 0):.3f}")
            print(f"  Avg Response Time: {metrics.get('avg_response_time', 0):.2f}s")
            print(f"  Avg Sources per Query: {metrics.get('avg_sources_count', 0):.1f}")
            
            print(f"\n📈 Quality Distribution:")
            print(f"  High Quality: {metrics.get('high_quality_count', 0)}")
            print(f"  Medium Quality: {metrics.get('medium_quality_count', 0)}")
            print(f"  Low Quality: {metrics.get('low_quality_count', 0)}")
            
            # Topic Performance
            topic_metrics = metrics.get("topic_metrics", {})
            if topic_metrics:
                print(f"\n📋 Topic Performance:")
                for topic, topic_data in topic_metrics.items():
                    print(f"  {topic}: Quality {topic_data.get('avg_quality', 0):.3f}, Confidence {topic_data.get('avg_confidence', 0):.3f}")
        
        # Recommendations
        recommendations = summary.get("recommendations", [])
        if recommendations:
            print(f"\n💡 Recommendations:")
            for i, rec in enumerate(recommendations, 1):
                print(f"  {i}. {rec}")
        
        print("=" * 80)
    
    def save_results(self, filename: str = None):
        """Save test results to file."""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"api_pipeline_results_{timestamp}.json"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.test_results, f, indent=2, ensure_ascii=False)
            print(f"\n💾 Results saved to: {filename}")
        except Exception as e:
            print(f"\n❌ Error saving results: {str(e)}")
    
    def run_full_test(self, topics: List[str] = None) -> Dict[str, Any]:
        """Run the complete API-based pipeline test."""
        if not topics:
            topics = [
                "6sense Revenue AI",
                "Predictive Analytics",
                "B2B Marketing",
                "Customer Data Platform"
            ]
        
        print(f"🚀 Starting API Pipeline Test")
        print(f"Test Topics: {topics}")
        print("=" * 80)
        
        # Step 1: Health Check
        health = self.test_api_health()
        if not health or health.get("status") != "ready":
            print("❌ API not ready, aborting test")
            return self.test_results
        
        # Step 2: Collections Check
        collections = self.test_collections_endpoint()
        
        # Step 3: Generate Questions
        questions = self.generate_test_questions(topics)
        print(f"\n📝 Generated {len(questions)} test questions")
        
        # Step 4: Test Query Pipeline
        qa_results = self.test_query_pipeline(questions)
        
        # Step 5: Test Chat Pipeline
        chat_results = self.test_chat_pipeline(questions)
        
        # Step 6: Calculate Metrics
        metrics = self.calculate_comprehensive_metrics(qa_results)
        
        # Step 7: Generate Summary
        summary = self.generate_final_summary()
        
        # Display Results
        self.display_results()
        
        # Save Results
        self.save_results()
        
        return self.test_results


def main():
    """Main function to run API pipeline test."""
    print("🚀 Starting API-Based Full Pipeline Test")
    
    # Check if API is running
    tester = APIPipelineTester()
    
    # Define test topics
    test_topics = [
        "6sense Revenue AI Platform",
        "Predictive Analytics for B2B",
        "Customer Data Platform",
        "Marketing Automation"
    ]
    
    # Run test
    results = tester.run_full_test(test_topics)
    
    print("\n✅ API Pipeline Test Completed!")
    return results


if __name__ == "__main__":
    results = main()

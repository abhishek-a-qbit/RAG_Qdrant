"""
Lightweight Streamlit Frontend for RAG System
Optimized for fast loading and better performance
"""

import streamlit as st
import requests
import time
import os
from typing import List, Dict, Any, Optional

# API configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

# Custom CSS for better styling
st.markdown("""
<style>
.metric-card {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #1f77b4;
    margin: 0.5rem 0;
}
.question-card {
    background-color: #ffffff;
    padding: 1rem;
    border-radius: 0.5rem;
    border: 1px solid #e1e5e9;
    margin: 0.5rem 0;
    cursor: pointer;
    transition: all 0.3s ease;
}
.question-card:hover {
    background-color: #f8f9fa;
    border-color: #1f77b4;
    transform: translateY(-2px);
}
.source-card {
    background-color: #f8f9fa;
    padding: 0.8rem;
    border-radius: 0.3rem;
    border-left: 3px solid #28a745;
    margin: 0.3rem 0;
    font-size: 0.9rem;
}
.chat-message {
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 0.5rem 0;
    max-width: 80%;
}
.user-message {
    background-color: #007bff;
    color: white;
    margin-left: auto;
    text-align: right;
}
.bot-message {
    background-color: #f8f9fa;
    color: #333;
    border: 1px solid #e1e5e9;
}
.metric-value {
    font-size: 1.5rem;
    font-weight: bold;
    color: #1f77b4;
}
.metric-label {
    font-size: 0.9rem;
    color: #666;
    margin-top: 0.25rem;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
def initialize_session_state():
    """Initialize session state variables"""
    if 'session_id' not in st.session_state:
        st.session_state.session_id = f"session_{int(time.time())}"
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    if 'suggested_questions' not in st.session_state:
        st.session_state.suggested_questions = []
    if 'bot_status' not in st.session_state:
        st.session_state.bot_status = None
    if 'current_metrics' not in st.session_state:
        st.session_state.current_metrics = None

def call_api(endpoint: str, method: str = "GET", data: Dict = None) -> Optional[Dict]:
    """Make API call to backend"""
    try:
        url = f"{API_BASE_URL}{endpoint}"
        
        if method == "GET":
            response = requests.get(url, timeout=10)
        elif method == "POST":
            response = requests.post(url, json=data, timeout=30)
        else:
            return None
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
            
    except requests.exceptions.RequestException as e:
        st.error(f"Connection Error: {str(e)}")
        return None

def display_sources(sources: List[Dict[str, Any]]):
    """Display source documents with links"""
    if not sources:
        return
    
    st.markdown("### 📚 Source Documents")
    
    for i, source in enumerate(sources):
        with st.expander(f"Source {i+1} - Score: {source.get('score', 0):.2f}", expanded=i==0):
            filename = source.get('filename', 'Unknown')
            doc_id = source.get('document_id', 'Unknown')
            score = source.get('score', 0)
            link = source.get('link', 'No link')
            content = source.get('content', '')
            
            st.markdown(f"**Document {i+1}**: {filename}")
            st.markdown(f"**Score**: {score:.3f}")
            
            if link and link != 'No link':
                st.markdown(f"**🔗 Direct Link**: [{link}]({link})")
            
            if content:
                preview = content[:300] + "..." if len(content) > 300 else content
                st.text(preview)

def display_metrics(metrics: Dict[str, Any]):
    """Display comprehensive quality metrics"""
    if not metrics:
        return
    
    # Main metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Coverage",
            f"{metrics.get('coverage', 0):.2f}",
            help="How well does answer cover the question scope"
        )
    
    with col2:
        st.metric(
            "Specificity",
            f"{metrics.get('specificity', 0):.2f}",
            help="How specific and detailed is answer"
        )
    
    with col3:
        st.metric(
            "Insightfulness",
            f"{metrics.get('insightfulness', 0):.2f}",
            help="How insightful and valuable is answer"
        )
    
    with col4:
        st.metric(
            "Groundedness",
            f"{metrics.get('groundedness', 0):.2f}",
            help="How well is answer grounded in source documents"
        )
    
    # Overall score
    st.markdown("---")
    overall_score = metrics.get('overall_score', 0)
    st.metric(
        "Overall Quality",
        f"{overall_score:.3f}",
        help="Combined quality score across all metrics"
        )
    
    # Additional details
    with st.expander("📊 Detailed Analysis"):
        st.write(f"**Response Time**: {metrics.get('response_time', 0):.2f}s")
        st.write(f"**Sources Retrieved**: {len(metrics.get('sources', []))}")
        
        # Display source links if available
        sources = metrics.get('sources', [])
        if sources:
            st.write("**Source Links**:")
            for i, source in enumerate(sources[:3]):
                link = source.get('link', 'No link')
                filename = source.get('filename', 'Unknown')
                score = source.get('score', 0)
                st.markdown(f"- [{i+1}] **[{filename}]({score:.2f})**: {link}")

def process_user_query(query: str):
    """Process user query through API"""
    if not query.strip():
        return
    
    # Add user message to history
    st.session_state.conversation_history.append({
        "role": "user",
        "content": query,
        "timestamp": time.time()
    })
    
    # Process query
    data = {
        "query": query,
        "collection_name": "documents",
        "limit": 5,
        "similarity_threshold": 0.7
    }
    
    response = call_api("/chat", "POST", data)
    
    if response:
        # Add bot response to history
        st.session_state.conversation_history.append({
            "role": "assistant",
            "content": response.get('answer', 'No answer available'),
            "timestamp": time.time(),
            "metrics": response.get('metrics', {}),
            "sources": response.get('sources', [])
        })
        
        # Store current metrics for display
        st.session_state.current_metrics = response.get('metrics', {})
        
        st.success("✅ Query processed successfully!")
        return response

def main():
    """Main Streamlit application"""
    initialize_session_state()
    
    st.title("🤖 Enhanced RAG Bot")
    st.markdown("Lightweight interface with source links and quality metrics")
    
    # Main content area with tabs
    tab1, tab2, tab3 = st.tabs(["💬 Chat", "🤔 Generate Questions", "📊 Analytics"])
    
    with tab1:
        st.markdown("## 💬 Chat Interface")
        
        # Display conversation history
        if st.session_state.conversation_history:
            st.markdown("### Chat History")
            
            for message in reversed(st.session_state.conversation_history):
                if message['role'] == 'user':
                    st.markdown(f"""
                    <div class="chat-message user-message">
                        <strong>You:</strong> {message['content']}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    # Display bot response with metrics
                    metrics = message.get('metrics', {})
                    sources = message.get('sources', [])
                    
                    st.markdown(f"""
                    <div class="chat-message bot-message">
                        <strong>Bot:</strong> {message['content']}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Display metrics if available
                    if metrics:
                        display_metrics(metrics)
                    
                    # Display sources if available
                    if sources:
                        display_sources(sources)
        else:
            st.info("No conversation yet. Start by asking a question below!")
        
        # Chat input at bottom
        st.markdown("---")
        st.markdown("### Ask a Question")
        
        col1, col2 = st.columns([4, 1])
        with col1:
            user_input = st.text_input(
                "Your Question:",
                placeholder="Ask me anything about indexed documents...",
                key="chat_input"
            )
        with col2:
            if st.button("🚀 Send", type="primary"):
                process_user_query(user_input)
    
    with tab2:
        st.markdown("## 🤔 Question Generation")
        
        # Question Generation Interface
        col1, col2 = st.columns([2, 1])
        
        with col1:
            topic = st.text_input("Topic", placeholder="e.g., 6sense capabilities", key="gen_topic")
        
        with col2:
            num_questions = st.slider("Number of Questions", 1, 10, 5, key="gen_num")
        
        if st.button("🔄 Generate Questions", type="primary"):
            if topic.strip():
                with st.spinner("Generating questions..."):
                    # Generate questions using the suggested questions endpoint
                    params = {
                        "topic": topic,
                        "num_questions": num_questions
                    }
                    
                    try:
                        url = f"{API_BASE_URL}/suggested-questions"
                        response = requests.get(url, params=params, timeout=30)
                        
                        if response.status_code == 200:
                            questions = response.json()
                            st.success(f"✅ Generated {len(questions)} questions!")
                            
                            # Display generated questions with metrics
                            st.markdown("### 📋 Generated Questions")
                            
                            for i, question_data in enumerate(questions):
                                with st.expander(f"Question {i+1}", expanded=True):
                                    question_text = question_data.get('question', 'No question generated')
                                    confidence = question_data.get('confidence', 0)
                                    topic_name = question_data.get('topic', topic)
                                    
                                    st.markdown(f"**Question**: {question_text}")
                                    st.markdown(f"**Topic**: {topic_name}")
                                    st.markdown(f"**Confidence**: {confidence:.3f}")
                                    
                                    # Get metrics for this specific question
                                    if question_text.strip():
                                        metrics_data = {
                                            "query": question_text,
                                            "collection_name": "documents", 
                                            "limit": 5,
                                            "similarity_threshold": 0.7
                                        }
                                        
                                        metrics_response = call_api("/query", "POST", metrics_data)
                                        
                                        if metrics_response and 'metrics' in metrics_response:
                                            metrics = metrics_response['metrics']
                                            st.markdown("**📊 Quality Metrics**:")
                                            st.write(f"- Coverage: {metrics.get('coverage', 0):.2f}")
                                            st.write(f"- Specificity: {metrics.get('specificity', 0):.2f}")
                                            st.write(f"- Insightfulness: {metrics.get('insightfulness', 0):.2f}")
                                            st.write(f"- Groundedness: {metrics.get('groundedness', 0):.2f}")
                                            st.write(f"- Overall Score: {metrics.get('overall_score', 0):.3f}")
                                            
                                            # Show sources used for this question
                                            if 'sources' in metrics_response:
                                                sources = metrics_response['sources']
                                                st.write(f"**Sources Used**: {len(sources)}")
                                                for j, source in enumerate(sources[:2]):
                                                    filename = source.get('filename', 'Unknown')
                                                    score = source.get('score', 0)
                                                    st.write(f"  - {filename} (Score: {score:.3f})")
                        else:
                            st.error(f"Failed to generate questions: {response.status_code}")
                    except Exception as e:
                        st.error(f"Error generating questions: {str(e)}")
            else:
                st.warning("Please enter a topic")
    
    with tab3:
        st.markdown("## 📊 Analytics Dashboard")
        
        if st.session_state.conversation_history:
            # Extract metrics from conversation history
            metrics_history = []
            for msg in st.session_state.conversation_history:
                if msg['role'] == 'assistant' and 'metrics' in msg:
                    metrics_history.append(msg['metrics'])
            
            if metrics_history:
                # Statistics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    avg_score = sum(m.get('overall_score', 0) for m in metrics_history) / len(metrics_history)
                    st.metric("Avg Overall Score", f"{avg_score:.3f}")
                
                with col2:
                    total_sources = sum(len(m.get('sources', [])) for m in metrics_history)
                    st.metric("Total Sources Used", total_sources)
                
                with col3:
                    avg_time = sum(m.get('response_time', 0) for m in metrics_history) / len(metrics_history)
                    st.metric("Avg Response Time", f"{avg_time:.2f}s")
                
                with col4:
                    st.metric("Total Questions", len(metrics_history))
                
                # Detailed metrics
                st.markdown("### Detailed Metrics History")
                for i, metrics in enumerate(metrics_history[-5:]):  # Show last 5
                    with st.expander(f"Response {i+1}", expanded=False):
                        st.json(metrics)
            else:
                st.info("No metrics available yet. Start chatting to see analytics!")
        else:
            st.info("No conversation history yet. Start chatting to see analytics!")
    
    # Sidebar - API Status and Controls
    with st.sidebar:
        st.markdown("## 🎛️ System Status")
        
        # API Status
        status = call_api("/health")
        if status:
            st.success("🟢 API Connected")
            services = status.get('services', {})
            for svc, svc_status in services.items():
                if svc_status == "connected" or svc_status == "configured":
                    st.success(f"{svc}: {svc_status}")
                else:
                    st.error(f"{svc}: {svc_status}")
        else:
            st.error("🔴 API Disconnected")
        
        st.markdown("---")
        
        # Clear conversation
        if st.button("🗑️ Clear All Data"):
            st.session_state.conversation_history = []
            st.session_state.current_metrics = None
            st.success("✅ All data cleared!")
            st.rerun()
        
        # Session info
        st.markdown("### 📋 Session Info")
        st.text(f"Session ID: {st.session_state.session_id}")
        st.text(f"Messages: {len(st.session_state.conversation_history)}")
    
    # Current metrics display at bottom
    if st.session_state.current_metrics:
        st.markdown("---")
        st.markdown("### 📊 Current Response Metrics")
        display_metrics(st.session_state.current_metrics)

if __name__ == "__main__":
    main()

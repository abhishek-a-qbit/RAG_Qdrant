"""
Streamlit Frontend for Amazon Rufus-style Conversational RAG Bot
Lightweight and optimized for better performance
"""

import streamlit as st
import requests
import json
import time
import os
from typing import List, Dict, Any, Optional

# Configure page
st.set_page_config(
    page_title="Conversational RAG Bot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    """Make API call to the backend"""
    try:
        url = f"{API_BASE_URL}{endpoint}"
        
        if method == "GET":
            response = requests.get(url, timeout=30)
        elif method == "POST":
            # Send as form data for FastAPI compatibility
            response = requests.post(url, json=data, timeout=30)
        elif method == "DELETE":
            response = requests.delete(url, timeout=30)
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

def display_metrics(metrics: Dict[str, Any], title: str = "Quality Metrics"):
    """Display comprehensive quality metrics"""
    if not metrics:
        return
    
    # Main metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Coverage",
            f"{metrics.get('coverage', 0):.2f}",
            delta=None,
            help="How well the answer covers the question scope"
        )
    
    with col2:
        st.metric(
            "Specificity",
            f"{metrics.get('specificity', 0):.2f}",
            delta=None,
            help="How specific and detailed the answer is"
        )
    
    with col3:
        st.metric(
            "Insightfulness",
            f"{metrics.get('insightfulness', 0):.2f}",
            delta=None,
            help="How insightful and valuable the answer is"
        )
    
    with col4:
        st.metric(
            "Groundedness",
            f"{metrics.get('groundedness', 0):.2f}",
            delta=None,
            help="How well the answer is grounded in source documents"
        )
    
    # Overall score
    st.markdown("---")
    overall_score = metrics.get('overall_score', 0)
    st.metric(
        "Overall Quality",
        f"{overall_score:.3f}",
        delta=None,
        help="Combined quality score across all metrics"
    )
    
    # Detailed metrics in expandable section
    with st.expander("📊 Detailed Analysis"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Response Analysis")
            st.write(f"**Answer Length**: {len(metrics.get('answer', ''))} characters")
            st.write(f"**Response Time**: {metrics.get('response_time', 0):.2f}s")
            
        with col2:
            st.subheader("Source Analysis")
            sources = metrics.get('sources', [])
            st.write(f"**Sources Retrieved**: {len(sources)}")
            if sources:
                avg_score = sum([s.get('score', 0) for s in sources]) / len(sources)
                st.write(f"**Average Source Score**: {avg_score:.3f}")
                
                # Display source links
                st.write("**Source Links**:")
                for i, source in enumerate(sources[:3]):  # Show top 3 sources
                    link = source.get('link', 'No link')
                    filename = source.get('filename', 'Unknown')
                    score = source.get('score', 0)
                    st.markdown(f"- [{i+1}] **[{filename}]({score:.2f})**: {link}")
            else:
                st.write("No sources available for this response")

def display_sources(sources: List[Dict[str, Any]]):
    """Display source documents with links"""
    if not sources:
        return
    
    st.markdown("### 📚 Source Documents")
    
    for i, source in enumerate(sources):
        with st.expander(f"Source {i+1} - Score: {source.get('score', 0):.2f}", expanded=i==0):
            # Display source info
            filename = source.get('filename', 'Unknown')
            doc_id = source.get('document_id', 'Unknown')
            score = source.get('score', 0)
            link = source.get('link', 'No link')
            content = source.get('content', '')
            
            st.markdown(f"**Document {i+1}**: {filename}")
            st.markdown(f"**Score**: {score:.3f}")
            
            # Display content preview
            if content:
                preview = content[:300] + "..." if len(content) > 300 else content
                st.text(preview)
            
            # Display link if available
            if link and link != 'No link':
                st.markdown(f"**🔗 Direct Link**: [{link}]({link})")
            
            st.markdown("---")

def display_suggested_questions():
    """Display suggested questions as clickable cards"""
    if not st.session_state.suggested_questions:
        return
    
    st.markdown("### 💡 Suggested Questions")
    
    # Create columns for question cards
    cols = st.columns(2)
    
    for i, question in enumerate(st.session_state.suggested_questions):
        col = cols[i % 2]
        
        with col:
            # Question card with click handler
            question_text = question.get('question', '')
            confidence = question.get('confidence', 0)
            topic = question.get('topic', 'General')
            
            # Create clickable card
            card_html = f"""
            <div class="question-card" onclick="handleQuestionClick('{question_text.replace("'", "\\'")}')">
                <h4>{question_text}</h4>
                <small><strong>Topic:</strong> {topic} | <strong>Confidence:</strong> {confidence:.2f}</small>
            </div>
            """
            st.markdown(card_html, unsafe_allow_html=True)
            
            # Add button for mobile compatibility
            if st.button(f"Ask: {question_text[:50]}...", key=f"question_{i}"):
                # Store in a different session state variable to avoid widget conflict
                st.session_state.selected_question = question_text
                st.rerun()

def display_chat_history():
    """Display conversation history"""
    if not st.session_state.conversation_history:
        return
    
    st.markdown("### 💬 Conversation History")
    
    for message in st.session_state.conversation_history:
        if message['role'] == 'user':
            st.markdown(f"""
            <div class="chat-message user-message">
                <strong>You:</strong> {message['content']}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="chat-message bot-message">
                <strong>Bot:</strong> {message['content']}
            </div>
            """, unsafe_allow_html=True)

def display_bot_status():
    """Display bot status in sidebar"""
    status = call_api("/health")
    
    if status:
        st.session_state.bot_status = status
        
        # Status indicator
        if status.get('status') == 'healthy':
            st.success("🤖 Bot is Ready")
        else:
            st.warning("🔄 Bot is Initializing")
        
        # Status details
        with st.expander("Bot Details"):
            st.json(status)
            
            # Check services
            services = status.get('services', {})
            for svc, svc_status in services.items():
                if svc_status == "connected" or svc_status == "configured":
                    st.success(f"{svc}: {svc_status}")
                else:
                    st.error(f"{svc}: {svc_status}")

def load_suggested_questions(topic: str = "", num_questions: int = 5):
    """Load suggested questions from API"""
    # Send as query parameters to match FastAPI endpoint
    params = {
        "topic": topic,
        "num_questions": num_questions
    }
    
    try:
        url = f"{API_BASE_URL}/suggested-questions"
        response = requests.get(url, params=params, timeout=30)
        
        if response.status_code == 200:
            st.session_state.suggested_questions = response.json()
            return True
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return False
    except requests.exceptions.RequestException as e:
        st.error(f"Connection Error: {str(e)}")
        return False

def process_user_query(query: str):
    """Process user query through API"""
    data = {
        "query": query,
        "collection_name": "documents",
        "limit": 5,
        "similarity_threshold": 0.7
    }
    
    response = call_api("/chat", "POST", data)
    
    if response:
        # Add to conversation history
        st.session_state.conversation_history.append({
            "role": "user",
            "content": query,
            "timestamp": time.time()
        })
        
        st.session_state.conversation_history.append({
            "role": "assistant",
            "content": response.get('answer', 'No answer available'),
            "timestamp": time.time(),
            "metrics": response.get('metrics', {}),
            "sources": response.get('sources', []),
            "timing": {"response_time": response.get('response_time', 0.0)}
        })
        
        # Store current metrics for display
        st.session_state.current_metrics = {"overall_score": response.get('confidence_score', 0.0)}
        
        return response
    return None

def create_metrics_chart(metrics_history: List[Dict]):
    """Create metrics visualization chart"""
    if not metrics_history:
        return
    
    # Prepare data for plotting
    df = pd.DataFrame(metrics_history)
    
    # Create line chart for metrics over time
    fig = go.Figure()
    
    metric_columns = ['overall_score', 'question_quality', 'answer_quality', 'faithfulness']
    
    for metric in metric_columns:
        if metric in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df[metric],
                mode='lines+markers',
                name=metric.replace('_', ' ').title(),
                line=dict(width=2)
            ))
    
    fig.update_layout(
        title="Quality Metrics Over Time",
        xaxis_title="Conversation Turn",
        yaxis_title="Score",
        hovermode='x unified',
        legend=dict(x=0, y=1)
    )
    
    st.plotly_chart(fig, use_container_width=True)
        
    # User input
    st.markdown("### 💬 Ask Question")
    user_input = st.text_input(
        "Your Question:",
        placeholder="Ask me anything about  indexed documents...",
        key="user_query"
    )
    
    # Send button
    if st.button("🚀 Send Query", type="primary"):
        process_user_query(user_input)

# Display conversation history
if st.session_state.conversation_history:
    st.markdown("### 💬 Chat History")
    
    for message in reversed(st.session_state.conversation_history):
        if message['role'] == 'user':
            st.markdown(f"""
            <div class="chat-message user-message">
                <strong>You:</strong> {message['content']}
            </div>
            """)
        else:
            # Display bot response with metrics
            metrics = message.get('metrics', {})
            sources = message.get('sources', [])
            
            st.markdown(f"""
            <div class="chat-message bot-message">
                <strong>Bot:</strong> {message['content']}
            </div>
            """)
            
            # Display metrics if available
            if metrics:
                st.markdown(f"**📊 Quality Score**: {metrics.get('overall_score', 0):.3f}")
            
            # Display sources if available
            if sources:
                st.markdown("**📚 Sources:**")
                for i, source in enumerate(sources[:3]):
                    link = source.get('link', 'No link')
                    filename = source.get('filename', 'Unknown')
                    score = source.get('score', 0)
                    st.markdown(f"- [{i+1}] **[{filename}]({score:.2f})**: {link}")

    # Current metrics
    if st.session_state.current_metrics:
        st.markdown("### 📊 Current Response Metrics")
        display_metrics(st.session_state.current_metrics)
        
        # Current metrics
        if st.session_state.current_metrics:
            display_metrics(st.session_state.current_metrics, "Current Response Metrics")
        
    # Header
    # st.title("🤖 Conversational RAG Bot")
    # st.markdown("Amazon Rufus-style AI assistant with comprehensive evaluation metrics")
    
    # Sidebar
    # with st.sidebar:
    #     st.markdown("## 🎛️ Control Panel")
        
    #     # Bot status
    #     display_bot_status()
        
    #     # Configuration
    #     st.markdown("### ⚙️ Configuration")
        
    #     topic = st.text_input("Topic for Questions", placeholder="e.g., 6sense capabilities")
    #     num_questions = st.slider("Number of Questions", 1, 10, 5)
        
    #     # Load questions button
    #     if st.button("🔄 Load Suggested Questions"):
    #         with st.spinner("Loading suggested questions..."):
    #             load_suggested_questions(topic, num_questions)
    #             if len(st.session_state.suggested_questions) > 0:
    #                 st.success(f"Loaded {len(st.session_state.suggested_questions)} questions")
    #             else:
    #                 st.info("No suggested questions available. Please ask your own questions in the chat interface.")
        
    #     # Clear conversation
    #     if st.button("🗑️ Clear Conversation"):
    #         st.session_state.conversation_history = []
    #         st.session_state.current_metrics = None
    #         st.rerun()
        
    #     # Session info
    #     st.markdown("### 📋 Session Info")
    #     st.text(f"Session ID: {st.session_state.session_id}")
    #     st.text(f"Messages: {len(st.session_state.conversation_history)}")
    
    # Main content area
    # col1, col2 = st.columns([3, 2])
    
    # with col1:
    #     # Chat interface
    #     st.markdown("## 💬 Chat Interface")
        
    #     # Display conversation history
    #     display_chat_history()
        
    #     # User input
    #     st.markdown("### Type your question:")
        
    #     # Check if a question was selected from suggested questions
    #     if 'selected_question' in st.session_state:
    #         default_value = st.session_state.selected_question
    #         # Clear the selected question after using it
    #         del st.session_state.selected_question
    #     else:
    #         default_value = ""
        
    #     user_input = st.text_input(
    #         "Your Question:",
    #         key="user_input",
    #         value=default_value,
    #         placeholder="Ask me anything about the indexed documents...",
    #         label_visibility="collapsed"
    #     )
        
    #     # Send button
    #     col_send, col_clear = st.columns([1, 1])
        
    #     with col_send:
    #         if st.button("📤 Send", type="primary"):
    #             if user_input.strip():
    #                 with st.spinner("Processing your question..."):
    #                     response = process_user_query(user_input)
    #                     if response:
    #                         st.success("Response generated!")
    #                         st.rerun()
    #                     else:
    #                         st.error("Failed to process question")
    #             else:
    #                 st.warning("Please enter a question")
        
    #     with col_clear:
    #         if st.button("🔄 Clear Input"):
    #             st.session_state.user_input = ""
    #             st.rerun()
    
    # with col2:
    #     # Suggested questions
    #     display_suggested_questions()
        
    #     # Current metrics
    #     if st.session_state.current_metrics:
    #         display_metrics(st.session_state.current_metrics, "Current Response Metrics")
        
    #     # Sources
    #     if st.session_state.conversation_history:
    #         last_message = st.session_state.conversation_history[-1]
    #         if last_message['role'] == 'assistant' and 'sources' in last_message:
    #             display_sources(last_message['sources'])
        if st.session_state.conversation_history:
            last_message = st.session_state.conversation_history[-1]
            if last_message['role'] == 'assistant' and 'sources' in last_message:
                display_sources(last_message['sources'])
    
    # Bottom section for analytics
    if st.session_state.conversation_history:
        st.markdown("---")
        st.markdown("## 📊 Analytics Dashboard")
        
        # Extract metrics history
        metrics_history = []
        for msg in st.session_state.conversation_history:
            if msg['role'] == 'assistant' and 'metrics' in msg:
                metrics_history.append(msg['metrics'])
        
        if metrics_history:
            # Metrics chart
            create_metrics_chart(metrics_history)
            
            # Statistics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                avg_score = sum(m.get('overall_score', 0) for m in metrics_history) / len(metrics_history)
                st.metric("Average Overall Score", f"{avg_score:.2f}")
            
            with col2:
                total_sources = sum(m.get('sources_count', 0) for m in metrics_history)
                st.metric("Total Sources Used", total_sources)
            
            with col3:
                avg_time = sum(m.get('response_time', 0) for m in metrics_history) / len(metrics_history)
                st.metric("Avg Response Time", f"{avg_time:.2f}s")

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; font-size: 0.8rem;'>
            Conversational RAG Bot • Powered by Qdrant, LangChain, and RAGAS • 
            Real-time Evaluation Metrics
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()

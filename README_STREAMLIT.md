# Streamlit Frontend for Conversational RAG Bot

A beautiful, interactive Streamlit interface for the Amazon Rufus-style conversational RAG bot with real-time metrics and suggested questions.

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- API server running on `http://localhost:8001`
- All dependencies installed

### Installation & Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm

# Start the API server (in one terminal)
python services/conversational_api.py

# Start the Streamlit frontend (in another terminal)
streamlit run streamlit_app.py

# Or use the runner script
python run_streamlit.py
```

### Access the App
- **Streamlit Frontend**: http://localhost:8501
- **API Server**: http://localhost:8001

## 🎨 Features

### 💬 Interactive Chat Interface
- **Real-time Conversation**: Natural language interaction with the RAG bot
- **Session Management**: Maintains conversation context across multiple turns
- **Message History**: Complete conversation history with timestamps
- **Responsive Design**: Works on desktop and mobile devices

### 💡 Suggested Questions
- **Auto-Generated**: Questions created from indexed documents
- **Clickable Cards**: Click on any suggested question to ask it
- **Confidence Scores**: Each question shows confidence level
- **Topic Categorization**: Questions grouped by relevant topics
- **Customizable**: Specify topics and number of questions

### 📊 Real-time Metrics
- **Quality Scores**: Overall, question, and answer quality metrics
- **RAGAS Evaluation**: Faithfulness, relevancy, precision metrics
- **Performance Tracking**: Response times and token usage
- **Visual Charts**: Interactive plots showing metrics over time
- **Source Attribution**: Document sources with relevance scores

### 📚 Source Documents
- **Document Preview**: See source content used for answers
- **Metadata Display**: File information and document details
- **Relevance Scores**: How relevant each source was to the query
- **Expandable Cards**: Click to expand/collapse source details

### 🎛️ Control Panel
- **Bot Status**: Real-time status and health checks
- **Configuration**: Customize topics and question counts
- **Session Info**: View session ID and message counts
- **Clear Functions**: Reset conversation or input fields

## 🖥️ Interface Layout

### Main Chat Area (Left Panel)
```
┌─────────────────────────────────────┐
│ 💬 Chat Interface                  │
│ ┌─────────────────────────────────┐ │
│ │ Conversation History           │ │
│ │ ┌─ User Message ──────────────┐ │ │
│ │ │ ┌─ Bot Response ───────────┐ │ │
│ │ │ │ • Metrics                │ │ │
│ │ │ │ • Sources                │ │ │
│ │ │ └─────────────────────────┘ │ │
│ └─────────────────────────────────┘ │
│ ┌─────────────────────────────────┐ │
│ │ 📝 Your Question: [_________] │ │
│ │ [📤 Send] [🔄 Clear]          │ │
│ └─────────────────────────────────┘ │
└─────────────────────────────────────┘
```

### Side Panel (Right)
```
┌─────────────────┐
│ 🎛️ Control Panel│
│ ┌─────────────┐ │
│ │ 🤖 Bot Status│ │
│ │ ⚙️ Config   │ │
│ │ 📋 Session  │ │
│ └─────────────┘ │
│ ┌─────────────┐ │
│ │ 💡 Suggested │ │
│ │ Questions   │ │
│ │ ┌─ Q Card ─┐ │ │
│ │ └───────────┘ │ │
│ └─────────────┘ │
│ ┌─────────────┐ │
│ │ 📊 Metrics  │ │
│ │ • Overall  │ │
│ │ • Quality  │ │
│ │ • Sources  │ │
│ └─────────────┘ │
└─────────────────┘
```

### Analytics Dashboard (Bottom)
```
┌─────────────────────────────────────────────────────────┐
│ 📊 Analytics Dashboard                                  │
│ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐         │
│ │ 📈 Metrics  │ │ 📚 Sources  │ │ ⏱️ Timing   │         │
│ │   Chart     │ │   Used      │ │   Stats     │         │
│ └─────────────┘ └─────────────┘ └─────────────┘         │
└─────────────────────────────────────────────────────────┘
```

## 🎯 Usage Guide

### 1. Start a Conversation
1. **Load Suggested Questions**: Click "🔄 Load Suggested Questions" in the sidebar
2. **Ask a Question**: Either click a suggested question or type your own
3. **View Response**: See the bot's answer with metrics and sources
4. **Continue Chat**: Ask follow-up questions to maintain context

### 2. Understand the Metrics
- **Overall Score**: Combined quality assessment (0-1)
- **Question Quality**: How good your question was (0-1)
- **Answer Quality**: How complete and accurate the response was (0-1)
- **Faithfulness**: How well the answer sticks to sources (0-1)
- **Answer Relevancy**: How relevant the answer is to your question (0-1)

### 3. Explore Sources
- Click on any source card to expand and see full content
- Check the relevance score to understand source importance
- View metadata to see which document provided the information

### 4. Monitor Performance
- **Response Time**: How long each answer took to generate
- **Token Usage**: How many tokens were consumed
- **Source Count**: How many documents were used
- **Metrics Trends**: How quality changes over conversation

## 🎨 Customization

### Styling
The app uses custom CSS for a modern look:
- **Question Cards**: Hover effects and smooth transitions
- **Chat Messages**: Different colors for user vs bot messages
- **Metric Cards**: Clean, professional styling
- **Responsive Design**: Works on all screen sizes

### Configuration Options
In the sidebar control panel:
- **Topic**: Specify topic for suggested questions
- **Number of Questions**: How many suggestions to generate (1-10)
- **Clear Conversation**: Reset the entire chat history
- **Clear Input**: Just clear the current input field

### API Configuration
Update the `API_BASE_URL` in `streamlit_app.py` if your API runs on a different port:
```python
API_BASE_URL = "http://localhost:8001"  # Change if needed
```

## 🔧 Troubleshooting

### Common Issues

#### "Connection Error"
- **Solution**: Make sure the API server is running on port 8001
- **Check**: `python services/conversational_api.py`

#### "Bot is Initializing"
- **Solution**: Wait for the bot to finish loading documents
- **Check**: The status panel in the sidebar for progress

#### "No Suggested Questions"
- **Solution**: Click "🔄 Load Suggested Questions" button
- **Check**: Ensure documents are properly indexed

#### "Poor Quality Metrics"
- **Solution**: Try rephrasing your question
- **Check**: Ensure your question is specific and clear

#### "Slow Response Times"
- **Solution**: Monitor token usage and optimize questions
- **Check**: The analytics dashboard for performance trends

### Debug Mode
Enable detailed logging by setting environment variables:
```bash
export STREAMLIT_LOG_LEVEL=debug
streamlit run streamlit_app.py
```

### Performance Tips
- **Keep Questions Specific**: More specific questions get better results
- **Use Suggested Questions**: They're optimized for the available content
- **Monitor Metrics**: Use the analytics to understand what works well
- **Clear Sessions**: Start fresh sessions for different topics

## 📱 Mobile Support

The Streamlit app is fully responsive:
- **Touch-Friendly**: Large buttons and touch targets
- **Adaptive Layout**: Columns stack on small screens
- **Readable Text**: Optimized font sizes for mobile
- **Smooth Scrolling**: Works well on touch devices

## 🚀 Advanced Features

### Session Persistence
- **Session IDs**: Unique identifiers for each conversation
- **History Storage**: Complete conversation history maintained
- **Context Awareness**: Bot remembers previous messages
- **Cross-Session**: Sessions persist across page refreshes

### Real-time Updates
- **Live Metrics**: Scores update instantly with each response
- **Dynamic Charts**: Visualizations update as you chat
- **Status Monitoring**: Real-time bot status updates
- **Error Handling**: Graceful error messages and recovery

### Export & Sharing
- **Conversation Export**: Save chat history (future feature)
- **Metrics Export**: Download performance data (future feature)
- **Session Sharing**: Share conversation links (future feature)

## 🎯 Best Practices

### For Best Results
1. **Use Specific Questions**: More specific = better answers
2. **Check Sources**: Verify information in source documents
3. **Monitor Metrics**: Use quality scores to guide interactions
4. **Provide Feedback**: Use the metrics to understand system performance

### Question Examples
✅ **Good**: "What are the key features of 6sense Revenue AI?"  
✅ **Good**: "How does 6sense help with B2B sales targeting?"  
❌ **Poor**: "Tell me about the system"  
❌ **Poor**: "help"  

### Conversation Flow
1. **Start Broad**: Begin with general questions
2. **Get Specific**: Ask follow-up questions for details
3. **Verify Sources**: Check source documents for accuracy
4. **Use Context**: Build on previous answers

## 📞 Support

### Getting Help
- **Check Logs**: Look at the terminal output for errors
- **API Status**: Verify the API server is running
- **Documentation**: Refer to the main README for API details
- **Issues**: Report problems with detailed error messages

### Contributing
- **UI Improvements**: Submit styling and layout suggestions
- **New Features**: Request additional functionality
- **Bug Reports**: Report issues with steps to reproduce
- **Performance**: Suggest optimizations for speed/usability

---

**Enjoy your conversational RAG experience! 🤖✨**

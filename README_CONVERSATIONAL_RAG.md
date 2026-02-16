# Amazon Rufus-Style Conversational RAG Bot

A sophisticated conversational AI system that generates suggested questions and provides answers with comprehensive evaluation metrics using RAGAS framework.

## Features

### 🤖 Core Capabilities
- **Suggested Questions**: Automatically generates relevant questions based on indexed documents
- **Interactive Chat**: Users can click suggested questions or type their own queries
- **Real-time Metrics**: Comprehensive evaluation metrics displayed throughout conversation
- **Source Tracking**: Shows source documents and relevance scores for each response
- **Session Management**: Maintains conversation context across multiple interactions

### 📊 Evaluation Metrics
- **Question Quality**: Clarity, specificity, and relevance scoring
- **Answer Quality**: Completeness, accuracy, and structure evaluation
- **Retrieval Metrics**: Precision and recall of document retrieval
- **RAGAS Metrics**: Faithfulness, relevancy, precision, and correctness
- **Performance Metrics**: Response time and token usage tracking

### 📁 Document Support
- **JSON**: Structured data files
- **CSV**: Tabular data with pandas integration
- **PDF**: Text extraction with PyPDF2
- **DOCX**: Word document processing
- **TXT/MD**: Plain text and markdown files
- **XLSX**: Excel spreadsheet support

## Architecture

### System Components
```
├── services/
│   ├── conversational_rag_bot.py    # Main bot logic and evaluation
│   ├── conversational_api.py         # FastAPI endpoints
│   ├── document_indexer.py           # Qdrant vector indexing
│   ├── langchain_orchestrator.py     # LLM orchestration
│   └── data_driven_question_generator.py  # Question generation
├── utils/
│   ├── file_extractor.py             # Multi-format document processing
│   └── langchain_utils.py            # LangChain utilities
└── uploads/                          # Document storage
    ├── dataset_1/                     # JSON datasets
    ├── dataset_2/                     # Additional datasets
    └── News/                          # CSV news data
```

### Data Flow
1. **Document Loading**: Enhanced data loader processes various file formats
2. **Vector Indexing**: Documents are chunked and indexed in Qdrant cloud
3. **Question Generation**: Data-driven questions created from indexed content
4. **Query Processing**: User queries processed with context retrieval
5. **Response Generation**: LLM generates answers with source attribution
6. **Evaluation**: RAGAS and custom metrics calculate quality scores

## Installation

### Prerequisites
- Python 3.8+
- OpenAI API key
- Qdrant cloud instance

### Setup
```bash
# Clone and navigate to project
cd RAG_Qdrant

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
# Edit .env with your API keys and endpoints

# Download spaCy model
python -m spacy download en_core_web_sm
```

### Environment Variables
```bash
# Qdrant Cloud Configuration
qdrant_db_path=https://your-cluster.cloud.qdrant.io:6333
qdrant_api_key=your-api-key

# OpenAI Configuration
OPENAI_API_KEY=your-openai-api-key
model=gpt-4o-mini
temperature=0.1

# RAG Configuration
chunk_size=2000
no_of_chunks=3
```

## Usage

### Starting the API Server
```bash
# Start the conversational bot API
python services/conversational_api.py

# Or use uvicorn directly
uvicorn services.conversational_api:app --host 0.0.0.0 --port 8001 --reload
```

### API Endpoints

#### Get Suggested Questions
```bash
POST /suggested-questions
{
    "topic": "6sense capabilities",
    "num_questions": 5
}
```

#### Chat with Bot
```bash
POST /chat
{
    "query": "What are the key features of 6sense?",
    "session_id": "optional-session-id",
    "username": "user"
}
```

#### Get Bot Status
```bash
GET /status
```

#### Get Conversation History
```bash
GET /chat/sessions/{session_id}/history
```

### Response Format

#### Suggested Questions Response
```json
[
    {
        "id": "suggested_0",
        "question": "What are the main capabilities of 6sense Revenue AI?",
        "preview_answer": "6sense Revenue AI provides comprehensive B2B revenue intelligence...",
        "topic": "Platform Capabilities",
        "metrics": {
            "overall_score": 0.85,
            "coverage_final": 0.90,
            "specificity_final": 0.80
        },
        "sources_count": 3,
        "confidence": 0.85
    }
]
```

#### Chat Response
```json
{
    "response": "6sense Revenue AI offers several key capabilities...",
    "refined_query": "What are the main capabilities of 6sense Revenue AI platform?",
    "metrics": {
        "question_quality": 0.85,
        "answer_quality": 0.90,
        "faithfulness": 0.88,
        "answer_relevancy": 0.92,
        "overall_score": 0.89
    },
    "sources": [
        {
            "id": "source_0",
            "content": "6sense Revenue AI provides predictive analytics...",
            "metadata": {"source": "dataset_01_capabilities.json"},
            "score": 0.92
        }
    ],
    "token_usage": {
        "prompt_tokens": 150,
        "completion_tokens": 200,
        "total_tokens": 350
    },
    "timing": {
        "response_time": 2.5,
        "total_time": 3.2
    },
    "session_id": "session-123",
    "timestamp": 1678901234.56
}
```

## Evaluation Metrics

### Question Metrics
- **Coverage**: Relevance to available content
- **Specificity**: Clarity and focus of the question
- **Insightfulness**: Depth and analytical value
- **Groundedness**: Answerability from indexed documents

### Answer Metrics
- **Quality**: Completeness and accuracy
- **Faithfulness**: Alignment with retrieved context
- **Relevancy**: Direct response to user query
- **Correctness**: Factual accuracy

### Retrieval Metrics
- **Precision**: Relevance of retrieved documents
- **Recall**: Coverage of relevant documents
- **Context Precision**: Quality of context retrieval
- **Context Relevancy**: Alignment with query

### RAGAS Integration
When available, the system uses RAGAS framework for advanced evaluation:
- **Faithfulness**: Answer adherence to context
- **Answer Relevancy**: Response relevance to question
- **Context Relevancy**: Context relevance to question
- **Context Precision**: Precision of retrieved context
- **Answer Correctness**: Factual correctness

## Document Processing

### Supported Formats
- **JSON**: Structured data with metadata extraction
- **CSV**: Tabular data with pandas processing
- **PDF**: Text extraction with metadata
- **DOCX**: Word document parsing
- **TXT/MD**: Plain text and markdown
- **XLSX**: Excel spreadsheet reading

### Data Loading Process
1. **File Detection**: Scans uploads directory recursively
2. **Format Identification**: Determines file type and appropriate loader
3. **Content Extraction**: Extracts text and metadata
4. **Quality Validation**: Ensures content quality before indexing
5. **Vector Indexing**: Chunks and indexes in Qdrant

### Example Document Structure
```
uploads/
├── dataset_1/
│   ├── dataset_01_capabilities.json
│   ├── dataset_02_customerProfiles.json
│   └── ...
├── dataset_2/
│   ├── additional_data.json
│   └── ...
└── News/
    └── 6Sense_News.csv
```

## Performance Optimization

### Caching
- Document embeddings cached for faster retrieval
- Conversation history maintained in memory
- Question generation results cached

### Async Processing
- Non-blocking document loading
- Parallel evaluation metrics calculation
- Async LLM calls for better throughput

### Resource Management
- Connection pooling for Qdrant
- Efficient memory usage for large documents
- Token usage monitoring and optimization

## Monitoring and Debugging

### Logging
- Comprehensive logging at all levels
- Performance metrics tracking
- Error handling and recovery

### Health Checks
- Bot initialization status
- Document loading progress
- API health and response times

### Metrics Dashboard
- Real-time evaluation metrics
- Conversation quality trends
- System performance indicators

## Troubleshooting

### Common Issues
1. **Bot Initialization Fails**: Check Qdrant connection and API keys
2. **No Documents Loaded**: Verify uploads directory structure
3. **Poor Quality Questions**: Ensure documents contain relevant content
4. **Slow Responses**: Monitor token usage and optimize prompts

### Debug Mode
Enable debug logging for detailed troubleshooting:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Contributing

### Adding New Document Types
1. Extend `EnhancedDataLoader` with new format support
2. Add appropriate metadata extraction
3. Update supported formats list
4. Add tests for new format

### Custom Metrics
1. Implement new metric calculation methods
2. Add to `ConversationMetrics` dataclass
3. Update overall score calculation
4. Add metric descriptions to API

## License

This project is licensed under the MIT License - see the LICENSE file for details.

import os
from dotenv import load_dotenv
load_dotenv()
from utils.langchain_utils import LangChainManager
from utils.qdrant_utils import QdrantManager
from utils.utils import get_embeddings_client

# Test full RAG pipeline
print('Testing full RAG pipeline...')

# Initialize components
lm = LangChainManager(os.getenv('OPENAI_API_KEY'))
qm = QdrantManager(os.getenv('qdrant_db_path'), os.getenv('qdrant_api_key'))
embeddings_client = get_embeddings_client()

# Test query
test_query = 'What is 6sense and how does it help companies?'
print(f'Query: {test_query}')

# Generate embedding
query_embedding = lm.embed_query(test_query)
print(f'Generated embedding: {len(query_embedding)} dimensions')

# Search in Qdrant
search_results = qm.search('documents', query_embedding, 3, 0.5)
print(f'Found {len(search_results)} search results')

# Create LangChain vectorstore
vectorstore = qm.create_langchain_vectorstore('documents', embeddings_client)

# Create retriever
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3, "score_threshold": 0.5}
)

# Create QA chain
qa_chain = lm.create_qa_chain(retriever)

# Generate answer
result = lm.generate_answer(qa_chain, test_query)

print(f'Answer: {result["answer"]}')
print(f'Response time: {result["response_time"]:.2f}s')
print(f'Success: {result["success"]}')

import os
from dotenv import load_dotenv
load_dotenv()
from utils.qdrant_utils import QdrantManager
from utils.langchain_utils import LangChainManager

# Test exactly like API does
print("Testing API-like query...")

# Initialize managers like API
qdrant_manager = QdrantManager(os.getenv('qdrant_db_path'), os.getenv('qdrant_api_key'))
langchain_manager = LangChainManager(os.getenv('OPENAI_API_KEY'), os.getenv('model'), float(os.getenv('temperature')))

# Test query
test_query = "What is 6sense?"
print(f"Query: {test_query}")

# Generate embedding
query_embedding = langchain_manager.embed_query(test_query)
print(f"Generated embedding: {len(query_embedding)} dimensions")

# Search with same parameters as API
search_results = qdrant_manager.search(
    "documents",  # collection_name
    query_embedding,
    3,  # limit
    0.5  # similarity_threshold
)

print(f"Search results: {len(search_results)}")
for i, result in enumerate(search_results[:2]):
    print(f"Result {i+1}: Score={result['score']:.3f}, Preview={result['text'][:100]}...")

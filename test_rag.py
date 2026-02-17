import os
from dotenv import load_dotenv
load_dotenv()
from utils.langchain_utils import LangChainManager
from utils.qdrant_utils import QdrantManager

# Test LangChain manager
print('Testing LangChain manager...')
lm = LangChainManager(os.getenv('OPENAI_API_KEY'))
test_query = 'What is 6sense?'
embedding = lm.embed_query(test_query)
print(f'Query embedding generated: {len(embedding)} dimensions')

# Test Qdrant search
print('Testing Qdrant search...')
qm = QdrantManager(os.getenv('qdrant_db_path'), os.getenv('qdrant_api_key'))
results = qm.search('documents', embedding, 3, 0.5)
print(f'Found {len(results)} results')
for i, result in enumerate(results[:2]):
    print(f'Result {i+1}: Score={result["score"]:.3f}, Preview={result["text"][:100]}...')

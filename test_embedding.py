from utils.langchain_utils import LangChainManager
from utils.utils import OPENAI_API_KEY, MODEL, TEMPERATURE

print('🔍 Testing embedding generation...')

try:
    langchain = LangChainManager(OPENAI_API_KEY, MODEL, TEMPERATURE)
    
    # Test embedding generation
    test_query = '6sense benefits'
    embedding = langchain.embed_query(test_query)
    
    print(f'Query: {test_query}')
    print(f'Embedding Type: {type(embedding)}')
    print(f'Embedding Length: {len(embedding) if hasattr(embedding, "__len__") else "Unknown"}')
    print(f'First 5 values: {list(embedding)[:5] if hasattr(embedding, "__getitem__") else "Cannot access"}')
    
except Exception as e:
    print(f'Error: {e}')

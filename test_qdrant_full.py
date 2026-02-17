import os
from dotenv import load_dotenv
load_dotenv()
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, SearchRequest

# Test Qdrant client methods
url = os.getenv('qdrant_db_path')
api_key = os.getenv('qdrant_api_key')
client = QdrantClient(url=url, api_key=api_key)

print('All available methods:')
methods = [method for method in dir(client) if not method.startswith('_')]
for method in sorted(methods):
    print(f'  {method}')

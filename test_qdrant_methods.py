import os
from dotenv import load_dotenv
load_dotenv()
from qdrant_client import QdrantClient

# Test Qdrant client methods
url = os.getenv('qdrant_db_path')
api_key = os.getenv('qdrant_api_key')
client = QdrantClient(url=url, api_key=api_key)

print('Available methods:')
methods = [method for method in dir(client) if not method.startswith('_')]
for method in sorted(methods):
    if 'search' in method.lower():
        print(f'  {method}')

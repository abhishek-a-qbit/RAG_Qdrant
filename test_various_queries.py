import requests

# Test with different queries that might match our documents
test_queries = [
    '6sense',
    'revenue',
    'B2B',
    'predictive analytics',
    'customer data',
    'account-based marketing'
]

for query in test_queries:
    try:
        response = requests.post('http://localhost:8000/query', json={
            'query': query,
            'collection_name': 'documents',
            'limit': 5,
            'similarity_threshold': 0.5  # Lower threshold
        }, timeout=15)
        
        print(f'Query: {query}')
        print(f'Status: {response.status_code}')
        if response.status_code == 200:
            result = response.json()
            sources = result.get('sources', [])
            print(f'Sources: {len(sources)}')
            if len(sources) > 0:
                print('🎉 FOUND DOCUMENTS!')
                for source in sources[:2]:
                    print(f'  - {source.get("filename", "unknown")}')
            else:
                print('No sources found')
        else:
            print(f'Error: {response.text[:100]}')
    except Exception as e:
        print(f'Failed: {e}')
    print('---')

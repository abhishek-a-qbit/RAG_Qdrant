import requests
import time

time.sleep(3)
print('🔍 Checking API Status...')

try:
    health = requests.get('http://localhost:8000/health', timeout=5)
    print('📊 Health:', health.status_code)
    if health.status_code == 200:
        print('✅ Health Response:', health.json())
    
    # Test a simple query
    test_query = requests.post('http://localhost:8000/query', json={
        'query': '6sense',
        'collection_name': 'documents',
        'limit': 3,
        'similarity_threshold': 0.5
    }, timeout=20)
    
    print('🔍 Simple Query Status:', test_query.status_code)
    if test_query.status_code == 200:
        result = test_query.json()
        print('✅ Response received')
        print('📝 Answer preview:', result.get('answer', 'NO ANSWER')[:100])
        print('📚 Sources found:', len(result.get('sources', [])))
        if result.get('sources') and len(result.get('sources', [])) > 0:
            print('🎉 REAL RAG SOURCES!')
            for i, source in enumerate(result.get('sources', [])[:2]):
                print(f'  📄 {source.get("filename", "unknown")}')
        else:
            print('⚠️ Still no sources - checking search method...')
    else:
        print('❌ Query failed:', test_query.text[:200])
        
except Exception as e:
    print('❌ Connection error:', e)

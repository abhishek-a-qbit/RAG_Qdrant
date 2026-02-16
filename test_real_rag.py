import requests
import time

time.sleep(5)
print('🚀 Testing REAL RAG with 294 Documents...')

try:
    # Test query about 6sense
    test_query = requests.post('http://localhost:8000/query', json={
        'query': 'What are the main benefits of 6sense for B2B companies?',
        'collection_name': 'documents',
        'limit': 5,
        'similarity_threshold': 0.7
    }, timeout=30)
    
    print('🔍 RAG Query Test:')
    print('Status:', test_query.status_code)
    if test_query.status_code == 200:
        result = test_query.json()
        print('✅ SUCCESS!')
        print('📝 Answer:', result.get('answer', 'NO ANSWER')[:200])
        print('📚 Sources:', len(result.get('sources', [])))
        print('🎯 Confidence:', result.get('confidence_score', 0))
        print('⏱️ Response Time:', result.get('response_time', 0))
        
        # Check if we got real RAG results
        if result.get('sources') and len(result.get('sources', [])) > 0:
            print('🎉 REAL RAG WORKING!')
            for i, source in enumerate(result.get('sources', [])[:2]):
                print(f'  Source {i+1}: {source.get("filename", "unknown")}')
        else:
            print('⚠️ Using fallback (no sources found)')
            
    else:
        print('❌ Error:', test_query.text[:300])
        
except Exception as e:
    print('❌ Test failed:', e)

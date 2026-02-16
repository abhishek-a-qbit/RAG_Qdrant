import requests
import time

time.sleep(8)
print('🔍 Final API Test...')

try:
    health = requests.get('http://localhost:8000/health', timeout=5)
    print('📊 Health:', health.json())
    
    # Test with very specific query about B2B benefits
    test_query = requests.post('http://localhost:8000/query', json={
        'query': 'B2B revenue intelligence platform',
        'collection_name': 'documents',
        'limit': 5,
        'similarity_threshold': 0.3  # Very low threshold
    }, timeout=30)
    
    print('🔍 Final RAG Test:')
    print('Status:', test_query.status_code)
    if test_query.status_code == 200:
        result = test_query.json()
        print('✅ SUCCESS!')
        print('📝 Answer:', result.get('answer', 'NO ANSWER')[:200])
        print('📚 Sources:', len(result.get('sources', [])))
        print('🎯 Working:', 'REAL RAG' if len(result.get('sources', [])) > 0 else 'FALLBACK')
        
        if len(result.get('sources', [])) > 0:
            print('🎉 AMAZING! REAL RAG IS WORKING!')
            print('📄 Document Sources:')
            for i, source in enumerate(result.get('sources', [])[:3]):
                filename = source.get('filename', 'unknown')
                content = source.get('text', '')[:150]
                print(f'  {i+1}. {filename}')
                print(f'     Content: {content}...')
        else:
            print('🤔 Still using fallback - documents not found')
    else:
        print('❌ Error:', test_query.text[:300])
        
except Exception as e:
    print('❌ Test failed:', e)

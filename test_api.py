import requests
import time

time.sleep(3)
print('🚀 Testing Updated API with Streamlit Compatibility...')

# Test health
health = requests.get('http://localhost:8003/health')
print('📊 Health:', health.json())

# Test suggested questions
questions = requests.get('http://localhost:8003/suggested-questions?topic=6sense&num_questions=3')
print('📝 Suggested Questions Status:', questions.status_code)
if questions.status_code == 200:
    print('✅ Questions endpoint working!')
    qs = questions.json()
    for i, q in enumerate(qs):
        print(f'  {i+1}. {q["question"][:50]}...')

# Test query endpoint
query_resp = requests.post('http://localhost:8003/query', json={
    'query': 'What is benefit of 6sense?',
    'username': 'test-user'
})
print('🔍 Query Status:', query_resp.status_code)
if query_resp.status_code == 200:
    result = query_resp.json()
    print('✅ Query working!')
    print('📝 Response:', result['response'][:100] + '...')
    print('📚 Sources:', len(result['sources']))
else:
    print('❌ Query failed:', query_resp.text)

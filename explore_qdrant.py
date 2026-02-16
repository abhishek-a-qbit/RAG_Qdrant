# Let's explore what collections and documents we actually have
from utils.qdrant_utils import QdrantManager
from utils.utils import QDRANT_DB_PATH, QDRANT_API_KEY

print('🔍 Exploring Qdrant Collections...')

try:
    qdrant = QdrantManager(QDRANT_DB_PATH, QDRANT_API_KEY)
    
    # Get all collections
    collections = qdrant.client.get_collections()
    print('📚 Available Collections:', [col.name for col in collections.collections])
    
    # Check documents collection
    try:
        collection_info = qdrant.client.get_collection('documents')
        print('📊 Documents Collection Info:')
        print('  - Points:', collection_info.points_count)
        print('  - Status:', 'ACTIVE' if collection_info.status else 'INACTIVE')
        
        # Try to get vector count - different versions might use different attribute names
        try:
            vectors_count = collection_info.vectors_count
            print('  - Vectors:', vectors_count)
        except:
            try:
                vectors_count = collection_info.config.params.vectors.size
                print('  - Vector Config Size:', vectors_count)
            except:
                print('  - Vectors: Available but count method varies')
        
        # Sample some documents if available
        if collection_info.points_count > 0:
            print('🔍 Sampling documents...')
            try:
                # Try different search methods
                search_result = qdrant.client.search(
                    collection_name='documents',
                    query_vector=[0.1] * 1536,  # Dummy vector for sampling
                    limit=3,
                    with_payload=True
                )
                print('📝 Sample Documents Found:', len(search_result))
                for i, hit in enumerate(search_result):
                    if hasattr(hit, 'payload') and hit.payload:
                        print(f'  {i+1}. File: {hit.payload.get("filename", "unknown")}')
                        print(f'     Content Preview: {str(hit.payload.get("text", ""))[:100]}...')
            except Exception as search_error:
                print(f'❌ Search method error: {search_error}')
                # Try scroll method instead
                try:
                    scroll_result = qdrant.client.scroll(
                        collection_name='documents',
                        limit=3,
                        with_payload=True
                    )
                    print('📜 Scroll Results Found:', len(scroll_result[0]))
                    for i, point in enumerate(scroll_result[0]):
                        if hasattr(point, 'payload'):
                            print(f'  {i+1}. File: {point.payload.get("filename", "unknown")}')
                            print(f'     Content: {str(point.payload.get("text", ""))[:100]}...')
                except Exception as scroll_error:
                    print(f'❌ Scroll error: {scroll_error}')
        
    except Exception as e:
        print('❌ Error exploring collection:', e)
        
except Exception as e:
    print('❌ Error connecting to Qdrant:', e)

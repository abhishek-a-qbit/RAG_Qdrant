"""
Test script to verify Qdrant cloud connection
"""
import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from services.logger import setup_logger

# Load environment variables
load_dotenv()

# Initialize logger
logger = setup_logger("qdrant_test")

def test_qdrant_connection():
    """Test connection to Qdrant cloud instance"""
    try:
        # Get configuration from environment
        qdrant_url = os.getenv("qdrant_db_path")
        qdrant_api_key = os.getenv("qdrant_api_key")
        
        logger.info(f"Testing connection to Qdrant at: {qdrant_url}")
        
        if not qdrant_url or not qdrant_api_key:
            logger.error("Qdrant URL or API key not found in environment variables")
            return False
        
        # Initialize Qdrant client
        qdrant_client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key
        )
        
        # Test connection by getting collections
        logger.info("Fetching collections...")
        collections = qdrant_client.get_collections()
        
        logger.info("✅ Successfully connected to Qdrant!")
        logger.info(f"Available collections: {[col.name for col in collections.collections]}")
        
        # Test collection info if any exist
        if collections.collections:
            first_collection = collections.collections[0]
            logger.info(f"First collection: {first_collection.name}")
            
            try:
                collection_info = qdrant_client.get_collection(first_collection.name)
                logger.info(f"Collection info: {collection_info}")
            except Exception as e:
                logger.warning(f"Could not get collection info: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to connect to Qdrant: {str(e)}")
        return False

def test_manual_connection():
    """Test with hardcoded credentials"""
    try:
        logger.info("Testing with hardcoded credentials...")
        
        qdrant_client = QdrantClient(
            url="https://87467d62-d449-45a1-ba82-d1e731a529e0.sa-east-1-0.aws.cloud.qdrant.io:6333", 
            api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.T5vsq3-eUrX_6Gw8YldVLZhrVEwuuloLxv7XE_8mDOk"
        )
        
        collections = qdrant_client.get_collections()
        logger.info("✅ Manual connection successful!")
        logger.info(f"Collections: {[col.name for col in collections.collections]}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Manual connection failed: {str(e)}")
        return False

if __name__ == "__main__":
    print("Testing Qdrant Cloud Connection")
    print("=" * 50)
    
    # Test with environment variables
    print("\n1. Testing with environment variables:")
    env_success = test_qdrant_connection()
    
    # Test with hardcoded credentials
    print("\n2. Testing with hardcoded credentials:")
    manual_success = test_manual_connection()
    
    print("\n" + "=" * 50)
    if env_success or manual_success:
        print("✅ At least one connection method successful!")
    else:
        print("❌ All connection methods failed!")
        print("\nPlease check:")
        print("- Network connectivity")
        print("- API key validity")
        print("- URL correctness")
        print("- Firewall/proxy settings")

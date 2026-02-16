"""
Fix Qdrant collection dimension mismatch
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.qdrant_utils import QdrantManager
from utils.utils import QDRANT_DB_PATH, QDRANT_API_KEY

def fix_collection():
    """Fix the collection dimension issue"""
    print("Fixing Qdrant collection dimension mismatch...")
    
    # Initialize Qdrant manager
    qdrant_manager = QdrantManager(QDRANT_DB_PATH, QDRANT_API_KEY)
    
    # Delete existing collection
    if qdrant_manager.collection_exists("documents"):
        print("Deleting existing 'documents' collection...")
        success = qdrant_manager.delete_collection("documents")
        print(f"Collection deleted: {success}")
    
    # Create new collection with correct dimension (1536 for text-embedding-ada-002)
    print("Creating new 'documents' collection with 1536 dimensions...")
    success = qdrant_manager.create_collection("documents", vector_size=1536)
    print(f"Collection created: {success}")
    
    # Check collection info
    info = qdrant_manager.get_collection_info("documents")
    if info:
        print(f"Collection info: {info}")
    else:
        print("Could not get collection info")

if __name__ == "__main__":
    fix_collection()

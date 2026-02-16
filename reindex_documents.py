"""
Script to reindex all documents in the uploads directory
"""

import os
import json
import asyncio
from pathlib import Path

from utils.qdrant_utils import QdrantManager
from utils.langchain_utils import LangChainManager
from utils.utils import OPENAI_API_KEY, MODEL, TEMPERATURE, QDRANT_DB_PATH, QDRANT_API_KEY
from utils.file_extractor import extract_text_with_metadata

async def reindex_documents():
    """Reindex all documents in uploads directory"""
    
    print("🚀 Starting document reindexing...")
    
    # Initialize managers
    qm = QdrantManager(QDRANT_DB_PATH, QDRANT_API_KEY)
    lm = LangChainManager(OPENAI_API_KEY, MODEL, TEMPERATURE)
    
    # Delete and recreate collection
    print("🗑️ Deleting existing collection...")
    try:
        qm.delete_collection("documents")
        print("✅ Deleted existing collection")
    except Exception as e:
        print(f"⚠️ Could not delete collection: {e}")
    
    print("📁 Creating new collection...")
    qm.create_collection("documents")
    
    # Process all files
    uploads_dir = Path("uploads")
    total_files = 0
    processed_files = 0
    
    for file_path in uploads_dir.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in ['.json', '.csv', '.txt', '.md']:
            total_files += 1
    
    print(f"📊 Found {total_files} files to process")
    
    for file_path in uploads_dir.rglob("*"):
        if not file_path.is_file() or file_path.suffix.lower() not in ['.json', '.csv', '.txt', '.md']:
            continue
            
        try:
            print(f"📖 Processing: {file_path.name}")
            
            # Read file content
            with open(file_path, 'rb') as f:
                file_content = f.read()
            
            # Extract text and metadata
            result = await extract_text_with_metadata(file_content, file_path.name)
            text = result.get('text', '')
            metadata = result.get('metadata', {})
            
            if not text.strip():
                print(f"⚠️ No text extracted from {file_path.name}")
                continue
            
            # Update metadata
            metadata.update({
                'filename': file_path.name,
                'file_path': str(file_path),
                'file_size': len(file_content)
            })
            
            # Split text into chunks
            chunks = lm.split_text(text, chunk_size=2000, chunk_overlap=200)
            
            if not chunks:
                print(f"⚠️ No chunks created from {file_path.name}")
                continue
            
            # Generate embeddings
            embeddings = lm.embed_texts(chunks)
            
            # Prepare metadata for each chunk
            chunk_metadata = []
            for i, chunk in enumerate(chunks):
                chunk_meta = metadata.copy()
                chunk_meta.update({
                    'chunk_index': i,
                    'total_chunks': len(chunks),
                    'chunk_text': chunk[:200] + "..." if len(chunk) > 200 else chunk
                })
                chunk_metadata.append(chunk_meta)
            
            # Add to Qdrant
            success = qm.add_documents("documents", chunks, embeddings, chunk_metadata)
            
            if success:
                processed_files += 1
                print(f"✅ Successfully indexed {file_path.name} ({len(chunks)} chunks)")
            else:
                print(f"❌ Failed to index {file_path.name}")
                
        except Exception as e:
            print(f"❌ Error processing {file_path.name}: {e}")
    
    # Get final collection info
    try:
        info = qm.get_collection_info("documents")
        print(f"📊 Final collection info: {info}")
    except Exception as e:
        print(f"⚠️ Could not get collection info: {e}")
    
    print(f"🎉 Reindexing complete! Processed {processed_files}/{total_files} files")

if __name__ == "__main__":
    asyncio.run(reindex_documents())

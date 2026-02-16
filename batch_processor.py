"""
Batch Document Processor using existing functions
Processes all documents in upload folders using existing chunking, embedding, and indexing functions
"""

import os
import json
import time
import pandas as pd
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging
import asyncio
from uuid import uuid4

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BatchDocumentProcessor:
    """Batch processor using existing document processing functions"""
    
    def __init__(self, langchain_manager, qdrant_manager):
        self.langchain_manager = langchain_manager
        self.qdrant_manager = qdrant_manager
        self.processed_docs = 0
        self.failed_docs = 0
        self.total_chunks = 0
        
    async def process_all_documents(self, base_path: str = "uploads"):
        """Process all documents in the upload directory using working Qdrant manager"""
        logger.info("Starting batch document processing...")
        
        # Ensure collection exists
        if not self.qdrant_manager.collection_exists("documents"):
            logger.info("Creating documents collection...")
            self.qdrant_manager.create_collection("documents")
        
        # Collect all documents
        documents = []
        folders = ["dataset_1", "dataset_2", "News"]
        
        for folder in folders:
            folder_path = os.path.join(base_path, folder)
            if os.path.exists(folder_path):
                logger.info(f"Collecting documents from: {folder}")
                folder_docs = await self._collect_documents_from_folder(folder_path, folder)
                documents.extend(folder_docs)
        
        logger.info(f"Collected {len(documents)} documents total")
        
        # Process documents individually using the working Qdrant manager
        for i, doc_data in enumerate(documents):
            logger.info(f"Processing document {i+1}/{len(documents)}: {doc_data['filename']}")
            
            try:
                success = await self._process_single_document(doc_data)
                if success:
                    self.processed_docs += 1
                else:
                    self.failed_docs += 1
                
                # Small delay to avoid rate limits
                await asyncio.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error processing {doc_data['filename']}: {str(e)}")
                self.failed_docs += 1
        
        logger.info(f"Batch processing complete!")
        logger.info(f"Processed: {self.processed_docs} documents")
        logger.info(f"Failed: {self.failed_docs} documents") 
        logger.info(f"Total chunks: {self.total_chunks}")
        
        return {
            "processed": self.processed_docs,
            "failed": self.failed_docs,
            "total_chunks": self.total_chunks
        }
    
    async def _process_single_document(self, doc_data: Dict[str, Any]) -> bool:
        """Process a single document using the working Qdrant manager"""
        try:
            text_content = doc_data['text']
            metadata = doc_data['metadata']
            
            # Split text into chunks
            chunks = self.langchain_manager.split_text(text_content, 2000, 200)
            
            if not chunks:
                logger.warning(f"No chunks created for {metadata['filename']}")
                return False
            
            # Generate embeddings
            embeddings = self.langchain_manager.embed_texts(chunks)
            
            # Create metadata for each chunk
            chunk_metadata_list = []
            for i, chunk in enumerate(chunks):
                chunk_metadata = metadata.copy()
                chunk_metadata.update({
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "chunk_text": chunk[:500] + "..." if len(chunk) > 500 else chunk,
                    "chunk_length": len(chunk)
                })
                chunk_metadata_list.append(chunk_metadata)
            
            # Add to Qdrant using the working manager
            success = self.qdrant_manager.add_documents(
                "documents", chunks, embeddings, chunk_metadata_list
            )
            
            if success:
                self.total_chunks += len(chunks)
                logger.info(f"Successfully ingested {len(chunks)} chunks from {metadata['filename']}")
                return True
            else:
                logger.error(f"Failed to ingest {metadata['filename']}")
                return False
                
        except Exception as e:
            logger.error(f"Error ingesting {doc_data['filename']}: {str(e)}")
            return False
    
    async def _collect_documents_from_folder(self, folder_path: str, folder_name: str) -> List[Dict[str, Any]]:
        """Collect all documents from a folder with proper text extraction"""
        documents = []
        
        for file_path in Path(folder_path).glob("*"):
            if file_path.is_file():
                try:
                    doc_data = await self._extract_document_data(file_path, folder_name)
                    if doc_data:
                        documents.append(doc_data)
                except Exception as e:
                    logger.error(f"Error collecting {file_path}: {str(e)}")
                    self.failed_docs += 1
        
        return documents
    
    async def _extract_document_data(self, file_path: Path, folder_name: str) -> Optional[Dict[str, Any]]:
        """Extract text and metadata from a single file"""
        file_extension = file_path.suffix.lower()
        filename = file_path.name
        
        logger.info(f"Extracting text from: {filename}")
        
        # Extract text based on file type
        text_content = await self._extract_text_from_file(file_path, file_extension)
        
        if not text_content or len(text_content.strip()) < 50:
            logger.warning(f"Skipping {filename} - insufficient content")
            return None
        
        # Create metadata
        metadata = {
            "filename": filename,
            "folder": folder_name,
            "file_type": file_extension,
            "file_size": file_path.stat().st_size,
            "processed_at": time.time(),
            "source": f"uploads/{folder_name}/{filename}",
            "document_id": str(uuid4())
        }
        
        # Add folder-specific metadata
        if folder_name == "dataset_1":
            metadata["dataset_type"] = "structured_data"
            metadata["content_category"] = self._categorize_dataset1_file(filename)
        elif folder_name == "dataset_2":
            metadata["dataset_type"] = "reports_analysis"
            metadata["content_category"] = self._categorize_dataset2_file(filename)
        elif folder_name == "News":
            metadata["dataset_type"] = "news"
            metadata["content_category"] = "company_news"
        
        return {
            "text": text_content,
            "filename": filename,
            "file_type": file_extension,
            "metadata": metadata
        }
    
    async def _extract_text_from_file(self, file_path: Path, file_extension: str) -> str:
        """Extract text from various file types"""
        try:
            if file_extension == '.json':
                return self._extract_from_json(file_path)
            elif file_extension == '.txt':
                return self._extract_from_text(file_path)
            elif file_extension == '.csv':
                return self._extract_from_csv(file_path)
            elif file_extension == '.tsv':
                return self._extract_from_tsv(file_path)
            else:
                logger.warning(f"Unsupported file type: {file_extension}")
                return ""
        except Exception as e:
            logger.error(f"Error extracting text from {file_path}: {str(e)}")
            return ""
    
    def _extract_from_json(self, file_path: Path) -> str:
        """Extract text from JSON files"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle different JSON structures
        if isinstance(data, list):
            # Array of objects
            texts = []
            for item in data[:100]:  # Limit to first 100 items
                if isinstance(item, dict):
                    texts.append(json.dumps(item, indent=2))
                else:
                    texts.append(str(item))
            return "\n\n".join(texts)
        elif isinstance(data, dict):
            # Single object
            return json.dumps(data, indent=2)
        else:
            return str(data)
    
    def _extract_from_text(self, file_path: Path) -> str:
        """Extract text from text files"""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    
    def _extract_from_csv(self, file_path: Path) -> str:
        """Extract text from CSV files"""
        try:
            df = pd.read_csv(file_path)
            # Convert DataFrame to text representation
            text_parts = []
            
            # Add column headers
            text_parts.append("Columns: " + ", ".join(df.columns.tolist()))
            
            # Add first few rows as text
            for _, row in df.head(50).iterrows():  # Limit to first 50 rows
                row_text = " | ".join([str(val) for val in row.values])
                text_parts.append(row_text)
            
            return "\n".join(text_parts)
        except Exception as e:
            logger.error(f"Error reading CSV {file_path}: {str(e)}")
            return ""
    
    def _extract_from_tsv(self, file_path: Path) -> str:
        """Extract text from TSV files"""
        try:
            df = pd.read_csv(file_path, sep='\t')
            # Similar to CSV but with tab separation
            text_parts = []
            text_parts.append("Columns: " + ", ".join(df.columns.tolist()))
            
            for _, row in df.head(50).iterrows():
                row_text = " | ".join([str(val) for val in row.values])
                text_parts.append(row_text)
            
            return "\n".join(text_parts)
        except Exception as e:
            logger.error(f"Error reading TSV {file_path}: {str(e)}")
            return ""
    
    def _categorize_dataset1_file(self, filename: str) -> str:
        """Categorize dataset_1 files"""
        if "capabilities" in filename.lower():
            return "product_capabilities"
        elif "customer" in filename.lower():
            return "customer_information"
        elif "metrics" in filename.lower():
            return "performance_metrics"
        elif "integrations" in filename.lower():
            return "platform_integrations"
        elif "security" in filename.lower():
            return "security_compliance"
        elif "pricing" in filename.lower():
            return "pricing_information"
        elif "competitors" in filename.lower():
            return "competitive_analysis"
        elif "faq" in filename.lower():
            return "frequently_asked_questions"
        else:
            return "general_information"
    
    def _categorize_dataset2_file(self, filename: str) -> str:
        """Categorize dataset_2 files"""
        if "case study" in filename.lower():
            return "case_studies"
        elif "report" in filename.lower() or "wave" in filename.lower():
            return "industry_reports"
        elif "review" in filename.lower():
            return "customer_reviews"
        elif "seo" in filename.lower():
            return "seo_keywords"
        elif "stack" in filename.lower():
            return "technology_stack"
        else:
            return "market_analysis"

async def main():
    """Main processing function"""
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    from utils.langchain_utils import LangChainManager
    from utils.qdrant_utils import QdrantManager
    from utils.utils import OPENAI_API_KEY, MODEL, TEMPERATURE, QDRANT_DB_PATH, QDRANT_API_KEY
    
    # Initialize managers
    langchain_manager = LangChainManager(OPENAI_API_KEY, MODEL, TEMPERATURE)
    qdrant_manager = QdrantManager(QDRANT_DB_PATH, QDRANT_API_KEY)
    
    # Create processor
    processor = BatchDocumentProcessor(langchain_manager, qdrant_manager)
    
    # Process all documents
    results = await processor.process_all_documents()
    
    print(f"\n🎉 Processing Complete!")
    print(f"✅ Documents processed: {results['processed']}")
    print(f"❌ Documents failed: {results['failed']}")
    print(f"📊 Total chunks created: {results['total_chunks']}")

if __name__ == "__main__":
    asyncio.run(main())

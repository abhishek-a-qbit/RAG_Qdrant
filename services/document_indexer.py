import os
from typing import List, Dict, Any, Optional
from qdrant_client import AsyncQdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from services.logger import setup_logger
from utils.utils import OPENAI_API_KEY, create_text_splitter, generate_uuid, QDRANT_API_KEY, QDRANT_DB_PATH
from uuid import uuid4
import asyncio

class DocumentIndexer:
    """Handles document indexing and vector storage operations"""
    
    def __init__(self, qdrant_db_path: str = None, collection_name: str = "documents"):
        # Use centralized QDRANT_DB_PATH if not provided
        self.db_path = qdrant_db_path or QDRANT_DB_PATH
        self.collection_name = collection_name
        self.embedding_function = OpenAIEmbeddings(
            model="text-embedding-3-large", 
            api_key=OPENAI_API_KEY
        )
        self.vector_store = None
        self.client = AsyncQdrantClient(self.db_path, api_key=QDRANT_API_KEY)
        self.logger = setup_logger("document_indexer")
        self.text_splitter = create_text_splitter()
        
    async def initialize_collection(self, vector_size: int = 3072) -> bool:
        """Initialize Qdrant collection if it doesn't exist"""
        try:
            collections = await self.client.get_collections()
            collection_exists = any(
                col.name == self.collection_name 
                for col in collections.collections
            )
            
            if not collection_exists:
                await self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=vector_size,
                        distance=Distance.COSINE
                    )
                )
                self.logger.info(f"Created collection: {self.collection_name}")
            else:
                self.logger.info(f"Collection already exists: {self.collection_name}")
            
            # Initialize vector store
            self.vector_store = QdrantVectorStore(
                client=self.client,
                collection_name=self.collection_name,
                embedding=self.embedding_function
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing collection: {str(e)}")
            return False
    
    async def index_documents(self, documents: List[Document], 
                             batch_size: int = 100) -> Dict[str, Any]:
        """Index documents into vector store"""
        try:
            if not self.vector_store:
                await self.initialize_collection()
            
            # Split documents into chunks
            chunks = []
            metadatas = []
            
            for doc in documents:
                doc_chunks = self.text_splitter.split_documents([doc])
                for i, chunk in enumerate(doc_chunks):
                    chunk.metadata.update({
                        "chunk_index": i,
                        "document_id": doc.metadata.get("document_id", generate_uuid()),
                        "source": doc.metadata.get("source", "unknown")
                    })
                chunks.extend(doc_chunks)
            
            # Process in batches to avoid rate limits
            total_chunks = len(chunks)
            processed_chunks = 0
            
            for i in range(0, total_chunks, batch_size):
                batch = chunks[i:i + batch_size]
                
                # Add to vector store
                await self.vector_store.aadd_documents(batch)
                
                processed_chunks += len(batch)
                self.logger.info(f"Processed {processed_chunks}/{total_chunks} chunks")
            
            return {
                "success": True,
                "total_documents": len(documents),
                "total_chunks": total_chunks,
                "collection": self.collection_name
            }
            
        except Exception as e:
            self.logger.error(f"Error indexing documents: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def index_single_document(self, document: Document) -> Dict[str, Any]:
        """Index a single document"""
        return await self.index_documents([document])
    
    async def search_similar(self, query: str, k: int = 5, 
                           score_threshold: float = 0.7) -> List[Document]:
        """Search for similar documents"""
        try:
            if not self.vector_store:
                await self.initialize_collection()
            
            # Perform similarity search
            results = await self.vector_store.asimilarity_search(
                query, k=k, score_threshold=score_threshold
            )
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error searching similar documents: {str(e)}")
            return []
    
    async def search_with_scores(self, query: str, k: int = 5) -> List[tuple]:
        """Search for similar documents with scores"""
        try:
            if not self.vector_store:
                await self.initialize_collection()
            
            # Perform similarity search with scores
            results = await self.vector_store.asimilarity_search_with_score(query, k=k)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error searching with scores: {str(e)}")
            return []
    
    async def delete_document(self, document_id: str) -> bool:
        """Delete a document from the vector store"""
        try:
            if not self.vector_store:
                await self.initialize_collection()
            
            # Delete by metadata filter
            await self.vector_store.adelete(
                filter={
                    "must": [
                        {"key": "document_id", "match": {"value": document_id}}
                    ]
                }
            )
            
            self.logger.info(f"Deleted document: {document_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error deleting document: {str(e)}")
            return False
    
    async def get_collection_info(self) -> Optional[Dict[str, Any]]:
        """Get collection information"""
        try:
            info = await self.client.get_collection(self.collection_name)
            return {
                "name": self.collection_name,
                "vectors_count": info.vectors_count,
                "indexed_vectors_count": info.indexed_vectors_count,
                "points_count": info.points_count,
                "status": info.status
            }
        except Exception as e:
            self.logger.error(f"Error getting collection info: {str(e)}")
            return None
    
    async def clear_collection(self) -> bool:
        """Clear all documents from the collection"""
        try:
            await self.client.delete_collection(self.collection_name)
            await self.initialize_collection()
            self.logger.info(f"Cleared collection: {self.collection_name}")
            return True
        except Exception as e:
            self.logger.error(f"Error clearing collection: {str(e)}")
            return False
    
    def create_document_from_text(self, text: str, metadata: Dict[str, Any] = None) -> Document:
        """Create a Document object from text"""
        if metadata is None:
            metadata = {}
        
        return Document(
            page_content=text,
            metadata=metadata
        )
    
    async def get_document_count(self) -> int:
        """Get total number of documents in the collection"""
        try:
            info = await self.get_collection_info()
            return info.get("vectors_count", 0) if info else 0
        except Exception as e:
            self.logger.error(f"Error getting document count: {str(e)}")
            return 0
    
    async def index_in_qdrantdb(self, extracted_text: str, file_name: str, 
                               doc_type: str, chunk_size: int = None) -> bool:
        """Index extracted text into Qdrant database"""
        try:
            # Use dynamic chunk size from environment if not provided
            if chunk_size is None:
                chunk_size = int(os.getenv("chunk_size", "2000"))
            
            self.logger.info(f"Using dynamic chunk size: {chunk_size}")

            # Create a Document object
            doc = Document(
                page_content=extracted_text,
                metadata={
                    "file_name": file_name,
                    "doc_type": doc_type,
                    "document_id": generate_uuid()
                }
            )

            # Split the document
            text_splitter = RecursiveCharacterTextSplitter(
                separators=['\n\n', '\n', ','],
                chunk_size=chunk_size,
                chunk_overlap=200
            )
            docus = text_splitter.split_documents([doc])

            # Generate UUIDs for all chunks
            uuids = [str(uuid4()) for _ in range(len(docus))]
            collection = "rag_demo_collection"

            # Check if collection exists
            collections = await self.client.get_collections()

            if collection in [collection_name.name for collection_name in collections.collections]:
                self.logger.info(f"Collection {collection} already exists in QdrantDB")
            else:
                await self.client.create_collection(
                    collection_name=collection,
                    vectors_config=VectorParams(size=3072, distance=Distance.COSINE))
                self.logger.info(f"Created new collection: {collection}")

            # Initialize vector store
            self.vector_store = QdrantVectorStore.from_existing_collection(
                collection_name=collection, 
                embedding=self.embedding_function, 
                url=self.db_path,
                api_key=QDRANT_API_KEY
            )

            # Add documents to vector store
            await self.vector_store.aadd_documents(documents=docus, ids=uuids)

            self.logger.info(f"Successfully indexed {len(docus)} chunks from {file_name} in QdrantDB")
            return True

        except Exception as e:
            self.logger.error(f"Error indexing document in QdrantDB: {e}")
            raise

    async def get_retriever(self, top_k: int = 5):
        """Get retriever for similarity search"""
        try:
            collection = "rag_demo_collection"
            if self.vector_store is None:
                self.vector_store = QdrantVectorStore.from_existing_collection(
                    collection_name=collection, 
                    embedding=self.embedding_function, 
                    url=self.db_path,
                    api_key=QDRANT_API_KEY
                )

            return self.vector_store.as_retriever(
                search_type="similarity", 
                search_kwargs={"k": top_k}
            )
        except Exception as e:
            self.logger.error(f"Error creating retriever: {e}")
            raise

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on the indexer"""
        try:
            collection_info = await self.get_collection_info()
            document_count = await self.get_document_count()
            
            return {
                "status": "healthy",
                "collection_exists": collection_info is not None,
                "document_count": document_count,
                "collection_name": self.collection_name,
                "embedding_model": self.embedding_function.model
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }

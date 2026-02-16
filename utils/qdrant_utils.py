import os
from typing import List, Dict, Any, Optional, Tuple
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from langchain_qdrant import QdrantVectorStore as LangChainQdrant
from langchain_openai import OpenAIEmbeddings
import uuid
from datetime import datetime

class QdrantManager:
    """Manager for Qdrant vector database operations"""
    
    def __init__(self, url: str = "http://localhost:6333", api_key: Optional[str] = None):
        self.url = url
        self.api_key = api_key
        self.client = QdrantClient(url=url, api_key=api_key)
    
    def create_collection(self, collection_name: str, vector_size: int = 1536, 
                         distance: str = "Cosine") -> bool:
        """Create a new collection in Qdrant"""
        try:
            distance_map = {
                "Cosine": Distance.COSINE,
                "Euclidean": Distance.EUCLID,
                "Dot": Distance.DOT
            }
            
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=distance_map.get(distance, Distance.COSINE)
                )
            )
            return True
        except Exception as e:
            print(f"Error creating collection: {e}")
            return False
    
    def delete_collection(self, collection_name: str) -> bool:
        """Delete a collection"""
        try:
            self.client.delete_collection(collection_name=collection_name)
            return True
        except Exception as e:
            print(f"Error deleting collection: {e}")
            return False
    
    def collection_exists(self, collection_name: str) -> bool:
        """Check if collection exists"""
        try:
            collections = self.client.get_collections()
            return any(col.name == collection_name for col in collections.collections)
        except Exception as e:
            print(f"Error checking collection existence: {e}")
            return False
    
    def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """Get collection information"""
        try:
            collection = self.client.get_collection(collection_name)
            return {
                "name": collection_name,
                "points_count": collection.points_count,
                "status": "active" if collection.status else "inactive",
                "vector_size": collection.config.params.vectors.size if hasattr(collection.config, 'params') else 1536
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {}
    
    def add_points(self, collection_name: str, points: List[PointStruct]) -> bool:
        """Add points to collection"""
        try:
            self.client.upsert(
                collection_name=collection_name,
                points=points
            )
            return True
        except Exception as e:
            print(f"Error adding points: {e}")
            return False
    
    def add_documents(self, collection_name: str, documents: List[str], 
                     embeddings: List[List[float]], 
                     metadata: Optional[List[Dict[str, Any]]] = None) -> bool:
        """Add documents with embeddings to collection"""
        try:
            points = []
            for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
                point_id = str(uuid.uuid4())
                point_metadata = metadata[i] if metadata and i < len(metadata) else {}
                point_metadata.update({
                    "text": doc,
                    "created_at": datetime.now().isoformat()
                })
                
                point = PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=point_metadata
                )
                points.append(point)
            
            return self.add_points(collection_name, points)
        except Exception as e:
            print(f"Error adding documents: {e}")
            return False
    
    def search(self, collection_name: str, query_vector: List[float], 
               limit: int = 5, score_threshold: float = 0.7,
               filter_conditions: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search for similar vectors"""
        try:
            search_filter = None
            if filter_conditions:
                conditions = []
                for field, value in filter_conditions.items():
                    conditions.append(FieldCondition(key=field, match=MatchValue(value=value)))
                search_filter = Filter(must=conditions)
            
            results = self.client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=limit,
                score_threshold=score_threshold,
                query_filter=search_filter,
                with_payload=True
            )
            
            return [
                {
                    "id": str(point.id),
                    "score": score,
                    "payload": point.payload,
                    "text": point.payload.get("text", "")
                }
                for point, score in results
            ]
        except Exception as e:
            print(f"Error searching: {e}")
            return []
    
    def get_point(self, collection_name: str, point_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific point by ID"""
        try:
            result = self.client.retrieve(
                collection_name=collection_name,
                ids=[point_id]
            )
            
            if result:
                point = result[0]
                return {
                    "id": str(point.id),
                    "payload": point.payload,
                    "vector": point.vector if hasattr(point, 'vector') else None
                }
            return None
        except Exception as e:
            print(f"Error getting point: {e}")
            return None
    
    def delete_points(self, collection_name: str, point_ids: List[str]) -> bool:
        """Delete points by IDs"""
        try:
            self.client.delete(
                collection_name=collection_name,
                points_selector=point_ids
            )
            return True
        except Exception as e:
            print(f"Error deleting points: {e}")
            return False
    
    def scroll_collection(self, collection_name: str, limit: int = 100, 
                         offset: Optional[int] = None) -> List[Dict[str, Any]]:
        """Scroll through collection points"""
        try:
            results, next_page_offset = self.client.scroll(
                collection_name=collection_name,
                limit=limit,
                offset=offset
            )
            
            return [
                {
                    "id": str(point.id),
                    "payload": point.payload
                }
                for point in results
            ]
        except Exception as e:
            print(f"Error scrolling collection: {e}")
            return []
    
    def count_points(self, collection_name: str) -> int:
        """Count points in collection"""
        try:
            result = self.client.count(collection_name=collection_name)
            return result.count
        except Exception as e:
            print(f"Error counting points: {e}")
            return 0
    
    def create_langchain_vectorstore(self, collection_name: str, embeddings: OpenAIEmbeddings) -> LangChainQdrant:
        """Create LangChain Qdrant vectorstore"""
        try:
            return LangChainQdrant(
                client=self.client,
                collection_name=collection_name,
                embedding=embeddings,
                api_key=self.api_key  # Add API key here!
            )
        except Exception as e:
            print(f"Error creating LangChain vectorstore: {e}")
            raise
    
    def get_collections(self) -> List[Dict[str, Any]]:
        """Get all collections"""
        try:
            collections = self.client.get_collections()
            result = []
            for col in collections.collections:
                # Get collection info for each collection
                try:
                    info = self.client.get_collection(col.name)
                    result.append({
                        "name": col.name,
                        "vectors_count": info.vectors_count if hasattr(info, 'vectors_count') else 0,
                        "indexed_vectors_count": info.indexed_vectors_count if hasattr(info, 'indexed_vectors_count') else 0,
                        "points_count": info.points_count if hasattr(info, 'points_count') else 0,
                        "status": str(col.status) if hasattr(col, 'status') else "unknown"
                    })
                except Exception:
                    # If we can't get info, just return basic info
                    result.append({
                        "name": col.name,
                        "vectors_count": 0,
                        "indexed_vectors_count": 0,
                        "points_count": 0,
                        "status": "unknown"
                    })
            return result
        except Exception as e:
            print(f"Error getting collections: {e}")
            return []

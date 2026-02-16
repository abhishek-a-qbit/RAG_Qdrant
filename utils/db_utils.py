import os
import sqlite3
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path

class DatabaseManager:
    """Database manager for storing document metadata and tracking operations"""
    
    def __init__(self, db_path: str = "rag_system.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Documents table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    file_type TEXT NOT NULL,
                    file_size INTEGER NOT NULL,
                    upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    collection_name TEXT NOT NULL,
                    chunk_count INTEGER DEFAULT 0,
                    status TEXT DEFAULT 'uploaded'
                )
            ''')
            
            # Chunks table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    chunk_text TEXT NOT NULL,
                    vector_id TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (document_id) REFERENCES documents (id)
                )
            ''')
            
            # Query history table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS query_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query TEXT NOT NULL,
                    answer TEXT,
                    sources TEXT,
                    response_time REAL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
    
    def add_document(self, doc_id: str, filename: str, file_type: str, 
                    file_size: int, collection_name: str) -> bool:
        """Add document metadata to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO documents (id, filename, file_type, file_size, collection_name)
                    VALUES (?, ?, ?, ?, ?)
                ''', (doc_id, filename, file_type, file_size, collection_name))
                conn.commit()
                return True
        except sqlite3.Error as e:
            print(f"Error adding document: {e}")
            return False
    
    def update_document_chunks(self, doc_id: str, chunk_count: int) -> bool:
        """Update document chunk count"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE documents SET chunk_count = ?, status = 'processed'
                    WHERE id = ?
                ''', (chunk_count, doc_id))
                conn.commit()
                return True
        except sqlite3.Error as e:
            print(f"Error updating document chunks: {e}")
            return False
    
    def add_chunk(self, chunk_id: str, document_id: str, chunk_index: int, 
                  chunk_text: str, vector_id: Optional[str] = None) -> bool:
        """Add chunk metadata to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO chunks (id, document_id, chunk_index, chunk_text, vector_id)
                    VALUES (?, ?, ?, ?, ?)
                ''', (chunk_id, document_id, chunk_index, chunk_text, vector_id))
                conn.commit()
                return True
        except sqlite3.Error as e:
            print(f"Error adding chunk: {e}")
            return False
    
    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get document metadata"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM documents WHERE id = ?', (doc_id,))
                row = cursor.fetchone()
                return dict(row) if row else None
        except sqlite3.Error as e:
            print(f"Error getting document: {e}")
            return None
    
    def get_documents_by_collection(self, collection_name: str) -> List[Dict[str, Any]]:
        """Get all documents in a collection"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM documents WHERE collection_name = ?', (collection_name,))
                return [dict(row) for row in cursor.fetchall()]
        except sqlite3.Error as e:
            print(f"Error getting documents: {e}")
            return []
    
    def log_query(self, query: str, answer: str, sources: str, response_time: float) -> bool:
        """Log query to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO query_history (query, answer, sources, response_time)
                    VALUES (?, ?, ?, ?)
                ''', (query, answer, sources, response_time))
                conn.commit()
                return True
        except sqlite3.Error as e:
            print(f"Error logging query: {e}")
            return False
    
    def get_query_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get query history"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT * FROM query_history 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                ''', (limit,))
                return [dict(row) for row in cursor.fetchall()]
        except sqlite3.Error as e:
            print(f"Error getting query history: {e}")
            return []

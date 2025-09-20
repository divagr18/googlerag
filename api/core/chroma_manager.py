# api/core/chroma_manager.py

import hashlib
import os
import time
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any
from urllib.parse import urlparse

import chromadb
from chromadb.config import Settings
import numpy as np
from .embedding_manager import OptimizedEmbeddingManager

class ChromaDocumentManager:
    """
    Persistent document storage and retrieval using ChromaDB.
    Handles document embeddings with rich metadata for citations.
    """
    
    def __init__(self, persist_directory: str = "./chroma_db"):
        """Initialize ChromaDB client with persistent storage."""
        self.persist_directory = persist_directory
        
        # Ensure the directory exists
        os.makedirs(persist_directory, exist_ok=True)
        
        # Initialize ChromaDB client with persistent storage
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Use a single collection for all documents for now
        # This allows cross-document search and simpler management
        self.collection_name = "document_chunks"
        self.collection = self._get_or_create_collection()
        
        print(f"📚 ChromaDB initialized with persistent storage at: {persist_directory}")
    
    def _get_or_create_collection(self):
        """Get or create the main document collection."""
        try:
            collection = self.client.get_collection(
                name=self.collection_name,
                embedding_function=None  # We'll provide embeddings manually
            )
            print(f"📖 Using existing collection: {self.collection_name}")
        except Exception:
            collection = self.client.create_collection(
                name=self.collection_name,
                embedding_function=None,
                metadata={"description": "Document chunks with citations metadata"}
            )
            print(f"📝 Created new collection: {self.collection_name}")
        
        return collection
    
    def _generate_document_id(self, url: str) -> str:
        """Generate a unique document ID from URL."""
        return hashlib.md5(url.encode('utf-8')).hexdigest()
    
    def _generate_chunk_id(self, document_id: str, chunk_index: int) -> str:
        """Generate a unique chunk ID."""
        return f"{document_id}_{chunk_index:04d}"
    
    def _extract_document_title(self, url: str) -> str:
        """Extract document title from URL."""
        parsed = urlparse(url)
        
        # Handle our custom uploaded:// scheme
        if parsed.scheme == 'uploaded':
            filename = parsed.netloc + parsed.path  # For uploaded://filename.pdf
        else:
            filename = os.path.basename(parsed.path)
            
        if filename:
            # Remove file extension for cleaner title
            name, _ = os.path.splitext(filename)
            return name.replace('_', ' ').replace('-', ' ').title()
        return "Unknown Document"
    
    def _get_file_type(self, url: str) -> str:
        """Extract file type from URL."""
        parsed = urlparse(url)
        
        # Handle our custom uploaded:// scheme
        if parsed.scheme == 'uploaded':
            filename = parsed.netloc + parsed.path  # For uploaded://filename.pdf
        else:
            filename = parsed.path
            
        _, ext = os.path.splitext(filename)
        return ext.lower().lstrip('.') or 'unknown'
    
    def document_exists(self, url: str) -> bool:
        """Check if a document already exists in the database."""
        document_id = self._generate_document_id(url)
        
        try:
            # Get any chunk from this document using where filter
            results = self.collection.get(
                where={"document_id": document_id},
                limit=1
            )
            return len(results['ids']) > 0
        except Exception as e:
            print(f"⚠️ Error checking document existence: {e}")
            return False
    
    def store_document_chunks(
        self, 
        url: str, 
        chunks: List[Dict], 
        embeddings: np.ndarray,
        force_update: bool = False
    ) -> str:
        """
        Store document chunks with embeddings and metadata.
        
        Args:
            url: Document URL
            chunks: List of chunk dictionaries with text and metadata
            embeddings: Numpy array of embeddings
            force_update: Whether to update if document already exists
            
        Returns:
            document_id: The generated document ID
        """
        document_id = self._generate_document_id(url)
        
        # Check if document already exists
        if not force_update and self.document_exists(url):
            print(f"📋 Document already exists in database: {url[:50]}...")
            return document_id
        
        # If forcing update, delete existing chunks first
        if force_update and self.document_exists(url):
            self.delete_document(url)
            print(f"🔄 Updating existing document: {url[:50]}...")
        
        # Prepare data for ChromaDB
        chunk_ids = []
        chunk_texts = []
        chunk_embeddings = []
        chunk_metadatas = []
        
        document_title = self._extract_document_title(url)
        file_type = self._get_file_type(url)
        processing_timestamp = datetime.now().isoformat()
        
        print(f"📝 Processing document: '{document_title}' (type: {file_type})")
        
        for i, chunk in enumerate(chunks):
            chunk_id = self._generate_chunk_id(document_id, i)
            chunk_ids.append(chunk_id)
            chunk_texts.append(chunk['text'])
            chunk_embeddings.append(embeddings[i].tolist())
            
            # Enhanced metadata with citation information
            metadata = {
                "document_id": document_id,
                "document_url": url,
                "document_title": document_title,
                "file_type": file_type,
                "page_number": chunk.get('metadata', {}).get('page', 1),
                "chunk_index": i,
                "chunk_size": len(chunk['text']),
                "processed_timestamp": processing_timestamp,
                "processing_version": "v1.0"
            }
            chunk_metadatas.append(metadata)
        
        # Store in ChromaDB
        try:
            self.collection.add(
                ids=chunk_ids,
                embeddings=chunk_embeddings,
                documents=chunk_texts,
                metadatas=chunk_metadatas
            )
            
            print(f"💾 Stored {len(chunks)} chunks for document: {document_title}")
            print(f"📊 Document ID: {document_id}")
            
        except Exception as e:
            print(f"❌ Error storing document chunks: {e}")
            raise
        
        return document_id
    
    def search_documents(
        self, 
        query_embedding: np.ndarray, 
        n_results: int = 10,
        document_ids: Optional[List[str]] = None,
        file_types: Optional[List[str]] = None
    ) -> List[Tuple[Dict, float]]:
        """
        Search for relevant document chunks.
        
        Args:
            query_embedding: Query embedding vector
            n_results: Number of results to return
            document_ids: Optional list of document IDs to filter by
            file_types: Optional list of file types to filter by
            
        Returns:
            List of (chunk_dict, similarity_score) tuples
        """
        # Build where filter
        where_filter = {}
        if document_ids:
            where_filter["document_id"] = {"$in": document_ids}
        if file_types:
            where_filter["file_type"] = {"$in": file_types}
        
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=n_results,
                where=where_filter if where_filter else None,
                include=["documents", "metadatas", "distances"]
            )
            
            # Convert results to expected format
            chunks_with_scores = []
            for i in range(len(results['ids'][0])):
                chunk_dict = {
                    "text": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i]
                }
                # ChromaDB returns distances, convert to similarity (1 - distance)
                similarity = 1.0 - results['distances'][0][i]
                chunks_with_scores.append((chunk_dict, similarity))
            
            return chunks_with_scores
            
        except Exception as e:
            print(f"❌ Error searching documents: {e}")
            return []
    
    def get_document_info(self, url: str) -> Optional[Dict[str, Any]]:
        """Get information about a stored document."""
        document_id = self._generate_document_id(url)
        
        try:
            results = self.collection.get(
                where={"document_id": document_id},
                limit=1,
                include=["metadatas"]
            )
            
            if results['metadatas']:
                metadata = results['metadatas'][0]
                
                # Get total chunk count
                count_results = self.collection.get(
                    where={"document_id": document_id}
                )
                
                return {
                    "document_id": document_id,
                    "document_url": metadata["document_url"],
                    "document_title": metadata["document_title"],
                    "file_type": metadata["file_type"],
                    "total_chunks": len(count_results['ids']),
                    "processed_timestamp": metadata["processed_timestamp"],
                    "processing_version": metadata["processing_version"]
                }
            
        except Exception as e:
            print(f"⚠️ Error getting document info: {e}")
        
        return None
    
    def delete_document(self, url: str) -> bool:
        """Delete all chunks for a document by URL."""
        document_id = self._generate_document_id(url)
        return self.delete_document_by_id(document_id)
    
    def delete_document_by_id(self, document_id: str) -> bool:
        """Delete all chunks for a document by document ID."""
        try:
            # Get all chunk IDs for this document
            results = self.collection.get(
                where={"document_id": document_id}
            )
            
            if results['ids']:
                self.collection.delete(ids=results['ids'])
                print(f"🗑️ Deleted {len(results['ids'])} chunks for document ID: {document_id}")
                return True
                
        except Exception as e:
            print(f"❌ Error deleting document: {e}")
        
        return False
    
    def list_documents(self) -> List[Dict[str, Any]]:
        """List all documents in the database."""
        try:
            # Get all documents metadata
            results = self.collection.get(
                include=["metadatas"]
            )
            
            documents = {}
            for metadata in results['metadatas']:
                doc_id = metadata["document_id"]
                if doc_id not in documents:
                    documents[doc_id] = {
                        "document_id": doc_id,
                        "document_url": metadata["document_url"],
                        "document_title": metadata["document_title"],
                        "file_type": metadata["file_type"],
                        "processed_timestamp": metadata["processed_timestamp"],
                        "chunk_count": 0
                    }
                documents[doc_id]["chunk_count"] += 1
            
            return list(documents.values())
            
        except Exception as e:
            print(f"❌ Error listing documents: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        try:
            # Get all documents using get() instead of query()
            results = self.collection.get(
                include=["metadatas"]
            )
            
            if not results['metadatas']:
                return {
                    "collection_name": self.collection_name,
                    "total_chunks": 0,
                    "total_documents": 0,
                    "file_types": {},
                    "last_updated": "Never"
                }
            
            total_chunks = len(results['metadatas'])
            unique_documents = len(set(
                m["document_id"] for m in results['metadatas']
            ))
            
            file_types = {}
            last_updated = datetime.min
            
            for metadata in results['metadatas']:
                file_type = metadata["file_type"]
                file_types[file_type] = file_types.get(file_type, 0) + 1
                
                # Track most recent update
                if "created_at" in metadata:
                    doc_time = datetime.fromisoformat(metadata["created_at"])
                    if doc_time > last_updated:
                        last_updated = doc_time
            
            return {
                "collection_name": self.collection_name,
                "total_chunks": total_chunks,
                "total_documents": unique_documents,
                "file_types": file_types,
                "last_updated": last_updated.isoformat() if last_updated != datetime.min else "Never",
                "persist_directory": self.persist_directory
            }
            
        except Exception as e:
            print(f"❌ Error getting stats: {e}")
            return {
                "collection_name": self.collection_name,
                "total_chunks": 0,
                "total_documents": 0,
                "file_types": {},
                "last_updated": "Error",
                "persist_directory": self.persist_directory
            }
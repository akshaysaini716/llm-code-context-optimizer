"""
Abstract base class for vector store implementations
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

from rag.models import CodeBaseChunk, RetrievalResult


class VectorStore(ABC):
    """Abstract base class for vector store implementations"""
    
    @abstractmethod
    def upsert_chunks(self, chunks: List[CodeBaseChunk]) -> bool:
        """
        Insert or update chunks in the vector store
        
        Args:
            chunks: List of CodeBaseChunk objects with embeddings
            
        Returns:
            bool: True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def search_similar(
        self, 
        query_vector: List[float], 
        project_path: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None, 
        top_k: int = 10
    ) -> List[RetrievalResult]:
        """
        Search for similar vectors
        
        Args:
            query_vector: Query embedding vector
            project_path: Optional project path filter
            filters: Additional filters
            top_k: Number of results to return
            
        Returns:
            List[RetrievalResult]: Search results with similarity scores
        """
        pass
    
    @abstractmethod
    def search_by_filter(
        self, 
        filters: Dict[str, Any], 
        limit: int = 10
    ) -> List[CodeBaseChunk]:
        """
        Search chunks by filter conditions only (no vector similarity)
        
        Args:
            filters: Filter conditions
            limit: Maximum number of results
            
        Returns:
            List[CodeBaseChunk]: Matching chunks
        """
        pass
    
    @abstractmethod
    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about the vector collection/table
        
        Returns:
            Dict[str, Any]: Collection statistics and info
        """
        pass
    
    @abstractmethod
    def delete_chunks(self, chunk_ids: List[str]) -> bool:
        """
        Delete chunks by their IDs
        
        Args:
            chunk_ids: List of chunk IDs to delete
            
        Returns:
            bool: True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def delete_chunks_by_project_path(self, project_path: str) -> bool:
        """
        Delete all chunks for a specific project path
        
        Args:
            project_path: The project path to delete chunks for
            
        Returns:
            bool: True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def clear_collection(self) -> bool:
        """
        Clear all data from the collection/table
        
        Returns:
            bool: True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def health_check(self) -> Dict[str, Any]:
        """
        Check the health status of the vector store
        
        Returns:
            Dict[str, Any]: Health status information
        """
        pass

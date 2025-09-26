"""
Factory for creating vector store instances based on configuration
"""

import logging
from typing import Optional

from rag.configs import VectorStoreConfig
from rag.vector_store.base_vector_store import VectorStore
from rag.vector_store.qdrant_client_impl import QdrantClientImpl
from rag.vector_store.postgresql_vector_store import PostgreSQLVectorStore

logger = logging.getLogger(__name__)


class VectorStoreFactory:
    """Factory class for creating vector store instances"""
    
    @staticmethod
    def create_vector_store(config: VectorStoreConfig) -> VectorStore:
        """
        Create a vector store instance based on configuration
        
        Args:
            config: VectorStoreConfig with provider and settings
            
        Returns:
            VectorStore: Configured vector store instance
            
        Raises:
            ValueError: If provider is not supported
            RuntimeError: If vector store creation fails
        """
        provider = config.provider.lower()
        
        try:
            if provider == "qdrant":
                return VectorStoreFactory._create_qdrant_store(config)
            elif provider == "postgresql":
                return VectorStoreFactory._create_postgresql_store(config)
            else:
                raise ValueError(f"Unsupported vector store provider: {provider}")
                
        except Exception as e:
            logger.error(f"Failed to create {provider} vector store: {e}")
            raise RuntimeError(f"Vector store creation failed: {e}")
    
    @staticmethod
    def _create_qdrant_store(config: VectorStoreConfig) -> QdrantClientImpl:
        """Create Qdrant vector store instance"""
        logger.info("Creating Qdrant vector store")
        
        # For now, use the existing QdrantClientImpl constructor
        # TODO: Update QdrantClientImpl to accept configuration
        store = QdrantClientImpl()
        
        # Override collection name if different
        if config.collection_name != "code_chunks":
            store.collection_name = config.collection_name
        
        logger.info("Qdrant vector store created successfully")
        return store
    
    @staticmethod
    def _create_postgresql_store(config: VectorStoreConfig) -> PostgreSQLVectorStore:
        """Create PostgreSQL vector store instance"""
        logger.info("Creating PostgreSQL vector store")
        
        # Create PostgreSQL store with configuration
        store = PostgreSQLVectorStore(config.postgresql)
        store.embedding_dimension = config.embedding_dimension
        
        logger.info("PostgreSQL vector store created successfully")
        return store
    
    @staticmethod
    def get_supported_providers() -> list:
        """Get list of supported vector store providers"""
        return ["qdrant", "postgresql"]
    
    @staticmethod
    def validate_config(config: VectorStoreConfig) -> bool:
        """
        Validate vector store configuration
        
        Args:
            config: VectorStoreConfig to validate
            
        Returns:
            bool: True if valid, False otherwise
        """
        try:
            # Check provider
            if config.provider not in VectorStoreFactory.get_supported_providers():
                logger.error(f"Unsupported provider: {config.provider}")
                return False
            
            # Check embedding dimension
            if config.embedding_dimension <= 0:
                logger.error("Embedding dimension must be positive")
                return False
            
            # Check distance metric
            valid_metrics = ["cosine", "euclidean", "dot_product"]
            if config.distance_metric not in valid_metrics:
                logger.error(f"Invalid distance metric: {config.distance_metric}")
                return False
            
            # Provider-specific validation
            if config.provider == "postgresql":
                pg_config = config.postgresql
                if not all([pg_config.host, pg_config.database, pg_config.username]):
                    logger.error("PostgreSQL configuration missing required fields")
                    return False
            
            elif config.provider == "qdrant":
                qdrant_config = config.qdrant
                if not qdrant_config.host:
                    logger.error("Qdrant host is required")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Configuration validation error: {e}")
            return False


# Singleton instance for global use
_vector_store_instance: Optional[VectorStore] = None
_current_config: Optional[VectorStoreConfig] = None


def get_vector_store(config: Optional[VectorStoreConfig] = None) -> VectorStore:
    """
    Get or create a vector store instance (singleton pattern)
    
    Args:
        config: VectorStoreConfig. If None, uses existing instance
        
    Returns:
        VectorStore: Configured vector store instance
    """
    global _vector_store_instance, _current_config
    
    # If config provided and different from current, recreate instance
    if config and config != _current_config:
        logger.info("Vector store configuration changed, recreating instance")
        _vector_store_instance = VectorStoreFactory.create_vector_store(config)
        _current_config = config
    
    # Create instance if none exists
    elif _vector_store_instance is None:
        if config is None:
            # Use default configuration
            from rag.configs import VectorStoreConfig
            config = VectorStoreConfig()
            logger.info("Using default vector store configuration")
        
        _vector_store_instance = VectorStoreFactory.create_vector_store(config)
        _current_config = config
    
    return _vector_store_instance


def reset_vector_store():
    """Reset the singleton vector store instance"""
    global _vector_store_instance, _current_config
    
    if _vector_store_instance and hasattr(_vector_store_instance, 'close'):
        _vector_store_instance.close()
    
    _vector_store_instance = None
    _current_config = None
    logger.info("Vector store instance reset")

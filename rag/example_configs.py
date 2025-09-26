"""
Example configurations for different vector store setups
"""

from rag.configs import VectorStoreConfig, QdrantConfig, PostgreSQLConfig

# Default Qdrant configuration (development)
DEFAULT_QDRANT = VectorStoreConfig(
    provider="qdrant",
    collection_name="code_chunks",
    embedding_dimension=768,
    distance_metric="cosine",
    qdrant=QdrantConfig(
        host="localhost",
        port=6333,
        https=False,
        api_key=None
    )
)

# PostgreSQL configuration (local development)
LOCAL_POSTGRESQL = VectorStoreConfig(
    provider="postgresql",
    collection_name="code_chunks",
    embedding_dimension=768,
    distance_metric="cosine",
    postgresql=PostgreSQLConfig(
        host="localhost",
        port=5432,
        database="rag_dev",
        username="rag_user",
        password="rag_password",
        max_connections=10,
        table_name="code_chunks",
        enable_ivfflat_index=True,
        ivfflat_lists=50
    )
)

# Production PostgreSQL configuration
PRODUCTION_POSTGRESQL = VectorStoreConfig(
    provider="postgresql",
    collection_name="production_code_chunks",
    embedding_dimension=768,
    distance_metric="cosine",
    postgresql=PostgreSQLConfig(
        host="prod-db.example.com",
        port=5432,
        database="rag_production",
        username="rag_prod_user",
        password="secure_prod_password",
        max_connections=50,
        connection_timeout=60,
        table_name="code_chunks",
        enable_ivfflat_index=True,
        ivfflat_lists=200,
        maintenance_work_mem="512MB",
        effective_cache_size="4GB"
    )
)

# Qdrant Cloud configuration
QDRANT_CLOUD = VectorStoreConfig(
    provider="qdrant",
    collection_name="cloud_code_chunks",
    embedding_dimension=768,
    distance_metric="cosine",
    qdrant=QdrantConfig(
        host="your-cluster.qdrant.io",
        port=6333,
        https=True,
        api_key="your_api_key_here",
        search_params={
            "exact": False,
            "hnsw_ef": 256,
            "indexed_only": True
        }
    )
)

# High-performance local Qdrant
HIGH_PERF_QDRANT = VectorStoreConfig(
    provider="qdrant",
    collection_name="high_perf_chunks",
    embedding_dimension=768,
    distance_metric="cosine",
    qdrant=QdrantConfig(
        host="localhost",
        port=6333,
        https=False,
        on_disk_payload=False,  # Keep payloads in memory for speed
        index_threshold=50000,
        search_params={
            "exact": False,
            "hnsw_ef": 512,  # Higher for better recall
            "indexed_only": True
        }
    )
)

def get_config_by_environment(env: str = "development") -> VectorStoreConfig:
    """
    Get configuration based on environment
    
    Args:
        env: Environment name ("development", "staging", "production")
        
    Returns:
        VectorStoreConfig: Configuration for the environment
    """
    configs = {
        "development": DEFAULT_QDRANT,
        "dev_postgres": LOCAL_POSTGRESQL,
        "staging": LOCAL_POSTGRESQL,
        "production": PRODUCTION_POSTGRESQL,
        "cloud": QDRANT_CLOUD,
        "high_perf": HIGH_PERF_QDRANT
    }
    
    return configs.get(env, DEFAULT_QDRANT)

def get_config_from_env():
    """
    Create configuration from environment variables
    """
    import os
    
    provider = os.getenv("VECTOR_STORE_PROVIDER", "qdrant").lower()
    
    if provider == "postgresql":
        return VectorStoreConfig(
            provider="postgresql",
            collection_name=os.getenv("VECTOR_COLLECTION_NAME", "code_chunks"),
            embedding_dimension=int(os.getenv("EMBEDDING_DIMENSION", "768")),
            distance_metric=os.getenv("DISTANCE_METRIC", "cosine"),
            postgresql=PostgreSQLConfig(
                host=os.getenv("POSTGRES_HOST", "localhost"),
                port=int(os.getenv("POSTGRES_PORT", "5432")),
                database=os.getenv("POSTGRES_DB", "rag_db"),
                username=os.getenv("POSTGRES_USER", "rag_user"),
                password=os.getenv("POSTGRES_PASSWORD", ""),
                max_connections=int(os.getenv("POSTGRES_MAX_CONNECTIONS", "20")),
                table_name=os.getenv("POSTGRES_TABLE_NAME", "code_chunks"),
                enable_ivfflat_index=os.getenv("ENABLE_IVFFLAT_INDEX", "true").lower() == "true",
                ivfflat_lists=int(os.getenv("IVFFLAT_LISTS", "100"))
            )
        )
    else:  # Qdrant
        return VectorStoreConfig(
            provider="qdrant",
            collection_name=os.getenv("VECTOR_COLLECTION_NAME", "code_chunks"),
            embedding_dimension=int(os.getenv("EMBEDDING_DIMENSION", "768")),
            distance_metric=os.getenv("DISTANCE_METRIC", "cosine"),
            qdrant=QdrantConfig(
                host=os.getenv("QDRANT_HOST", "localhost"),
                port=int(os.getenv("QDRANT_PORT", "6333")),
                https=os.getenv("QDRANT_HTTPS", "false").lower() == "true",
                api_key=os.getenv("QDRANT_API_KEY")
            )
        )

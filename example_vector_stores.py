#!/usr/bin/env python3
"""
Example script demonstrating how to use different vector stores
"""

import os
from pathlib import Path

# Add the project root to the Python path
import sys
sys.path.append(str(Path(__file__).parent))

from rag.configs import VectorStoreConfig, QdrantConfig, PostgreSQLConfig
from rag.core.rag_system import RAGSystem
from rag.example_configs import get_config_by_environment, get_config_from_env


def example_qdrant():
    """Example using Qdrant vector store"""
    print("=== Qdrant Example ===")
    
    # Configuration
    config = VectorStoreConfig(
        provider="qdrant",
        collection_name="example_chunks",
        qdrant=QdrantConfig(
            host="localhost",
            port=6333
        )
    )
    
    # Create RAG system
    rag = RAGSystem(vector_store_config=config)
    
    # Health check
    health = rag.vector_store.health_check()
    print(f"Qdrant Health: {health}")
    
    return rag


def example_postgresql():
    """Example using PostgreSQL vector store"""
    print("=== PostgreSQL Example ===")
    
    # Configuration
    config = VectorStoreConfig(
        provider="postgresql",
        collection_name="example_chunks",
        postgresql=PostgreSQLConfig(
            host="localhost",
            port=5432,
            database="rag_db",
            username="rag_user",
            password="rag_password",
            max_connections=10
        )
    )
    
    try:
        # Create RAG system
        rag = RAGSystem(vector_store_config=config)
        
        # Health check
        health = rag.vector_store.health_check()
        print(f"PostgreSQL Health: {health}")
        
        return rag
        
    except Exception as e:
        print(f"PostgreSQL connection failed: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure PostgreSQL is running with pgvector extension")
        print("2. Run setup script: ./rag/scripts/setup_postgresql_docker.sh")
        print("3. Or use Docker: docker run --name postgres-pgvector -e POSTGRES_PASSWORD=rag_password -e POSTGRES_DB=rag_db -e POSTGRES_USER=rag_user -p 5432:5432 -d pgvector/pgvector:pg16")
        print("4. Test setup: python test_postgresql_setup.py")
        return None


def example_environment_config():
    """Example using environment variables"""
    print("=== Environment Config Example ===")
    
    # Set some environment variables for demo
    os.environ.setdefault("VECTOR_STORE_PROVIDER", "qdrant")
    os.environ.setdefault("QDRANT_HOST", "localhost")
    os.environ.setdefault("QDRANT_PORT", "6333")
    
    # Get config from environment
    config = get_config_from_env()
    print(f"Using provider: {config.provider}")
    
    # Create RAG system
    rag = RAGSystem(vector_store_config=config)
    return rag


def example_indexing_and_query(rag_system):
    """Example of indexing and querying"""
    if not rag_system:
        return
        
    print(f"\n=== Indexing and Query with {rag_system.vector_store.health_check()['provider']} ===")
    
    # Index a small sample project
    sample_project = Path(__file__).parent / "sample_project"
    
    if sample_project.exists():
        print(f"Indexing sample project: {sample_project}")
        
        try:
            result = rag_system.index_codebase(
                project_path=str(sample_project),
                file_patterns=["*.py"],
                force_reindex=False
            )
            
            print(f"Indexing result:")
            print(f"  Files processed: {result['files_processed']}")
            print(f"  Chunks created: {result['chunks_created']}")
            print(f"  Chunks embedded: {result['chunks_embedded']}")
            print(f"  Time taken: {result['indexing_time_seconds']:.2f}s")
            
            # Try a query
            print(f"\nTesting query...")
            response = rag_system.query(
                query="How to calculate sum of two numbers?",
                project_path=str(sample_project),
                top_k=3
            )
            
            print(f"Query response:")
            print(f"  Chunks found: {len(response.chunks_used)}")
            print(f"  Total tokens: {response.total_tokens}")
            print(f"  Retrieval time: {response.retrieval_time_ms:.2f}ms")
            print(f"  Context preview: {response.context[:200]}...")
            
        except Exception as e:
            print(f"Error during indexing/query: {e}")
    else:
        print(f"Sample project not found at {sample_project}")


def compare_vector_stores():
    """Compare performance of different vector stores"""
    print("\n=== Vector Store Comparison ===")
    
    configs = {
        "qdrant": get_config_by_environment("development"),
        "postgresql": get_config_by_environment("dev_postgres")
    }
    
    for name, config in configs.items():
        print(f"\nTesting {name}:")
        try:
            rag = RAGSystem(vector_store_config=config)
            health = rag.vector_store.health_check()
            
            if health["status"] == "healthy":
                info = rag.vector_store.get_collection_info()
                print(f"  Status: ✓ Healthy")
                print(f"  Total chunks: {info.get('total_points', 0)}")
                print(f"  Provider: {health['provider']}")
            else:
                print(f"  Status: ✗ {health.get('error', 'Unhealthy')}")
                
        except Exception as e:
            print(f"  Status: ✗ Error: {e}")


def main():
    """Main example function"""
    print("Vector Store Examples for RAG System")
    print("=" * 50)
    
    # Example 1: Qdrant (should work if Qdrant is running)
    try:
        qdrant_rag = example_qdrant()
        example_indexing_and_query(qdrant_rag)
    except Exception as e:
        print(f"Qdrant example failed: {e}")
    
    print("\n" + "=" * 50)
    
    # Example 2: PostgreSQL (requires setup)
    try:
        postgresql_rag = example_postgresql()
        example_indexing_and_query(postgresql_rag)
    except Exception as e:
        print(f"PostgreSQL example failed: {e}")
    
    print("\n" + "=" * 50)
    
    # Example 3: Environment configuration
    try:
        env_rag = example_environment_config()
    except Exception as e:
        print(f"Environment config example failed: {e}")
    
    print("\n" + "=" * 50)
    
    # Example 4: Compare vector stores
    compare_vector_stores()
    
    print("\n" + "=" * 50)
    print("Examples completed!")
    print("\nNext steps:")
    print("1. For Qdrant: docker run -p 6333:6333 qdrant/qdrant:latest")
    print("2. For PostgreSQL: ./rag/scripts/setup_postgresql_docker.sh (or see POSTGRESQL_QUICKSTART.md)")
    print("3. Test your setup: python test_postgresql_setup.py")  
    print("4. Set environment variables as needed")
    print("5. Use rag/example_configs.py for different setups")


if __name__ == "__main__":
    main()

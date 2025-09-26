#!/usr/bin/env python3
"""
Quick test script to verify PostgreSQL with pgvector is working
"""

import sys
import psycopg2
from psycopg2.extras import RealDictCursor

def test_postgresql_connection():
    """Test basic PostgreSQL connection"""
    try:
        conn = psycopg2.connect(
            host="localhost",
            port=5432,
            database="rag_db",
            user="rag_user",
            password="rag_password"
        )
        print("✅ PostgreSQL connection successful")
        return conn
    except Exception as e:
        print(f"❌ PostgreSQL connection failed: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure PostgreSQL is running")
        print("2. Check connection details (host, port, database, user, password)")
        print("3. Run: docker ps | grep postgres")
        return None

def test_pgvector_extension(conn):
    """Test pgvector extension"""
    try:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # Try to create extension
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
        print("✅ pgvector extension created/verified")
        
        # Check extension version
        cursor.execute("SELECT extname, extversion FROM pg_extension WHERE extname = 'vector'")
        result = cursor.fetchone()
        if result:
            print(f"✅ pgvector version: {result['extversion']}")
        else:
            print("❌ pgvector extension not found")
            return False
        
        # Test vector operations
        cursor.execute("SELECT '[1,2,3]'::vector <-> '[4,5,6]'::vector as cosine_distance")
        result = cursor.fetchone()
        print(f"✅ Vector operations work: cosine_distance = {result['cosine_distance']}")
        
        # Test vector version function
        cursor.execute("SELECT vector_version()")
        result = cursor.fetchone()
        print(f"✅ Vector version function: {result['vector_version']}")
        
        conn.commit()
        return True
        
    except Exception as e:
        print(f"❌ pgvector extension test failed: {e}")
        print("\nThis means pgvector is not properly installed.")
        print("Solutions:")
        print("1. Use Docker: docker run --name postgres-pgvector -e POSTGRES_PASSWORD=rag_password -e POSTGRES_DB=rag_db -e POSTGRES_USER=rag_user -p 5432:5432 -d pgvector/pgvector:pg16")
        print("2. Install pgvector: brew install pgvector (macOS) or apt install postgresql-14-pgvector (Ubuntu)")
        print("3. Build from source: https://github.com/pgvector/pgvector")
        return False

def test_rag_integration():
    """Test integration with RAG system"""
    try:
        # Import RAG components
        import os
        import sys
        from pathlib import Path
        
        # Add project root to path
        project_root = Path(__file__).parent
        sys.path.insert(0, str(project_root))
        
        from rag.configs import VectorStoreConfig, PostgreSQLConfig
        from rag.vector_store.vector_store_factory import get_vector_store
        
        # Create config
        config = VectorStoreConfig(
            provider="postgresql",
            postgresql=PostgreSQLConfig(
                host="localhost",
                port=5432,
                database="rag_db",
                username="rag_user",
                password="rag_password"
            )
        )
        
        # Test vector store creation
        vector_store = get_vector_store(config)
        print("✅ RAG vector store created successfully")
        
        # Test health check
        health = vector_store.health_check()
        print(f"✅ Vector store health check: {health['status']}")
        
        # Test collection info
        info = vector_store.get_collection_info()
        print(f"✅ Collection info: {info.get('total_points', 0)} chunks stored")
        
        return True
        
    except Exception as e:
        print(f"❌ RAG integration test failed: {e}")
        print("This might be normal if you haven't run the full setup yet.")
        return False

def main():
    print("🧪 Testing PostgreSQL + pgvector Setup")
    print("=" * 50)
    
    # Test 1: PostgreSQL Connection
    print("\n1. Testing PostgreSQL connection...")
    conn = test_postgresql_connection()
    if not conn:
        return 1
    
    # Test 2: pgvector Extension
    print("\n2. Testing pgvector extension...")
    pgvector_ok = test_pgvector_extension(conn)
    conn.close()
    
    if not pgvector_ok:
        return 1
    
    # Test 3: RAG Integration (optional)
    print("\n3. Testing RAG integration...")
    rag_ok = test_rag_integration()
    
    print("\n" + "=" * 50)
    if pgvector_ok:
        print("🎉 PostgreSQL + pgvector setup is working!")
        print("\nYou can now use PostgreSQL as your vector store:")
        print("python example_vector_stores.py")
        if rag_ok:
            print("✅ RAG integration also working")
        else:
            print("⚠️  RAG integration needs setup (run: pip install -r requirements.txt)")
        return 0
    else:
        print("❌ Setup incomplete - follow the troubleshooting steps above")
        return 1

if __name__ == "__main__":
    exit(main())

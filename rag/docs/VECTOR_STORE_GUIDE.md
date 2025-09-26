# Vector Store Configuration Guide

This guide explains how to configure and use different vector database backends in the RAG system.

## Supported Vector Stores

- **Qdrant**: High-performance vector search engine (default)
- **PostgreSQL**: Popular relational database with pgvector extension

## Configuration

### Basic Configuration

```python
from rag.configs import VectorStoreConfig, QdrantConfig, PostgreSQLConfig

# Use Qdrant (default)
qdrant_config = VectorStoreConfig(
    provider="qdrant",
    collection_name="code_chunks",
    embedding_dimension=768,
    distance_metric="cosine"
)

# Use PostgreSQL
postgresql_config = VectorStoreConfig(
    provider="postgresql",
    collection_name="code_chunks",
    embedding_dimension=768,
    distance_metric="cosine",
    postgresql=PostgreSQLConfig(
        host="localhost",
        port=5432,
        database="rag_db",
        username="rag_user",
        password="rag_password"
    )
)
```

### Using the Factory Pattern

```python
from rag.vector_store.vector_store_factory import VectorStoreFactory, get_vector_store
from rag.configs import VectorStoreConfig

# Method 1: Direct factory usage
config = VectorStoreConfig(provider="postgresql")
vector_store = VectorStoreFactory.create_vector_store(config)

# Method 2: Singleton pattern (recommended)
vector_store = get_vector_store(config)

# Method 3: Use with RAG System
from rag.core.rag_system import RAGSystem
rag_system = RAGSystem(vector_store_config=config)
```

## Qdrant Setup

### Installation

```bash
# Using Docker
docker run -p 6333:6333 qdrant/qdrant:latest
```

### Configuration

```python
qdrant_config = VectorStoreConfig(
    provider="qdrant",
    qdrant=QdrantConfig(
        host="localhost",
        port=6333,
        https=False,
        api_key=None  # Set if using Qdrant Cloud
    )
)
```
docker run --name postgres-pgvector \
  -e POSTGRES_PASSWORD=password \
  -e POSTGRES_DB=rag_db \
  -e POSTGRES_USER=rag_user \
  -p 5432:5432 \
  -d pgvector/pgvector:pg16

## PostgreSQL Setup

### ⚠️ pgvector Installation Error Fix

If you get the error: `ERROR: could not open extension control file "/usr/local/share/postgresql/extension/vector.control": No such file or directory`

This means pgvector extension is not installed. Here are the solutions:

### **Solution 1: Docker (Recommended - Easiest)**

Use the official pgvector Docker image that has everything pre-installed:

```bash
# Run PostgreSQL with pgvector (official image)
docker run --name postgres-pgvector \
  -e POSTGRES_PASSWORD=rag_password \
  -e POSTGRES_DB=rag_db \
  -e POSTGRES_USER=rag_user \
  -p 5432:5432 \
  -d pgvector/pgvector:pg16

# Wait for startup, then test
sleep 10
docker exec postgres-pgvector psql -U rag_user -d rag_db -c "CREATE EXTENSION IF NOT EXISTS vector;"
docker exec postgres-pgvector psql -U rag_user -d rag_db -c "SELECT vector_version();"
```

Or use the automated setup script:
```bash
chmod +x rag/scripts/setup_postgresql_docker.sh
./rag/scripts/setup_postgresql_docker.sh
```

### **Solution 2: Install pgvector on Existing PostgreSQL**

#### Ubuntu/Debian:
```bash
sudo apt update
sudo apt install postgresql-contrib
sudo apt install postgresql-14-pgvector  # Replace 14 with your PG version

# Restart PostgreSQL
sudo systemctl restart postgresql
```

#### macOS (Homebrew):
```bash
brew install pgvector
# Restart PostgreSQL service
brew services restart postgresql
```

#### macOS (PostgreSQL.app):
```bash
# Download pgvector from releases: https://github.com/pgvector/pgvector/releases
# Extract and build:
cd pgvector-*
make
make install  # May need sudo
```

#### CentOS/RHEL/Fedora:
```bash
# Install from PGDG repository
sudo yum install postgresql14-devel  # Replace 14 with your version
git clone https://github.com/pgvector/pgvector.git
cd pgvector
make
sudo make install
```

#### Windows:
```bash
# Use WSL2 or Docker (recommended)
# Or download precompiled binaries from pgvector releases
```

### **Solution 3: Build from Source**

If package managers don't work:

```bash
git clone --branch v0.5.1 https://github.com/pgvector/pgvector.git
cd pgvector
make
sudo make install  # Add to PostgreSQL

# Restart PostgreSQL
sudo systemctl restart postgresql  # Linux
brew services restart postgresql   # macOS
```

### **Verify Installation**

After installing pgvector, connect to PostgreSQL and test:

```sql
-- Connect to your database
psql -h localhost -p 5432 -U rag_user -d rag_db

-- Create extension (should work now)
CREATE EXTENSION IF NOT EXISTS vector;

-- Verify
SELECT extname, extversion FROM pg_extension WHERE extname = 'vector';

-- Test vector operations
SELECT '[1,2,3]'::vector <-> '[4,5,6]'::vector as cosine_distance;
```

### Manual Database Setup (After pgvector is installed)

```sql
-- Create database (run as postgres superuser)
CREATE DATABASE rag_db;

-- Create user
CREATE USER rag_user WITH PASSWORD 'rag_password';
GRANT ALL PRIVILEGES ON DATABASE rag_db TO rag_user;

-- Connect to database
\c rag_db

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Grant schema permissions
GRANT USAGE ON SCHEMA public TO rag_user;
GRANT CREATE ON SCHEMA public TO rag_user;
```

### Configuration

```python
postgresql_config = VectorStoreConfig(
    provider="postgresql",
    postgresql=PostgreSQLConfig(
        host="localhost",
        port=5432,
        database="rag_db",
        username="rag_user",
        password="rag_password",
        max_connections=20,
        table_name="code_chunks",
        enable_ivfflat_index=True,
        ivfflat_lists=100
    )
)
```

## Environment Configuration

Create a `.env` file for database credentials:

```bash
# PostgreSQL
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=rag_db
POSTGRES_USER=rag_user
POSTGRES_PASSWORD=rag_password

# Qdrant
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_API_KEY=your_api_key_if_needed
```

## Configuration Examples

### Development (Default Qdrant)

```python
config = VectorStoreConfig()  # Uses Qdrant by default
```

### Production PostgreSQL

```python
import os

config = VectorStoreConfig(
    provider="postgresql",
    collection_name="production_code_chunks",
    postgresql=PostgreSQLConfig(
        host=os.getenv("POSTGRES_HOST"),
        port=int(os.getenv("POSTGRES_PORT", 5432)),
        database=os.getenv("POSTGRES_DB"),
        username=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
        max_connections=50,
        enable_ivfflat_index=True
    )
)
```

### High Performance Qdrant Cloud

```python
config = VectorStoreConfig(
    provider="qdrant",
    qdrant=QdrantConfig(
        host="your-cluster.qdrant.io",
        port=6333,
        https=True,
        api_key=os.getenv("QDRANT_API_KEY")
    )
)
```

## Migration Between Backends

You can migrate data between vector stores:

```python
from rag.vector_store.vector_store_factory import get_vector_store

# Source (e.g., Qdrant)
source_config = VectorStoreConfig(provider="qdrant")
source_store = get_vector_store(source_config)

# Destination (e.g., PostgreSQL)
dest_config = VectorStoreConfig(provider="postgresql")
dest_store = get_vector_store(dest_config)

# Get all data from source
# Note: Implement pagination for large datasets
source_info = source_store.get_collection_info()
print(f"Migrating {source_info['total_points']} chunks...")

# TODO: Implement migration utility
# This would involve:
# 1. Scrolling through all chunks in source
# 2. Batch inserting into destination
# 3. Verifying data integrity
```

## Performance Considerations

### Qdrant
- **Pros**: Purpose-built for vector search, excellent performance
- **Cons**: Additional service to manage, separate from main database
- **Best for**: High-throughput vector searches, cloud deployments

### PostgreSQL
- **Pros**: Single database, familiar SQL interface, strong consistency
- **Cons**: May require more tuning for large-scale vector operations
- **Best for**: Existing PostgreSQL infrastructure, transactional requirements

### Tuning Tips

#### PostgreSQL
```sql
-- Optimize for vector operations
SET maintenance_work_mem = '256MB';
SET effective_cache_size = '1GB';
SET random_page_cost = 1.0;  -- For SSDs

-- Create optimal indexes
CREATE INDEX CONCURRENTLY code_chunks_embedding_idx 
ON code_chunks 
USING ivfflat (embedding vector_cosine_ops) 
WITH (lists = 100);
```

#### Qdrant
```python
qdrant_config = QdrantConfig(
    search_params={
        "exact": False,
        "hnsw_ef": 128,  # Higher for better recall
        "indexed_only": True
    }
)
```

## Troubleshooting

### PostgreSQL Issues

**pgvector not found**:
```bash
# Ubuntu/Debian
sudo apt install postgresql-14-pgvector

# macOS with Homebrew
brew install pgvector
```

**Connection refused**:
- Check PostgreSQL is running
- Verify host/port configuration
- Check firewall settings

### Qdrant Issues

**Service not available**:
```bash
# Check if Qdrant is running
curl http://localhost:6333/health

# Start with Docker
docker run -p 6333:6333 qdrant/qdrant:latest
```

**Collection not found**:
- Collections are created automatically
- Check collection name in configuration

## API Reference

All vector stores implement the `VectorStore` interface:

```python
class VectorStore(ABC):
    def upsert_chunks(self, chunks: List[CodeBaseChunk]) -> bool
    def search_similar(self, query_vector: List[float], **kwargs) -> List[RetrievalResult]
    def search_by_filter(self, filters: Dict[str, Any], limit: int = 10) -> List[CodeBaseChunk]
    def get_collection_info(self) -> Dict[str, Any]
    def delete_chunks(self, chunk_ids: List[str]) -> bool
    def clear_collection(self) -> bool
    def health_check(self) -> Dict[str, Any]
```

This ensures consistent behavior regardless of the backend used.

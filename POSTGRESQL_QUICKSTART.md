# PostgreSQL + pgvector Quick Start

## 🚨 Fix for pgvector Extension Error

If you get this error:
```
ERROR: could not open extension control file "/usr/local/share/postgresql/extension/vector.control": No such file or directory
```

**This means pgvector extension is not installed.** Here's how to fix it:

## ⚡ Quick Fix (Docker - Recommended)

The easiest way is to use the official pgvector Docker image:

```bash
# 1. Stop any existing PostgreSQL
docker stop postgres-pgvector 2>/dev/null || true
docker rm postgres-pgvector 2>/dev/null || true

# 2. Run PostgreSQL with pgvector pre-installed
docker run --name postgres-pgvector \
  -e POSTGRES_PASSWORD=rag_password \
  -e POSTGRES_DB=rag_db \
  -e POSTGRES_USER=rag_user \
  -p 5432:5432 \
  -d pgvector/pgvector:pg16

# 3. Wait for startup
sleep 10

# 4. Test it works
docker exec postgres-pgvector psql -U rag_user -d rag_db -c "CREATE EXTENSION IF NOT EXISTS vector; SELECT vector_version();"
```

## ✅ Verify Setup

Run the test script:
```bash
python test_postgresql_setup.py
```

You should see:
```
✅ PostgreSQL connection successful
✅ pgvector extension created/verified
✅ Vector operations work
✅ RAG vector store created successfully
```

## 🎯 Use with RAG System

```python
from rag.configs import VectorStoreConfig, PostgreSQLConfig
from rag.core.rag_system import RAGSystem

# Configure PostgreSQL
config = VectorStoreConfig(
    provider="postgresql",
    postgresql=PostgreSQLConfig(
        host="localhost",
        database="rag_db",
        username="rag_user",
        password="rag_password"
    )
)

# Create RAG system
rag = RAGSystem(vector_store_config=config)

# Index and query
result = rag.index_codebase(
    project_path="/path/to/your/project",
    file_patterns=["*.py", "*.js"],
    force_reindex=False
)
```

## 🔧 Alternative Solutions

### macOS with Homebrew
```bash
brew install pgvector
brew services restart postgresql
```

### Ubuntu/Debian
```bash
sudo apt install postgresql-14-pgvector
sudo systemctl restart postgresql
```

### Build from Source
```bash
git clone --branch v0.5.1 https://github.com/pgvector/pgvector.git
cd pgvector
make
sudo make install
sudo systemctl restart postgresql
```

## 🚀 Complete Examples

```bash
# Run all examples
python example_vector_stores.py

# Test just PostgreSQL
python -c "
from rag.configs import VectorStoreConfig, PostgreSQLConfig
from rag.vector_store.vector_store_factory import get_vector_store

config = VectorStoreConfig(
    provider='postgresql',
    postgresql=PostgreSQLConfig(
        host='localhost',
        database='rag_db', 
        username='rag_user',
        password='rag_password'
    )
)

store = get_vector_store(config)
print('Health:', store.health_check())
"
```

## 📚 More Help

- **Full documentation**: `rag/docs/VECTOR_STORE_GUIDE.md`
- **Configuration examples**: `rag/example_configs.py`
- **Setup scripts**: `rag/scripts/`

## 🐛 Troubleshooting

**Container won't start:**
```bash
docker logs postgres-pgvector
```

**Connection refused:**
```bash
docker ps | grep postgres
telnet localhost 5432
```

**Extension still not found:**
- Make sure you're using the `pgvector/pgvector:pg16` image
- Try restarting the container: `docker restart postgres-pgvector`
- Check PostgreSQL logs for errors

**Need different credentials:**
```bash
docker run --name postgres-pgvector \
  -e POSTGRES_PASSWORD=your_password \
  -e POSTGRES_DB=your_db \
  -e POSTGRES_USER=your_user \
  -p 5432:5432 \
  -d pgvector/pgvector:pg16
```

That's it! You should now have PostgreSQL with pgvector working for your RAG system. 🎉

#!/bin/bash

# PostgreSQL with pgvector Docker setup script
# This script sets up a PostgreSQL container with pgvector extension

set -e

echo "Setting up PostgreSQL with pgvector using Docker..."

# Stop and remove existing container if it exists
docker stop postgres-pgvector 2>/dev/null || true
docker rm postgres-pgvector 2>/dev/null || true

# Run PostgreSQL with pgvector
echo "Starting PostgreSQL container..."
docker run --name postgres-pgvector \
  -e POSTGRES_PASSWORD=rag_password \
  -e POSTGRES_DB=rag_db \
  -e POSTGRES_USER=rag_user \
  -p 5432:5432 \
  -d pgvector/pgvector:pg16

# Wait for PostgreSQL to start
echo "Waiting for PostgreSQL to start..."
sleep 10

# Test connection and create extension
echo "Setting up pgvector extension..."
docker exec -i postgres-pgvector psql -U rag_user -d rag_db << 'EOF'
-- Create the pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Verify extension is created
SELECT extname, extversion FROM pg_extension WHERE extname = 'vector';

-- Test vector operations
SELECT '[1,2,3]'::vector <-> '[4,5,6]'::vector as cosine_distance;

-- Show vector version
SELECT vector_version();

EOF

echo "✅ PostgreSQL with pgvector is ready!"
echo ""
echo "Connection details:"
echo "  Host: localhost"
echo "  Port: 5432"
echo "  Database: rag_db"
echo "  Username: rag_user"
echo "  Password: rag_password"
echo ""
echo "To connect manually:"
echo "  psql -h localhost -p 5432 -U rag_user -d rag_db"
echo ""
echo "To stop the container:"
echo "  docker stop postgres-pgvector"
echo ""
echo "To start the container again:"
echo "  docker start postgres-pgvector"

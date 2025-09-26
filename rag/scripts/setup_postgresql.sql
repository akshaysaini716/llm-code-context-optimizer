-- PostgreSQL setup script for RAG system
-- Run this script to set up PostgreSQL with pgvector for the RAG system

-- Create database (run this as postgres user)
-- CREATE DATABASE rag_db;

-- Connect to the database
\c rag_db;

-- Enable the pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create user for RAG system
CREATE USER rag_user WITH PASSWORD 'rag_password';

-- Grant necessary permissions
GRANT CONNECT ON DATABASE rag_db TO rag_user;
GRANT USAGE ON SCHEMA public TO rag_user;
GRANT CREATE ON SCHEMA public TO rag_user;

-- Create the code_chunks table (this will be done automatically by the app)
-- But you can create it manually if needed:

/*
CREATE TABLE code_chunks (
    id VARCHAR(255) PRIMARY KEY,
    file_path TEXT NOT NULL,
    project_path TEXT NOT NULL DEFAULT '',
    content TEXT NOT NULL,
    language VARCHAR(100) NOT NULL,
    chunk_type VARCHAR(100) NOT NULL,
    start_byte INTEGER NOT NULL,
    end_byte INTEGER NOT NULL,
    start_line INTEGER NOT NULL,
    end_line INTEGER NOT NULL,
    embedding vector(768),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Grant permissions on the table
GRANT ALL PRIVILEGES ON TABLE code_chunks TO rag_user;

-- Create indexes for better performance
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_code_chunks_file_path ON code_chunks (file_path);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_code_chunks_project_path ON code_chunks (project_path);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_code_chunks_language ON code_chunks (language);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_code_chunks_chunk_type ON code_chunks (chunk_type);

-- Create vector index (IVFFlat) for similarity search
-- Note: This requires data in the table to work effectively
-- The application will create this automatically when needed
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_code_chunks_embedding_ivfflat 
ON code_chunks 
USING ivfflat (embedding vector_cosine_ops) 
WITH (lists = 100);
*/

-- Optimize PostgreSQL settings for vector operations
-- Add these to postgresql.conf or set them per session:

-- For better vector performance
-- shared_preload_libraries = 'pg_stat_statements,vector'
-- maintenance_work_mem = 256MB
-- effective_cache_size = 1GB
-- random_page_cost = 1.0  -- For SSDs

-- Show current settings
SELECT name, setting, unit, context 
FROM pg_settings 
WHERE name IN (
    'maintenance_work_mem',
    'effective_cache_size', 
    'random_page_cost',
    'max_connections'
);

-- Show pgvector version
SELECT vector_version();

-- Test vector operations
SELECT 
    '[1,2,3]'::vector <-> '[4,5,6]'::vector as cosine_distance,
    '[1,2,3]'::vector <+> '[4,5,6]'::vector as inner_product,
    '[1,2,3]'::vector <#> '[4,5,6]'::vector as negative_inner_product;

ECHO 'PostgreSQL setup completed successfully!';
ECHO 'You can now use PostgreSQL as your vector store backend.';

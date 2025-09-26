"""
PostgreSQL vector store implementation using pgvector
"""

import logging
import uuid
import json
from typing import List, Dict, Any, Optional
import psycopg2
from psycopg2.extras import RealDictCursor, execute_values
from psycopg2 import pool
import numpy as np

from rag.models import CodeBaseChunk, RetrievalResult
from rag.vector_store.base_vector_store import VectorStore
from rag.configs import PostgreSQLConfig

logger = logging.getLogger(__name__)


class PostgreSQLVectorStore(VectorStore):
    """PostgreSQL vector store implementation using pgvector extension"""
    
    def __init__(self, config: PostgreSQLConfig):
        self.config = config
        self.table_name = config.table_name
        self.embedding_dimension = 768  # Will be configured later
        
        # Create connection pool
        self.connection_pool = psycopg2.pool.ThreadedConnectionPool(
            minconn=1,
            maxconn=config.max_connections,
            host=config.host,
            port=config.port,
            database=config.database,
            user=config.username,
            password=config.password,
            connect_timeout=config.connection_timeout
        )
        
        # Initialize database schema
        self._initialize_database()
    
    def _get_connection(self):
        """Get a connection from the pool"""
        return self.connection_pool.getconn()
    
    def _return_connection(self, conn):
        """Return a connection to the pool"""
        self.connection_pool.putconn(conn)
    
    def _initialize_database(self):
        """Initialize database schema with pgvector support"""
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Enable pgvector extension
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
            
            # Create table if not exists
            create_table_sql = f"""
            CREATE TABLE IF NOT EXISTS {self.table_name} (
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
                embedding vector({self.embedding_dimension}),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
            cursor.execute(create_table_sql)
            
            # Create indexes for better performance
            indexes = [
                f"CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_{self.table_name}_file_path ON {self.table_name} (file_path)",
                f"CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_{self.table_name}_project_path ON {self.table_name} (project_path)",
                f"CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_{self.table_name}_language ON {self.table_name} (language)",
                f"CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_{self.table_name}_chunk_type ON {self.table_name} (chunk_type)"
            ]
            
            for index_sql in indexes:
                try:
                    cursor.execute(index_sql)
                except Exception as e:
                    # Index might already exist, that's okay
                    logger.debug(f"Index creation skipped: {e}")
            
            # Create vector index if enabled
            if self.config.enable_ivfflat_index:
                try:
                    vector_index_sql = f"""
                    CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_{self.table_name}_embedding_ivfflat 
                    ON {self.table_name} 
                    USING ivfflat (embedding vector_cosine_ops) 
                    WITH (lists = {self.config.ivfflat_lists})
                    """
                    cursor.execute(vector_index_sql)
                except Exception as e:
                    logger.warning(f"Vector index creation failed: {e}")
            
            conn.commit()
            logger.info(f"Database initialized successfully with table: {self.table_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            if conn:
                conn.rollback()
            raise
        finally:
            if conn:
                self._return_connection(conn)
    
    def upsert_chunks(self, chunks: List[CodeBaseChunk]) -> bool:
        """Insert or update chunks in PostgreSQL"""
        if not chunks:
            return False
        
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Prepare data for batch upsert with proper deduplication
            chunk_data = []
            seen_ids = {}  # Use dict to track chunk_id -> chunk mapping
            for chunk in chunks:
                if chunk.embedding:
                    chunk_id = chunk.id or str(uuid.uuid4())
                    
                    # Handle duplicate IDs within the same batch by deduplication
                    if chunk_id in seen_ids:
                        logger.debug(f"Duplicate chunk ID detected: {chunk_id}, keeping latest version")
                        # Replace with the newer chunk (assuming later chunks are more recent)
                        seen_ids[chunk_id] = chunk
                        continue
                    
                    seen_ids[chunk_id] = chunk
            
            # Build chunk_data from deduplicated chunks
            for chunk_id, chunk in seen_ids.items():
                chunk_data.append((
                    chunk_id,
                    chunk.file_path,
                    chunk.project_path or '',
                    chunk.content,
                    chunk.language,
                    chunk.chunk_type,
                    chunk.start_byte,
                    chunk.end_byte,
                    chunk.start_line,
                    chunk.end_line,
                    chunk.embedding
                ))
            
            if not chunk_data:
                logger.warning("No chunks with embeddings to upsert")
                return False
            
            # Use UPSERT (INSERT ... ON CONFLICT)
            upsert_sql = f"""
            INSERT INTO {self.table_name} 
            (id, file_path, project_path, content, language, chunk_type, 
             start_byte, end_byte, start_line, end_line, embedding)
            VALUES %s
            ON CONFLICT (id) DO UPDATE SET
                file_path = EXCLUDED.file_path,
                project_path = EXCLUDED.project_path,
                content = EXCLUDED.content,
                language = EXCLUDED.language,
                chunk_type = EXCLUDED.chunk_type,
                start_byte = EXCLUDED.start_byte,
                end_byte = EXCLUDED.end_byte,
                start_line = EXCLUDED.start_line,
                end_line = EXCLUDED.end_line,
                embedding = EXCLUDED.embedding,
                updated_at = CURRENT_TIMESTAMP
            """
            
            # Process chunks in batches to avoid overwhelming PostgreSQL
            batch_size = 20  # Process 20 chunks at a time
            total_processed = 0
            
            for i in range(0, len(chunk_data), batch_size):
                batch = chunk_data[i:i + batch_size]
                execute_values(cursor, upsert_sql, batch, template=None)
                total_processed += len(batch)
                logger.info(f"Processed batch {i//batch_size + 1}: {total_processed}/{len(chunk_data)} chunks")
            
            conn.commit()
            
            logger.info(f"Successfully upserted {total_processed} chunks in batches")
            return True
            
        except Exception as e:
            logger.error(f"Failed to upsert chunks: {e}")
            if conn:
                conn.rollback()
            return False
        finally:
            if conn:
                self._return_connection(conn)
    
    def search_similar(
        self, 
        query_vector: List[float], 
        project_path: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None, 
        top_k: int = 10
    ) -> List[RetrievalResult]:
        """Search for similar vectors using cosine distance"""
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Convert query vector to PostgreSQL vector format
            vector_str = '[' + ','.join(map(str, query_vector)) + ']'
            
            # Build the WHERE clause - use simple string formatting to avoid parameter issues
            where_conditions = []
            
            if project_path:
                # Escape single quotes in project_path for SQL safety
                escaped_path = project_path.replace("'", "''")
                where_conditions.append(f"project_path = '{escaped_path}'")
            
            if filters:
                for key, value in filters.items():
                    if hasattr(CodeBaseChunk, key):  # Only allow valid chunk attributes
                        if isinstance(value, dict):
                            # Handle range queries
                            if "$gte" in value:
                                where_conditions.append(f"{key} >= {value['$gte']}")
                            if "$lte" in value:
                                where_conditions.append(f"{key} <= {value['$lte']}")
                            if "$gt" in value:
                                where_conditions.append(f"{key} > {value['$gt']}")
                            if "$lt" in value:
                                where_conditions.append(f"{key} < {value['$lt']}")
                        else:
                            # Simple equality - handle strings vs numbers
                            if isinstance(value, str):
                                escaped_value = value.replace("'", "''")
                                where_conditions.append(f"{key} = '{escaped_value}'")
                            else:
                                where_conditions.append(f"{key} = {value}")
            
            where_clause = ""
            if where_conditions:
                where_clause = "WHERE " + " AND ".join(where_conditions)
            
            # Use cosine distance for similarity search - avoid parameterizing LIMIT to prevent issues
            search_sql = f"""
            SELECT *,
                   1 - (embedding <=> '{vector_str}'::vector) as similarity_score
            FROM {self.table_name}
            {where_clause}
            ORDER BY embedding <=> '{vector_str}'::vector
            LIMIT {top_k}
            """
            
            cursor.execute(search_sql)
            rows = cursor.fetchall()
            
            results = []
            for row in rows:
                chunk = CodeBaseChunk(
                    id=row['id'],
                    file_path=row['file_path'],
                    project_path=row['project_path'],
                    content=row['content'],
                    language=row['language'],
                    chunk_type=row['chunk_type'],
                    start_byte=row['start_byte'],
                    end_byte=row['end_byte'],
                    start_line=row['start_line'],
                    end_line=row['end_line']
                )
                
                result = RetrievalResult(
                    chunk=chunk,
                    relevance_score=float(row['similarity_score'])
                )
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to search similar vectors: {e}")
            return []
        finally:
            if conn:
                self._return_connection(conn)
    
    def search_by_filter(self, filters: Dict[str, Any], limit: int = 10) -> List[CodeBaseChunk]:
        """Search chunks by filter conditions only"""
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Build WHERE clause from filters
            where_conditions = []
            params = []
            param_index = 0
            
            for key, value in filters.items():
                if hasattr(CodeBaseChunk, key):  # Only allow valid chunk attributes
                    if isinstance(value, dict):
                        # Handle range queries
                        if "$gte" in value:
                            where_conditions.append(f"{key} >= ${param_index + 1}")
                            params.append(value["$gte"])
                            param_index += 1
                        if "$lte" in value:
                            where_conditions.append(f"{key} <= ${param_index + 1}")
                            params.append(value["$lte"])
                            param_index += 1
                        if "$gt" in value:
                            where_conditions.append(f"{key} > ${param_index + 1}")
                            params.append(value["$gt"])
                            param_index += 1
                        if "$lt" in value:
                            where_conditions.append(f"{key} < ${param_index + 1}")
                            params.append(value["$lt"])
                            param_index += 1
                    else:
                        # Simple equality
                        where_conditions.append(f"{key} = ${param_index + 1}")
                        params.append(value)
                        param_index += 1
            
            if not where_conditions:
                return []
            
            params.append(limit)
            where_clause = "WHERE " + " AND ".join(where_conditions)
            
            filter_sql = f"""
            SELECT * FROM {self.table_name}
            {where_clause}
            LIMIT ${param_index + 1}
            """
            
            cursor.execute(filter_sql, params)
            rows = cursor.fetchall()
            
            chunks = []
            for row in rows:
                chunk = CodeBaseChunk(
                    id=row['id'],
                    file_path=row['file_path'],
                    project_path=row['project_path'],
                    content=row['content'],
                    language=row['language'],
                    chunk_type=row['chunk_type'],
                    start_byte=row['start_byte'],
                    end_byte=row['end_byte'],
                    start_line=row['start_line'],
                    end_line=row['end_line']
                )
                chunks.append(chunk)
            
            return chunks
            
        except Exception as e:
            logger.error(f"Failed to search by filter: {e}")
            return []
        finally:
            if conn:
                self._return_connection(conn)
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the PostgreSQL table"""
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Get table statistics
            cursor.execute(f"SELECT COUNT(*) FROM {self.table_name}")
            total_rows = cursor.fetchone()[0]
            
            # Get table size
            cursor.execute(f"""
                SELECT pg_size_pretty(pg_total_relation_size('{self.table_name}')) as table_size,
                       pg_size_pretty(pg_relation_size('{self.table_name}')) as data_size
            """)
            size_info = cursor.fetchone()
            
            return {
                "total_points": total_rows,
                "embedding_dimension": self.embedding_dimension,
                "table_size": size_info[0] if size_info else "unknown",
                "data_size": size_info[1] if size_info else "unknown",
                "provider": "postgresql"
            }
            
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return {}
        finally:
            if conn:
                self._return_connection(conn)
    
    def delete_chunks(self, chunk_ids: List[str]) -> bool:
        """Delete chunks by their IDs"""
        if not chunk_ids:
            return True
        
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Use ANY to delete multiple IDs efficiently
            delete_sql = f"DELETE FROM {self.table_name} WHERE id = ANY(%s)"
            cursor.execute(delete_sql, (chunk_ids,))
            
            deleted_count = cursor.rowcount
            conn.commit()
            
            logger.info(f"Deleted {deleted_count} chunks")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete chunks: {e}")
            if conn:
                conn.rollback()
            return False
        finally:
            if conn:
                self._return_connection(conn)
    
    def delete_chunks_by_project_path(self, project_path: str) -> bool:
        """Delete all chunks for a specific project path"""
        if not project_path:
            return True
        
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            delete_sql = f"DELETE FROM {self.table_name} WHERE project_path = %s"
            cursor.execute(delete_sql, (project_path,))
            
            deleted_count = cursor.rowcount
            conn.commit()
            
            logger.info(f"Deleted {deleted_count} chunks for project path: {project_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete chunks by project path: {e}")
            if conn:
                conn.rollback()
            return False
        finally:
            if conn:
                self._return_connection(conn)
    
    def clear_collection(self) -> bool:
        """Clear all data from the table"""
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute(f"TRUNCATE TABLE {self.table_name}")
            conn.commit()
            
            logger.info(f"Cleared all data from {self.table_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear collection: {e}")
            if conn:
                conn.rollback()
            return False
        finally:
            if conn:
                self._return_connection(conn)
    
    def health_check(self) -> Dict[str, Any]:
        """Check the health status of PostgreSQL"""
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Test basic connection and table existence
            cursor.execute(f"SELECT 1 FROM {self.table_name} LIMIT 1")
            
            # Get basic info
            info = self.get_collection_info()
            
            return {
                "status": "healthy",
                "provider": "postgresql",
                "table_exists": True,
                "total_points": info.get("total_points", 0),
                "connection_pool_size": self.connection_pool.minconn
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "provider": "postgresql",
                "error": str(e)
            }
        finally:
            if conn:
                self._return_connection(conn)
    
    def close(self):
        """Close all connections in the pool"""
        if hasattr(self, 'connection_pool'):
            self.connection_pool.closeall()
            logger.info("PostgreSQL connection pool closed")

"""
Database Manager
Handles database connections, queries, and data operations
"""

import sqlite3
import json
import logging
from typing import Any, Dict, List, Optional, Union
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from datetime import datetime
import threading

logger = logging.getLogger(__name__)

@dataclass
class QueryResult:
    """Result of a database query"""
    success: bool
    data: Optional[List[Dict]] = None
    affected_rows: int = 0
    error_message: Optional[str] = None
    execution_time: float = 0.0

class ConnectionPool:
    """Thread-safe database connection pool"""
    
    def __init__(self, database_path: str, max_connections: int = 10):
        self.database_path = database_path
        self.max_connections = max_connections
        self.connections = []
        self.lock = threading.Lock()
        self._initialize_pool()
    
    def _initialize_pool(self):
        """Initialize the connection pool"""
        for _ in range(self.max_connections):
            conn = sqlite3.connect(self.database_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row  # Enable dict-like access
            self.connections.append(conn)
    
    @contextmanager
    def get_connection(self):
        """Get a connection from the pool"""
        with self.lock:
            if not self.connections:
                # Create temporary connection if pool is empty
                conn = sqlite3.connect(self.database_path, check_same_thread=False)
                conn.row_factory = sqlite3.Row
            else:
                conn = self.connections.pop()
        
        try:
            yield conn
        finally:
            with self.lock:
                if len(self.connections) < self.max_connections:
                    self.connections.append(conn)
                else:
                    conn.close()
    
    def close_all(self):
        """Close all connections in the pool"""
        with self.lock:
            for conn in self.connections:
                conn.close()
            self.connections.clear()

class QueryBuilder:
    """SQL query builder with method chaining"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset the query builder"""
        self._select_fields = []
        self._from_table = ""
        self._joins = []
        self._where_conditions = []
        self._group_by = []
        self._having = []
        self._order_by = []
        self._limit_count = None
        self._offset_count = None
        self._parameters = {}
        return self
    
    def select(self, *fields):
        """Add SELECT fields"""
        self._select_fields.extend(fields)
        return self
    
    def from_table(self, table_name):
        """Set FROM table"""
        self._from_table = table_name
        return self
    
    def join(self, table, on_condition, join_type="INNER"):
        """Add JOIN clause"""
        self._joins.append(f"{join_type} JOIN {table} ON {on_condition}")
        return self
    
    def where(self, condition, **params):
        """Add WHERE condition"""
        self._where_conditions.append(condition)
        self._parameters.update(params)
        return self
    
    def group_by(self, *fields):
        """Add GROUP BY fields"""
        self._group_by.extend(fields)
        return self
    
    def having(self, condition):
        """Add HAVING condition"""
        self._having.append(condition)
        return self
    
    def order_by(self, field, direction="ASC"):
        """Add ORDER BY field"""
        self._order_by.append(f"{field} {direction}")
        return self
    
    def limit(self, count):
        """Set LIMIT"""
        self._limit_count = count
        return self
    
    def offset(self, count):
        """Set OFFSET"""
        self._offset_count = count
        return self
    
    def build(self):
        """Build the final SQL query"""
        if not self._from_table:
            raise ValueError("FROM table is required")
        
        # Build SELECT clause
        if self._select_fields:
            select_clause = f"SELECT {', '.join(self._select_fields)}"
        else:
            select_clause = "SELECT *"
        
        # Build query parts
        query_parts = [
            select_clause,
            f"FROM {self._from_table}"
        ]
        
        # Add JOINs
        if self._joins:
            query_parts.extend(self._joins)
        
        # Add WHERE
        if self._where_conditions:
            where_clause = "WHERE " + " AND ".join(self._where_conditions)
            query_parts.append(where_clause)
        
        # Add GROUP BY
        if self._group_by:
            query_parts.append(f"GROUP BY {', '.join(self._group_by)}")
        
        # Add HAVING
        if self._having:
            query_parts.append(f"HAVING {' AND '.join(self._having)}")
        
        # Add ORDER BY
        if self._order_by:
            query_parts.append(f"ORDER BY {', '.join(self._order_by)}")
        
        # Add LIMIT and OFFSET
        if self._limit_count is not None:
            query_parts.append(f"LIMIT {self._limit_count}")
        
        if self._offset_count is not None:
            query_parts.append(f"OFFSET {self._offset_count}")
        
        return " ".join(query_parts), self._parameters

class DatabaseManager:
    """Main database manager class"""
    
    def __init__(self, database_path: str = "app.db", pool_size: int = 10):
        self.database_path = database_path
        self.connection_pool = ConnectionPool(database_path, pool_size)
        self.query_builder = QueryBuilder()
        self._setup_database()
    
    def _setup_database(self):
        """Setup initial database schema"""
        with self.connection_pool.get_connection() as conn:
            # Enable foreign keys
            conn.execute("PRAGMA foreign_keys = ON")
            conn.commit()
    
    def execute_query(self, query: str, parameters: Dict = None) -> QueryResult:
        """
        Execute a SQL query
        
        Args:
            query: SQL query string
            parameters: Query parameters
            
        Returns:
            QueryResult object
        """
        start_time = datetime.now()
        parameters = parameters or {}
        
        try:
            with self.connection_pool.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, parameters)
                
                # Check if it's a SELECT query
                if query.strip().upper().startswith('SELECT'):
                    data = [dict(row) for row in cursor.fetchall()]
                    affected_rows = len(data)
                else:
                    data = None
                    affected_rows = cursor.rowcount
                    conn.commit()
                
                execution_time = (datetime.now() - start_time).total_seconds()
                
                return QueryResult(
                    success=True,
                    data=data,
                    affected_rows=affected_rows,
                    execution_time=execution_time
                )
        
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Database query failed: {e}")
            
            return QueryResult(
                success=False,
                error_message=str(e),
                execution_time=execution_time
            )
    
    def create_table(self, table_name: str, columns: Dict[str, str], 
                    constraints: List[str] = None) -> QueryResult:
        """
        Create a database table
        
        Args:
            table_name: Name of the table
            columns: Dict of column_name -> column_definition
            constraints: List of table constraints
            
        Returns:
            QueryResult object
        """
        column_definitions = [f"{name} {definition}" for name, definition in columns.items()]
        
        if constraints:
            column_definitions.extend(constraints)
        
        query = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            {', '.join(column_definitions)}
        )
        """
        
        return self.execute_query(query)
    
    def insert(self, table_name: str, data: Dict[str, Any]) -> QueryResult:
        """
        Insert data into a table
        
        Args:
            table_name: Name of the table
            data: Dictionary of column -> value
            
        Returns:
            QueryResult object
        """
        columns = list(data.keys())
        placeholders = [f":{col}" for col in columns]
        
        query = f"""
        INSERT INTO {table_name} ({', '.join(columns)})
        VALUES ({', '.join(placeholders)})
        """
        
        return self.execute_query(query, data)
    
    def insert_many(self, table_name: str, data_list: List[Dict[str, Any]]) -> QueryResult:
        """
        Insert multiple records
        
        Args:
            table_name: Name of the table
            data_list: List of dictionaries to insert
            
        Returns:
            QueryResult object
        """
        if not data_list:
            return QueryResult(success=True, affected_rows=0)
        
        columns = list(data_list[0].keys())
        placeholders = [f":{col}" for col in columns]
        
        query = f"""
        INSERT INTO {table_name} ({', '.join(columns)})
        VALUES ({', '.join(placeholders)})
        """
        
        try:
            with self.connection_pool.get_connection() as conn:
                cursor = conn.cursor()
                cursor.executemany(query, data_list)
                conn.commit()
                
                return QueryResult(
                    success=True,
                    affected_rows=cursor.rowcount
                )
        
        except Exception as e:
            logger.error(f"Batch insert failed: {e}")
            return QueryResult(
                success=False,
                error_message=str(e)
            )
    
    def update(self, table_name: str, data: Dict[str, Any], 
              where_condition: str, where_params: Dict = None) -> QueryResult:
        """
        Update records in a table
        
        Args:
            table_name: Name of the table
            data: Dictionary of column -> new_value
            where_condition: WHERE clause condition
            where_params: Parameters for WHERE clause
            
        Returns:
            QueryResult object
        """
        set_clauses = [f"{col} = :{col}" for col in data.keys()]
        
        query = f"""
        UPDATE {table_name}
        SET {', '.join(set_clauses)}
        WHERE {where_condition}
        """
        
        # Combine data and where parameters
        all_params = data.copy()
        if where_params:
            all_params.update(where_params)
        
        return self.execute_query(query, all_params)
    
    def delete(self, table_name: str, where_condition: str, 
              where_params: Dict = None) -> QueryResult:
        """
        Delete records from a table
        
        Args:
            table_name: Name of the table
            where_condition: WHERE clause condition
            where_params: Parameters for WHERE clause
            
        Returns:
            QueryResult object
        """
        query = f"DELETE FROM {table_name} WHERE {where_condition}"
        return self.execute_query(query, where_params or {})
    
    def select(self, table_name: str = None) -> QueryBuilder:
        """
        Start building a SELECT query
        
        Args:
            table_name: Optional table name to start with
            
        Returns:
            QueryBuilder instance for method chaining
        """
        self.query_builder.reset()
        if table_name:
            self.query_builder.from_table(table_name)
        return self.query_builder
    
    def execute_builder(self, builder: QueryBuilder = None) -> QueryResult:
        """
        Execute a query from QueryBuilder
        
        Args:
            builder: QueryBuilder instance (uses self.query_builder if None)
            
        Returns:
            QueryResult object
        """
        if builder is None:
            builder = self.query_builder
        
        query, parameters = builder.build()
        return self.execute_query(query, parameters)
    
    def get_table_info(self, table_name: str) -> QueryResult:
        """
        Get information about a table structure
        
        Args:
            table_name: Name of the table
            
        Returns:
            QueryResult with table schema information
        """
        query = f"PRAGMA table_info({table_name})"
        return self.execute_query(query)
    
    def backup_table(self, table_name: str, backup_file: str) -> bool:
        """
        Backup a table to JSON file
        
        Args:
            table_name: Name of the table to backup
            backup_file: Path to backup file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            result = self.execute_query(f"SELECT * FROM {table_name}")
            if result.success and result.data:
                with open(backup_file, 'w') as f:
                    json.dump(result.data, f, indent=2, default=str)
                return True
            return False
        
        except Exception as e:
            logger.error(f"Backup failed: {e}")
            return False
    
    def restore_table(self, table_name: str, backup_file: str) -> bool:
        """
        Restore a table from JSON backup
        
        Args:
            table_name: Name of the table to restore
            backup_file: Path to backup file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(backup_file, 'r') as f:
                data = json.load(f)
            
            if data:
                result = self.insert_many(table_name, data)
                return result.success
            return True
        
        except Exception as e:
            logger.error(f"Restore failed: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get database statistics
        
        Returns:
            Dictionary with database statistics
        """
        stats = {}
        
        # Get all tables
        tables_result = self.execute_query(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        
        if tables_result.success and tables_result.data:
            stats['tables'] = {}
            
            for table_row in tables_result.data:
                table_name = table_row['name']
                
                # Get row count
                count_result = self.execute_query(f"SELECT COUNT(*) as count FROM {table_name}")
                row_count = count_result.data[0]['count'] if count_result.success else 0
                
                stats['tables'][table_name] = {
                    'row_count': row_count
                }
        
        return stats
    
    def close(self):
        """Close all database connections"""
        self.connection_pool.close_all()

class ModelManager:
    """ORM-like model manager for database operations"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
    
    def create_model_table(self, model_class):
        """
        Create table for a dataclass model
        
        Args:
            model_class: Dataclass representing the model
        """
        # This is a simplified ORM - in practice, you'd use proper type mapping
        table_name = model_class.__name__.lower()
        
        # Basic column mapping
        type_mapping = {
            str: "TEXT",
            int: "INTEGER",
            float: "REAL",
            bool: "BOOLEAN",
            datetime: "TIMESTAMP"
        }
        
        columns = {}
        for field_name, field_type in model_class.__annotations__.items():
            sql_type = type_mapping.get(field_type, "TEXT")
            columns[field_name] = sql_type
        
        return self.db.create_table(table_name, columns)
    
    def save_model(self, model_instance):
        """
        Save a model instance to database
        
        Args:
            model_instance: Instance of a dataclass model
        """
        table_name = model_instance.__class__.__name__.lower()
        data = asdict(model_instance)
        
        return self.db.insert(table_name, data)
    
    def find_by_id(self, model_class, record_id):
        """
        Find a model by ID
        
        Args:
            model_class: Model class
            record_id: ID to search for
        """
        table_name = model_class.__name__.lower()
        result = self.db.execute_query(
            f"SELECT * FROM {table_name} WHERE id = :id",
            {'id': record_id}
        )
        
        if result.success and result.data:
            return model_class(**result.data[0])
        return None

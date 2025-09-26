"""
Configuration file for RAG system with enhanced chunking and retrieval settings
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

@dataclass
class ChunkingConfig:
    """Configuration for enhanced chunking behavior"""
    # Size limits
    max_chunk_size: int = 1500  # Maximum tokens per chunk
    min_chunk_size: int = 100   # Minimum tokens per chunk
    max_class_chunk_size: int = 3000  # Larger limit for complete classes
    
    # Overlapping configuration
    enable_overlapping: bool = True
    overlap_ratio: float = 0.2  # 20% overlap between chunks
    
    # Context preservation
    preserve_class_methods: bool = True  # Keep class methods together
    include_context_lines: int = 3  # Lines of context before/after chunks
    hierarchical_chunking: bool = True  # Create parent-child relationships
    
    # Import handling
    smart_imports: bool = True  # Only include relevant imports
    always_include_imports: List[str] = field(default_factory=lambda: [
        "typing", "dataclasses", "abc", "enum"  # Common stdlib imports
    ])
    
    # Sliding window
    sliding_window_size: int = 200  # Tokens for sliding window expansion
    
    # Language-specific settings
    language_configs: Dict[str, dict] = field(default_factory=lambda: {
        "python": {
            "preserve_decorators": True,
            "include_docstrings": True,
            "method_grouping": True
        },
        "javascript": {
            "preserve_exports": True,
            "include_jsdoc": True,
            "arrow_function_handling": True
        },
        "typescript": {
            "preserve_interfaces": True,
            "preserve_types": True,
            "include_jsdoc": True
        },
        "java": {
            "preserve_annotations": True,
            "include_javadoc": True,
            "interface_handling": True
        }
    })

@dataclass
class RetrievalConfig:
    """Configuration for enhanced retrieval"""
    # Basic retrieval
    default_top_k: int = 10
    search_multiplier: float = 2.0  # Retrieve top_k * multiplier for reranking
    
    # Context expansion
    enable_context_expansion: bool = True
    expansion_window: int = 300  # Tokens to expand on each side
    expand_lines_before: int = 10
    expand_lines_after: int = 10
    
    # Relationship-aware retrieval
    include_related_chunks: bool = True
    relationship_score_adjustments: Dict[str, float] = field(default_factory=lambda: {
        "parent": 0.9,      # Parent class relevance multiplier
        "child": 0.85,      # Child method relevance multiplier
        "sibling": 0.7,     # Sibling chunk relevance multiplier
        "overlap": 0.8      # Overlapping chunk relevance multiplier
    })
    
    # Reranking
    enable_reranking: bool = True
    rerank_factors: Dict[str, float] = field(default_factory=lambda: {
        "class_boost": 1.2,         # Boost for complete classes
        "function_boost": 1.15,     # Boost for functions/methods
        "overlap_penalty": 0.9,     # Penalty for overlap chunks
        "exact_match_boost": 0.1    # Per-term exact match boost
    })
    
    # Caching
    enable_file_cache: bool = True
    cache_size_mb: int = 100

@dataclass
class FusionConfig:
    """Configuration for context fusion"""
    # Structure preservation
    preserve_structure: bool = True
    group_by_file: bool = True
    merge_adjacent: bool = True
    merge_distance: int = 5  # Lines between chunks to consider adjacent
    
    # Content handling
    deduplicate_content: bool = True
    smart_truncation: bool = True
    include_file_context: bool = True
    
    # Token allocation
    min_tokens_per_group: int = 200
    formatting_reserve: int = 100  # Tokens reserved for formatting
    
    # Display settings
    show_line_numbers: bool = True
    show_metadata: bool = True
    use_separators: bool = True
    separator_style: str = "=" * 50

@dataclass
class QdrantConfig:
    """Configuration specific to Qdrant"""
    host: str = "localhost"
    port: int = 6333
    https: bool = False
    api_key: Optional[str] = None
    
    # Collection settings
    index_threshold: int = 10000
    on_disk_payload: bool = True
    
    # Search settings
    search_params: Dict[str, any] = field(default_factory=lambda: {
        "exact": False,
        "hnsw_ef": 128,
        "indexed_only": True
    })

@dataclass
class PostgreSQLConfig:
    """Configuration specific to PostgreSQL with pgvector"""
    host: str = "localhost"
    port: int = 5432
    database: str = "rag_db"
    username: str = "rag_user"
    password: str = "rag_password"
    
    # Connection settings
    max_connections: int = 20
    connection_timeout: int = 30
    
    # Table settings
    table_name: str = "code_chunks"
    enable_ivfflat_index: bool = True
    ivfflat_lists: int = 100
    
    # Performance settings
    maintenance_work_mem: str = "256MB"
    effective_cache_size: str = "1GB"

@dataclass
class VectorStoreConfig:
    """Configuration for vector store"""
    # Database type selection
    provider: str = "postgresql"  # "qdrant" or "postgresql"
    
    # Common settings
    collection_name: str = "code_chunks"
    embedding_dimension: int = 768  # For all-mpnet-base-v2
    distance_metric: str = "cosine"  # cosine, euclidean, dot_product
    
    # Provider-specific configs
    qdrant: QdrantConfig = field(default_factory=QdrantConfig)
    postgresql: PostgreSQLConfig = field(default_factory=PostgreSQLConfig)

@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation"""
    # Model settings
    code_model: str = "all-mpnet-base-v2"
    text_model: str = "all-mpnet-base-v2"
    
    # Processing settings
    batch_size: int = 32
    show_progress: bool = True
    normalize_embeddings: bool = True
    
    # Context enhancement
    include_file_context: bool = True
    include_type_info: bool = True
    include_symbols: bool = False  # Set to True when symbol extraction is implemented

@dataclass
class RAGConfig:
    """Main configuration combining all components"""
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    
    # Global settings
    max_workers: int = 4  # For parallel processing
    enable_metrics: bool = True
    log_level: str = "INFO"
    
    # File handling
    max_file_size_mb: int = 10
    supported_extensions: List[str] = field(default_factory=lambda: [
        ".py", ".js", ".jsx", ".ts", ".tsx", 
        ".java", ".kt", ".kts", ".go",
        ".rs", ".cpp", ".c", ".h", ".hpp"
    ])
    
    # Ignore patterns
    ignore_patterns: List[str] = field(default_factory=lambda: [
        "__pycache__", ".git", "node_modules", 
        ".venv", "venv", "dist", "build",
        "*.pyc", "*.pyo", ".DS_Store"
    ])

def load_config(config_path: Optional[str] = None) -> RAGConfig:
    """
    Load configuration from file or use defaults
    
    Args:
        config_path: Path to configuration file (JSON or YAML)
    
    Returns:
        RAGConfig object
    """
    if config_path:
        import json
        import yaml
        from pathlib import Path
        
        path = Path(config_path)
        if path.suffix == '.json':
            with open(path, 'r') as f:
                config_dict = json.load(f)
        elif path.suffix in ['.yml', '.yaml']:
            with open(path, 'r') as f:
                config_dict = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported config file format: {path.suffix}")
        
        # Create nested configs
        config = RAGConfig()
        if 'chunking' in config_dict:
            config.chunking = ChunkingConfig(**config_dict['chunking'])
        if 'retrieval' in config_dict:
            config.retrieval = RetrievalConfig(**config_dict['retrieval'])
        if 'fusion' in config_dict:
            config.fusion = FusionConfig(**config_dict['fusion'])
        if 'vector_store' in config_dict:
            config.vector_store = VectorStoreConfig(**config_dict['vector_store'])
        if 'embedding' in config_dict:
            config.embedding = EmbeddingConfig(**config_dict['embedding'])
        
        return config
    else:
        return RAGConfig()

def save_config(config: RAGConfig, config_path: str, format: str = 'json'):
    """
    Save configuration to file
    
    Args:
        config: RAGConfig object to save
        config_path: Path to save configuration
        format: Format to save ('json' or 'yaml')
    """
    import json
    import yaml
    from dataclasses import asdict
    
    config_dict = asdict(config)
    
    if format == 'json':
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    elif format in ['yml', 'yaml']:
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
    else:
        raise ValueError(f"Unsupported format: {format}")

# Create default configuration instance
default_config = RAGConfig()

# Example configurations for different use cases
CONFIGS = {
    "performance": RAGConfig(
        chunking=ChunkingConfig(
            max_chunk_size=2000,
            preserve_class_methods=True,
            hierarchical_chunking=True,
            overlap_ratio=0.25
        ),
        retrieval=RetrievalConfig(
            default_top_k=15,
            enable_context_expansion=True,
            expansion_window=400
        )
    ),
    "memory_efficient": RAGConfig(
        chunking=ChunkingConfig(
            max_chunk_size=1000,
            overlap_ratio=0.1,
            preserve_class_methods=False
        ),
        retrieval=RetrievalConfig(
            default_top_k=5,
            enable_context_expansion=False
        ),
        vector_store=VectorStoreConfig(
            qdrant=QdrantConfig(on_disk_payload=True)
        )
    ),
    "high_accuracy": RAGConfig(
        chunking=ChunkingConfig(
            preserve_class_methods=True,
            smart_imports=True,
            include_context_lines=5,
            overlap_ratio=0.3
        ),
        retrieval=RetrievalConfig(
            search_multiplier=3.0,
            enable_reranking=True,
            include_related_chunks=True
        ),
        fusion=FusionConfig(
            preserve_structure=True,
            deduplicate_content=True,
            smart_truncation=True
        )
    )
}

def get_config_preset(preset_name: str) -> RAGConfig:
    """Get a predefined configuration preset"""
    if preset_name in CONFIGS:
        return CONFIGS[preset_name]
    else:
        raise ValueError(f"Unknown preset: {preset_name}. Available: {list(CONFIGS.keys())}")

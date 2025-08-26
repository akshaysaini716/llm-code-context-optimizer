import logging
import re
from typing import List, Dict, Optional, Tuple, Set
from sentence_transformers import SentenceTransformer
import tiktoken
import numpy as np
from dataclasses import dataclass

from rag.models import CodeBaseChunk
from rag.configs import EmbeddingConfig

logger = logging.getLogger(__name__)

@dataclass
class EmbeddingContext:
    """Rich context for embedding generation"""
    chunk: CodeBaseChunk
    parent_context: Optional[str] = None
    sibling_context: List[str] = None
    symbols: Set[str] = None
    imports: Set[str] = None
    language_features: Dict[str, any] = None
    
    def __post_init__(self):
        self.sibling_context = self.sibling_context or []
        self.symbols = self.symbols or set()
        self.imports = self.imports or set()
        self.language_features = self.language_features or {}

class EnhancedCodeEmbedder:
    """Enhanced embedder with context-aware and relationship-aware embedding generation"""
    
    def __init__(self, config: Optional[EmbeddingConfig] = None):
        self.config = config or EmbeddingConfig()
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        # Initialize models
        self._initialize_models()
        
        # Caches for performance
        self.symbol_cache: Dict[str, Set[str]] = {}
        self.import_cache: Dict[str, Set[str]] = {}
        
        # Language-specific patterns
        self._initialize_language_patterns()

    def _initialize_models(self):
        """Initialize specialized embedding models"""
        try:
            # Primary code model - good for general code understanding
            self.code_model = SentenceTransformer(self.config.code_model)
            
            # Text model for documentation and comments
            self.text_model = SentenceTransformer(self.config.text_model)
            
            # Use the same model for specialized code understanding
            # In the future, you can replace this with actual specialized models
            self.specialized_code_model = self.code_model
            
            logger.info("Enhanced embedder models initialized successfully")
            logger.info(f"Using {self.config.code_model} for code embedding")
            
        except Exception as e:
            logger.error(f"Error initializing embedding models: {e}")
            raise

    def _initialize_language_patterns(self):
        """Initialize language-specific patterns for better understanding"""
        self.language_patterns = {
            "python": {
                "class_pattern": r"class\s+(\w+)",
                "function_pattern": r"def\s+(\w+)",
                "import_pattern": r"(?:from\s+[\w.]+\s+)?import\s+([\w\s,.*]+)",
                "decorator_pattern": r"@(\w+)",
                "docstring_pattern": r'"""(.*?)"""',
                "keywords": {
                    "async", "await", "class", "def", "if", "else", "elif", 
                    "for", "while", "try", "except", "finally", "with", "as",
                    "import", "from", "return", "yield", "lambda", "global", "nonlocal"
                }
            },
            "javascript": {
                "class_pattern": r"class\s+(\w+)",
                "function_pattern": r"(?:function\s+(\w+)|(\w+)\s*=\s*(?:async\s+)?(?:function|\([^)]*\)\s*=>))",
                "import_pattern": r"import\s+.*?from\s+['\"]([^'\"]+)['\"]",
                "export_pattern": r"export\s+(?:default\s+)?(?:class|function|const|let|var)\s+(\w+)",
                "keywords": {
                    "class", "function", "const", "let", "var", "if", "else",
                    "for", "while", "try", "catch", "finally", "async", "await",
                    "import", "export", "default", "return", "yield"
                }
            },
            "typescript": {
                "interface_pattern": r"interface\s+(\w+)",
                "type_pattern": r"type\s+(\w+)",
                "class_pattern": r"class\s+(\w+)",
                "function_pattern": r"(?:function\s+(\w+)|(\w+)\s*:\s*\([^)]*\)\s*=>)",
                "keywords": {
                    "interface", "type", "class", "function", "const", "let", "var",
                    "public", "private", "protected", "readonly", "static",
                    "extends", "implements", "namespace", "enum"
                }
            },
            "java": {
                "class_pattern": r"(?:public\s+)?class\s+(\w+)",
                "interface_pattern": r"(?:public\s+)?interface\s+(\w+)",
                "method_pattern": r"(?:public|private|protected)?\s*(?:static\s+)?[\w<>\[\]]+\s+(\w+)\s*\(",
                "annotation_pattern": r"@(\w+)",
                "keywords": {
                    "class", "interface", "public", "private", "protected", "static",
                    "final", "abstract", "extends", "implements", "package", "import"
                }
            }
        }

    def embed_chunks(self, chunks: List[CodeBaseChunk], chunk_relationships: Dict[str, any] = None) -> List[CodeBaseChunk]:
        """
        Enhanced chunk embedding with relationship awareness
        
        Args:
            chunks: List of chunks to embed
            chunk_relationships: Optional relationships between chunks
        """
        if not chunks:
            return chunks

        try:
            # Build embedding contexts with relationships
            contexts = self._build_embedding_contexts(chunks, chunk_relationships)
            
            # Group by embedding strategy
            grouped_contexts = self._group_contexts_by_strategy(contexts)
            
            # Embed each group with appropriate strategy
            for strategy, strategy_contexts in grouped_contexts.items():
                self._embed_context_group(strategy_contexts, strategy)
            
            # Apply post-processing enhancements
            self._apply_embedding_enhancements(chunks, contexts)
            
            logger.info(f"Successfully embedded {len(chunks)} chunks with enhanced context")
            return chunks
            
        except Exception as e:
            logger.error(f"Error in enhanced chunk embedding: {e}")
            return chunks

    def _build_embedding_contexts(
        self, 
        chunks: List[CodeBaseChunk], 
        chunk_relationships: Dict[str, any] = None
    ) -> List[EmbeddingContext]:
        """Build rich contexts for each chunk"""
        contexts = []
        chunk_map = {chunk.id: chunk for chunk in chunks}
        
        for chunk in chunks:
            context = EmbeddingContext(chunk=chunk)
            
            # Extract symbols and imports
            context.symbols = self._extract_symbols(chunk)
            context.imports = self._extract_imports(chunk)
            
            # Add relationship context if available
            if chunk_relationships and chunk.id in chunk_relationships:
                rel = chunk_relationships[chunk.id]
                
                # Add parent context
                if rel.parent_id and rel.parent_id in chunk_map:
                    parent_chunk = chunk_map[rel.parent_id]
                    context.parent_context = self._create_parent_context(parent_chunk)
                
                # Add sibling context
                for sibling_id in rel.sibling_ids[:3]:  # Limit to 3 siblings
                    if sibling_id in chunk_map:
                        sibling_chunk = chunk_map[sibling_id]
                        sibling_summary = self._create_sibling_context(sibling_chunk)
                        context.sibling_context.append(sibling_summary)
            
            # Extract language-specific features
            context.language_features = self._extract_language_features(chunk)
            
            contexts.append(context)
        
        return contexts

    def _group_contexts_by_strategy(self, contexts: List[EmbeddingContext]) -> Dict[str, List[EmbeddingContext]]:
        """Group contexts by embedding strategy"""
        groups = {
            "class_complete": [],
            "function_method": [],
            "documentation": [],
            "imports": [],
            "overlap": [],
            "general": []
        }
        
        for context in contexts:
            chunk = context.chunk
            
            if chunk.chunk_type == "class_complete":
                groups["class_complete"].append(context)
            elif chunk.chunk_type in ["function_definition", "method"]:
                groups["function_method"].append(context)
            elif chunk.chunk_type in ["docstring", "comment", "text"]:
                groups["documentation"].append(context)
            elif chunk.chunk_type == "imports":
                groups["imports"].append(context)
            elif chunk.chunk_type == "overlap":
                groups["overlap"].append(context)
            else:
                groups["general"].append(context)
        
        return {k: v for k, v in groups.items() if v}  # Remove empty groups

    def _embed_context_group(self, contexts: List[EmbeddingContext], strategy: str):
        """Embed a group of contexts using specific strategy"""
        if not contexts:
            return
        
        try:
            # Prepare texts based on strategy
            texts = []
            for context in contexts:
                text = self._prepare_text_for_strategy(context, strategy)
                texts.append(text)
            
            # Choose appropriate model
            model = self._select_model_for_strategy(strategy)
            
            # Generate embeddings
            embeddings = model.encode(
                texts,
                batch_size=self.config.batch_size,
                show_progress_bar=self.config.show_progress and len(texts) > 50,
                convert_to_numpy=True,
                normalize_embeddings=self.config.normalize_embeddings
            )
            
            # Assign embeddings back to chunks
            for context, embedding in zip(contexts, embeddings):
                context.chunk.embedding = embedding.tolist()
            
            logger.debug(f"Embedded {len(contexts)} chunks using {strategy} strategy")
            
        except Exception as e:
            logger.error(f"Error embedding {strategy} group: {e}")

    def _prepare_text_for_strategy(self, context: EmbeddingContext, strategy: str) -> str:
        """Prepare text for embedding based on strategy"""
        chunk = context.chunk
        
        if strategy == "class_complete":
            return self._prepare_class_text(context)
        elif strategy == "function_method":
            return self._prepare_function_text(context)
        elif strategy == "documentation":
            return self._prepare_documentation_text(context)
        elif strategy == "imports":
            return self._prepare_imports_text(context)
        elif strategy == "overlap":
            return self._prepare_overlap_text(context)
        else:
            return self._prepare_general_text(context)

    def _prepare_class_text(self, context: EmbeddingContext) -> str:
        """Prepare text for class chunks with full context"""
        chunk = context.chunk
        parts = []
        
        # File context
        if self.config.include_file_context:
            filename = chunk.file_path.split("/")[-1]
            parts.append(f"File: {filename}")
        
        # Type information
        parts.append(f"Type: Complete Class")
        
        # Extract class name and methods
        class_info = self._analyze_class_structure(chunk.content)
        if class_info:
            parts.append(f"Class: {class_info['name']}")
            if class_info['methods']:
                methods_str = ", ".join(class_info['methods'][:10])  # Limit to 10 methods
                parts.append(f"Methods: {methods_str}")
            if class_info['inheritance']:
                parts.append(f"Inherits: {class_info['inheritance']}")
        
        # Symbols and imports
        if context.symbols:
            symbols_str = ", ".join(list(context.symbols)[:15])
            parts.append(f"Symbols: {symbols_str}")
        
        # Language features
        if context.language_features:
            features = []
            for feature, value in context.language_features.items():
                if value:
                    features.append(feature)
            if features:
                parts.append(f"Features: {', '.join(features[:5])}")
        
        # Main content
        header = " | ".join(parts)
        return f"{header}\n\n{chunk.content}"

    def _prepare_function_text(self, context: EmbeddingContext) -> str:
        """Prepare text for function/method chunks"""
        chunk = context.chunk
        parts = []
        
        # Basic context
        if self.config.include_file_context:
            filename = chunk.file_path.split("/")[-1]
            parts.append(f"File: {filename}")
        
        parts.append(f"Type: {chunk.chunk_type}")
        
        # Function analysis
        func_info = self._analyze_function_structure(chunk.content)
        if func_info:
            parts.append(f"Function: {func_info['name']}")
            if func_info['parameters']:
                params_str = ", ".join(func_info['parameters'][:5])
                parts.append(f"Parameters: {params_str}")
            if func_info['returns']:
                parts.append(f"Returns: {func_info['returns']}")
        
        # Parent class context
        if context.parent_context:
            parts.append(f"Class Context: {context.parent_context}")
        
        # Sibling context (other methods in same class)
        if context.sibling_context:
            siblings = ", ".join(context.sibling_context[:3])
            parts.append(f"Related Methods: {siblings}")
        
        header = " | ".join(parts)
        return f"{header}\n\n{chunk.content}"

    def _prepare_documentation_text(self, context: EmbeddingContext) -> str:
        """Prepare text for documentation chunks"""
        chunk = context.chunk
        
        # For documentation, focus on the content with minimal metadata
        if self.config.include_file_context:
            filename = chunk.file_path.split("/")[-1]
            return f"Documentation from {filename}:\n\n{chunk.content}"
        else:
            return chunk.content

    def _prepare_imports_text(self, context: EmbeddingContext) -> str:
        """Prepare text for import chunks"""
        chunk = context.chunk
        
        # Analyze import dependencies
        import_analysis = self._analyze_imports(chunk.content)
        
        parts = [f"Type: Imports"]
        if import_analysis['modules']:
            modules_str = ", ".join(import_analysis['modules'][:10])
            parts.append(f"Modules: {modules_str}")
        if import_analysis['symbols']:
            symbols_str = ", ".join(import_analysis['symbols'][:15])
            parts.append(f"Imported Symbols: {symbols_str}")
        
        header = " | ".join(parts)
        return f"{header}\n\n{chunk.content}"

    def _prepare_overlap_text(self, context: EmbeddingContext) -> str:
        """Prepare text for overlap chunks"""
        chunk = context.chunk
        
        # Overlap chunks should emphasize their bridging nature
        parts = [
            f"Type: Context Bridge",
            f"File: {chunk.file_path.split('/')[-1]}"
        ]
        
        header = " | ".join(parts)
        return f"{header}\n\n{chunk.content}"

    def _prepare_general_text(self, context: EmbeddingContext) -> str:
        """Prepare text for general chunks"""
        chunk = context.chunk
        
        parts = []
        if self.config.include_file_context:
            filename = chunk.file_path.split("/")[-1]
            parts.append(f"File: {filename}")
        
        if self.config.include_type_info and chunk.chunk_type:
            parts.append(f"Type: {chunk.chunk_type}")
        
        if context.symbols and self.config.include_symbols:
            symbols_str = ", ".join(list(context.symbols)[:10])
            parts.append(f"Symbols: {symbols_str}")
        
        if parts:
            header = " | ".join(parts)
            return f"{header}\n\n{chunk.content}"
        else:
            return chunk.content

    def _select_model_for_strategy(self, strategy: str) -> SentenceTransformer:
        """Select appropriate model for embedding strategy"""
        if strategy in ["class_complete", "function_method", "imports"]:
            return self.specialized_code_model
        elif strategy == "documentation":
            return self.text_model
        else:
            return self.code_model

    def _apply_embedding_enhancements(self, chunks: List[CodeBaseChunk], contexts: List[EmbeddingContext]):
        """Apply post-processing enhancements to embeddings"""
        # This could include techniques like:
        # - Embedding averaging for related chunks
        # - Attention-based weighting
        # - Hierarchical embedding adjustments
        
        # For now, we'll implement a simple enhancement based on relationships
        chunk_map = {chunk.id: chunk for chunk in chunks}
        context_map = {ctx.chunk.id: ctx for ctx in contexts}
        
        for context in contexts:
            chunk = context.chunk
            if not chunk.embedding:
                continue
            
            # If this chunk has a parent, slightly adjust embedding towards parent
            if context.parent_context and hasattr(context, 'parent_id'):
                parent_chunk = chunk_map.get(context.parent_id)
                if parent_chunk and parent_chunk.embedding:
                    # Weighted average with small parent influence
                    chunk_emb = np.array(chunk.embedding)
                    parent_emb = np.array(parent_chunk.embedding)
                    enhanced_emb = 0.95 * chunk_emb + 0.05 * parent_emb
                    chunk.embedding = enhanced_emb.tolist()

    def embed_query(self, query: str, query_type: str = "mixed", context: Dict[str, any] = None) -> List[float]:
        """
        Enhanced query embedding with context awareness
        
        Args:
            query: The search query
            query_type: Type of query ("code", "documentation", "mixed")
            context: Additional context for query understanding
        """
        try:
            # Analyze query to understand intent
            query_analysis = self._analyze_query(query)
            
            # Prepare enhanced query text
            enhanced_query = self._prepare_query_text(query, query_analysis, context)
            
            # Select appropriate model
            if query_analysis['is_code_focused']:
                model = self.specialized_code_model
            elif query_analysis['is_documentation_focused']:
                model = self.text_model
            else:
                model = self.code_model
            
            # Generate embedding
            embedding = model.encode(
                [enhanced_query], 
                convert_to_numpy=True, 
                normalize_embeddings=self.config.normalize_embeddings
            )[0]
            
            return embedding.tolist()
            
        except Exception as e:
            logger.error(f"Error embedding query: {e}")
            return []

    def _analyze_query(self, query: str) -> Dict[str, any]:
        """Analyze query to understand search intent"""
        query_lower = query.lower()
        
        # Code-focused indicators
        code_indicators = [
            "function", "method", "class", "variable", "parameter",
            "implementation", "algorithm", "logic", "code", "syntax"
        ]
        
        # Documentation-focused indicators
        doc_indicators = [
            "documentation", "comment", "docstring", "readme", "guide",
            "explanation", "description", "overview", "summary"
        ]
        
        # Language-specific indicators
        language_indicators = {
            "python": ["python", "py", "def", "class", "import"],
            "javascript": ["javascript", "js", "function", "const", "let"],
            "typescript": ["typescript", "ts", "interface", "type"],
            "java": ["java", "public", "private", "static"]
        }
        
        analysis = {
            "is_code_focused": any(indicator in query_lower for indicator in code_indicators),
            "is_documentation_focused": any(indicator in query_lower for indicator in doc_indicators),
            "mentioned_languages": [],
            "mentioned_concepts": [],
            "is_specific_search": len(query.split()) <= 3,  # Short queries are usually specific
            "has_camelcase": bool(re.search(r'[a-z][A-Z]', query)),  # CamelCase suggests code
            "has_snake_case": '_' in query,  # snake_case suggests code
        }
        
        # Detect mentioned languages
        for lang, indicators in language_indicators.items():
            if any(indicator in query_lower for indicator in indicators):
                analysis["mentioned_languages"].append(lang)
        
        # Extract potential concepts/symbols
        words = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', query)
        analysis["mentioned_concepts"] = [w for w in words if len(w) > 2]
        
        return analysis

    def _prepare_query_text(self, query: str, analysis: Dict[str, any], context: Dict[str, any] = None) -> str:
        """Prepare enhanced query text for embedding"""
        parts = []
        
        # Add query type context
        if analysis["is_code_focused"]:
            parts.append("Code Search:")
        elif analysis["is_documentation_focused"]:
            parts.append("Documentation Search:")
        else:
            parts.append("General Search:")
        
        # Add language context if detected
        if analysis["mentioned_languages"]:
            lang_str = ", ".join(analysis["mentioned_languages"])
            parts.append(f"Languages: {lang_str}")
        
        # Add the original query
        parts.append(query)
        
        # Add additional context if provided
        if context:
            if "file_type" in context:
                parts.append(f"File Type: {context['file_type']}")
            if "project_context" in context:
                parts.append(f"Project: {context['project_context']}")
        
        return " | ".join(parts)

    def _extract_symbols(self, chunk: CodeBaseChunk) -> Set[str]:
        """Extract symbols from chunk content"""
        if chunk.file_path in self.symbol_cache:
            return self.symbol_cache[chunk.file_path]
        
        symbols = set()
        language = chunk.language
        
        if language in self.language_patterns:
            patterns = self.language_patterns[language]
            
            # Extract class names
            if "class_pattern" in patterns:
                matches = re.findall(patterns["class_pattern"], chunk.content)
                symbols.update(matches)
            
            # Extract function names
            if "function_pattern" in patterns:
                matches = re.findall(patterns["function_pattern"], chunk.content)
                symbols.update([m for m in matches if m])  # Filter empty matches
            
            # Extract other patterns specific to language
            for pattern_name, pattern in patterns.items():
                if pattern_name.endswith("_pattern") and pattern_name not in ["class_pattern", "function_pattern"]:
                    try:
                        matches = re.findall(pattern, chunk.content)
                        symbols.update([m for m in matches if m and isinstance(m, str)])
                    except:
                        continue
        
        # Cache the result
        self.symbol_cache[chunk.file_path] = symbols
        return symbols

    def _extract_imports(self, chunk: CodeBaseChunk) -> Set[str]:
        """Extract import information from chunk"""
        if chunk.file_path in self.import_cache:
            return self.import_cache[chunk.file_path]
        
        imports = set()
        language = chunk.language
        
        if language in self.language_patterns and "import_pattern" in self.language_patterns[language]:
            pattern = self.language_patterns[language]["import_pattern"]
            matches = re.findall(pattern, chunk.content)
            imports.update([m.strip() for m in matches if m])
        
        # Cache the result
        self.import_cache[chunk.file_path] = imports
        return imports

    def _extract_language_features(self, chunk: CodeBaseChunk) -> Dict[str, any]:
        """Extract language-specific features"""
        features = {}
        language = chunk.language
        content = chunk.content
        
        if language in self.language_patterns:
            patterns = self.language_patterns[language]
            
            # Check for keywords
            if "keywords" in patterns:
                keyword_count = 0
                for keyword in patterns["keywords"]:
                    keyword_count += len(re.findall(rf'\b{keyword}\b', content))
                features["keyword_density"] = keyword_count / max(len(content.split()), 1)
            
            # Language-specific features
            if language == "python":
                features["has_async"] = "async" in content
                features["has_decorators"] = "@" in content
                features["has_docstring"] = '"""' in content or "'''" in content
            elif language in ["javascript", "typescript"]:
                features["has_arrow_functions"] = "=>" in content
                features["has_async"] = "async" in content
                features["has_promises"] = "Promise" in content
            elif language == "java":
                features["has_annotations"] = "@" in content
                features["has_generics"] = "<" in content and ">" in content
                features["has_interfaces"] = "interface" in content
        
        return features

    def _analyze_class_structure(self, content: str) -> Optional[Dict[str, any]]:
        """Analyze class structure from content"""
        # Extract class name
        class_match = re.search(r'class\s+(\w+)', content)
        if not class_match:
            return None
        
        class_name = class_match.group(1)
        
        # Extract methods
        method_matches = re.findall(r'def\s+(\w+)', content)
        methods = [m for m in method_matches if not m.startswith('__') or m in ['__init__', '__str__', '__repr__']]
        
        # Extract inheritance
        inheritance_match = re.search(r'class\s+\w+\s*\(\s*([^)]+)\s*\)', content)
        inheritance = inheritance_match.group(1) if inheritance_match else None
        
        return {
            "name": class_name,
            "methods": methods,
            "inheritance": inheritance
        }

    def _analyze_function_structure(self, content: str) -> Optional[Dict[str, any]]:
        """Analyze function structure from content"""
        # Extract function name and signature
        func_match = re.search(r'def\s+(\w+)\s*\(([^)]*)\)', content)
        if not func_match:
            return None
        
        func_name = func_match.group(1)
        params_str = func_match.group(2)
        
        # Parse parameters
        parameters = []
        if params_str.strip():
            params = [p.strip().split(':')[0].split('=')[0].strip() for p in params_str.split(',')]
            parameters = [p for p in params if p and p != 'self']
        
        # Look for return type or return statements
        returns = None
        return_match = re.search(r'->\s*([^:]+):', content)
        if return_match:
            returns = return_match.group(1).strip()
        elif 'return ' in content:
            returns = "value"
        
        return {
            "name": func_name,
            "parameters": parameters,
            "returns": returns
        }

    def _analyze_imports(self, content: str) -> Dict[str, List[str]]:
        """Analyze import statements"""
        modules = []
        symbols = []
        
        # Standard imports
        import_matches = re.findall(r'import\s+([\w.]+)', content)
        modules.extend(import_matches)
        
        # From imports
        from_matches = re.findall(r'from\s+([\w.]+)\s+import\s+([^#\n]+)', content)
        for module, imported in from_matches:
            modules.append(module)
            # Parse imported symbols
            symbol_list = [s.strip().split(' as ')[0] for s in imported.split(',')]
            symbols.extend([s for s in symbol_list if s and s != '*'])
        
        return {
            "modules": modules,
            "symbols": symbols
        }

    def _create_parent_context(self, parent_chunk: CodeBaseChunk) -> str:
        """Create concise parent context string"""
        if parent_chunk.chunk_type == "class_complete":
            class_info = self._analyze_class_structure(parent_chunk.content)
            if class_info:
                return f"{class_info['name']} class"
        return f"{parent_chunk.chunk_type}"

    def _create_sibling_context(self, sibling_chunk: CodeBaseChunk) -> str:
        """Create concise sibling context string"""
        if sibling_chunk.chunk_type in ["function_definition", "method"]:
            func_info = self._analyze_function_structure(sibling_chunk.content)
            if func_info:
                return func_info['name']
        return f"{sibling_chunk.chunk_type}"

    def clear_caches(self):
        """Clear all caches"""
        self.symbol_cache.clear()
        self.import_cache.clear()

    def get_embedding_stats(self) -> Dict[str, any]:
        """Get statistics about embedding generation"""
        return {
            "symbol_cache_size": len(self.symbol_cache),
            "import_cache_size": len(self.import_cache),
            "models_loaded": {
                "code_model": self.config.code_model,
                "text_model": self.config.text_model,
                "specialized_available": hasattr(self, 'specialized_code_model')
            }
        }

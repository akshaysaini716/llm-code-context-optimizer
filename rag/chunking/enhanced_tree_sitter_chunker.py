import logging
import tiktoken
import mmap
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Set
from dataclasses import dataclass

from rag.configs import ChunkingConfig
from rag.models import CodeBaseChunk
from tree_sitter_language_pack import get_language, get_parser
from tree_sitter import Parser, Node, Tree, Query, QueryCursor
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import uuid

logger = logging.getLogger(__name__)

@dataclass
class ChunkConfig:
    """Configuration for chunking behavior"""
    max_chunk_size: int = 1500  # tokens
    min_chunk_size: int = 100   # tokens
    overlap_ratio: float = 0.2  # 20% overlap
    preserve_class_methods: bool = True  # Keep class methods together
    include_context_lines: int = 3  # Lines before/after for context
    smart_imports: bool = True  # Only include relevant imports
    hierarchical_chunking: bool = True  # Create parent-child relationships
    sliding_window_size: int = 200  # tokens for sliding window
    max_class_chunk_size: int = 3000  # larger limit for classes

@dataclass
class ChunkRelationship:
    """Tracks relationships between chunks"""
    chunk_id: str
    parent_id: Optional[str] = None
    child_ids: List[str] = None
    sibling_ids: List[str] = None
    overlaps_with: List[str] = None
    
    def __post_init__(self):
        self.child_ids = self.child_ids or []
        self.sibling_ids = self.sibling_ids or []
        self.overlaps_with = self.overlaps_with or []

class EnhancedTreeSitterChunker:
    def __init__(self, config: Optional[ChunkingConfig] = None):
        self.config = config or ChunkConfig()
        self.parsers = {}
        self.languages = {}
        self.queries = {}
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.chunk_relationships: Dict[str, ChunkRelationship] = {}
        
        self._initialize_parsers()
        self._initialize_queries()

    def _initialize_parsers(self):
        language_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.jsx': 'javascript',
            '.ts': 'typescript',
            '.tsx': 'typescript',
            '.java': 'java',
            '.kt': 'kotlin',
            '.kts': 'kotlin',
            '.go': 'go'
        }
        for ext, lang_name in language_map.items():
            try:
                language = get_language(lang_name)
                parser = get_parser(lang_name)
                self.languages[ext] = language
                self.parsers[ext] = parser
                logger.info(f"Initialized {ext} parser for {lang_name}")
            except Exception as e:
                logger.warning(f"Failed to initialize parser for {lang_name}: {e}")

    def _initialize_queries(self):
        if '.py' in self.languages:
            python_queries = {
                'functions': '''
                    (function_definition 
                        name: (identifier) @func-name) @func-def
                ''',
                'classes': '''
                    (class_definition 
                        name: (identifier) @class-name) @class-def
                ''',
                'imports': '''
                    [
                        (import_statement) @import
                        (import_from_statement) @import
                    ]
                ''',
                'methods': '''
                    (class_definition 
                        body: (block
                            (function_definition
                                name: (identifier) @method-name) @method-def))
                ''',
                'decorators': '''
                    (decorator) @decorator
                ''',
                'docstrings': '''
                    (expression_statement
                        (string) @docstring)
                ''',
                'class_with_methods': '''
                    (class_definition
                        name: (identifier) @class-name
                        body: (block) @class-body) @class-def
                '''
            }
            self.queries['.py'] = {}
            for name, query_text in python_queries.items():
                try:
                    self.queries['.py'][name] = Query(self.languages['.py'], query_text)
                except Exception as e:
                    logger.warning(f"Failed to initialize query for {name}: {e}")

    def chunk_file(self, file_path: Path) -> List[CodeBaseChunk]:
        """Main entry point for chunking a file"""
        try:
            if file_path.stat().st_size > 10 * 1024 * 1024:  # 10 MB
                chunks = self._chunk_large_file(file_path)
            else:
                chunks = self._chunk_regular_file(file_path)

            # Add relationships between chunks
            self._establish_chunk_relationships(chunks)
            
            # Apply overlapping if configured
            if self.config.overlap_ratio > 0:
                chunks = self._add_overlapping_chunks(chunks, file_path)
            
            return chunks
        except Exception as e:
            logger.error(f"Error chunking file {file_path}: {e}")
            return []

    def _chunk_large_file(self, file_path: Path) -> List[CodeBaseChunk]:
        try:
            with open(file_path, 'rb') as file:
                with mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                    content = mm.read().decode('utf-8', errors='ignore')
                    return self._parse_and_chunk(file_path, content)
        except Exception as e:
            logger.error(f"Error chunking large file {file_path}: {e}")
            return []

    def _chunk_regular_file(self, file_path: Path) -> List[CodeBaseChunk]:
        try:
            content = file_path.read_text(encoding='utf-8', errors='ignore')
            return self._parse_and_chunk(file_path, content)
        except Exception as e:
            logger.error(f"Error chunking regular file {file_path}: {e}")
            return []

    def _parse_and_chunk(self, file_path: Path, content: str) -> List[CodeBaseChunk]:
        ext = file_path.suffix.lower()
        if ext not in self.parsers:
            return self._fallback_chunking(file_path, content)

        try:
            parser = self.parsers[ext]
            tree = parser.parse(content.encode('utf-8'))
            
            if ext == '.py':
                return self._chunk_python_enhanced(file_path, content, tree)
            elif ext in ['.js', '.jsx', '.ts', '.tsx']:
                return self._chunk_javascript_enhanced(file_path, content, tree)
            elif ext == '.java':
                return self._chunk_java_enhanced(file_path, content, tree)
            else:
                return self._fallback_chunking(file_path, content)
                
        except Exception as e:
            logger.error(f"Tree-sitter parsing failed for {file_path}: {e}")
            return self._fallback_chunking(file_path, content)

    def _chunk_python_enhanced(self, file_path: Path, content: str, tree: Tree) -> List[CodeBaseChunk]:
        """Enhanced Python chunking that preserves class context"""
        chunks = []
        lines = content.split('\n')
        
        # Extract imports with smart analysis
        imports = self._extract_imports_with_query(tree, content, '.py')
        import_symbols = self._analyze_import_symbols(imports)
        
        # Process classes with their methods kept together
        if self.config.preserve_class_methods:
            classes = self._extract_complete_classes(tree, content, '.py')
            for class_name, class_node, methods in classes:
                class_chunk = self._create_class_chunk_with_methods(
                    file_path, content, class_node, class_name, methods, imports, import_symbols
                )
                if class_chunk:
                    chunks.append(class_chunk)
        else:
            # Original behavior - separate chunks
            classes = self._extract_classes_with_query(tree, content, '.py')
            for class_name, class_node in classes:
                chunk = self._create_chunk(
                    file_path, content, class_node, class_name, 
                    self._get_relevant_imports(class_node, imports, import_symbols, content), tree
                )
                if chunk:
                    chunks.append(chunk)
        
        # Process standalone functions
        functions = self._extract_standalone_functions(tree, content, '.py')
        for func_name, func_node in functions:
            relevant_imports = self._get_relevant_imports(func_node, imports, import_symbols, content)
            chunk = self._create_chunk(
                file_path, content, func_node, func_name, relevant_imports, tree
            )
            if chunk:
                chunks.append(chunk)
        
        # Add module-level code if significant
        module_chunk = self._create_module_chunk(file_path, content, tree, imports)
        if module_chunk:
            chunks.append(module_chunk)
        
        return chunks

    def _extract_complete_classes(self, tree: Tree, content: str, ext: str) -> List[Tuple[str, Node, List[Tuple[str, Node]]]]:
        """Extract classes with all their methods"""
        classes = []
        if ext in self.queries and 'class_with_methods' in self.queries[ext]:
            query = self.queries[ext]['class_with_methods']
            cursor = QueryCursor(query)
            captures = cursor.captures(tree.root_node)
            
            class_names = captures.get('class-name', [])
            class_defs = captures.get('class-def', [])
            class_bodies = captures.get('class-body', [])
            
            for name_node, def_node, body_node in zip(class_names, class_defs, class_bodies):
                class_name = content[name_node.start_byte:name_node.end_byte]
                
                # Extract all methods within this class
                methods = []
                for child in body_node.children:
                    if child.type == 'function_definition':
                        method_name_node = None
                        for subchild in child.children:
                            if subchild.type == 'identifier':
                                method_name_node = subchild
                                break
                        if method_name_node:
                            method_name = content[method_name_node.start_byte:method_name_node.end_byte]
                            methods.append((method_name, child))
                
                classes.append((class_name, def_node, methods))
        
        return classes

    def _extract_standalone_functions(self, tree: Tree, content: str, ext: str) -> List[Tuple[str, Node]]:
        """Extract functions that are not inside classes"""
        standalone_functions = []
        all_functions = self._extract_functions_with_query(tree, content, ext)
        
        # Get all class boundaries
        class_ranges = []
        classes = self._extract_classes_with_query(tree, content, ext)
        for _, class_node in classes:
            class_ranges.append((class_node.start_byte, class_node.end_byte))
        
        # Filter out functions inside classes
        for func_name, func_node in all_functions:
            is_inside_class = False
            for start, end in class_ranges:
                if start <= func_node.start_byte < end:
                    is_inside_class = True
                    break
            if not is_inside_class:
                standalone_functions.append((func_name, func_node))
        
        return standalone_functions

    def _create_class_chunk_with_methods(
        self, file_path: Path, content: str, class_node: Node, 
        class_name: str, methods: List[Tuple[str, Node]], 
        imports: List[str], import_symbols: Set[str]
    ) -> Optional[CodeBaseChunk]:
        """Create a single chunk containing the entire class with all its methods"""
        
        # Get the full class content
        class_content = content[class_node.start_byte:class_node.end_byte]
        
        # Check token count
        token_count = len(self.tokenizer.encode(class_content))
        
        # If class is too large, we might need to split it
        if token_count > self.config.max_class_chunk_size:
            # For now, include it anyway but log a warning
            logger.warning(f"Class {class_name} has {token_count} tokens, exceeding max of {self.config.max_class_chunk_size}")
        
        # Get relevant imports for this class
        relevant_imports = self._get_relevant_imports(class_node, imports, import_symbols, content)
        
        # Add context lines before and after if configured
        if self.config.include_context_lines > 0:
            class_content = self._add_context_lines(content, class_node, self.config.include_context_lines)
        
        # Combine imports and class content
        if relevant_imports:
            full_content = f"{relevant_imports}\n\n{class_content}"
        else:
            full_content = class_content
        
        chunk_id = self._generate_chunk_id(file_path, class_name, "class")
        
        return CodeBaseChunk(
            id=chunk_id,
            file_path=str(file_path),
            content=full_content.strip(),
            language=file_path.suffix.lower().lstrip('.'),
            chunk_type="class_complete",  # New type to indicate complete class
            start_byte=class_node.start_byte,
            end_byte=class_node.end_byte,
            start_line=class_node.start_point[0] + 1,
            end_line=class_node.end_point[0] + 1
        )

    def _add_overlapping_chunks(self, chunks: List[CodeBaseChunk], file_path: Path) -> List[CodeBaseChunk]:
        """Add overlapping chunks for better context continuity"""
        if not chunks or len(chunks) < 2:
            return chunks
        
        overlapping_chunks = []
        content = Path(file_path).read_text(encoding='utf-8', errors='ignore')
        lines = content.split('\n')
        
        for i in range(len(chunks) - 1):
            current_chunk = chunks[i]
            next_chunk = chunks[i + 1]
            
            # Calculate overlap region
            overlap_start_line = max(
                current_chunk.start_line,
                current_chunk.end_line - int(self.config.overlap_ratio * (current_chunk.end_line - current_chunk.start_line))
            )
            overlap_end_line = min(
                next_chunk.end_line,
                next_chunk.start_line + int(self.config.overlap_ratio * (next_chunk.end_line - next_chunk.start_line))
            )
            
            if overlap_end_line > overlap_start_line:
                overlap_content = '\n'.join(lines[overlap_start_line - 1:overlap_end_line])
                
                # Create overlap chunk
                overlap_chunk_id = self._generate_chunk_id(file_path, f"overlap_{i}", "overlap")
                overlap_chunk = CodeBaseChunk(
                    id=overlap_chunk_id,
                    file_path=str(file_path),
                    content=overlap_content,
                    language=file_path.suffix.lower().lstrip('.'),
                    chunk_type="overlap",
                    start_byte=0,  # Calculate if needed
                    end_byte=0,    # Calculate if needed
                    start_line=overlap_start_line,
                    end_line=overlap_end_line
                )
                
                overlapping_chunks.append(overlap_chunk)
                
                # Track relationships
                self._add_overlap_relationship(current_chunk.id, overlap_chunk_id)
                self._add_overlap_relationship(next_chunk.id, overlap_chunk_id)
        
        # Combine original and overlapping chunks
        all_chunks = chunks + overlapping_chunks
        return all_chunks

    def _establish_chunk_relationships(self, chunks: List[CodeBaseChunk]):
        """Establish parent-child and sibling relationships between chunks"""
        if not self.config.hierarchical_chunking:
            return
        
        # Group chunks by type and file
        file_chunks = {}
        for chunk in chunks:
            if chunk.file_path not in file_chunks:
                file_chunks[chunk.file_path] = []
            file_chunks[chunk.file_path].append(chunk)
        
        for file_path, file_chunk_list in file_chunks.items():
            # Sort by start position
            file_chunk_list.sort(key=lambda c: c.start_line)
            
            # Find parent-child relationships
            for i, chunk in enumerate(file_chunk_list):
                chunk_rel = ChunkRelationship(chunk_id=chunk.id)
                
                # Find parent (containing chunk)
                for j, potential_parent in enumerate(file_chunk_list):
                    if i != j and self._is_contained_in(chunk, potential_parent):
                        chunk_rel.parent_id = potential_parent.id
                        break
                
                # Find siblings (same level, same parent)
                if chunk_rel.parent_id:
                    for other_chunk in file_chunk_list:
                        if other_chunk.id != chunk.id:
                            other_rel = self.chunk_relationships.get(other_chunk.id)
                            if other_rel and other_rel.parent_id == chunk_rel.parent_id:
                                chunk_rel.sibling_ids.append(other_chunk.id)
                
                self.chunk_relationships[chunk.id] = chunk_rel

    def _is_contained_in(self, inner: CodeBaseChunk, outer: CodeBaseChunk) -> bool:
        """Check if inner chunk is contained within outer chunk"""
        return (outer.start_line <= inner.start_line and 
                outer.end_line >= inner.end_line and
                outer.start_byte <= inner.start_byte and
                outer.end_byte >= inner.end_byte)

    def _add_overlap_relationship(self, chunk_id1: str, chunk_id2: str):
        """Add overlap relationship between chunks"""
        if chunk_id1 not in self.chunk_relationships:
            self.chunk_relationships[chunk_id1] = ChunkRelationship(chunk_id=chunk_id1)
        if chunk_id2 not in self.chunk_relationships:
            self.chunk_relationships[chunk_id2] = ChunkRelationship(chunk_id=chunk_id2)
        
        self.chunk_relationships[chunk_id1].overlaps_with.append(chunk_id2)
        self.chunk_relationships[chunk_id2].overlaps_with.append(chunk_id1)

    def _analyze_import_symbols(self, imports: List[str]) -> Set[str]:
        """Extract symbols from import statements"""
        symbols = set()
        for import_stmt in imports:
            # Handle different import patterns
            if 'from' in import_stmt and 'import' in import_stmt:
                # from module import symbol1, symbol2
                parts = import_stmt.split('import')
                if len(parts) > 1:
                    imported = parts[1].strip()
                    for symbol in imported.split(','):
                        symbol = symbol.strip().split(' as ')[0]
                        if symbol and symbol != '*':
                            symbols.add(symbol)
            elif 'import' in import_stmt:
                # import module
                parts = import_stmt.split('import')
                if len(parts) > 1:
                    module = parts[1].strip().split(' as ')[0]
                    if module:
                        symbols.add(module.split('.')[0])
        return symbols

    def _get_relevant_imports(
        self, node: Node, imports: List[str], 
        import_symbols: Set[str], content: str
    ) -> str:
        """Get only relevant imports for a given node"""
        if not self.config.smart_imports:
            return '\n'.join(imports)
        
        node_content = content[node.start_byte:node.end_byte]
        relevant_imports = []
        
        for import_stmt in imports:
            # Check if any imported symbol is used in the node
            for symbol in import_symbols:
                if symbol in node_content:
                    if import_stmt not in relevant_imports:
                        relevant_imports.append(import_stmt)
                    break
        
        return '\n'.join(relevant_imports)

    def _add_context_lines(self, content: str, node: Node, context_lines: int) -> str:
        """Add context lines before and after a node"""
        lines = content.split('\n')
        start_line = max(0, node.start_point[0] - context_lines)
        end_line = min(len(lines), node.end_point[0] + 1 + context_lines)
        
        context_content = []
        
        # Add context before
        if start_line < node.start_point[0]:
            context_before = lines[start_line:node.start_point[0]]
            if context_before:
                context_content.append("# ... context before ...")
                context_content.extend(context_before)
        
        # Add main content
        main_content = lines[node.start_point[0]:node.end_point[0] + 1]
        context_content.extend(main_content)
        
        # Add context after
        if end_line > node.end_point[0] + 1:
            context_after = lines[node.end_point[0] + 1:end_line]
            if context_after:
                context_content.extend(context_after)
                context_content.append("# ... context after ...")
        
        return '\n'.join(context_content)

    def _generate_chunk_id(self, file_path: Path, identifier: str, chunk_type: str) -> str:
        """Generate unique chunk ID"""
        unique_str = f"{file_path}_{identifier}_{chunk_type}"
        return hashlib.md5(unique_str.encode()).hexdigest()

    def _create_module_chunk(
        self, file_path: Path, content: str, 
        tree: Tree, imports: List[str]
    ) -> Optional[CodeBaseChunk]:
        """Create chunk for module-level code (globals, constants, etc.)"""
        # This would extract module-level code that's not in functions/classes
        # For now, returning None as a placeholder
        return None

    def _fallback_chunking(self, file_path: Path, content: str) -> List[CodeBaseChunk]:
        """Fallback chunking for unsupported file types"""
        chunks = []
        lines = content.split('\n')
        
        # Simple line-based chunking with overlap
        chunk_size = 50  # lines per chunk
        overlap = int(chunk_size * self.config.overlap_ratio)
        
        for i in range(0, len(lines), chunk_size - overlap):
            chunk_lines = lines[i:i + chunk_size]
            chunk_content = '\n'.join(chunk_lines)
            
            chunk_id = self._generate_chunk_id(file_path, f"fallback_{i}", "text")
            chunk = CodeBaseChunk(
                id=chunk_id,
                file_path=str(file_path),
                content=chunk_content,
                language="text",
                chunk_type="text",
                start_byte=0,
                end_byte=0,
                start_line=i + 1,
                end_line=min(i + chunk_size, len(lines))
            )
            chunks.append(chunk)
        
        return chunks

    def _chunk_javascript_enhanced(self, file_path: Path, content: str, tree: Tree) -> List[CodeBaseChunk]:
        """Enhanced JavaScript/TypeScript chunking"""
        # Placeholder for JavaScript/TypeScript implementation
        return self._fallback_chunking(file_path, content)

    def _chunk_java_enhanced(self, file_path: Path, content: str, tree: Tree) -> List[CodeBaseChunk]:
        """Enhanced Java chunking"""
        # Placeholder for Java implementation
        return self._fallback_chunking(file_path, content)

    def _extract_imports_with_query(self, tree: Tree, content: str, ext: str) -> List[str]:
        """Extract import statements"""
        imports = []
        if ext in self.queries and 'imports' in self.queries[ext]:
            query = self.queries[ext]['imports']
            cursor = QueryCursor(query)
            captures = cursor.captures(tree.root_node)
            if 'import' in captures:
                for node in captures['import']:
                    import_text = content[node.start_byte:node.end_byte]
                    imports.append(import_text)
        return imports

    def _extract_functions_with_query(self, tree: Tree, content: str, ext: str) -> List[Tuple[str, Node]]:
        """Extract function definitions"""
        functions = []
        if ext in self.queries and 'functions' in self.queries[ext]:
            query = self.queries[ext]['functions']
            cursor = QueryCursor(query)
            captures = cursor.captures(tree.root_node)
            
            func_names = captures.get('func-name', [])
            func_defs = captures.get('func-def', [])
            
            for node_name, def_node in zip(func_names, func_defs):
                func_name = content[node_name.start_byte:node_name.end_byte]
                functions.append((func_name, def_node))
        
        return functions

    def _extract_classes_with_query(self, tree: Tree, content: str, ext: str) -> List[Tuple[str, Node]]:
        """Extract class definitions"""
        classes = []
        if ext in self.queries and 'classes' in self.queries[ext]:
            query = self.queries[ext]['classes']
            cursor = QueryCursor(query)
            captures = cursor.captures(tree.root_node)
            
            class_names = captures.get('class-name', [])
            class_defs = captures.get('class-def', [])
            
            for node_name, def_node in zip(class_names, class_defs):
                class_name = content[node_name.start_byte:node_name.end_byte]
                classes.append((class_name, def_node))
        
        return classes

    def _create_chunk(
        self, file_path: Path, content: str, node: Node, 
        symbol_name: str, imports_text: str, tree: Tree
    ) -> Optional[CodeBaseChunk]:
        """Create a chunk from a node"""
        node_content = content[node.start_byte:node.end_byte]
        
        # Add context lines if configured
        if self.config.include_context_lines > 0:
            node_content = self._add_context_lines(content, node, self.config.include_context_lines)
        
        full_content = f"{imports_text}\n\n{node_content}" if imports_text else node_content
        
        chunk_id = self._generate_chunk_id(file_path, symbol_name, node.type)
        
        return CodeBaseChunk(
            id=chunk_id,
            file_path=str(file_path),
            content=full_content.strip(),
            language=file_path.suffix.lower().lstrip('.'),
            chunk_type=node.type,
            start_byte=node.start_byte,
            end_byte=node.end_byte,
            start_line=node.start_point[0] + 1,
            end_line=node.end_point[0] + 1
        )

    def _validate_chunk_quality(self, chunk: CodeBaseChunk) -> bool:
        """Validate chunk quality"""
        # Check minimum size
        token_count = len(self.tokenizer.encode(chunk.content))
        if token_count < self.config.min_chunk_size:
            return False
        
        # Check maximum size
        if token_count > self.config.max_chunk_size and chunk.chunk_type != "class_complete":
            logger.warning(f"Chunk exceeds max size: {token_count} tokens")
        
        return True

    def chunk_files_parallel(self, file_paths: List[Path]) -> List[CodeBaseChunk]:
        """Chunk multiple files in parallel"""
        if len(file_paths) < 2:
            chunks = []
            for path in file_paths:
                chunks.extend(self.chunk_file(path))
            return chunks

        all_chunks = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_path = {executor.submit(self.chunk_file, path): path for path in file_paths}
            for future in as_completed(future_to_path):
                path = future_to_path[future]
                try:
                    chunks = future.result()
                    all_chunks.extend(chunks)
                except Exception as e:
                    logger.error(f"Error chunking file {path}: {e}")

        return all_chunks

    def get_chunk_relationships(self, chunk_id: str) -> Optional[ChunkRelationship]:
        """Get relationships for a specific chunk"""
        return self.chunk_relationships.get(chunk_id)

    def get_related_chunks(self, chunk_id: str, relationship_type: str = "all") -> List[str]:
        """Get related chunk IDs based on relationship type"""
        rel = self.chunk_relationships.get(chunk_id)
        if not rel:
            return []
        
        if relationship_type == "parent":
            return [rel.parent_id] if rel.parent_id else []
        elif relationship_type == "children":
            return rel.child_ids
        elif relationship_type == "siblings":
            return rel.sibling_ids
        elif relationship_type == "overlaps":
            return rel.overlaps_with
        else:  # "all"
            related = []
            if rel.parent_id:
                related.append(rel.parent_id)
            related.extend(rel.child_ids)
            related.extend(rel.sibling_ids)
            related.extend(rel.overlaps_with)
            return related

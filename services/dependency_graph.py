"""
Dependency Graph Builder
Creates and manages dependency relationships between code symbols and files
"""

from typing import Dict, List, Set, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque
import os
from pathlib import Path

from .ast_parser import ASTParser, FileAnalysis, Symbol

@dataclass
class DependencyNode:
    """Represents a node in the dependency graph"""
    name: str
    type: str  # 'file', 'function', 'class', 'variable'
    file_path: str
    dependencies: Set[str] = field(default_factory=set)  # nodes this depends on
    dependents: Set[str] = field(default_factory=set)   # nodes that depend on this
    importance_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DependencyPath:
    """Represents a path between two nodes in the dependency graph"""
    source: str
    target: str
    path: List[str]
    distance: int
    relevance_score: float = 0.0

class DependencyGraph:
    """Builds and manages dependency graphs for codebases"""
    
    def __init__(self):
        self.ast_parser = ASTParser()
        self.nodes: Dict[str, DependencyNode] = {}
        self.file_analyses: Dict[str, FileAnalysis] = {}
        self.symbol_to_file: Dict[str, str] = {}  # symbol name -> file path
        self.file_to_symbols: Dict[str, List[str]] = defaultdict(list)  # file -> symbols
    
    def build_graph(self, directory: str, extensions: Optional[List[str]] = None) -> None:
        """Build dependency graph for the entire codebase"""
        # First pass: analyze all files
        self.file_analyses = self.ast_parser.analyze_codebase(directory, extensions)
        
        # Second pass: build nodes
        self._build_nodes()
        
        # Third pass: build dependencies
        self._build_dependencies()
        
        # Fourth pass: calculate importance scores
        self._calculate_importance_scores()
    
    def _build_nodes(self) -> None:
        """Build nodes from file analyses"""
        self.nodes.clear()
        self.symbol_to_file.clear()
        self.file_to_symbols.clear()
        
        # Create file nodes
        for file_path, analysis in self.file_analyses.items():
            file_node = DependencyNode(
                name=file_path,
                type='file',
                file_path=file_path,
                metadata={
                    'language': analysis.language,
                    'symbol_count': len(analysis.symbols),
                    'import_count': len(analysis.imports),
                    'export_count': len(analysis.exports)
                }
            )
            self.nodes[file_path] = file_node
            
            # Create symbol nodes
            for symbol in analysis.symbols:
                symbol_key = f"{file_path}::{symbol.name}"
                symbol_node = DependencyNode(
                    name=symbol.name,
                    type=symbol.type,
                    file_path=file_path,
                    metadata={
                        'scope': symbol.scope,
                        'start_line': symbol.start_line,
                        'end_line': symbol.end_line,
                        'parameters': symbol.parameters,
                        'docstring': symbol.docstring,
                        'return_type': symbol.return_type
                    }
                )
                self.nodes[symbol_key] = symbol_node
                
                # Update mappings
                self.symbol_to_file[symbol.name] = file_path
                self.file_to_symbols[file_path].append(symbol_key)
    
    def _build_dependencies(self) -> None:
        """Build dependency relationships between nodes"""
        for file_path, analysis in self.file_analyses.items():
            file_node = self.nodes[file_path]
            
            # File-level dependencies (imports)
            for import_name in analysis.imports:
                # Try to resolve import to actual file
                resolved_files = self._resolve_import(import_name, file_path)
                for resolved_file in resolved_files:
                    if resolved_file in self.nodes:
                        file_node.dependencies.add(resolved_file)
                        self.nodes[resolved_file].dependents.add(file_path)
            
            # Symbol-level dependencies
            for symbol in analysis.symbols:
                symbol_key = f"{file_path}::{symbol.name}"
                symbol_node = self.nodes[symbol_key]
                
                # Symbol depends on its file
                symbol_node.dependencies.add(file_path)
                file_node.dependents.add(symbol_key)
                
                # Find symbol references in other symbols
                self._find_symbol_references(symbol_key, symbol_node)
    
    def _resolve_import(self, import_name: str, current_file: str) -> List[str]:
        """Resolve import statement to actual file paths"""
        resolved_files = []
        current_dir = os.path.dirname(current_file)
        
        # Handle different import patterns
        if import_name.startswith('from '):
            # Python: from module import ...
            module_name = import_name.split(' import ')[0].replace('from ', '')
            resolved_files.extend(self._find_module_files(module_name, current_dir))
        elif import_name.startswith('import '):
            # Python: import module
            module_name = import_name.replace('import ', '')
            resolved_files.extend(self._find_module_files(module_name, current_dir))
        elif import_name.startswith('#include'):
            # C/C++: #include "file.h"
            header_name = import_name.replace('#include ', '').strip('<>"')
            resolved_files.extend(self._find_header_files(header_name, current_dir))
        else:
            # JavaScript/TypeScript: import from "module"
            if 'from ' in import_name:
                module_name = import_name.split('from ')[-1].strip('"\'')
                resolved_files.extend(self._find_js_module_files(module_name, current_dir))
        
        return resolved_files
    
    def _find_module_files(self, module_name: str, current_dir: str) -> List[str]:
        """Find Python module files"""
        candidates = []
        
        # Relative imports
        if module_name.startswith('.'):
            relative_path = module_name.replace('.', '/') + '.py'
            full_path = os.path.join(current_dir, relative_path)
            if os.path.exists(full_path):
                candidates.append(full_path)
        else:
            # Absolute imports - check if it's a local module
            module_path = module_name.replace('.', '/')
            
            # Check for .py file
            py_file = os.path.join(current_dir, module_path + '.py')
            if os.path.exists(py_file):
                candidates.append(py_file)
            
            # Check for __init__.py in directory
            init_file = os.path.join(current_dir, module_path, '__init__.py')
            if os.path.exists(init_file):
                candidates.append(init_file)
        
        return candidates
    
    def _find_header_files(self, header_name: str, current_dir: str) -> List[str]:
        """Find C/C++ header files"""
        candidates = []
        
        # Local headers (quoted includes)
        local_header = os.path.join(current_dir, header_name)
        if os.path.exists(local_header):
            candidates.append(local_header)
        
        return candidates
    
    def _find_js_module_files(self, module_name: str, current_dir: str) -> List[str]:
        """Find JavaScript/TypeScript module files"""
        candidates = []
        
        if module_name.startswith('./') or module_name.startswith('../'):
            # Relative imports
            module_path = os.path.join(current_dir, module_name)
            
            # Try different extensions
            for ext in ['.js', '.ts', '.jsx', '.tsx']:
                file_path = module_path + ext
                if os.path.exists(file_path):
                    candidates.append(file_path)
            
            # Try index files
            for ext in ['.js', '.ts']:
                index_path = os.path.join(module_path, 'index' + ext)
                if os.path.exists(index_path):
                    candidates.append(index_path)
        
        return candidates
    
    def _find_symbol_references(self, symbol_key: str, symbol_node: DependencyNode) -> None:
        """Find references to this symbol in other files"""
        symbol_name = symbol_node.name
        
        # Search for symbol usage in other files
        for file_path, analysis in self.file_analyses.items():
            if file_path == symbol_node.file_path:
                continue
            
            # Check if symbol is imported/used
            for import_stmt in analysis.imports:
                if symbol_name in import_stmt:
                    # This file depends on the symbol
                    file_node = self.nodes[file_path]
                    file_node.dependencies.add(symbol_key)
                    symbol_node.dependents.add(file_path)
    
    def _calculate_importance_scores(self) -> None:
        """Calculate importance scores for nodes using PageRank-like algorithm"""
        # Initialize scores
        for node in self.nodes.values():
            node.importance_score = 1.0
        
        # Iterative improvement (simplified PageRank)
        for _ in range(10):  # 10 iterations
            new_scores = {}
            
            for node_key, node in self.nodes.items():
                score = 0.15  # Base score
                
                # Add score from dependents
                for dependent_key in node.dependents:
                    if dependent_key in self.nodes:
                        dependent = self.nodes[dependent_key]
                        if len(dependent.dependencies) > 0:
                            score += 0.85 * dependent.importance_score / len(dependent.dependencies)
                
                new_scores[node_key] = score
            
            # Update scores
            for node_key, score in new_scores.items():
                self.nodes[node_key].importance_score = score
    
    def find_relevant_context(self, query: str, max_files: int = 10) -> List[Tuple[str, float]]:
        """Find most relevant files for a given query"""
        query_terms = query.lower().split()
        file_scores = {}
        
        for file_path, analysis in self.file_analyses.items():
            score = 0.0
            
            # Score based on symbol names
            for symbol in analysis.symbols:
                symbol_relevance = self._calculate_symbol_relevance(symbol.name, query_terms)
                if symbol_relevance > 0:
                    symbol_key = f"{file_path}::{symbol.name}"
                    if symbol_key in self.nodes:
                        importance = self.nodes[symbol_key].importance_score
                        score += symbol_relevance * importance
            
            # Score based on file name with enhanced matching
            file_name = os.path.basename(file_path).lower()
            file_name_no_ext = os.path.splitext(file_name)[0]
            
            filename_score = 0.0
            for term in query_terms:
                # Exact filename match (highest priority)
                if term == file_name or term == file_name_no_ext:
                    filename_score += 10.0  # Much higher score for exact filename match
                elif term in file_name:
                    filename_score += 3.0   # Higher score for partial filename match
                # Check for filename with extension mentioned
                elif term.endswith('.py') and term == file_name:
                    filename_score += 10.0
            
            score += filename_score
            
            # Score based on imports/exports
            for export in analysis.exports:
                export_relevance = self._calculate_symbol_relevance(export, query_terms)
                score += export_relevance * 0.5
            

            
            if score > 0:
                file_scores[file_path] = score
        
        # Sort by score and return top files
        sorted_files = sorted(file_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_files[:max_files]
    
    def get_all_files_with_importance(self) -> List[Tuple[str, float]]:
        """Get all files sorted by importance score (for full context mode)"""
        file_scores = []
        
        for file_path, analysis in self.file_analyses.items():
            # Use importance score from dependency graph, or default to 1.0
            importance_score = 1.0
            if file_path in self.nodes:
                importance_score = self.nodes[file_path].importance_score
            
            file_scores.append((file_path, importance_score))
        
        # Sort by importance score
        return sorted(file_scores, key=lambda x: x[1], reverse=True)
    
    def _calculate_symbol_relevance(self, symbol_name: str, query_terms: List[str]) -> float:
        """Calculate relevance score between symbol name and query terms"""
        symbol_lower = symbol_name.lower()
        score = 0.0
        
        for term in query_terms:
            if term == symbol_lower:
                score += 3.0  # Exact match
            elif term in symbol_lower:
                score += 2.0  # Substring match
            elif self._fuzzy_match(term, symbol_lower):
                score += 1.0  # Fuzzy match
        
        return score
    
    def _fuzzy_match(self, term: str, symbol: str) -> bool:
        """Simple fuzzy matching (edit distance based)"""
        if len(term) < 3:
            return False
        
        # Simple substring with typo tolerance
        for i in range(len(symbol) - len(term) + 1):
            substring = symbol[i:i + len(term)]
            differences = sum(1 for a, b in zip(term, substring) if a != b)
            if differences <= 1:  # Allow 1 character difference
                return True
        
        return False
    
    def get_dependency_path(self, source: str, target: str) -> Optional[DependencyPath]:
        """Find shortest dependency path between two nodes"""
        if source not in self.nodes or target not in self.nodes:
            return None
        
        # BFS to find shortest path
        queue = deque([(source, [source])])
        visited = {source}
        
        while queue:
            current, path = queue.popleft()
            
            if current == target:
                return DependencyPath(
                    source=source,
                    target=target,
                    path=path,
                    distance=len(path) - 1
                )
            
            # Explore dependencies
            current_node = self.nodes[current]
            for dep in current_node.dependencies:
                if dep not in visited and dep in self.nodes:
                    visited.add(dep)
                    queue.append((dep, path + [dep]))
        
        return None
    
    def get_related_symbols(self, symbol_name: str, max_results: int = 20) -> List[Tuple[str, float]]:
        """Get symbols related to the given symbol"""
        related = []
        
        # Find the symbol in the graph
        symbol_keys = [key for key in self.nodes.keys() if key.endswith(f"::{symbol_name}")]
        
        if not symbol_keys:
            return related
        
        symbol_key = symbol_keys[0]  # Take first match
        symbol_node = self.nodes[symbol_key]
        
        # Get direct dependencies and dependents
        all_related = set(symbol_node.dependencies) | set(symbol_node.dependents)
        
        # Score related symbols
        for related_key in all_related:
            if related_key in self.nodes:
                related_node = self.nodes[related_key]
                score = related_node.importance_score
                
                # Boost score for same file
                if related_node.file_path == symbol_node.file_path:
                    score *= 1.5
                
                related.append((related_key, score))
        
        # Sort by score and return top results
        related.sort(key=lambda x: x[1], reverse=True)
        return related[:max_results]
    
    def get_graph_stats(self) -> Dict[str, Any]:
        """Get statistics about the dependency graph"""
        file_nodes = [n for n in self.nodes.values() if n.type == 'file']
        symbol_nodes = [n for n in self.nodes.values() if n.type != 'file']
        
        return {
            'total_nodes': len(self.nodes),
            'file_nodes': len(file_nodes),
            'symbol_nodes': len(symbol_nodes),
            'total_dependencies': sum(len(n.dependencies) for n in self.nodes.values()),
            'avg_dependencies_per_node': sum(len(n.dependencies) for n in self.nodes.values()) / len(self.nodes) if self.nodes else 0,
            'most_important_files': sorted(
                [(n.name, n.importance_score) for n in file_nodes],
                key=lambda x: x[1],
                reverse=True
            )[:10],
            'languages': list(set(analysis.language for analysis in self.file_analyses.values()))
        } 
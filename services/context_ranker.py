"""
Context Ranking System
Uses dependency graphs and symbol relevance to rank and filter code context
"""

from typing import Dict, List, Set, Optional, Tuple, Any
from dataclasses import dataclass, field
import os
import re
from collections import defaultdict

from .dependency_graph import DependencyGraph, DependencyNode
from .ast_parser import FileAnalysis

@dataclass
class ContextItem:
    """Represents a piece of context (file, function, class, etc.)"""
    file_path: str
    content: str
    relevance_score: float
    importance_score: float
    context_type: str  # 'file', 'function', 'class', 'snippet'
    start_line: Optional[int] = None
    end_line: Optional[int] = None
    symbol_name: Optional[str] = None
    dependencies: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RankedContext:
    """Ranked context result"""
    items: List[ContextItem]
    total_score: float
    token_estimate: int
    query_coverage: float
    metadata: Dict[str, Any] = field(default_factory=dict)

class ContextRanker:
    """Ranks and filters code context using dependency graphs and relevance scoring"""
    
    def __init__(self, dependency_graph: DependencyGraph):
        self.dependency_graph = dependency_graph
        self.query_cache = {}  # Cache for query results
    
    def rank_context(
        self,
        query: str,
        max_tokens: int = 8000,
        max_files: int = 20,
        include_dependencies: bool = True,
        context_types: Optional[List[str]] = None
    ) -> RankedContext:
        """
        Rank and filter context based on query relevance and dependency analysis
        
        Args:
            query: The user query or task description
            max_tokens: Maximum tokens to include in context
            max_files: Maximum number of files to include
            include_dependencies: Whether to include dependency files
            context_types: Types of context to include ['file', 'function', 'class']
        """
        if context_types is None:
            context_types = ['file', 'function', 'class']
        
        # Extract query features
        query_features = self._extract_query_features(query)
        
        # Find relevant files using dependency graph
        relevant_files = self.dependency_graph.find_relevant_context(query, max_files * 2)
        
        # Generate context items
        context_items = []
        
        for file_path, base_score in relevant_files:
            if file_path in self.dependency_graph.file_analyses:
                analysis = self.dependency_graph.file_analyses[file_path]
                
                # Add file-level context if requested
                if 'file' in context_types:
                    file_item = self._create_file_context_item(
                        file_path, analysis, query_features, base_score
                    )
                    if file_item:
                        context_items.append(file_item)
                
                # Add symbol-level context if requested
                if any(t in context_types for t in ['function', 'class']):
                    symbol_items = self._create_symbol_context_items(
                        file_path, analysis, query_features, base_score, context_types
                    )
                    context_items.extend(symbol_items)
        
        # Add dependency context if requested
        if include_dependencies:
            dependency_items = self._add_dependency_context(
                context_items, query_features, max_files // 4
            )
            context_items.extend(dependency_items)
        
        # Sort by combined score
        context_items.sort(key=lambda x: x.relevance_score * x.importance_score, reverse=True)
        
        # Filter by token budget
        filtered_items = self._filter_by_token_budget(context_items, max_tokens)
        
        # Calculate final metrics
        total_score = sum(item.relevance_score * item.importance_score for item in filtered_items)
        token_estimate = sum(self._estimate_tokens(item.content) for item in filtered_items)
        query_coverage = self._calculate_query_coverage(filtered_items, query_features)
        
        return RankedContext(
            items=filtered_items,
            total_score=total_score,
            token_estimate=token_estimate,
            query_coverage=query_coverage,
            metadata={
                'query_features': query_features,
                'total_candidates': len(context_items),
                'files_included': len(set(item.file_path for item in filtered_items)),
                'context_types': list(set(item.context_type for item in filtered_items))
            }
        )
    
    def rank_full_context(
        self,
        all_files: List[Tuple[str, float]],
        max_tokens: int = 8000
    ) -> RankedContext:
        """
        Create full context with all files (for relevant_only=False)
        
        Args:
            all_files: List of (file_path, importance_score) tuples
            max_tokens: Maximum tokens to include
        """
        context_items = []
        
        # Create file-level context items for all files
        for file_path, importance_score in all_files:
            if file_path in self.dependency_graph.file_analyses:
                analysis = self.dependency_graph.file_analyses[file_path]
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    item = ContextItem(
                        file_path=file_path,
                        content=content,
                        relevance_score=1.0,  # All files are equally relevant in full mode
                        importance_score=importance_score,
                        context_type='file',
                        metadata={
                            'language': analysis.language,
                            'symbol_count': len(analysis.symbols),
                            'import_count': len(analysis.imports),
                            'line_count': len(content.split('\n'))
                        }
                    )
                    context_items.append(item)
                    
                except Exception:
                    continue
        
        # Filter by token budget (sorted by importance)
        context_items.sort(key=lambda x: x.importance_score, reverse=True)
        filtered_items = self._filter_by_token_budget(context_items, max_tokens)
        
        # Calculate final metrics
        total_score = sum(item.importance_score for item in filtered_items)
        token_estimate = sum(self._estimate_tokens(item.content) for item in filtered_items)
        
        return RankedContext(
            items=filtered_items,
            total_score=total_score,
            token_estimate=token_estimate,
            query_coverage=1.0,  # Full context covers everything
            metadata={
                'mode': 'full_context',
                'total_files': len(all_files),
                'files_included': len(filtered_items),
                'context_types': ['file']
            }
        )
    
    def _extract_query_features(self, query: str) -> Dict[str, Any]:
        """Extract features from the query to guide context selection"""
        query_lower = query.lower()
        
        # Extract potential symbols (camelCase, snake_case, etc.)
        symbol_patterns = [
            r'\b[a-z][a-zA-Z0-9_]*[A-Z][a-zA-Z0-9_]*\b',  # camelCase
            r'\b[a-zA-Z][a-zA-Z0-9_]*_[a-zA-Z0-9_]+\b',   # snake_case
            r'\b[A-Z][a-zA-Z0-9_]*\b',                     # PascalCase
        ]
        
        symbols = set()
        for pattern in symbol_patterns:
            symbols.update(re.findall(pattern, query))
        
        # Extract action keywords
        action_keywords = {
            'create': ['create', 'add', 'implement', 'build', 'make', 'generate'],
            'modify': ['modify', 'change', 'update', 'edit', 'alter', 'refactor'],
            'debug': ['debug', 'fix', 'error', 'bug', 'issue', 'problem'],
            'analyze': ['analyze', 'understand', 'explain', 'how', 'what', 'why'],
            'test': ['test', 'testing', 'unit test', 'integration'],
            'optimize': ['optimize', 'performance', 'speed', 'memory', 'efficient']
        }
        
        detected_actions = set()
        for action, keywords in action_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                detected_actions.add(action)
        
        # Extract file type hints
        file_extensions = re.findall(r'\.([a-zA-Z0-9]+)', query)
        
        # Extract domain-specific terms
        domain_terms = {
            'web': ['html', 'css', 'javascript', 'react', 'angular', 'vue', 'dom'],
            'backend': ['api', 'server', 'database', 'sql', 'rest', 'graphql'],
            'data': ['data', 'dataset', 'analysis', 'pandas', 'numpy', 'sql'],
            'ml': ['machine learning', 'model', 'training', 'prediction', 'tensorflow', 'pytorch'],
            'mobile': ['android', 'ios', 'mobile', 'app', 'kotlin', 'swift'],
            'devops': ['docker', 'kubernetes', 'ci/cd', 'deployment', 'infrastructure']
        }
        
        detected_domains = set()
        for domain, terms in domain_terms.items():
            if any(term in query_lower for term in terms):
                detected_domains.add(domain)
        
        return {
            'symbols': list(symbols),
            'actions': list(detected_actions),
            'file_extensions': file_extensions,
            'domains': list(detected_domains),
            'query_terms': query_lower.split(),
            'query_length': len(query.split())
        }
    
    def _create_file_context_item(
        self,
        file_path: str,
        analysis: FileAnalysis,
        query_features: Dict[str, Any],
        base_score: float
    ) -> Optional[ContextItem]:
        """Create a file-level context item"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception:
            return None
        
        # Calculate relevance score
        relevance_score = base_score
        
        # Boost score for file extension matches
        file_ext = os.path.splitext(file_path)[1][1:]  # Remove the dot
        if file_ext in query_features['file_extensions']:
            relevance_score *= 1.5
        
        # Boost score for domain matches
        file_content_lower = content.lower()
        for domain in query_features['domains']:
            domain_terms = {
                'web': ['html', 'css', 'javascript', 'react', 'angular', 'vue'],
                'backend': ['api', 'server', 'database', 'sql', 'rest'],
                'data': ['pandas', 'numpy', 'matplotlib', 'seaborn'],
                'ml': ['tensorflow', 'pytorch', 'sklearn', 'model'],
            }.get(domain, [])
            
            if any(term in file_content_lower for term in domain_terms):
                relevance_score *= 1.3
        
        # Get importance score from dependency graph
        importance_score = 1.0
        if file_path in self.dependency_graph.nodes:
            importance_score = self.dependency_graph.nodes[file_path].importance_score
        
        # Boost importance for direct file queries (when filename is mentioned in query)
        file_name = os.path.basename(file_path).lower()
        file_name_no_ext = os.path.splitext(file_name)[0]
        query_terms_lower = [term.lower() for term in query_features['query_terms']]
        
        if (file_name in query_terms_lower or 
            file_name_no_ext in query_terms_lower or
            any(term.endswith('.py') and term == file_name for term in query_terms_lower)):
            # This is a direct file query - boost importance significantly
            importance_score = max(importance_score, 5.0)  # Ensure high importance for direct queries
        
        return ContextItem(
            file_path=file_path,
            content=content,
            relevance_score=relevance_score,
            importance_score=importance_score,
            context_type='file',
            metadata={
                'language': analysis.language,
                'symbol_count': len(analysis.symbols),
                'import_count': len(analysis.imports),
                'line_count': len(content.split('\n'))
            }
        )
    
    def _create_symbol_context_items(
        self,
        file_path: str,
        analysis: FileAnalysis,
        query_features: Dict[str, Any],
        base_score: float,
        context_types: List[str]
    ) -> List[ContextItem]:
        """Create symbol-level context items (functions, classes)"""
        items = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
        except Exception:
            return items
        
        for symbol in analysis.symbols:
            if symbol.type not in context_types:
                continue
            
            # Extract symbol content
            start_line = max(0, symbol.start_line - 1)
            end_line = min(len(lines), symbol.end_line)
            symbol_content = '\n'.join(lines[start_line:end_line])
            
            # Calculate relevance score
            relevance_score = self._calculate_symbol_relevance(symbol, query_features, base_score)
            
            # Get importance score from dependency graph
            symbol_key = f"{file_path}::{symbol.name}"
            importance_score = 1.0
            if symbol_key in self.dependency_graph.nodes:
                importance_score = self.dependency_graph.nodes[symbol_key].importance_score
            
            # Get dependencies
            dependencies = set()
            if symbol_key in self.dependency_graph.nodes:
                dependencies = self.dependency_graph.nodes[symbol_key].dependencies
            
            item = ContextItem(
                file_path=file_path,
                content=symbol_content,
                relevance_score=relevance_score,
                importance_score=importance_score,
                context_type=symbol.type,
                start_line=symbol.start_line,
                end_line=symbol.end_line,
                symbol_name=symbol.name,
                dependencies=dependencies,
                metadata={
                    'scope': symbol.scope,
                    'parameters': symbol.parameters,
                    'docstring': symbol.docstring,
                    'return_type': symbol.return_type
                }
            )
            items.append(item)
        
        return items
    
    def _calculate_symbol_relevance(
        self,
        symbol,
        query_features: Dict[str, Any],
        base_score: float
    ) -> float:
        """Calculate relevance score for a symbol"""
        score = base_score * 0.1  # Start with a fraction of file score
        
        # Exact symbol name matches
        if symbol.name in query_features['symbols']:
            score += 5.0
        
        # Fuzzy symbol name matches
        symbol_lower = symbol.name.lower()
        for query_symbol in query_features['symbols']:
            if query_symbol.lower() in symbol_lower or symbol_lower in query_symbol.lower():
                score += 2.0
        
        # Query term matches in symbol name
        for term in query_features['query_terms']:
            if term in symbol_lower:
                score += 1.0
        
        # Action-based scoring
        if 'create' in query_features['actions'] and symbol.type == 'class':
            score *= 1.2
        elif 'modify' in query_features['actions'] and symbol.type == 'function':
            score *= 1.2
        elif 'test' in query_features['actions'] and 'test' in symbol_lower:
            score *= 1.5
        
        # Boost for symbols with docstrings (likely important)
        if symbol.docstring:
            score *= 1.1
        
        # Boost for public symbols (not starting with _)
        if not symbol.name.startswith('_'):
            score *= 1.05
        
        return score
    
    def _add_dependency_context(
        self,
        existing_items: List[ContextItem],
        query_features: Dict[str, Any],
        max_dependencies: int
    ) -> List[ContextItem]:
        """Add context from dependencies of existing items"""
        dependency_items = []
        added_files = set(item.file_path for item in existing_items)
        
        # Collect all dependencies
        all_dependencies = set()
        for item in existing_items:
            all_dependencies.update(item.dependencies)
        
        # Score and rank dependencies
        dependency_scores = []
        for dep_key in all_dependencies:
            if dep_key in self.dependency_graph.nodes:
                dep_node = self.dependency_graph.nodes[dep_key]
                if dep_node.file_path not in added_files:
                    score = dep_node.importance_score
                    dependency_scores.append((dep_key, score))
        
        # Sort by score and take top dependencies
        dependency_scores.sort(key=lambda x: x[1], reverse=True)
        
        for dep_key, score in dependency_scores[:max_dependencies]:
            dep_node = self.dependency_graph.nodes[dep_key]
            
            if dep_node.type == 'file':
                # Add file dependency
                if dep_node.file_path in self.dependency_graph.file_analyses:
                    analysis = self.dependency_graph.file_analyses[dep_node.file_path]
                    dep_item = self._create_file_context_item(
                        dep_node.file_path, analysis, query_features, score * 0.5
                    )
                    if dep_item:
                        dependency_items.append(dep_item)
                        added_files.add(dep_node.file_path)
        
        return dependency_items
    
    def _filter_by_token_budget(
        self,
        items: List[ContextItem],
        max_tokens: int
    ) -> List[ContextItem]:
        """Filter context items to fit within token budget"""
        filtered_items = []
        current_tokens = 0
        
        # Sort by combined score (relevance * importance)
        sorted_items = sorted(
            items,
            key=lambda x: x.relevance_score * x.importance_score,
            reverse=True
        )
        
        for item in sorted_items:
            item_tokens = self._estimate_tokens(item.content)
            
            if current_tokens + item_tokens <= max_tokens:
                filtered_items.append(item)
                current_tokens += item_tokens
            else:
                # Try to include partial content for files
                if item.context_type == 'file' and current_tokens < max_tokens * 0.9:
                    remaining_tokens = max_tokens - current_tokens
                    partial_content = self._truncate_content(item.content, remaining_tokens)
                    if partial_content:
                        item.content = partial_content
                        filtered_items.append(item)
                        break
        
        return filtered_items
    
    def _estimate_tokens(self, content: str) -> int:
        """Estimate token count for content"""
        # Rough estimation: 1 token ≈ 4 characters
        return max(1, len(content) // 4)
    
    def _truncate_content(self, content: str, max_tokens: int) -> Optional[str]:
        """Truncate content to fit within token budget"""
        max_chars = max_tokens * 4
        if len(content) <= max_chars:
            return content
        
        # Try to truncate at natural boundaries (functions, classes)
        lines = content.split('\n')
        truncated_lines = []
        current_chars = 0
        
        for line in lines:
            if current_chars + len(line) + 1 <= max_chars:
                truncated_lines.append(line)
                current_chars += len(line) + 1
            else:
                break
        
        if truncated_lines:
            return '\n'.join(truncated_lines) + '\n# ... (truncated)'
        
        return None
    
    def _calculate_query_coverage(
        self,
        items: List[ContextItem],
        query_features: Dict[str, Any]
    ) -> float:
        """Calculate how well the context covers the query"""
        if not query_features['query_terms']:
            return 1.0
        
        # Combine all content
        all_content = ' '.join(item.content.lower() for item in items)
        
        # Check coverage of query terms
        covered_terms = 0
        for term in query_features['query_terms']:
            if term in all_content:
                covered_terms += 1
        
        # Check coverage of symbols
        covered_symbols = 0
        for symbol in query_features['symbols']:
            if symbol.lower() in all_content:
                covered_symbols += 1
        
        # Calculate overall coverage
        total_terms = len(query_features['query_terms']) + len(query_features['symbols'])
        if total_terms == 0:
            return 1.0
        
        coverage = (covered_terms + covered_symbols) / total_terms
        return min(1.0, coverage)
    
    def get_context_summary(self, ranked_context: RankedContext) -> Dict[str, Any]:
        """Get a summary of the ranked context"""
        files_by_type = defaultdict(int)
        symbols_by_type = defaultdict(int)
        languages = set()
        
        for item in ranked_context.items:
            if item.context_type == 'file':
                files_by_type[item.metadata.get('language', 'unknown')] += 1
                languages.add(item.metadata.get('language', 'unknown'))
            else:
                symbols_by_type[item.context_type] += 1
        
        return {
            'total_items': len(ranked_context.items),
            'total_score': ranked_context.total_score,
            'token_estimate': ranked_context.token_estimate,
            'query_coverage': ranked_context.query_coverage,
            'files_by_language': dict(files_by_type),
            'symbols_by_type': dict(symbols_by_type),
            'languages': list(languages),
            'top_items': [
                {
                    'file': item.file_path,
                    'type': item.context_type,
                    'symbol': item.symbol_name,
                    'relevance': item.relevance_score,
                    'importance': item.importance_score
                }
                for item in ranked_context.items[:5]
            ]
        } 
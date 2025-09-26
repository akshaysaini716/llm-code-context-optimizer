"""
Multi-Stage Retrieval System
Enhances your existing retrieval with intelligent routing and multiple retrieval strategies
Works with your current Enhanced RAG system components
"""
import logging
import re
from typing import List, Dict, Optional, Set
from dataclasses import dataclass
from enum import Enum

from rag.models import RetrievalResult, CodeBaseChunk
from rag.retrieval.enhanced_retriever import EnhancedContextRetriever
from rag.vector_store.vector_store_factory import get_vector_store
from rag.configs import VectorStoreConfig

logger = logging.getLogger(__name__)

class QueryType(Enum):
    SYMBOL_LOOKUP = "symbol_lookup"        # "DatabaseManager", "authenticate_user"
    IMPLEMENTATION = "implementation"       # "how does authentication work"
    DEFINITION = "definition"              # "what is UserService"
    USAGE_EXAMPLES = "usage_examples"      # "examples of using Redis"
    ERROR_CONTEXT = "error_context"        # "why is this failing"
    EXPLORATION = "exploration"            # "related to payment processing"

@dataclass
class QueryAnalysis:
    """Analysis result for a query"""
    original_query: str
    query_type: QueryType
    symbols_mentioned: List[str]
    confidence: float
    should_prioritize_exact: bool
    should_expand_context: bool
    should_include_usage: bool

class MultiStageRetriever:
    """
    Multi-stage retrieval that combines different strategies:
    
    Stage 1: Query Analysis - understand what user wants
    Stage 2: Exact Match Search - find exact symbol/name matches  
    Stage 3: Semantic Search - find conceptually similar code
    Stage 4: Context Expansion - get surrounding relevant code
    Stage 5: Result Fusion & Ranking - combine and rank all results
    """
    
    def __init__(self, vector_store_config: Optional[VectorStoreConfig] = None):
        self.enhanced_retriever = EnhancedContextRetriever(vector_store_config=vector_store_config)
        self.vector_store = get_vector_store(vector_store_config)
        
        # Pre-compiled patterns for efficiency
        self.symbol_patterns = [
            r'\b[A-Z][a-zA-Z0-9]*\b',        # CamelCase (DatabaseManager)
            r'\b[a-z_][a-z0-9_]*[a-z0-9]\b', # snake_case (user_service)
            r'\b[a-zA-Z_][a-zA-Z0-9_]*\(\)', # Function calls (getData())
        ]
        
        self.query_type_indicators = {
            QueryType.DEFINITION: [
                r'\bwhat\s+is\b', r'\bdefine\b', r'\bclass\s+\w+', r'\binterface\s+\w+'
            ],
            QueryType.IMPLEMENTATION: [
                r'\bhow\s+does\b', r'\bhow\s+to\b', r'\bimplementation\b', r'\bworks?\b', r'\bprocess\b'
            ],
            QueryType.USAGE_EXAMPLES: [
                r'\bexample\b', r'\busage\b', r'\bhow\s+to\s+use\b', r'\bcall\b', r'\binvoke\b'
            ],
            QueryType.ERROR_CONTEXT: [
                r'\berror\b', r'\bbug\b', r'\bissue\b', r'\bproblem\b', r'\bfail\b', r'\bexception\b'
            ],
            QueryType.EXPLORATION: [
                r'\brelated\b', r'\bsimilar\b', r'\blike\b', r'\bconnected\b', r'\baround\b'
            ],
        }
    
    def retrieve(self, query: str, top_k: int = 10, project_path: Optional[str] = None) -> List[RetrievalResult]:
        """
        Main multi-stage retrieval method with project context
        """
        logger.info(f"Multi-stage retrieval for: '{query}' in project: {project_path}")
        
        # Stage 1: Analyze the query
        analysis = self._analyze_query(query)
        logger.debug(f"Query analysis: {analysis.query_type.value}, confidence: {analysis.confidence:.2f}")
        
        # Stage 2: Execute multi-stage search strategy
        all_results = []
        
        # Stage 2a: Exact Symbol/Name Matching (if symbols detected)
        if analysis.symbols_mentioned and analysis.should_prioritize_exact:
            exact_results = self._exact_symbol_search(analysis.symbols_mentioned, top_k // 2, project_path)
            all_results.extend(exact_results)
            logger.debug(f"Exact search found {len(exact_results)} results")
        
        # Stage 2b: Enhanced Semantic Search (your existing system)
        semantic_k = top_k if not all_results else max(5, top_k - len(all_results))
        semantic_results = self._semantic_search(query, semantic_k, analysis, project_path)
        all_results.extend(semantic_results)
        logger.debug(f"Semantic search found {len(semantic_results)} results")
        
        # Stage 2c: Context Expansion (for implementation queries)
        if analysis.should_expand_context and len(all_results) < top_k:
            context_results = self._expand_context_search(query, analysis, top_k - len(all_results), project_path)
            all_results.extend(context_results)
            logger.debug(f"Context expansion found {len(context_results)} results")
        
        # Stage 3: Deduplicate and fuse results
        fused_results = self._fuse_results(all_results)
        
        # Stage 4: Re-rank based on query analysis
        final_results = self._rerank_for_query_type(fused_results, analysis)
        
        logger.info(f"Multi-stage retrieval returned {len(final_results[:top_k])} results")
        return final_results[:top_k]
    
    def _analyze_query(self, query: str) -> QueryAnalysis:
        """Analyze query to understand user intent"""
        query_lower = query.lower()
        
        # Detect query type
        detected_type = QueryType.IMPLEMENTATION  # default
        max_score = 0
        
        for query_type, patterns in self.query_type_indicators.items():
            score = sum(1 for pattern in patterns if re.search(pattern, query_lower))
            if score > max_score:
                max_score = score
                detected_type = query_type
        
        # Special case: short queries with symbols are likely lookups
        if len(query.split()) <= 3:
            detected_type = QueryType.SYMBOL_LOOKUP
            
        # Extract symbols mentioned in query
        symbols = []
        for pattern in self.symbol_patterns:
            matches = re.findall(pattern, query)
            symbols.extend(matches)
        
        # Remove common words that match patterns
        common_words = {'the', 'and', 'for', 'are', 'with', 'this', 'that', 'from', 'they', 'have', 'been'}
        symbols = [s for s in symbols if s.lower() not in common_words and len(s) > 2]
        symbols = list(set(symbols))  # deduplicate
        
        # Calculate confidence
        confidence = 0.5  # base confidence
        if max_score > 0:
            confidence += 0.2  # boost for clear intent patterns
        if symbols:
            confidence += 0.2  # boost for detected symbols
        if len(query.split()) <= 5:
            confidence += 0.1  # boost for specific queries
        
        confidence = min(1.0, confidence)
        
        return QueryAnalysis(
            original_query=query,
            query_type=detected_type,
            symbols_mentioned=symbols,
            confidence=confidence,
            should_prioritize_exact=(detected_type in [QueryType.SYMBOL_LOOKUP, QueryType.DEFINITION] and len(symbols) > 0),
            should_expand_context=(detected_type in [QueryType.IMPLEMENTATION, QueryType.EXPLORATION]),
            should_include_usage=(detected_type == QueryType.USAGE_EXAMPLES)
        )
    
    def _exact_symbol_search(self, symbols: List[str], max_results: int, project_path: Optional[str] = None) -> List[RetrievalResult]:
        """Search for exact symbol matches using content filtering"""
        results = []
        
        for symbol in symbols[:3]:  # Limit to top 3 symbols
            try:
                # Use semantic search but with exact symbol as query
                # This leverages your existing vector search but focuses on the symbol
                symbol_results = self.enhanced_retriever.retrieve_with_context(
                    query=symbol,
                    top_k=max_results // len(symbols) + 2,
                    include_related=False,  # Don't expand yet
                    expand_context=False,
                    project_path=project_path
                )
                
                # Boost scores for exact content matches
                for result in symbol_results:
                    content_lower = result.chunk.content.lower()
                    symbol_lower = symbol.lower()
                    
                    # Exact name match gets high boost
                    if f"class {symbol_lower}" in content_lower or f"def {symbol_lower}" in content_lower:
                        result.relevance_score *= 1.5
                        results.append(result)
                    # Symbol appears in content gets moderate boost  
                    elif symbol_lower in content_lower:
                        result.relevance_score *= 1.2
                        results.append(result)
                        
            except Exception as e:
                logger.warning(f"Error in exact symbol search for '{symbol}': {e}")
        
        # Sort by boosted scores
        results.sort(key=lambda r: r.relevance_score, reverse=True)
        return results[:max_results]
    
    def _semantic_search(self, query: str, top_k: int, analysis: QueryAnalysis, project_path: Optional[str] = None) -> List[RetrievalResult]:
        """Enhanced semantic search using your existing retriever"""
        try:
            # Customize search parameters based on query analysis
            include_related = analysis.should_expand_context
            expand_context = analysis.query_type != QueryType.SYMBOL_LOOKUP
            
            results = self.enhanced_retriever.retrieve_with_context(
                query=query,
                top_k=top_k,
                include_related=include_related,
                expand_context=expand_context,
                project_path=project_path
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            return []
    
    def _expand_context_search(self, query: str, analysis: QueryAnalysis, max_results: int, project_path: Optional[str] = None) -> List[RetrievalResult]:
        """Search for broader context around the query topic"""
        if not analysis.should_expand_context:
            return []
        
        try:
            # Create broader search terms
            broader_queries = self._generate_broader_queries(query, analysis)
            results = []
            
            for broader_query in broader_queries[:2]:  # Limit to 2 broader searches
                broader_results = self.enhanced_retriever.retrieve_with_context(
                    query=broader_query,
                    top_k=max_results // 2,
                    include_related=True,
                    expand_context=True,
                    project_path=project_path
                )
                
                # Lower scores since these are broader matches
                for result in broader_results:
                    result.relevance_score *= 0.7
                
                results.extend(broader_results)
            
            return results[:max_results]
            
        except Exception as e:
            logger.warning(f"Error in context expansion: {e}")
            return []
    
    def _generate_broader_queries(self, query: str, analysis: QueryAnalysis) -> List[str]:
        """Generate broader search queries for context expansion"""
        broader_queries = []
        
        # Extract key concepts
        words = query.lower().split()
        key_words = [w for w in words if len(w) > 3 and w not in {'how', 'does', 'what', 'the', 'and', 'with', 'this'}]
        
        if analysis.query_type == QueryType.IMPLEMENTATION:
            for word in key_words[:2]:
                broader_queries.append(f"{word} implementation")
                broader_queries.append(f"{word} process")
        
        elif analysis.query_type == QueryType.USAGE_EXAMPLES:
            for word in key_words[:2]:
                broader_queries.append(f"using {word}")
                broader_queries.append(f"{word} example")
        
        return broader_queries
    
    def _fuse_results(self, all_results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Deduplicate and fuse results from different stages"""
        seen_chunks = {}
        chunk_discovery_counts = {}  # Track how many times each chunk was found
        fused_results = []
        
        for result in all_results:
            chunk_id = result.chunk.id
            
            if chunk_id not in seen_chunks:
                # First time seeing this chunk
                seen_chunks[chunk_id] = result
                chunk_discovery_counts[chunk_id] = 1
                fused_results.append(result)
            else:
                # Seen before - combine scores intelligently
                existing_result = seen_chunks[chunk_id]
                chunk_discovery_counts[chunk_id] += 1
                
                # Use weighted average of scores with boost for multiple discoveries
                current_score = existing_result.relevance_score
                new_score = result.relevance_score
                
                # Weighted combination favoring higher scores
                combined_score = (current_score + new_score) / 2
                
                # Apply multi-discovery boost (more discoveries = higher relevance)
                discovery_boost = 1.0 + (chunk_discovery_counts[chunk_id] - 1) * 0.15
                existing_result.relevance_score = combined_score * discovery_boost
        
        # Sort by final scores
        fused_results.sort(key=lambda x: x.relevance_score, reverse=True)
        return fused_results
    
    def _rerank_for_query_type(self, results: List[RetrievalResult], analysis: QueryAnalysis) -> List[RetrievalResult]:
        """Re-rank results based on query type and analysis"""
        
        for result in results:
            chunk = result.chunk
            base_score = result.relevance_score
            
            # Query-type specific boosts
            if analysis.query_type == QueryType.DEFINITION:
                if chunk.chunk_type == "class_complete":
                    base_score *= 1.3
                elif chunk.chunk_type in ["class_definition", "interface"]:
                    base_score *= 1.2
            
            elif analysis.query_type == QueryType.IMPLEMENTATION:
                if chunk.chunk_type in ["function_definition", "method"]:
                    base_score *= 1.15
                elif chunk.chunk_type == "class_complete":
                    base_score *= 1.1
            
            elif analysis.query_type == QueryType.USAGE_EXAMPLES:
                # Boost chunks that look like they have examples/calls
                content_lower = chunk.content.lower()
                if any(word in content_lower for word in ['example', 'test', 'demo', 'sample']):
                    base_score *= 1.2
            
            # Symbol mention boost
            for symbol in analysis.symbols_mentioned:
                if symbol.lower() in chunk.content.lower():
                    base_score *= 1.1
                    break
            
            # File type preferences  
            if chunk.file_path.endswith('.py'):
                if analysis.query_type != QueryType.USAGE_EXAMPLES or 'test' not in chunk.file_path:
                    base_score *= 1.05  # Slight preference for Python files
            
            result.relevance_score = base_score
        
        # Final sort by adjusted scores
        results.sort(key=lambda r: r.relevance_score, reverse=True)
        return results
    
    def get_query_analysis(self, query: str) -> QueryAnalysis:
        """Public method to get query analysis without retrieval"""
        return self._analyze_query(query)

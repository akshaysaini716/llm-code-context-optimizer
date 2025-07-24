import os
from typing import Optional, Dict, Any, List
from utils.file_utils import read_code_files
from .dependency_graph import DependencyGraph
from .context_ranker import ContextRanker

# Global cache for dependency graphs (in production, use Redis or similar)
_graph_cache: Dict[str, DependencyGraph] = {}

def get_project_context(
    project_path: str, 
    query: str, 
    relevant_only: bool,
    max_tokens: int = 8000,
    use_tree_sitter: bool = True
):
    """
    Get project context using either tree-sitter analysis or simple file reading
    
    Args:
        project_path: Path to the project directory
        query: User query for context relevance
        relevant_only: Whether to filter for relevant files only
        max_tokens: Maximum tokens to include in context
        use_tree_sitter: Whether to use tree-sitter based analysis
    """
    

    
    if not use_tree_sitter:
        # Use simple approach when tree-sitter is disabled
        return _get_simple_context(project_path, query, relevant_only)
    

    
    try:
        # Try tree-sitter approach
        graph = _get_or_create_graph(project_path)
        
        # Check if tree-sitter analysis worked
        if not graph.file_analyses:
            print(f"Warning: Tree-sitter failed, falling back to simple mode")
            return _get_simple_context(project_path, query, relevant_only)
        
        print(f"🎉 Tree-sitter analysis successful - {len(graph.file_analyses)} files analyzed")
        
        # Use context ranker for intelligent context selection
        ranker = ContextRanker(graph)
        
        if relevant_only:
            # Check if this is a direct file query (mentions specific filename)
            query_lower = query.lower()
            is_direct_file_query = any(
                filename in query_lower 
                for filename in ['.py', '.js', '.ts', '.java', '.kt']
            ) and ('content' in query_lower or 'file' in query_lower)
            
            if is_direct_file_query:
                # Very focused mode for direct file queries
                ranked_context = ranker.rank_context(
                    query=query,
                    max_tokens=max_tokens // 4,  # Use even fewer tokens for direct file queries
                    max_files=2,  # Only 1-2 files for direct queries
                    include_dependencies=False,
                    context_types=['file']
                )
                mode_info = "tree_sitter_direct_file"
            else:
                # Regular relevant mode - focused, fewer results
                ranked_context = ranker.rank_context(
                    query=query,
                    max_tokens=max_tokens // 2,  # Use half the tokens for relevant mode
                    max_files=8,  # Fewer files for relevant mode  
                    include_dependencies=False,  # Don't include dependencies
                    context_types=['file']  # Only file-level context, no functions/classes
                )
                mode_info = "tree_sitter_relevant"
        else:
            # Full context mode - include ALL files from the project
            all_files = graph.get_all_files_with_importance()
            ranked_context = ranker.rank_full_context(
                all_files=all_files,
                max_tokens=max_tokens  # Use full token budget
            )
            mode_info = "tree_sitter_full"
        
        # Convert to expected format
        files = []
        for item in ranked_context.items:
            files.append({
                "file": item.file_path,
                "content": item.content,
                "relevance_score": item.relevance_score,
                "importance_score": item.importance_score,
                "context_type": item.context_type,
                "symbol_name": item.symbol_name,
                "start_line": item.start_line,
                "end_line": item.end_line
            })
        
        print(f"🚀 Tree-sitter mode '{mode_info}' returning {len(files)} files, {ranked_context.token_estimate} tokens")
        
        return {
            "mode": mode_info,
            "files": files,
            "metadata": {
                "total_score": ranked_context.total_score,
                "token_estimate": ranked_context.token_estimate,
                "query_coverage": ranked_context.query_coverage,
                "files_analyzed": len(graph.file_analyses),
                "symbols_found": sum(len(analysis.symbols) for analysis in graph.file_analyses.values()),
                "graph_stats": graph.get_graph_stats(),
                "relevant_only": relevant_only
            }
        }
        
    except Exception as e:
        print(f"Tree-sitter analysis failed: {e}")
        # Fall back to simple approach
        return _get_simple_context(project_path, query, relevant_only)

def _get_or_create_graph(project_path: str) -> DependencyGraph:
    """Get cached dependency graph or create new one"""
    abs_path = os.path.abspath(project_path)
    
    # Check if we have a cached graph
    if abs_path in _graph_cache:
        return _graph_cache[abs_path]
    
    # Create new graph
    graph = DependencyGraph()
    graph.build_graph(project_path)
    
    # Cache it (in production, implement proper cache invalidation)
    _graph_cache[abs_path] = graph
    
    return graph

def _get_simple_context(project_path: str, query: str, relevant_only: bool):
    """Original simple context extraction (fallback)"""
    all_files = read_code_files(project_path)
    


    if not relevant_only:
        # FULL CONTEXT MODE - return ALL files
        files = [{"file": f, "content": content} for f, content in all_files.items()]
        total_tokens = sum(len(content) // 4 for content in all_files.values())  # Rough estimate

        return {
            "mode": "simple_full",
            "files": files,
            "metadata": {
                "token_estimate": total_tokens,
                "total_files": len(files),
                "relevant_only": False
            }
        }

    # RELEVANT CONTEXT MODE - filter by query terms
    relevant_files = []
    query_terms = query.lower().split()
    
    for fpath, content in all_files.items():
        content_lower = content.lower()
        # Check if any query term appears in the file content
        matching_terms = [term for term in query_terms if term in content_lower]
        if matching_terms:
            relevant_files.append({"file": fpath, "content": content})

    total_tokens = sum(len(f["content"]) // 4 for f in relevant_files)  # Rough estimate
    return {
        "mode": "simple_relevant",
        "files": relevant_files,
        "metadata": {
            "token_estimate": total_tokens,
            "total_files": len(relevant_files),
            "relevant_only": True
        }
    }

def clear_graph_cache():
    """Clear the dependency graph cache"""
    global _graph_cache
    _graph_cache.clear()

def get_graph_stats(project_path: str) -> Optional[Dict[str, Any]]:
    """Get statistics about the dependency graph for a project"""
    try:
        graph = _get_or_create_graph(project_path)
        return graph.get_graph_stats()
    except Exception as e:
        print(f"Error getting graph stats: {e}")
        return None

def find_related_symbols(project_path: str, symbol_name: str, max_results: int = 20) -> List[Dict[str, Any]]:
    """Find symbols related to the given symbol"""
    try:
        graph = _get_or_create_graph(project_path)
        related = graph.get_related_symbols(symbol_name, max_results)
        
        result = []
        for symbol_key, score in related:
            if symbol_key in graph.nodes:
                node = graph.nodes[symbol_key]
                result.append({
                    "symbol_name": node.name,
                    "symbol_type": node.type,
                    "file_path": node.file_path,
                    "score": score,
                    "metadata": node.metadata
                })
        
        return result
    except Exception as e:
        print(f"Error finding related symbols: {e}")
        return []


from .embedding_service import EmbeddingService


class ContextService:
    def __init__(self):
        self.embedding_service = EmbeddingService()

    def extract_code_chunks(self, request):
        """Extract code chunks from codebase using existing functions"""
        code_chunks = []

        # Use existing get_project_context function
        context_data = get_project_context(
            project_path=request.project_path,
            query=request.message,
            relevant_only=request.relevant_only,
            use_tree_sitter=True
        )

        for file_info in context_data.get('files', []):
            print(f"file_info keys: {file_info.keys()}")
            file_path = file_info['file']

            # Extract functions and classes
            for func in file_info.get('functions', []):
                code_chunks.append({
                    'content': func.get('code', ''),
                    'file_path': file_path,
                    'type': 'function',
                    'name': func.get('name', ''),
                    'line_start': func.get('line_start', 0)
                })

            for cls in file_info.get('classes', []):
                code_chunks.append({
                    'content': cls.get('code', ''),
                    'file_path': file_path,
                    'type': 'class',
                    'name': cls.get('name', ''),
                    'line_start': cls.get('line_start', 0)
                })

        return code_chunks

    def filter_and_rank(self, similar_chunks, user_prompt):
        """Filter chunks by relevance and rank them"""
        # Handle different possible key names for similarity score
        filtered_chunks = []
        for chunk in similar_chunks:
            # Check various possible similarity score keys
            score = chunk.get('similarity_score', chunk.get('score', chunk.get('distance', 0)))
            if score > 0.3:
                filtered_chunks.append(chunk)

        return filtered_chunks

    def build_final_context(self, filtered_chunks, max_context_length):
        """Build final context within token limit"""
        context_parts = []
        total_length = 0

        for chunk in filtered_chunks:
            chunk_text = f"// {chunk['metadata']['file_path']} - {chunk['metadata']['type']}: {chunk['metadata']['name']}\n{chunk['content']}\n\n"
            if total_length + len(chunk_text) > max_context_length:
                break
            context_parts.append(chunk_text)
            total_length += len(chunk_text)

        return ''.join(context_parts)

    def build_semantic_index(self, request):
        """Build FAISS index from codebase"""
        # Use the tree-sitter analysis that's already working
        context_data = get_project_context(
            project_path=request.project_path,
            query=request.message,
            relevant_only=request.relevant_only,
            use_tree_sitter=True
        )

        print(f"Tree-sitter returned: {type(context_data)}")

        # Convert tree-sitter results to chunks format
        enhanced_chunks = []

        # Extract the actual files from the context_data
        if isinstance(context_data, dict) and 'files' in context_data:
            files = context_data['files']
            print(f"Processing {len(files)} files")

            for file_info in files:
                enhanced_chunks.append({
                    'content': file_info['content'],
                    'file_path': file_info['file'],
                    'type': 'code',
                    'name': file_info['file'].split('/')[-1],  # get filename
                    'line_start': 1
                })

        print(f"Created {len(enhanced_chunks)} enhanced chunks")
        return self.embedding_service.create_embeddings(enhanced_chunks)

    def get_relevant_context(self, user_prompt, max_context_length=8000):
        """Get most relevant code context for the prompt"""
        # Search for similar code
        similar_chunks = self.embedding_service.search_similar(user_prompt, k=10)

        # Filter and rank based on your existing logic
        filtered_chunks = self.filter_and_rank(similar_chunks, user_prompt)

        # Build final context within token limit
        final_context = self.build_final_context(filtered_chunks, max_context_length)

        return final_context
import logging
from typing import Optional, List, Dict, Tuple
import tiktoken
from pathlib import Path

from rag.configs import EmbeddingConfig
from rag.embedding.embedder import CodeEmbedder
from rag.embedding.enhanced_embedder import EnhancedCodeEmbedder
from rag.models import RetrievalResult, CodeBaseChunk
from rag.vector_store.qdrant_client_impl import QdrantClientImpl

logger = logging.getLogger(__name__)

class EnhancedContextRetriever:
    """Enhanced retriever with sliding window context expansion and relationship-aware retrieval"""
    
    def __init__(self, enable_context_expansion: bool = True, expansion_window: int = 300):
        self.qdrant_client = QdrantClientImpl()
        config = EmbeddingConfig(include_file_context=True)
        self.embedder = EnhancedCodeEmbedder(config)
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.enable_context_expansion = enable_context_expansion
        self.expansion_window = expansion_window  # tokens to expand on each side
        self.file_cache: Dict[str, str] = {}  # Cache for file contents

    def retrieve_with_context(
        self, 
        query: str, 
        top_k: int,
        include_related: bool = True,
        expand_context: bool = True,
        project_path: Optional[str] = None
    ) -> List[RetrievalResult]:
        """
        Retrieve chunks with enhanced context awareness
        
        Args:
            query: Search query
            top_k: Number of top results to return
            include_related: Include related chunks (parent, siblings, overlaps)
            expand_context: Apply sliding window expansion
        """
        try:
            # Step 1: Get initial results from vector search
            query_embedding = self.embedder.embed_query(query)
            if not query_embedding:
                return []

            initial_results = self.qdrant_client.search_similar(
                query_vector=query_embedding,
                project_path=project_path,
                filters=None,
                top_k=top_k * 2  # Get more for filtering
            )
            
            if not initial_results:
                return []
            
            # Step 2: Enhance results with relationships
            if include_related:
                initial_results = self._include_related_chunks(initial_results, top_k)
            
            # Step 3: Apply sliding window context expansion
            if expand_context and self.enable_context_expansion:
                initial_results = self._expand_context_windows(initial_results)
            
            # Step 4: Re-rank based on enhanced context
            final_results = self._rerank_results(initial_results, query)
            
            # Step 5: Return top-k results
            return final_results[:top_k]
            
        except Exception as e:
            logger.error(f"Error in enhanced retrieval: {e}")
            return []

    def _include_related_chunks(
        self, 
        results: List[RetrievalResult], 
        max_results: int
    ) -> List[RetrievalResult]:
        """Include related chunks based on relationships"""
        enhanced_results = []
        seen_chunk_ids = set()
        
        for result in results:
            # Add the main result
            if result.chunk.id not in seen_chunk_ids:
                enhanced_results.append(result)
                seen_chunk_ids.add(result.chunk.id)
            
            # Try to fetch related chunks
            related_chunks = self._fetch_related_chunks(result.chunk)
            
            for related_chunk, relationship_type in related_chunks:
                if related_chunk.id not in seen_chunk_ids:
                    # Adjust relevance score based on relationship
                    adjusted_score = self._adjust_relevance_score(
                        result.relevance_score, 
                        relationship_type
                    )
                    
                    related_result = RetrievalResult(
                        chunk=related_chunk,
                        relevance_score=adjusted_score
                    )
                    enhanced_results.append(related_result)
                    seen_chunk_ids.add(related_chunk.id)
                    
                    if len(enhanced_results) >= max_results * 1.5:
                        break
            
            if len(enhanced_results) >= max_results * 1.5:
                break
        
        return enhanced_results

    def _fetch_related_chunks(self, chunk: CodeBaseChunk) -> List[Tuple[CodeBaseChunk, str]]:
        """Fetch chunks related to the given chunk"""
        related = []
        
        # Simplified approach to avoid Qdrant filtering issues
        # We'll use vector similarity search with file-based filtering instead
        try:
            # Skip related chunk fetching for complete classes as they already contain everything
            if chunk.chunk_type == "class_complete":
                return related
            
            # For now, disable related chunk fetching to avoid the Qdrant issues
            # This can be re-enabled once the Qdrant client issues are resolved
            logger.debug(f"Skipping related chunk fetching for {chunk.id} due to Qdrant filtering issues")
            return related
            
        except Exception as e:
            logger.warning(f"Error fetching related chunks: {e}")
        
        return related

    def _adjust_relevance_score(self, base_score: float, relationship_type: str) -> float:
        """Adjust relevance score based on relationship type"""
        adjustments = {
            "parent": 0.9,      # Parent class is highly relevant
            "sibling": 0.7,     # Sibling chunks are moderately relevant
            "overlap": 0.8,     # Overlapping chunks are relevant
            "child": 0.85       # Child chunks are relevant
        }
        return base_score * adjustments.get(relationship_type, 0.5)

    def _expand_context_windows(self, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Expand context using sliding window approach"""
        expanded_results = []
        
        for result in results:
            chunk = result.chunk
            
            # Skip if already a large chunk
            token_count = len(self.tokenizer.encode(chunk.content))
            if token_count > 1500:
                expanded_results.append(result)
                continue
            
            try:
                # Get the original file content
                file_content = self._get_file_content(chunk.file_path)
                if not file_content:
                    expanded_results.append(result)
                    continue
                
                lines = file_content.split('\n')
                
                # Calculate expanded boundaries
                expand_lines_before = max(10, self.expansion_window // 20)  # Roughly estimate lines
                expand_lines_after = expand_lines_before
                
                start_line = max(1, chunk.start_line - expand_lines_before)
                end_line = min(len(lines), chunk.end_line + expand_lines_after)
                
                # Extract expanded content
                expanded_content = self._extract_expanded_content(
                    lines, 
                    start_line, 
                    end_line,
                    chunk.start_line,
                    chunk.end_line
                )
                
                # Create expanded chunk
                expanded_chunk = CodeBaseChunk(
                    id=f"{chunk.id}_expanded",
                    file_path=chunk.file_path,
                    content=expanded_content,
                    language=chunk.language,
                    chunk_type=f"{chunk.chunk_type}_expanded",
                    start_byte=chunk.start_byte,  # Keep original for reference
                    end_byte=chunk.end_byte,
                    start_line=start_line,
                    end_line=end_line
                )
                
                expanded_result = RetrievalResult(
                    chunk=expanded_chunk,
                    relevance_score=result.relevance_score
                )
                expanded_results.append(expanded_result)
                
            except Exception as e:
                logger.warning(f"Error expanding context for chunk {chunk.id}: {e}")
                expanded_results.append(result)
        
        return expanded_results

    def _get_file_content(self, file_path: str) -> Optional[str]:
        """Get file content with caching"""
        if file_path in self.file_cache:
            return self.file_cache[file_path]
        
        try:
            path = Path(file_path)
            if path.exists():
                content = path.read_text(encoding='utf-8', errors='ignore')
                self.file_cache[file_path] = content
                return content
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
        
        return None

    def _extract_expanded_content(
        self, 
        lines: List[str], 
        start_line: int, 
        end_line: int,
        original_start: int,
        original_end: int
    ) -> str:
        """Extract expanded content with markers"""
        expanded_lines = []
        
        # Add context before marker if we're adding context
        if start_line < original_start:
            expanded_lines.append("# === Context Before ===")
            expanded_lines.extend(lines[start_line - 1:original_start - 1])
            expanded_lines.append("# === Main Content ===")
        
        # Add main content
        expanded_lines.extend(lines[original_start - 1:original_end])
        
        # Add context after marker if we're adding context
        if end_line > original_end:
            expanded_lines.append("# === Context After ===")
            expanded_lines.extend(lines[original_end:end_line])
        
        return '\n'.join(expanded_lines)

    def _rerank_results(
        self, 
        results: List[RetrievalResult], 
        query: str
    ) -> List[RetrievalResult]:
        """Re-rank results based on enhanced scoring"""
        scored_results = []
        
        for result in results:
            chunk = result.chunk
            
            # Calculate various scoring factors
            base_score = result.relevance_score
            
            # Boost score for complete classes when query mentions class
            if "class" in query.lower() and chunk.chunk_type == "class_complete":
                base_score *= 1.2
            
            # Boost score for functions when query mentions function/method
            if any(term in query.lower() for term in ["function", "method", "def"]):
                if chunk.chunk_type in ["function_definition", "method"]:
                    base_score *= 1.15
            
            # Penalize overlapping chunks slightly
            if chunk.chunk_type == "overlap":
                base_score *= 0.9
            
            # Boost for exact name matches
            chunk_text_lower = chunk.content.lower()
            query_terms = query.lower().split()
            exact_matches = sum(1 for term in query_terms if term in chunk_text_lower)
            if exact_matches > 0:
                base_score *= (1 + 0.1 * exact_matches)
            
            # Store the adjusted score
            result.relevance_score = base_score
            scored_results.append(result)
        
        # Sort by final score
        scored_results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        return scored_results

    def retrieve_with_feedback(
        self,
        query: str,
        top_k: int,
        positive_feedback: List[str] = None,
        negative_feedback: List[str] = None
    ) -> List[RetrievalResult]:
        """
        Retrieve with relevance feedback
        
        Args:
            query: Search query
            top_k: Number of results
            positive_feedback: List of chunk IDs that were helpful
            negative_feedback: List of chunk IDs that were not helpful
        """
        results = self.retrieve_with_context(query, top_k * 2)
        
        if not positive_feedback and not negative_feedback:
            return results[:top_k]
        
        # Adjust scores based on feedback
        for result in results:
            if positive_feedback and result.chunk.id in positive_feedback:
                result.relevance_score *= 1.5
            elif negative_feedback and result.chunk.id in negative_feedback:
                result.relevance_score *= 0.5
        
        # Re-sort and return
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        return results[:top_k]

    def clear_cache(self):
        """Clear the file content cache"""
        self.file_cache.clear()

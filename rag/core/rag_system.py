import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
import hashlib
import time as time_module

from rag.chunking.enhanced_tree_sitter_chunker import EnhancedTreeSitterChunker
from rag.chunking.tree_sitter_chunker import TreeSitterChunker
from rag.configs import ChunkingConfig
from rag.embedding.embedder import CodeEmbedder
from rag.embedding.enhanced_embedder import EnhancedCodeEmbedder
from rag.models import RAGResponse
from rag.retrieval.context_fusion import ContextFusion
from rag.retrieval.enhanced_context_fusion import EnhancedContextFusion
from rag.retrieval.enhanced_multi_stage_retriever import EnhancedMultiStageRetriever
from rag.retrieval.enhanced_retriever import EnhancedContextRetriever
from rag.retrieval.multi_stage_retriever import MultiStageRetriever
from rag.retrieval.retriever import ContextRetriever
from rag.vector_store.vector_store_factory import get_vector_store
from rag.configs import VectorStoreConfig

logger = logging.getLogger(__name__)

class RAGSystem:

    def __init__(self, vector_store_config: Optional[VectorStoreConfig] = None):
        config = ChunkingConfig(preserve_class_methods=True)
        self.chunker = EnhancedTreeSitterChunker(config)
        self.embedder = EnhancedCodeEmbedder()
        self.vector_store = get_vector_store(vector_store_config)
        self.retriever = EnhancedContextRetriever(vector_store_config=vector_store_config)
        self.multi_stage_retriever = MultiStageRetriever(vector_store_config=vector_store_config)
        self.context_fuser = EnhancedContextFusion()
        
        # Session tracking for context expansion without chunk IDs
        self.session_cache: Dict[str, Dict[str, Any]] = {}
        self.max_session_cache_size = 100  # Prevent memory bloat

    def index_codebase(self, project_path, file_patterns, force_reindex) -> Dict[str, Any]:
        start_time = time.time()
        project_path = Path(project_path)
        file_patterns = file_patterns or ["*.py", "*.js", "*.ts", "*.java", "*.kt", "*.go", "*.scala"]
        
        # Handle force reindex - delete existing chunks for this project
        if force_reindex:
            logger.info(f"Force reindex requested - deleting existing chunks for project: {project_path}")
            deletion_success = self.vector_store.delete_chunks_by_project_path(str(project_path))
            if deletion_success:
                logger.info("Successfully deleted existing chunks")
            else:
                logger.warning("Failed to delete existing chunks, but continuing with indexing")
        
        all_files = []
        for pattern in file_patterns:
            files = list(project_path.rglob(pattern))
            all_files.extend(files)
        logger.info(f"Found {len(all_files)} code files to index")

        # Chunking the code files with original project path
        all_chunks = self.chunker.chunk_files_parallel(all_files, str(project_path))
        logger.info(f"Generated {len(all_chunks)} chunks")

        # Creating the embeddings
        chunk_relationships = self.chunker.chunk_relationships
        chunks_with_embeddings = self.embedder.embed_chunks(all_chunks, chunk_relationships=chunk_relationships)
        embedded_count = sum(1 for chunk in chunks_with_embeddings if chunk.embedding)
        logger.info(f"Created embeddings for {embedded_count}/{len(all_chunks)} chunks")
        logger.info(f"Chunk relationships: {len(chunk_relationships)}")
        relationship_stats = {
            "chunks_with_parents": sum(1 for rel in chunk_relationships.values() if rel.parent_id),
            "chunks_with_siblings": sum(1 for rel in chunk_relationships.values() if rel.sibling_ids),
            "chunks_with_overlaps": sum(1 for rel in chunk_relationships.values() if rel.overlaps_with)
        }
        logger.info(f"Relationship stats: {relationship_stats}")

        # Store the embeddings in vector DB
        success = self.vector_store.upsert_chunks(chunks_with_embeddings)
        if not success:
            raise Exception("Failed to upsert chunks into vector DB")

        indexing_time = time.time() - start_time

        # TODO: Add the metrics here

        # Validate chunk quality
        quality_metrics = self._validate_chunk_quality(all_chunks)
        
        return {
            "status": "success",
            "files_processed": len(all_files),
            "chunks_created": len(all_chunks),
            "chunks_embedded": embedded_count,
            "indexing_time_seconds": indexing_time,
            "collection_info": self.vector_store.get_collection_info(),
            "chunk_quality_metrics": quality_metrics
        }

    def query(
        self, 
        query: str, 
        project_path: Optional[str] = None, 
        max_context_tokens: int = 8000, 
        top_k: int = 10,
        expand_window: bool = False,
        expansion_level: str = "moderate",
        previous_chunk_ids: Optional[List[str]] = None,
        session_id: Optional[str] = None
    ) -> RAGResponse:
        start_time = time.time()
        
        # Generate session tracking
        query_signature = self._generate_query_signature(query, project_path)
        if not session_id:
            session_id = query_signature[:16]  # Use query signature as default session
        
        # Get previous chunks from session cache if available
        session_previous_chunks = None
        if expand_window and session_id in self.session_cache:
            session_data = self.session_cache[session_id]
            if self._is_similar_query(query, session_data.get('last_query', '')):
                session_previous_chunks = session_data.get('chunk_ids', [])
                logger.info(f"Found {len(session_previous_chunks)} previous chunks in session cache")
        
        # Use provided previous_chunk_ids or session cache
        effective_previous_chunks = previous_chunk_ids or session_previous_chunks
        
        # Apply context window expansion if requested
        if expand_window:
            expanded_params = self._get_expansion_parameters(
                max_context_tokens, top_k, expansion_level
            )
            max_context_tokens = expanded_params["max_tokens"]
            top_k = expanded_params["top_k"]
            logger.info(f"Context expansion enabled - Level: {expansion_level}, Tokens: {max_context_tokens}, Top-K: {top_k}")
        
        logger.info(f"Process Query Retrieval for query: {query}")
        try:
            # Use expanded retrieval if window expansion is enabled
            if expand_window:
                retrieval_result = self._retrieve_with_expansion(
                    query, top_k, project_path, expansion_level, effective_previous_chunks
                )
            else:
                retrieval_result = self.multi_stage_retriever.retrieve(query, top_k, project_path)
            if not retrieval_result:
                return RAGResponse(
                    context="",
                    chunks_used=[],
                    total_tokens=0,
                    retrieval_time_ms=(time.time() - start_time) * 1000
                )

            final_results = retrieval_result[:top_k]
            final_context = self.context_fuser.fuse_context(final_results, max_context_tokens)
            chunks_used = [result.chunk for result in final_results]
            
            # Store results in session cache for future expansion
            current_chunk_ids = [chunk.id for chunk in chunks_used]
            self._update_session_cache(session_id, query, current_chunk_ids, project_path)
            
            # Calculate expansion metrics
            chunks_before_expansion = 0
            tokens_before_expansion = 0
            if expand_window and effective_previous_chunks:
                # Use previous session data if available
                if session_id in self.session_cache:
                    session_data = self.session_cache[session_id]
                    chunks_before_expansion = session_data.get('previous_chunk_count', 0)
                    tokens_before_expansion = session_data.get('previous_token_count', 0)
            
            response = RAGResponse(
                context=final_context,
                chunks_used=chunks_used,
                total_tokens=len(self.context_fuser.tokenizer.encode(final_context)),
                retrieval_time_ms=(time.time() - start_time) * 1000,
                was_expanded=expand_window,
                expansion_level=expansion_level if expand_window else None,
                chunks_before_expansion=chunks_before_expansion,
                tokens_before_expansion=tokens_before_expansion
            )

            # TODO: Update metrics and Cache
            return response
        except Exception as e:
            raise Exception(f"Failed to process query retrieval: {str(e)}")
    
    def _validate_chunk_quality(self, chunks) -> Dict[str, Any]:
        """Validate and analyze chunk quality for context optimization"""
        if not chunks:
            return {"error": "No chunks to analyze"}
        
        try:
            # Analyze chunk size distribution
            chunk_sizes = [len(self.context_fuser.tokenizer.encode(chunk.content)) for chunk in chunks]
            
            # Analyze chunk types
            chunk_types = {}
            for chunk in chunks:
                chunk_type = chunk.chunk_type or "unknown"
                chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
            
            # Check for problematic patterns
            issues = []
            
            # Very small chunks (likely incomplete context)
            tiny_chunks = sum(1 for size in chunk_sizes if size < 50)
            if tiny_chunks > len(chunks) * 0.1:  # More than 10% tiny chunks
                issues.append(f"High number of tiny chunks ({tiny_chunks}/{len(chunks)}) - may indicate poor chunking boundaries")
            
            # Very large chunks (likely too much context)
            huge_chunks = sum(1 for size in chunk_sizes if size > 2000)
            if huge_chunks > 0:
                issues.append(f"Found {huge_chunks} overly large chunks - may lose focus")
            
            # Missing important chunk types
            if "class_complete" not in chunk_types and "class_definition" not in chunk_types:
                if any(chunk.language == "py" for chunk in chunks):
                    issues.append("No class chunks found in Python code - chunking may be missing class boundaries")
            
            # Calculate quality score
            avg_size = sum(chunk_sizes) / len(chunk_sizes) if chunk_sizes else 0
            size_variance = sum((size - avg_size) ** 2 for size in chunk_sizes) / len(chunk_sizes) if chunk_sizes else 0
            
            quality_score = 100
            if tiny_chunks > len(chunks) * 0.15:
                quality_score -= 30
            if huge_chunks > 0:
                quality_score -= 20
            if size_variance > 1000000:  # High variance in chunk sizes
                quality_score -= 10
            
            return {
                "total_chunks": len(chunks),
                "chunk_types": chunk_types,
                "size_stats": {
                    "min_tokens": min(chunk_sizes) if chunk_sizes else 0,
                    "max_tokens": max(chunk_sizes) if chunk_sizes else 0,
                    "avg_tokens": round(avg_size, 2),
                    "size_variance": round(size_variance, 2)
                },
                "quality_issues": issues,
                "quality_score": max(0, quality_score),
                "recommendations": self._get_chunking_recommendations(chunk_types, chunk_sizes, issues)
            }
            
        except Exception as e:
            logger.error(f"Error validating chunk quality: {e}")
            return {"error": f"Failed to validate chunk quality: {str(e)}"}
    
    def _get_chunking_recommendations(self, chunk_types, chunk_sizes, issues) -> List[str]:
        """Generate recommendations for improving chunking quality"""
        recommendations = []
        
        if not chunk_sizes:
            return ["No chunks to analyze"]
        
        avg_size = sum(chunk_sizes) / len(chunk_sizes)
        
        if avg_size < 200:
            recommendations.append("Consider increasing min_chunk_size to capture more context")
        elif avg_size > 1800:
            recommendations.append("Consider decreasing max_chunk_size to maintain focus")
        
        if sum(1 for size in chunk_sizes if size < 100) > len(chunk_sizes) * 0.1:
            recommendations.append("Enable hierarchical_chunking to better preserve context boundaries")
        
        if "class_complete" not in chunk_types and len([t for t in chunk_types if "class" in t]) > 0:
            recommendations.append("Enable preserve_class_methods to keep class structure intact")
        
        if max(chunk_sizes) - min(chunk_sizes) > 1500:
            recommendations.append("Consider adjusting overlap_ratio or chunk size limits for more consistent chunks")
        
        return recommendations or ["Chunking quality appears good - no specific recommendations"]
    
    def _get_expansion_parameters(self, base_tokens: int, base_top_k: int, expansion_level: str) -> Dict[str, int]:
        """Get expanded parameters based on expansion level"""
        
        expansion_configs = {
            "conservative": {
                "token_multiplier": 1.5,
                "top_k_multiplier": 1.5,
                "max_tokens_cap": 16000
            },
            "moderate": {
                "token_multiplier": 2.0,
                "top_k_multiplier": 2.0,
                "max_tokens_cap": 24000
            },
            "aggressive": {
                "token_multiplier": 3.0,
                "top_k_multiplier": 2.5,
                "max_tokens_cap": 32000
            }
        }
        
        config = expansion_configs.get(expansion_level, expansion_configs["moderate"])
        
        expanded_tokens = min(
            int(base_tokens * config["token_multiplier"]),
            config["max_tokens_cap"]
        )
        
        expanded_top_k = int(base_top_k * config["top_k_multiplier"])
        
        return {
            "max_tokens": expanded_tokens,
            "top_k": expanded_top_k
        }
    
    def _retrieve_with_expansion(
        self, 
        query: str, 
        top_k: int, 
        project_path: Optional[str], 
        expansion_level: str,
        previous_chunk_ids: Optional[List[str]] = None
    ) -> List:
        """Retrieve chunks with context window expansion strategy"""
        
        logger.info(f"Using expanded retrieval with level: {expansion_level}")
        
        # Step 1: Get base results from multi-stage retriever (with higher top_k)
        base_results = self.multi_stage_retriever.retrieve(query, top_k, project_path)
        
        # Step 2: Apply expansion strategies based on level
        if expansion_level == "conservative":
            expanded_results = self._apply_conservative_expansion(base_results, query, project_path)
        elif expansion_level == "moderate":
            expanded_results = self._apply_moderate_expansion(base_results, query, project_path)
        elif expansion_level == "aggressive":
            expanded_results = self._apply_aggressive_expansion(base_results, query, project_path)
        else:
            expanded_results = base_results
        
        # Step 3: Apply sliding window expansion around retrieved chunks
        sliding_expanded = self._apply_sliding_window_expansion(expanded_results, expansion_level)
        
        # Step 4: Filter out previously seen chunks if provided, otherwise use semantic deduplication
        if previous_chunk_ids:
            filtered_results = [
                result for result in sliding_expanded 
                if result.chunk.id not in previous_chunk_ids
            ]
            logger.info(f"Filtered out {len(sliding_expanded) - len(filtered_results)} previously seen chunks")
        else:
            # Use semantic deduplication when no previous chunk IDs available
            filtered_results = self._semantic_deduplication(sliding_expanded, query)
            logger.info(f"Applied semantic deduplication, kept {len(filtered_results)} unique results")
        
        # Step 5: Final deduplication and ranking
        final_results = self._deduplicate_and_rerank_expanded(filtered_results, query)
        
        logger.info(f"Expanded retrieval returned {len(final_results)} chunks")
        return final_results
    
    def _apply_conservative_expansion(self, base_results, query: str, project_path: Optional[str]):
        """Conservative expansion: Add related chunks and nearby context"""
        expanded_results = list(base_results)
        
        # Add related chunks for each base result
        for result in base_results[:5]:  # Limit to top 5 to avoid explosion
            try:
                related_chunks = self.retriever.vector_store.search_by_filter(
                    filters={"file_path": result.chunk.file_path}, 
                    limit=8
                )
                
                for chunk in related_chunks[:3]:  # Max 3 related per chunk
                    if chunk.id not in [r.chunk.id for r in expanded_results]:
                        from rag.models import RetrievalResult
                        expanded_results.append(RetrievalResult(
                            chunk=chunk,
                            relevance_score=result.relevance_score * 0.7  # Lower score for related
                        ))
            except Exception as e:
                logger.warning(f"Error adding related chunks: {e}")
                
        return expanded_results
    
    def _apply_moderate_expansion(self, base_results, query: str, project_path: Optional[str]):
        """Moderate expansion: Broader semantic search + file context"""
        expanded_results = list(base_results)
        
        # Generate broader search queries
        broader_queries = self._generate_expansion_queries(query)
        
        for broad_query in broader_queries[:2]:  # Limit to 2 expansion queries
            try:
                additional_results = self.multi_stage_retriever.retrieve(
                    broad_query, 8, project_path
                )
                
                for result in additional_results[:4]:  # Max 4 per expansion query
                    if result.chunk.id not in [r.chunk.id for r in expanded_results]:
                        result.relevance_score *= 0.8  # Lower score for expanded results
                        expanded_results.append(result)
                        
            except Exception as e:
                logger.warning(f"Error in moderate expansion: {e}")
        
        return expanded_results
    
    def _apply_aggressive_expansion(self, base_results, query: str, project_path: Optional[str]):
        """Aggressive expansion: Full file context + cross-file relationships"""
        expanded_results = list(base_results)
        
        # Get full file context for top chunks
        files_to_expand = set()
        for result in base_results[:3]:  # Top 3 chunks
            files_to_expand.add(result.chunk.file_path)
        
        # Add more context from same files
        for file_path in files_to_expand:
            try:
                file_chunks = self.retriever.vector_store.search_by_filter(
                    filters={"file_path": file_path}, 
                    limit=15
                )
                
                for chunk in file_chunks:
                    if chunk.id not in [r.chunk.id for r in expanded_results]:
                        from rag.models import RetrievalResult
                        expanded_results.append(RetrievalResult(
                            chunk=chunk,
                            relevance_score=0.5  # Lower base score for file context
                        ))
                        
            except Exception as e:
                logger.warning(f"Error in aggressive file expansion: {e}")
        
        # Also apply moderate expansion
        expanded_results = self._apply_moderate_expansion(expanded_results, query, project_path)
        
        return expanded_results
    
    def _apply_sliding_window_expansion(self, results, expansion_level: str):
        """Apply sliding window expansion around chunks"""
        if not results:
            return results
        
        expansion_sizes = {
            "conservative": 150,
            "moderate": 300, 
            "aggressive": 500
        }
        
        window_size = expansion_sizes.get(expansion_level, 300)
        
        # Use the enhanced retriever's expansion capability
        try:
            expanded_results = self.retriever._expand_context_windows(results)
            return expanded_results
        except Exception as e:
            logger.warning(f"Error in sliding window expansion: {e}")
            return results
    
    def _generate_expansion_queries(self, original_query: str) -> List[str]:
        """Generate broader queries for expansion"""
        # Extract key terms
        terms = original_query.lower().split()
        key_terms = [term for term in terms if len(term) > 3 and term not in {
            'how', 'what', 'where', 'when', 'does', 'the', 'and', 'with', 'this', 'that'
        }]
        
        expansion_queries = []
        
        # Individual term searches
        for term in key_terms[:3]:
            expansion_queries.append(f"{term} implementation")
            expansion_queries.append(f"{term} usage")
        
        # Combined term searches
        if len(key_terms) >= 2:
            expansion_queries.append(f"{key_terms[0]} {key_terms[1]}")
            
        # Context-based expansions
        if 'error' in original_query.lower() or 'bug' in original_query.lower():
            expansion_queries.append("error handling debugging")
            
        if 'test' in original_query.lower():
            expansion_queries.append("test cases examples")
            
        return expansion_queries[:4]  # Limit expansions
    
    def _deduplicate_and_rerank_expanded(self, results, query: str):
        """Deduplicate and rerank expanded results"""
        # Remove duplicates
        seen_ids = set()
        unique_results = []
        
        for result in results:
            if result.chunk.id not in seen_ids:
                seen_ids.add(result.chunk.id)
                unique_results.append(result)
        
        # Sort by relevance score
        unique_results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        return unique_results
    
    def _generate_query_signature(self, query: str, project_path: Optional[str]) -> str:
        """Generate a signature for query session tracking"""
        content = f"{query}:{project_path or ''}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _is_similar_query(self, query1: str, query2: str, threshold: float = 0.7) -> bool:
        """Check if two queries are similar enough for expansion context"""
        if not query1 or not query2:
            return False
        
        # Simple similarity check based on common words
        words1 = set(query1.lower().split())
        words2 = set(query2.lower().split())
        
        # Remove common stop words
        stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those', 'how', 'what', 'where', 'when', 'why'}
        
        words1 = {w for w in words1 if len(w) > 2 and w not in stop_words}
        words2 = {w for w in words2 if len(w) > 2 and w not in stop_words}
        
        if not words1 or not words2:
            return False
        
        # Calculate Jaccard similarity
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        similarity = intersection / union if union > 0 else 0
        return similarity >= threshold
    
    def _update_session_cache(
        self, 
        session_id: str, 
        query: str, 
        chunk_ids: List[str], 
        project_path: Optional[str]
    ):
        """Update session cache with current query results"""
        
        # Clean old sessions if cache is too large
        if len(self.session_cache) >= self.max_session_cache_size:
            # Remove oldest 20% of sessions
            oldest_sessions = sorted(
                self.session_cache.items(), 
                key=lambda x: x[1].get('timestamp', 0)
            )[:self.max_session_cache_size // 5]
            
            for old_session_id, _ in oldest_sessions:
                del self.session_cache[old_session_id]
        
        # Store previous data for expansion metrics
        previous_data = self.session_cache.get(session_id, {})
        previous_chunk_count = previous_data.get('chunk_count', 0)
        previous_token_count = previous_data.get('token_count', 0)
        
        # Update session cache
        self.session_cache[session_id] = {
            'last_query': query,
            'chunk_ids': chunk_ids,
            'chunk_count': len(chunk_ids),
            'token_count': sum(len(self.context_fuser.tokenizer.encode(chunk_id)) for chunk_id in chunk_ids[:10]),  # Estimate
            'project_path': project_path,
            'timestamp': time_module.time(),
            'previous_chunk_count': previous_chunk_count,
            'previous_token_count': previous_token_count
        }
        
        logger.debug(f"Updated session cache for {session_id} with {len(chunk_ids)} chunks")
    
    def _semantic_deduplication(self, results, query: str, similarity_threshold: float = 0.8):
        """Remove semantically similar chunks when no previous chunk IDs available"""
        if not results:
            return results
        
        unique_results = []
        seen_contents = []
        
        for result in results:
            chunk_content = result.chunk.content.lower()
            
            # Check if this content is too similar to any previously added content
            is_duplicate = False
            for seen_content in seen_contents:
                if self._content_similarity(chunk_content, seen_content) > similarity_threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_results.append(result)
                seen_contents.append(chunk_content)
                
                # Don't store too many content strings in memory
                if len(seen_contents) > 50:
                    seen_contents = seen_contents[-25:]  # Keep recent 25
        
        return unique_results
    
    def _content_similarity(self, content1: str, content2: str) -> float:
        """Calculate similarity between two pieces of content"""
        if not content1 or not content2:
            return 0.0
        
        # Use a simple approach based on common lines/words
        lines1 = set(line.strip() for line in content1.split('\n') if line.strip())
        lines2 = set(line.strip() for line in content2.split('\n') if line.strip())
        
        if not lines1 or not lines2:
            return 0.0
        
        # Calculate line-based Jaccard similarity
        intersection = len(lines1.intersection(lines2))
        union = len(lines1.union(lines2))
        
        return intersection / union if union > 0 else 0.0



import logging
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
import tiktoken
from dataclasses import dataclass

from rag.models import RetrievalResult, CodeBaseChunk

logger = logging.getLogger(__name__)

@dataclass
class FusionConfig:
    """Configuration for context fusion"""
    preserve_structure: bool = True  # Preserve code structure in output
    group_by_file: bool = True  # Group chunks by file
    deduplicate_content: bool = True  # Remove duplicate content
    smart_truncation: bool = True  # Intelligently truncate when exceeding token limit
    include_file_context: bool = True  # Include file path and structure info
    merge_adjacent: bool = True  # Merge adjacent chunks from same file

class EnhancedContextFusion:
    """Enhanced context fusion with structure preservation and intelligent merging"""
    
    def __init__(self, config: Optional[FusionConfig] = None):
        self.config = config or FusionConfig()
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def fuse_context(
        self, 
        results: List[RetrievalResult], 
        max_tokens: int,
        preserve_order: bool = False
    ) -> str:
        """
        Fuse retrieved chunks into coherent context
        
        Args:
            results: List of retrieval results
            max_tokens: Maximum token budget
            preserve_order: Preserve file order instead of relevance order
        """
        if not results:
            return ""

        try:
            # Step 1: Deduplicate overlapping content
            if self.config.deduplicate_content:
                results = self._deduplicate_chunks(results)
            
            # Step 2: Group chunks by file and structure
            grouped_chunks = self._group_and_structure_chunks(results)
            
            # Step 3: Merge adjacent chunks if configured
            if self.config.merge_adjacent:
                grouped_chunks = self._merge_adjacent_chunks(grouped_chunks)
            
            # Step 4: Allocate tokens intelligently
            allocated_groups = self._allocate_tokens_intelligently(
                grouped_chunks, 
                max_tokens
            )
            
            # Step 5: Build final context with structure preservation
            final_context = self._build_structured_context(
                allocated_groups,
                preserve_order
            )
            
            # Verify token count
            actual_tokens = len(self.tokenizer.encode(final_context))
            if actual_tokens > max_tokens:
                final_context = self._smart_truncate(final_context, max_tokens)
            
            logger.info(f"Context fused: {actual_tokens}/{max_tokens} tokens")
            return final_context
            
        except Exception as e:
            logger.error(f"Error in enhanced context fusion: {e}")
            return self._fallback_fusion(results, max_tokens)

    def _deduplicate_chunks(self, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Remove duplicate and overlapping content"""
        deduplicated = []
        seen_content = set()
        seen_ranges = defaultdict(list)  # file_path -> [(start, end)]
        
        for result in results:
            chunk = result.chunk
            
            # Check for exact duplicates
            content_hash = hash(chunk.content.strip())
            if content_hash in seen_content:
                continue
            
            # Check for overlapping ranges in the same file
            if chunk.file_path in seen_ranges:
                is_overlapping = False
                for start, end in seen_ranges[chunk.file_path]:
                    if self._ranges_overlap(
                        (chunk.start_line, chunk.end_line),
                        (start, end)
                    ):
                        # Keep the larger chunk
                        if (chunk.end_line - chunk.start_line) <= (end - start):
                            is_overlapping = True
                            break
                
                if is_overlapping:
                    continue
            
            # Add to deduplicated list
            deduplicated.append(result)
            seen_content.add(content_hash)
            seen_ranges[chunk.file_path].append((chunk.start_line, chunk.end_line))
        
        return deduplicated

    def _ranges_overlap(self, range1: Tuple[int, int], range2: Tuple[int, int]) -> bool:
        """Check if two line ranges overlap"""
        start1, end1 = range1
        start2, end2 = range2
        return not (end1 < start2 or end2 < start1)

    def _group_and_structure_chunks(
        self, 
        results: List[RetrievalResult]
    ) -> Dict[str, List[RetrievalResult]]:
        """Group chunks by file and maintain structure"""
        groups = defaultdict(list)
        
        for result in results:
            chunk = result.chunk
            
            # Create structured group key
            if self.config.group_by_file:
                # Group by file and potentially by class/module
                group_key = chunk.file_path
                
                # Sub-group by class if it's a class or method
                if chunk.chunk_type in ["class_complete", "class_definition"]:
                    # Extract class name from content if possible
                    class_name = self._extract_class_name(chunk.content)
                    if class_name:
                        group_key = f"{chunk.file_path}::{class_name}"
                elif chunk.chunk_type == "method":
                    # Try to find parent class
                    parent_class = self._find_parent_class(chunk, results)
                    if parent_class:
                        group_key = f"{chunk.file_path}::{parent_class}"
            else:
                group_key = "all"
            
            groups[group_key].append(result)
        
        # Sort chunks within each group by line number
        for group_key in groups:
            groups[group_key].sort(key=lambda r: r.chunk.start_line)
        
        return dict(groups)

    def _merge_adjacent_chunks(
        self, 
        grouped_chunks: Dict[str, List[RetrievalResult]]
    ) -> Dict[str, List[RetrievalResult]]:
        """Merge adjacent chunks from the same file"""
        merged_groups = {}
        
        for group_key, chunks in grouped_chunks.items():
            if len(chunks) <= 1:
                merged_groups[group_key] = chunks
                continue
            
            merged = []
            current_merged = None
            
            for result in chunks:
                chunk = result.chunk
                
                if current_merged is None:
                    current_merged = result
                    continue
                
                # Check if chunks are adjacent (within 5 lines)
                if (chunk.file_path == current_merged.chunk.file_path and
                    chunk.start_line <= current_merged.chunk.end_line + 5):
                    
                    # Merge chunks
                    merged_chunk = self._merge_two_chunks(
                        current_merged.chunk, 
                        chunk
                    )
                    current_merged = RetrievalResult(
                        chunk=merged_chunk,
                        relevance_score=max(
                            current_merged.relevance_score,
                            result.relevance_score
                        )
                    )
                else:
                    # Not adjacent, save current and start new
                    merged.append(current_merged)
                    current_merged = result
            
            if current_merged:
                merged.append(current_merged)
            
            merged_groups[group_key] = merged
        
        return merged_groups

    def _merge_two_chunks(self, chunk1: CodeBaseChunk, chunk2: CodeBaseChunk) -> CodeBaseChunk:
        """Merge two chunks into one while preserving code structure"""
        # Ensure chunks are from the same file
        if chunk1.file_path != chunk2.file_path:
            # Can't merge chunks from different files - return the higher scored one
            return chunk1 if chunk1.start_line <= chunk2.start_line else chunk2
        
        # Determine order based on line numbers
        if chunk1.start_line <= chunk2.start_line:
            first_chunk, second_chunk = chunk1, chunk2
        else:
            first_chunk, second_chunk = chunk2, chunk1
        
        # Check for overlap or adjacency
        if second_chunk.start_line <= first_chunk.end_line + 2:
            # Chunks overlap or are adjacent - merge by combining content
            if second_chunk.start_line <= first_chunk.end_line:
                # Overlapping - use the content from start of first to end of second
                combined_content = first_chunk.content
                
                # Add non-overlapping portion of second chunk
                second_lines = second_chunk.content.split('\n')
                first_lines = first_chunk.content.split('\n')
                
                # Calculate overlap
                overlap_lines = first_chunk.end_line - second_chunk.start_line + 1
                if overlap_lines > 0 and overlap_lines < len(second_lines):
                    # Add the non-overlapping part of the second chunk
                    non_overlap_content = '\n'.join(second_lines[overlap_lines:])
                    if non_overlap_content.strip():
                        combined_content += '\n' + non_overlap_content
                else:
                    # No meaningful overlap, just concatenate
                    combined_content = first_chunk.content + '\n' + second_chunk.content
            else:
                # Adjacent but not overlapping
                combined_content = first_chunk.content + '\n' + second_chunk.content
        else:
            # Chunks are far apart - just concatenate with separator
            combined_content = first_chunk.content + '\n\n# ... (gap) ...\n\n' + second_chunk.content
        
        return CodeBaseChunk(
            id=f"{chunk1.id}_merged_{chunk2.id}",
            file_path=chunk1.file_path,
            content=combined_content,
            language=chunk1.language,
            chunk_type="merged",
            start_byte=min(chunk1.start_byte, chunk2.start_byte),
            end_byte=max(chunk1.end_byte, chunk2.end_byte),
            start_line=min(chunk1.start_line, chunk2.start_line),
            end_line=max(chunk1.end_line, chunk2.end_line)
        )

    def _allocate_tokens_intelligently(
        self,
        grouped_chunks: Dict[str, List[RetrievalResult]],
        max_tokens: int
    ) -> List[Tuple[str, List[RetrievalResult], int]]:
        """Allocate token budget intelligently across groups"""
        # Calculate importance scores for each group
        group_scores = {}
        for group_key, results in grouped_chunks.items():
            # Average relevance score of chunks in group
            avg_score = sum(r.relevance_score for r in results) / len(results)
            
            # Boost for complete classes
            has_complete_class = any(
                r.chunk.chunk_type == "class_complete" 
                for r in results
            )
            if has_complete_class:
                avg_score *= 1.2
            
            # Boost for groups with multiple relevant chunks
            if len(results) > 1:
                avg_score *= (1 + 0.1 * min(len(results), 5))
            
            group_scores[group_key] = avg_score
        
        # Sort groups by score
        sorted_groups = sorted(
            group_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Allocate tokens
        allocated = []
        remaining_tokens = max_tokens - 100  # Reserve for formatting
        min_tokens_per_group = 200
        
        for group_key, score in sorted_groups:
            if remaining_tokens <= 0:
                break
            
            results = grouped_chunks[group_key]
            
            # Calculate desired tokens for this group
            total_score = sum(group_scores.values())
            if total_score > 0:
                proportion = score / total_score
                desired_tokens = int(remaining_tokens * proportion)
            else:
                desired_tokens = remaining_tokens // len(sorted_groups)
            
            # Ensure minimum allocation
            allocated_tokens = max(min_tokens_per_group, min(desired_tokens, remaining_tokens))
            
            allocated.append((group_key, results, allocated_tokens))
            remaining_tokens -= allocated_tokens
        
        return allocated

    def _build_structured_context(
        self,
        allocated_groups: List[Tuple[str, List[RetrievalResult], int]],
        preserve_order: bool
    ) -> str:
        """Build final context with structure preservation"""
        context_parts = []
        
        # Sort by file path if preserving order
        if preserve_order:
            allocated_groups.sort(key=lambda x: x[0])
        
        for group_key, results, token_budget in allocated_groups:
            group_context = self._build_group_context(
                group_key,
                results,
                token_budget
            )
            
            if group_context:
                context_parts.append(group_context)
        
        # Join with clear separators
        return "\n\n" + ("=" * 50) + "\n\n".join(context_parts)

    def _build_group_context(
        self,
        group_key: str,
        results: List[RetrievalResult],
        token_budget: int
    ) -> str:
        """Build context for a single group"""
        parts = []
        
        # Add file header if configured
        if self.config.include_file_context:
            if "::" in group_key:
                file_path, class_name = group_key.split("::", 1)
                filename = file_path.split('/')[-1]
                parts.append(f"# File: {filename}")
                parts.append(f"# Class: {class_name}")
            else:
                filename = group_key.split('/')[-1]
                parts.append(f"# File: {filename}")
            parts.append("")
        
        # Add chunks within token budget
        used_tokens = len(self.tokenizer.encode('\n'.join(parts)))
        
        for result in results:
            chunk = result.chunk
            chunk_content = self._format_chunk_content(chunk)
            chunk_tokens = len(self.tokenizer.encode(chunk_content))
            
            if used_tokens + chunk_tokens > token_budget:
                # Try to add a truncated version
                if self.config.smart_truncation:
                    truncated = self._truncate_chunk(
                        chunk_content, 
                        token_budget - used_tokens
                    )
                    if truncated:
                        parts.append(truncated)
                break
            
            parts.append(chunk_content)
            used_tokens += chunk_tokens
        
        return '\n'.join(parts)

    def _format_chunk_content(self, chunk: CodeBaseChunk) -> str:
        """Format chunk content for display"""
        lines = chunk.content.split('\n')
        
        # Add line numbers for code chunks
        if self.config.preserve_structure and chunk.chunk_type != "text":
            formatted_lines = []
            for i, line in enumerate(lines):
                line_num = chunk.start_line + i
                formatted_lines.append(f"{line_num:4d} | {line}")
            content = '\n'.join(formatted_lines)
        else:
            content = chunk.content
        
        # Add metadata as comment
        if chunk.chunk_type and chunk.chunk_type != "text":
            metadata = f"# Chunk Type: {chunk.chunk_type} | Lines: {chunk.start_line}-{chunk.end_line}"
            content = f"{metadata}\n{content}"
        
        return content

    def _truncate_chunk(self, content: str, max_tokens: int) -> Optional[str]:
        """Intelligently truncate chunk content"""
        if max_tokens < 50:
            return None
        
        lines = content.split('\n')
        
        # Try to keep complete statements/functions
        truncated_lines = []
        current_tokens = 0
        
        for line in lines:
            line_tokens = len(self.tokenizer.encode(line))
            if current_tokens + line_tokens > max_tokens:
                break
            truncated_lines.append(line)
            current_tokens += line_tokens
        
        if truncated_lines:
            result = '\n'.join(truncated_lines)
            result += "\n# ... (truncated)"
            return result
        
        return None

    def _smart_truncate(self, context: str, max_tokens: int) -> str:
        """Smart truncation of entire context"""
        lines = context.split('\n')
        truncated = []
        current_tokens = 0
        
        for line in lines:
            line_tokens = len(self.tokenizer.encode(line))
            if current_tokens + line_tokens > max_tokens - 20:  # Reserve space for truncation marker
                truncated.append("\n# === CONTEXT TRUNCATED DUE TO TOKEN LIMIT ===")
                break
            truncated.append(line)
            current_tokens += line_tokens
        
        return '\n'.join(truncated)

    def _extract_class_name(self, content: str) -> Optional[str]:
        """Extract class name from chunk content"""
        lines = content.split('\n')
        for line in lines:
            if 'class ' in line:
                # Simple extraction - can be improved with regex
                parts = line.split('class ')
                if len(parts) > 1:
                    class_name = parts[1].split('(')[0].split(':')[0].strip()
                    return class_name
        return None

    def _find_parent_class(
        self, 
        chunk: CodeBaseChunk, 
        all_results: List[RetrievalResult]
    ) -> Optional[str]:
        """Find parent class for a method chunk"""
        for result in all_results:
            other_chunk = result.chunk
            if (other_chunk.file_path == chunk.file_path and
                other_chunk.chunk_type in ["class_complete", "class_definition"] and
                other_chunk.start_line <= chunk.start_line and
                other_chunk.end_line >= chunk.end_line):
                return self._extract_class_name(other_chunk.content)
        return None

    def _fallback_fusion(self, results: List[RetrievalResult], max_tokens: int) -> str:
        """Simple fallback fusion method"""
        parts = []
        used_tokens = 0
        
        for result in results:
            chunk_content = result.chunk.content
            chunk_tokens = len(self.tokenizer.encode(chunk_content))
            
            if used_tokens + chunk_tokens > max_tokens:
                break
            
            parts.append(f"# From: {result.chunk.file_path}")
            parts.append(chunk_content)
            parts.append("")
            used_tokens += chunk_tokens
        
        return '\n'.join(parts)

import logging
from collections import defaultdict
from typing import List, Dict, Tuple

import tiktoken

from rag.models import RetrievalResult, CodeBaseChunk

logger = logging.getLogger(__name__)

class ContextFusion:
    def __init__(self):
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def fuse_context(self, results: List[RetrievalResult], max_tokens: int) -> str:
        if not results:
            return ""

        try:
            grouped_chunks = self._group_related_chunks(results)
            allocated_groups = self._allocate_tokens_to_groups(grouped_chunks, max_tokens)
            final_context = self._build_final_context(allocated_groups)
            actual_tokens = len(self.tokenizer.encode(final_context))
            logger.info(f"Context fused successfully. Actual tokens: {actual_tokens}/{max_tokens}")
            return final_context
        except Exception as e:
            logger.error(f"Failed to fuse context: {e}")
            return ""

    def _group_related_chunks(self, results: List[RetrievalResult]) -> Dict[str, List[RetrievalResult]]:
        """Group related chunks together"""
        groups = defaultdict(list)

        for result in results:
            chunk = result.chunk

            # Group by file primarily
            file_path = chunk.file_path

            # Sub-group by context (class, module, etc.)
            if chunk.chunk_type:
                file_path = f"{file_path}#{chunk.chunk_type}"

            groups[file_path].append(result)

        return dict(groups)

    def _allocate_tokens_to_groups(self, grouped_chunks: Dict[str, List[RetrievalResult]],
                                   max_tokens: int) -> List[
        Tuple[str, List[RetrievalResult], int]]:
        """Intelligently allocate token budget to chunk groups"""
        # Calculate relevance score for each group
        group_scores = {}
        for group_key, group_results in grouped_chunks.items():
            avg_score = sum(r.relevance_score for r in group_results) / len(group_results)
            group_scores[group_key] = avg_score

        # Sort groups by relevance
        sorted_groups = sorted(group_scores.items(), key=lambda x: x[1], reverse=True)

        # Allocate tokens proportionally to relevance
        total_score = sum(group_scores.values())
        allocated_groups = []
        remaining_tokens = max_tokens - 200  # Reserve tokens for formatting

        for group_key, score in sorted_groups:
            if remaining_tokens <= 0:
                break

            group_results = grouped_chunks[group_key]

            allocated_tokens = max(100, remaining_tokens)  # Minimum tokens per group
            allocated_groups.append((group_key, group_results, allocated_tokens))
            remaining_tokens -= allocated_tokens

        return allocated_groups

    def _build_final_context(self, allocated_groups: List[Tuple[str, List[RetrievalResult], int]]) -> str:
        """Build coherent context from allocated groups"""
        context_parts = []

        # Process each group
        for group_key, group_results, token_budget in allocated_groups:
            group_context = self._build_group_context(group_results, token_budget)
            if group_context:
                # Add group header
                file_path = group_key.split('#')[0]
                filename = file_path.split('/')[-1]
                context_parts.append(f"\n# {filename}")
                if '#' in group_key:
                    context_name = group_key.split('#')[1]
                    context_parts.append(f"# Context: {context_name}")

                context_parts.append(group_context)

        return '\n'.join(context_parts)

    def _build_group_context(self, group_results: List[RetrievalResult], token_budget: int) -> str:
        """Build context for a single group within token budget"""
        # Sort by relevance within group
        sorted_results = sorted(group_results, key=lambda x: x.relevance_score, reverse=True)

        group_parts = []
        used_tokens = 0

        for result in sorted_results:
            chunk = result.chunk

            # Prepare chunk content
            chunk_content = self._prepare_chunk_for_display(chunk)
            chunk_tokens = len(self.tokenizer.encode(chunk_content))

            if used_tokens + chunk_tokens > token_budget:
                # Try to fit a summary instead
                summary = self._summarize_chunk(chunk, token_budget - used_tokens)
                if summary:
                    group_parts.append(summary)
                break
            else:
                group_parts.append(chunk_content)
                used_tokens += chunk_tokens

        return '\n\n'.join(group_parts)

    def _prepare_chunk_for_display(self, chunk: CodeBaseChunk) -> str:
        """Prepare chunk for display in context"""
        lines = chunk.content.split('\n')

        # Add line numbers for code chunks
        if chunk.chunk_type in ['function', 'class', 'method']:
            numbered_lines = []
            start_line = chunk.start_line

            for i, line in enumerate(lines):
                line_num = start_line + i
                numbered_lines.append(f"{line_num:4d} | {line}")

            content = '\n'.join(numbered_lines)
        else:
            content = chunk.content

        # Add metadata comment
        # metadata = []
        # if chunk.symbols:
        #     symbol_names = [s.name for s in chunk.symbols]
        #     metadata.append(f"Symbols: {', '.join(symbol_names)}")
        #
        # if chunk.chunk_type:
        #     metadata.append(f"Type: {chunk.chunk_type}")
        #
        # if metadata:
        #     metadata_line = f"// {' | '.join(metadata)}"
        #     content = f"{metadata_line}\n{content}"

        return content

    def _summarize_chunk(self, chunk: CodeBaseChunk, max_tokens: int) -> str:
        """Create summary of chunk when full content doesn't fit"""
        if max_tokens < 50:
            return None

        # Simple summarization - show signature/definition only
        lines = chunk.content.split('\n')

        summary_lines = []
        for line in lines[:5]:  # First few lines usually have key info
            if line.strip():
                summary_lines.append(line)
                if len(summary_lines) >= 3:
                    break

        summary = '\n'.join(summary_lines)
        if len(summary_lines) < len(lines):
            summary += "\n    # ... (truncated)"

        # Check if summary fits in token budget
        if len(self.tokenizer.encode(summary)) <= max_tokens:
            return f"// Summarized content\n{summary}"

        return None





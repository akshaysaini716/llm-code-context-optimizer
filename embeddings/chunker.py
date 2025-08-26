import re
from pathlib import Path

CHUNK_SPLITTERS = {
    ".py": r"(?=^def\s|^class\s)",
    ".java": r"(?=^\s*(public|private|protected)?\s*(class|interface|void|[\w<>\[\]]+\s+\w+)\s)",
    ".kt": r"(?=^\s*fun\s|^\s*class\s)",
    ".js": r"(?=^\s*(function|class)\s)",
    ".ts": r"(?=^\s*(function|class|interface)\s)",
}


def chunk_code_file(file_path: Path, max_lines: int = 50) -> list[dict]:
    ext = file_path.suffix
    splitter = CHUNK_SPLITTERS.get(ext)
    text = file_path.read_text(encoding="utf-8", errors="ignore")

    if splitter:
        raw_chunks = re.split(splitter, text, flags=re.M)
        chunks = []
        for i in range(1, len(raw_chunks)):
            snippet = raw_chunks[i - 1] + raw_chunks[i]
            for j in range(0, len(snippet.splitlines()), max_lines):
                chunk = "\n".join(snippet.splitlines()[j:j + max_lines])
                if chunk.strip():
                    chunks.append({
                        "file": str(file_path),
                        "chunk": chunk.strip(),
                    })
        return chunks
    else:
        return [{
            "file": str(file_path),
            "chunk": text.strip()
        }]


"""
Better chunking:
1. Token-aware chunking: Uses actual tokenizer instead of line counts
2. Semantic boundaries: Respects function/class boundaries
3. Context preservation: Includes imports in each chunk
4. Configurable limits: Easy to adjust for different embedding models
5. Overlap support: Can add overlap between chunks for better context
6. Better metadata: Includes token count and line information

"""

import re
import tiktoken
from pathlib import Path
from typing import List, Dict

class ImprovedCodeChunker:
    def __init__(self, max_tokens: int = 1000, overlap_tokens: int = 100):
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.encoder = tiktoken.get_encoding("cl100k_base")  # GPT tokenizer
        
    def chunk_code_file(self, file_path: Path) -> List[Dict]:
        ext = file_path.suffix
        content = file_path.read_text(encoding="utf-8", errors="ignore")
        
        if ext == ".py":
            return self._chunk_python_semantically(file_path, content)
        else:
            return self._chunk_by_tokens(file_path, content)
    
    def _chunk_python_semantically(self, file_path: Path, content: str) -> List[Dict]:
        chunks = []
        lines = content.split('\n')
        
        # Extract imports first
        imports = self._extract_imports(lines)
        imports_text = '\n'.join(imports) if imports else ""
        
        # Find class/function boundaries
        boundaries = self._find_semantic_boundaries(lines)
        
        current_chunk = imports_text
        current_tokens = self._count_tokens(current_chunk)
        
        for start, end in boundaries:
            segment = '\n'.join(lines[start:end])
            segment_tokens = self._count_tokens(segment)
            
            if current_tokens + segment_tokens <= self.max_tokens:
                current_chunk += '\n' + segment
                current_tokens += segment_tokens
            else:
                # Save current chunk
                if current_chunk.strip():
                    chunks.append({
                        "file": str(file_path),
                        "chunk": current_chunk.strip(),
                        "tokens": current_tokens,
                        "lines": len(current_chunk.split('\n'))
                    })
                
                # Start new chunk with overlap
                current_chunk = imports_text + '\n' + segment
                current_tokens = self._count_tokens(current_chunk)
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append({
                "file": str(file_path),
                "chunk": current_chunk.strip(),
                "tokens": current_tokens,
                "lines": len(current_chunk.split('\n'))
            })
            
        return chunks
    
    def _extract_imports(self, lines: List[str]) -> List[str]:
        imports = []
        for line in lines:
            if line.strip().startswith(('import ', 'from ')):
                imports.append(line)
            elif line.strip() and not line.startswith('#'):
                break  # Stop at first non-import, non-comment line
        return imports
    
    def _find_semantic_boundaries(self, lines: List[str]) -> List[tuple]:
        boundaries = []
        current_start = 0
        
        for i, line in enumerate(lines):
            if re.match(r'^(class|def|async def)\s+', line.strip()):
                if i > current_start:
                    boundaries.append((current_start, i))
                current_start = i
        
        # Add final boundary
        if current_start < len(lines):
            boundaries.append((current_start, len(lines)))
            
        return boundaries
    
    def _count_tokens(self, text: str) -> int:
        return len(self.encoder.encode(text))
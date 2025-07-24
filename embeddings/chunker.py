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

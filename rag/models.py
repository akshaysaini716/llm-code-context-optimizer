from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import uuid

class ChatRequest(BaseModel):
    message: str
    project_path: str
    relevant_only: bool = False

class TokenUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatResponse(BaseModel):
    response: str
    token_usage: TokenUsage
    request: str

class IndexCodeBaseRequest(BaseModel):
    project_path: str
    file_patterns: Optional[List[str]] = None
    force_reindex: bool = False

class IndexCodeBaseResponse(BaseModel):
    status: str
    files_processed: int
    chunks_created: int
    chunks_embedded: int
    indexing_time_seconds: float
    collection_info: Dict[str, Any]

class CodeBaseChunk(BaseModel):
    id: str
    file_path: str
    project_path: str = ""  # Absolute path to project root
    content: str
    language: str
    chunk_type: str # this will function, class, module, import etc.
    start_byte: int
    end_byte: int
    start_line: int
    end_line: int
    embedding: Optional[List[float]] = None

    def to_qdrant_point(self) -> Dict[str, Any]:
        point_id = self.id if self.id else str(uuid.uuid4())

        return {
            "id": point_id,
            "vector": self.embedding,
            "payload": {
                "file_path": self.file_path,
                "project_path": self.project_path,
                "content": self.content,
                "language": self.language,
                "chunk_type": self.chunk_type,
                "start_byte": self.start_byte,
                "end_byte": self.end_byte,
                "start_line": self.start_line,
                "end_line": self.end_line
            }
        }

class QueryRequest(BaseModel):
    query: str
    project_path: Optional[str] = None
    max_context_tokens: int = 8000
    top_k: int = 10
    include_metadata: bool = False

class QueryResponse(BaseModel):
    context: str
    total_tokens: int
    retrieval_time_ms: float
    chunks_count: int

class RAGResponse(BaseModel):
    context: str
    chunks_used: List[CodeBaseChunk]
    total_tokens: int
    retrieval_time_ms: float

class RetrievalResult(BaseModel):
    chunk: CodeBaseChunk
    relevance_score: float

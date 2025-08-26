import logging
import time

import uvicorn
from fastapi import FastAPI, HTTPException

from rag.models import *
from pathlib import Path
from rag.services.gemini_service import GeminiService
from rag.services.context_service import ContextService
from rag.core.rag_system import RAGSystem


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title = "RAG Server, context optimization Service",
    version = "1.0.0"
)

gemini_service = GeminiService()
context_service = ContextService()
rag_system = RAGSystem()

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "RAG Server"}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        relevant_context = context_service.get_relevant_context(request)
        full_prompt = f"{request.message}\n\n{relevant_context}" if relevant_context else request.message
        result = gemini_service.chat_with_gemini(full_prompt)
        return ChatResponse(
            response = result["response"],
            token_usage = result["token_usage"],
            request = full_prompt
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing chat request: {str(e)}")


@app.post("/index", response_model=IndexCodeBaseResponse)
async def index_codebase(request: IndexCodeBaseRequest):
    try:
        project_path = Path(request.project_path)
        if not project_path.exists():
            raise HTTPException(status_code=404, detail=f"Project path {request.project_path} not found")

        if not project_path.is_dir():
            raise HTTPException(status_code=404, detail=f"Project path {request.project_path} is not a directory")

        logger.info(f"Starting codebase index for project path: {request.project_path}")
        result = rag_system.index_codebase(
            project_path=project_path,
            file_patterns=request.file_patterns,
            force_reindex=request.force_reindex
        )
        response = IndexCodeBaseResponse(**result)
        logger.info(f"Codebase index completed for project path: {request.project_path}")
        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing code baseindex request: {str(e)}")

@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    try:
        rag_response = rag_system.query(
            query  = request.query,
            max_context_tokens = request.max_context_tokens,
            top_k = request.top_k
        )

        response = QueryResponse(
            context = rag_response.context,
            total_tokens = rag_response.total_tokens,
            retrieval_time_ms = rag_response.retrieval_time_ms,
            chunks_count = len(rag_response.chunks_used)
        )
        # TODO:Include chunk metadata if required
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query request: {str(e)}")



if __name__ == '__main__':
    print("RAG server started")
    uvicorn.run(
        "rag.api.rag_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
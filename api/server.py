from fastapi import FastAPI, Query
from pydantic import BaseModel
import os
from typing import Optional, List
from services.context_service import (
    get_project_context, 
    get_graph_stats, 
    find_related_symbols,
    clear_graph_cache,
    ContextService
)
from services.gemini_service import chat_with_gemini
from embeddings.service import CodeSearchService

app = FastAPI(title="LLM Code Context Optimizer", version="1.0.0")

class ChatRequest(BaseModel):
    message: str
    include_context: bool = False
    project_path: str | None = None
    relevant_only: bool = True
    max_tokens: int = 8000
    use_tree_sitter: bool = True
    use_embedding: bool = False

### using FAISS + SentenceTransformer start here ###
@app.post("/chat-faiss")
def chat_faiss(request: ChatRequest):
    service = CodeSearchService(request.project_path)
    context = service.query(request.message)
    full_prompt = f"{context}\n\n{request.message}" if context else request.message
    return chat_with_gemini(full_prompt)

### using FAISS + SentenceTransformer end here ###

""" using embeddings start here """

@app.post("/chat-llm")
def chat_llm(request: ChatRequest):
    context_service = ContextService()

    # Set repository path from request
    context_service.repository_path = request.project_path
    print(f"Repository path set to: {context_service.repository_path}")

    # Build semantic index if not already built
    context_service.build_semantic_index(request)
    context_service.embedding_service.save_index("./faiss_index")

    # Try to load existing index
    context_service.embedding_service.load_index("./faiss_index")

    # Debug: Check what was loaded
    if hasattr(context_service.embedding_service, 'chunks'):
        print(f"Loaded {len(context_service.embedding_service.chunks)} chunks")
        for i, chunk in enumerate(context_service.embedding_service.chunks[:3]):  # Show first 3
            print(f"Chunk {i}: {chunk.get('metadata', {})} - {chunk.get('content', '')[:100]}...")

    # Get optimized context
    relevant_context = context_service.get_relevant_context(request.message)
    print(f"Relevant context length: {len(relevant_context) if relevant_context else 0}")
    full_prompt = f"{relevant_context}\n\n{request.message}" if relevant_context else request.message

    # Add context metadata to response
    response = chat_with_gemini(full_prompt)
    if request.include_context and request.project_path:
        response["context_metadata"] = {"request-context": full_prompt}

    return response

""" using embeddings end here """

@app.post("/chat")
def chat(req: ChatRequest):
    """Chat with LLM using optimized code context"""
    context_text = ""

    if req.include_context and req.project_path:
        ctx = get_project_context(
            req.project_path, 
            req.message, 
            req.relevant_only,
            req.max_tokens,
            req.use_tree_sitter
        )
        
        # Format context based on mode
        if ctx.get("mode") == "tree_sitter":
            # Enhanced formatting with metadata
            for file in ctx["files"]:
                if file.get("context_type") == "file":
                    context_text += f"\n\n# File: {file['file']}\n{file['content']}"
                else:
                    # Symbol-level context
                    symbol_info = f" ({file['context_type']}: {file['symbol_name']})" if file['symbol_name'] else ""
                    context_text += f"\n\n# {file['file']}{symbol_info} (lines {file['start_line']}-{file['end_line']})\n{file['content']}"
        else:
            # Simple formatting
            for file in ctx["files"]:
                context_text += f"\n\n# File: {file['file']}\n{file['content']}"

    full_prompt = f"{context_text}\n\n{req.message}" if context_text else req.message
    
    # Add context metadata to response
    response = chat_with_gemini(full_prompt)
    if req.include_context and req.project_path:
        metadata = ctx.get("metadata", {})
        metadata["request-context"] = full_prompt
        response["context_metadata"] = metadata
    
    return response

class ContextRequest(BaseModel):
    path: str
    query: str
    relevant_only: bool = True
    max_tokens: int = 8000
    use_tree_sitter: bool = True

@app.post("/context")
def get_context(req: ContextRequest):
    """Get optimized code context for a query"""
    return get_project_context(
        req.path, 
        req.query, 
        req.relevant_only,
        req.max_tokens,
        req.use_tree_sitter
    )

@app.get("/context")
def get_context_get(
    path: str,
    query: str,
    relevant_only: bool = True,
    max_tokens: int = 8000,
    use_tree_sitter: bool = True
):
    """Get optimized code context for a query (GET version)"""
    return get_project_context(path, query, relevant_only, max_tokens, use_tree_sitter)

@app.get("/graph/stats")
def get_dependency_graph_stats(project_path: str):
    """Get statistics about the dependency graph for a project"""
    stats = get_graph_stats(project_path)
    if stats is None:
        return {"error": "Could not analyze project"}
    return stats

@app.get("/symbols/related")
def get_related_symbols(
    project_path: str,
    symbol_name: str,
    max_results: int = Query(20, ge=1, le=100)
):
    """Find symbols related to the given symbol"""
    return {
        "symbol_name": symbol_name,
        "related_symbols": find_related_symbols(project_path, symbol_name, max_results)
    }

@app.post("/cache/clear")
def clear_cache():
    """Clear the dependency graph cache"""
    clear_graph_cache()
    return {"message": "Cache cleared successfully"}

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "LLM Code Context Optimizer"}

@app.get("/")
def root():
    """Root endpoint with API information"""
    return {
        "name": "LLM Code Context Optimizer",
        "version": "1.0.0",
        "description": "Intelligent code context extraction using tree-sitter and dependency analysis",
        "endpoints": {
            "POST /chat": "Chat with LLM using optimized context",
            "GET|POST /context": "Get optimized code context",
            "GET /graph/stats": "Get dependency graph statistics",
            "GET /symbols/related": "Find related symbols",
            "POST /cache/clear": "Clear dependency graph cache",
            "GET /health": "Health check"
        },
        "features": [
            "Tree-sitter based AST parsing",
            "Dependency graph analysis",
            "Intelligent context ranking",
            "Multi-language support (Python, JS, TS, Java, Go, C++)",
            "Token budget management",
            "Symbol relationship mapping"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.server:app",  # Use string format for auto-reload
        host="0.0.0.0", 
        port=8000,
        reload=True
    )

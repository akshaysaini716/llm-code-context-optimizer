from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import Optional, List
from services.context_service import (
    get_project_context, 
    get_graph_stats, 
    find_related_symbols,
    clear_graph_cache
)
from services.gemini_service import chat_with_gemini

app = FastAPI(title="LLM Code Context Optimizer", version="1.0.0")

class ChatRequest(BaseModel):
    message: str
    include_context: bool = False
    project_path: str | None = None
    relevant_only: bool = True
    max_tokens: int = 8000
    use_tree_sitter: bool = True

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
        response["context_metadata"] = ctx.get("metadata", {})
    
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

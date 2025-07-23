## ✅ Phase 1: MVP (COMPLETED)
 ✅ Setup tree-sitter for 5 languages (Python, JS, TS, Java, Kotlin) \
 ✅ Build symbol dependency graph with AST parsing \
 ✅ Create intelligent token budgeting system \
 ✅ Build REST API with optimized context generation \
 ✅ Implement relevance scoring and file prioritization \

Phase 2: Embedding Engine
 [] Integrate OpenAI/Cohere embeddings
 [] Chunk codebase by function/class/module
 [] Set up vector DB with retrieval

Phase 3: Token Budgeting & Ranking
 [] Implement scoring heuristics:
    - Symbol relevance
    - Recent file edits
    - Embedding similarity
 [] Add LLM-based summarizer for long tail files

Phase 4: Editor Integration
 [] Expose APIs for prompt generation
 [] Add to Monaco/VSCode extension
 [] Add telemetry: cost saved, tokens used, response time


-----------------------------------------------------------

## Ways
1. Code Parsing & Symbol Graph:
    - Using Tree-sitter, parse the code
    - Extract symbols, imports, functions, classes, and their relationships
    - Create a dependency graph (NetworkX)
    - Find related context using the graph based on the relevance score

2. Embedding Store:
   - Build a vector database (Pinecone/Weaviate)
   - Use Semantic Search to find similar embeddings (FAISS)

3. Chunkers:
   - Chunk the codebase by function/class/module
   - Use the chunker to create a context for each file





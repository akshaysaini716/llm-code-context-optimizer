## Start the server
uvicorn rag.api.rag_server:app --reload


## Tasks
1. Single File Update in Vector DB
2. Summarize the large files
3. Single File Codebase index
4. 



## Implementation
The implementation is divided into the following parts:
1. Creating the chunks, Embeddings, storing in db
   - Create Chunks:
      - Tree Sitter Parser for multiple programming languages
      - Using Tree Sitter Queries, a pattern matching system for extracting specific code structures from Abstract Syntax Trees (ASTs)
   - Embeddings
   - Store in Qdrant DB
2. Retrival 
    - Boosting the retrival by ranking the embeddings, reranking, again sliding window for more context
    - Context Fusion (Using multiple retrieved pieces together)
3. Full Prompt request and response


## References
1. Tree Sitter: https://aider.chat/2023/10/22/repomap.html
2. 
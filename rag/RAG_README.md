## Start the server
uvicorn rag.api.rag_server:app --reload


## Tasks
1. Single File Update in Vector DB
2. Summarize the large files
3. Single File Codebase index
4. Slidling Window Context Fusion 
5. Context Boosting - implemented recent edit files, open files, etc need to test.
6. 


Chunking:
1. Smart chunks, class methods with class context
2. overlap chunks
3. Context Preservation - including imports, parent-child relationship

Retrival:
1. Query Analysis - def, buh, error, implementation, etc
2. Exact sysmbol match
3. Semantic Search
4. Context Expansion - using session awareness
5. 



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
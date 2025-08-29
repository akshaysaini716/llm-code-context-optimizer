## Start the server
uvicorn rag.api.rag_server:app --reload

## Tasks
1. Token Approximation Approach improvements 
2. Chat with LLM using Previous Context
3. File watcher for real time updates
4. Implement Caching
5. Summarize the large files
6. Move the sliding window for more context
7. 



## Implementation
The implementation is divided into the following parts:
1. Creating the chunks, Embeddings
   - Create Chunks:
      - Tree Sitter Parser for multiple programming languages
      - Using Tree Sitter Queries, a pattern matching system for extracting specific code structures from Abstract Syntax Trees (ASTs)
   - Embeddings
   - Store in Qdrant DB
2. Retrival 
    - Boosting the retrival by ranking the embeddings, reranking, again sliding window for more context
    - Context Fusion (Using multiple retrieved pieces together)
3. Full Prompt request and response

graph TB
    A[File Change] --> B[Incremental Parser]
    B --> C[Multi-Index Update]
    C --> D[Symbol Index]
    C --> E[Text Index] 
    C --> F[Semantic Index]
    C --> G[Graph Index]
    
    H[Search Query] --> I[Context Builder]
    I --> J[Multi-Stage Search]
    J --> D
    J --> E
    J --> F
    J --> G
    
    J --> K[Result Ranker]
    K --> L[Context Expander]
    L --> M[Cache Manager]
    M --> N[Final Results]



1. Context Window
2. Postgres Implement
3. 


Using Qdrant as Vector Database and for sematic search
Why Qdrant?
1. Open Source and Cloud Both (minimal charges), self hosted
2. It also supports filtering
3. CRUD operations available
4. Can be used for production, monitoring/clustering available
5. Auto Scalable


## References
1. Tree Sitter: https://aider.chat/2023/10/22/repomap.html
2. 
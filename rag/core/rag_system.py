import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional

from rag.chunking.enhanced_tree_sitter_chunker import EnhancedTreeSitterChunker
from rag.chunking.tree_sitter_chunker import TreeSitterChunker
from rag.configs import ChunkingConfig
from rag.embedding.embedder import CodeEmbedder
from rag.embedding.enhanced_embedder import EnhancedCodeEmbedder
from rag.models import RAGResponse
from rag.retrieval.context_fusion import ContextFusion
from rag.retrieval.enhanced_context_fusion import EnhancedContextFusion
from rag.retrieval.enhanced_retriever import EnhancedContextRetriever
from rag.retrieval.retriever import ContextRetriever
from rag.vector_store.qdrant_client_impl import QdrantClientImpl

logger = logging.getLogger(__name__)

class RAGSystem:

    # TODO: Insert config here
    def __init__(self):
        config = ChunkingConfig(preserve_class_methods=True)
        self.chunker = EnhancedTreeSitterChunker(config)
        self.embedder = EnhancedCodeEmbedder()
        self.qdrant_client = QdrantClientImpl()
        self.retriever = EnhancedContextRetriever()
        self.context_fuser = EnhancedContextFusion()

    # TODO: force_reindex implementation
    def index_codebase(self, project_path, file_patterns, force_reindex) -> Dict[str, Any]:
        start_time = time.time()
        project_path = Path(project_path)
        file_patterns = file_patterns or ["*.py", "*.js", "*.ts", "*.java", "*.kt", "*.go"]
        all_files = []
        for pattern in file_patterns:
            files = list(project_path.rglob(pattern))
            all_files.extend(files)
        logger.info(f"Found {len(all_files)} code files to index")

        # Chunking the code files
        all_chunks = self.chunker.chunk_files_parallel(all_files)
        logger.info(f"Generated {len(all_chunks)} chunks")

        # Creating the embeddings
        chunks_with_embeddings = self.embedder.embed_chunks(all_chunks)
        embedded_count = sum(1 for chunk in chunks_with_embeddings if chunk.embedding)
        logger.info(f"Created embeddings for {embedded_count}/{len(all_chunks)} chunks")

        # Store the embeddings in vector DB, Qdrant here
        success = self.qdrant_client.upsert_chunks(chunks_with_embeddings)
        if not success:
            raise Exception("Failed to upsert chunks into vector DB")

        indexing_time = time.time() - start_time

        # TODO: Add the metrics here

        return {
            "status": "success",
            "files_processed": len(all_files),
            "chunks_created": len(all_chunks),
            "chunks_embedded": embedded_count,
            "indexing_time_seconds": indexing_time,
            "collection_info": self.qdrant_client.get_collection_info()
        }

    def query(self, query: str, max_context_tokens: int = 8000, top_k: int = 10) -> RAGResponse:
        start_time = time.time()
        logger.info(f"Process Query Retrieval for query: {query}")
        try:
            # TODO: Implement cache
            retrieval_result = self.retriever.retrieve_with_context(query, top_k*2) # get more for reranking
            if not retrieval_result:
                return RAGResponse(
                    context="",
                    chunks_used=[],
                    total_tokens=0,
                    retrieval_time_ms=(time.time() - start_time) * 1000
                )

            final_results = retrieval_result[:top_k]
            final_context = self.context_fuser.fuse_context(final_results, max_context_tokens)
            chunks_used = [result.chunk for result in final_results]
            response = RAGResponse(
                context=final_context,
                chunks_used=chunks_used,
                total_tokens=len(self.context_fuser.tokenizer.encode(final_context)),
                retrieval_time_ms=(time.time() - start_time) * 1000
            )

            # TODO: Update metrics and Cache
            return response
        except Exception as e:
            raise Exception(f"Failed to process query retrieval: {str(e)}")




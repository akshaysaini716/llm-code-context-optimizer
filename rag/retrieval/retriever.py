import logging
from typing import Optional, List

import tiktoken

from rag.embedding.embedder import CodeEmbedder
from rag.models import RAGResponse, RetrievalResult
from rag.vector_store.vector_store_factory import get_vector_store
from rag.configs import VectorStoreConfig

logger = logging.getLogger(__name__)

class ContextRetriever:
    def __init__(self, vector_store_config: Optional[VectorStoreConfig] = None):
        self.vector_store = get_vector_store(vector_store_config)
        self.embedder = CodeEmbedder()
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def retrieve(self, query: str, top_k: int) -> List[RetrievalResult]:
        try:
            query_embedding = self.embedder.embed_query(query)
            if not query_embedding:
                return []

            filters = None
            results = self.vector_store.search_similar(
                query_vector=query_embedding,
                filters = filters,
                top_k=top_k
            )
            return results
        except Exception as e:
            logger.error(f"Error in retrieve: {e}")
            return []




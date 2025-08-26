import logging
from typing import List, Dict, Any, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, Filter,
    FieldCondition, Match, Range, SearchRequest
)
try:
    from qdrant_client.models import Range
except ImportError:
    # Fallback for different qdrant versions
    Range = None
from rag.models import CodeBaseChunk, RetrievalResult

logger = logging.getLogger(__name__)

class QdrantClientImpl:
    def __init__(self):
        self.client = QdrantClient(
            host="localhost",
            port=6333,
            timeout=30
        )

        self.collection_name = "code_chunks"
        self._ensure_collection()

    def _ensure_collection(self):
        try:
            collection = self.client.get_collections()
            existing_collections = [col.name for col in collection.collections]
            if self.collection_name not in existing_collections:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=768,
                        distance=Distance.COSINE
                    )
                )
            else:
               # self.client.delete_collection(self.collection_name)
                logger.info(f"Collection {self.collection_name} already exists")
        except Exception as e:
            logger.error(f"Failed to ensure collection: {e}")

    def upsert_chunks(self, chunks: List[CodeBaseChunk]) -> bool:
        if not chunks:
            return False

        try:
            points = []
            for chunk in chunks:
                if chunk.embedding:
                    point_data = chunk.to_qdrant_point()
                    points.append(PointStruct(
                        id=point_data["id"],
                        vector=point_data["vector"],
                        payload=point_data["payload"]
                    ))
            if not points:
                logger.error("No chunks with embeddings to upsert")
                return False

            self.client.upsert(
                collection_name="code_chunks",
                points=points
            )
            return True
        except Exception as e:
            logger.error(f"Error upserting chunks: {e}")
            return False

    def get_collection_info(self) -> Dict[str, Any]:
        try:
            info = self.client.get_collection("code_chunks")
            return {
                "total_points": info.points_count,
                "vector_size": info.config.params.vectors.size,
                "distance": info.config.params.vectors.distance,
                "status": info.status
            }
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return {}

    def search_similar(self, query_vector: List[float], filters: Optional[Dict[str, Any]] = None, top_k: int = 10) -> List[RetrievalResult]:
        try:
            qdrant_filter = None
            if filters:
                # This is a simplified filter conversion - expand as needed
                filter_conditions = []
                for key, value in filters.items():
                    filter_conditions.append(
                        FieldCondition(key=key, match=Match(value=value))
                    )
                if filter_conditions:
                    qdrant_filter = Filter(must=filter_conditions)
            search_result = self.client.search(
                collection_name="code_chunks",
                query_vector=query_vector,
                query_filter=qdrant_filter,
                limit=top_k,
                with_payload=True,
                with_vectors=False
            )
            results = []
            for i, scored_point in enumerate(search_result):
                chunk = self._point_to_chunk(scored_point)
                if chunk:
                    result = RetrievalResult(
                        chunk=chunk,
                        relevance_score=scored_point.score
                    )
                    results.append(result)
            return results
        except Exception as e:
            logger.error(f"Failed to search result: {e}")
            return []

    def search_by_filter(self, filters: Dict[str, Any], limit: int = 10) -> List[CodeBaseChunk]:
        """Search chunks by filter conditions only (no vector similarity)"""
        try:
            # Bypass Qdrant filtering entirely and use manual filtering
            # This avoids the typing.Union instantiation issues
            logger.debug(f"Using manual filtering for: {filters}")
            return self._manual_filter_chunks(filters, limit)
            
        except Exception as e:
            logger.error(f"Failed to search by filter: {e}")
            return []

    def _manual_filter_chunks(self, filters: Dict[str, Any], limit: int) -> List[CodeBaseChunk]:
        """Manual filtering when Qdrant filters can't be used"""
        try:
            # Get all chunks and filter manually
            scroll_result = self.client.scroll(
                collection_name="code_chunks",
                limit=limit * 10,  # Get more chunks to filter from
                with_payload=True,
                with_vectors=False
            )
            
            results = []
            for point in scroll_result[0]:
                chunk = self._point_to_chunk_from_record(point)
                if chunk and self._matches_filters(chunk, filters):
                    results.append(chunk)
                    if len(results) >= limit:
                        break
            
            return results
            
        except Exception as e:
            logger.error(f"Failed manual filtering: {e}")
            return []

    def _matches_filters(self, chunk: CodeBaseChunk, filters: Dict[str, Any]) -> bool:
        """Check if a chunk matches the given filters"""
        try:
            for key, value in filters.items():
                chunk_value = getattr(chunk, key, None)
                
                if isinstance(value, dict):
                    # Handle range conditions
                    if "$gte" in value and chunk_value < value["$gte"]:
                        return False
                    if "$lte" in value and chunk_value > value["$lte"]:
                        return False
                    if "$gt" in value and chunk_value <= value["$gt"]:
                        return False
                    if "$lt" in value and chunk_value >= value["$lt"]:
                        return False
                else:
                    # Simple equality
                    if chunk_value != value:
                        return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error matching filters: {e}")
            return False

    def _point_to_chunk(self, scored_point) -> Optional[CodeBaseChunk]:
        try:
            payload = scored_point.payload
            return CodeBaseChunk(
                id = str(scored_point.id),
                file_path = payload.get("file_path", ""),
                content=payload.get("content", ""),
                language=payload.get("language", ""),
                chunk_type=payload.get("chunk_type", ""),
                start_byte=payload.get("start_byte", 0),
                end_byte=payload.get("end_byte", 0),
                start_line=payload.get("start_line", 0),
                end_line=payload.get("end_line", 0)
            )
        except Exception as e:
            logger.error(f"Failed to convert point to chunk: {e}")
            return None

    def _point_to_chunk_from_record(self, record) -> Optional[CodeBaseChunk]:
        """Convert a record from scroll results to CodeBaseChunk"""
        try:
            payload = record.payload
            return CodeBaseChunk(
                id = str(record.id),
                file_path = payload.get("file_path", ""),
                content=payload.get("content", ""),
                language=payload.get("language", ""),
                chunk_type=payload.get("chunk_type", ""),
                start_byte=payload.get("start_byte", 0),
                end_byte=payload.get("end_byte", 0),
                start_line=payload.get("start_line", 0),
                end_line=payload.get("end_line", 0)
            )
        except Exception as e:
            logger.error(f"Failed to convert record to chunk: {e}")
            return None

import logging
from typing import List
from sentence_transformers import SentenceTransformer
import tiktoken

from rag.models import CodeBaseChunk

logger = logging.getLogger(__name__)

class CodeEmbedder:
    def __init__(self):
        self.code_model = SentenceTransformer("all-mpnet-base-v2")
        self.text_model = SentenceTransformer("all-mpnet-base-v2")
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def embed_chunks(self, chunks: List[CodeBaseChunk]) -> List[CodeBaseChunk]:
        if not chunks:
            return chunks

        try:
            code_chunks = []
            text_chunks = []
            mixed_chunks = []
            for chunk in chunks:
                if self._is_pure_code(chunk):
                    code_chunks.append(chunk)
                elif self._is_pure_text(chunk):
                    text_chunks.append(chunk)
                else:
                    mixed_chunks.append(chunk)

            if code_chunks and self.code_model:
                self._embed_chunk_group(code_chunks, self.code_model, "code")

            if text_chunks and self.text_model:
                self._embed_chunk_group(text_chunks, self.text_model, "text")

            if mixed_chunks:
                model = self.code_model if self.code_model else self.text_model
                self._embed_chunk_group(mixed_chunks, model, "mixed")

        except Exception as e:
            logger.error(f"Error embedding chunks: {e}")
        return chunks

    def _embed_chunk_group(self, chunks: List[CodeBaseChunk], model: SentenceTransformer, content_type: str):
        try:
            texts = []
            for chunk in chunks:
                text = self._prepare_text_for_embedding(chunk, content_type)
                texts.append(text)

            embeddings = model.encode(
                texts,
                batch_size=32,
                show_progress_bar=len(texts) > 50,
                convert_to_numpy=True,
                normalize_embeddings=True
            )

            for chunk, embedding in zip(chunks, embeddings):
                chunk.embedding = embedding.tolist()

        except Exception as e:
            logger.error(f"Error embedding {content_type} chunks: {e}")


    def _prepare_text_for_embedding(self, chunk: CodeBaseChunk, content_type: str) -> str:
        if content_type == "code":
            context_parts = []
            if chunk.file_path:
                filename = chunk.file_path.split("/")[-1]
                context_parts.append(f"File: {filename}")
            # if chunk.symbols:
            #     symbol_names = [s.name for s in chunk.symbols]
            #     context_parts.append(f"Symbols: {', '.join(symbol_names)}")
            if chunk.chunk_type:
                context_parts.append(f"Type: {chunk.chunk_type}")

            context = " | ".join(context_parts)
            return f"{context}\n\n{chunk.content}"
        elif content_type == "text":
            return chunk.content
        else:
            return chunk.content

    def _is_pure_code(self, chunk: CodeBaseChunk) -> bool:
        # TODO: implement
        return chunk.chunk_type == "code"

    def _is_pure_text(self, chunk: CodeBaseChunk) -> bool:
        # TODO: implement
        return chunk.chunk_type == "text"

    def embed_query(self, query: str, query_type: str = "mixed") -> List[float]:
        try:
            model = self.text_model
            embedding = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Error embedding query: {e}")
            return []

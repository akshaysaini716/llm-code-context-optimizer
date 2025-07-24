import faiss
from sentence_transformers import SentenceTransformer
import pickle
import json
import os


class EmbeddingService:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.code_chunks = []
        self.metadata = []

    def create_embeddings(self, code_chunks):
        """Create embeddings for code chunks"""
        if not code_chunks:
            print("Warning: No code chunks to embed")
            return []

        # Extract content for embedding
        texts = [chunk['content'] for chunk in code_chunks if chunk.get('content', '').strip()]

        if not texts:
            print("Warning: No valid text content to embed")
            return []

        print(f"Creating embeddings for {len(texts)} chunks")

        # Create embeddings
        embeddings = self.model.encode(texts)

        # Handle empty or invalid embeddings
        if embeddings is None or len(embeddings) == 0:
            print("Warning: Failed to create embeddings")
            return []

        # Ensure embeddings is 2D
        if len(embeddings.shape) == 1:
            embeddings = embeddings.reshape(1, -1)

        dimension = embeddings.shape[1]

        # Create FAISS index
        if not hasattr(self, 'index') or self.index is None:
            self.index = faiss.IndexFlatIP(dimension)

        # Store chunks with metadata
        self.chunks = []
        for i, chunk in enumerate(code_chunks):
            if i < len(embeddings):  # Safety check
                self.chunks.append({
                    'content': chunk['content'],
                    'metadata': {
                        'file_path': chunk['file_path'],
                        'type': chunk['type'],
                        'name': chunk['name'],
                        'line_start': chunk.get('line_start', 0)
                    }
                })

        # Add embeddings to index
        self.index.add(embeddings.astype('float32'))

        return embeddings

    def search_similar(self, query, k=5):
        """Find k most similar code chunks"""
        if not hasattr(self, 'index') or self.index is None:
            print("Warning: No index available for search")
            return []

        if not hasattr(self, 'chunks') or not self.chunks:
            print("Warning: No chunks available for search")
            return []

        query_embedding = self.model.encode([query])

        # Ensure query_embedding is 2D
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)

        scores, indices = self.index.search(query_embedding.astype('float32'), k)

        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.chunks):
                results.append({
                    'content': self.chunks[idx]['content'],
                    'metadata': self.chunks[idx]['metadata'],
                    'score': scores[0][i]
                })

        return results

    def save_index(self, path):
        """Save FAISS index to disk"""

        # Create directory if it doesn't exist
        os.makedirs(path, exist_ok=True)

        if hasattr(self, 'index') and self.index is not None:
            faiss.write_index(self.index, f"{path}/faiss.index")

            # Save metadata
            with open(f"{path}/metadata.json", 'w') as f:
                json.dump({
                    'chunks': getattr(self, 'chunks', []),
                    'code_chunks': getattr(self, 'code_chunks', []),
                    'metadata': getattr(self, 'metadata', [])
                }, f)
            print(f"Index saved to {path}")
        else:
            print("Warning: No index to save")

    def load_index(self, path):
        """Load index and metadata"""
        index_file = f"{path}/faiss.index"
        metadata_file = f"{path}/metadata.json"

        if not os.path.exists(index_file):
            print(f"Warning: Index file {index_file} not found. Starting fresh.")
            return False

        try:
            self.index = faiss.read_index(index_file)

            # Load metadata if it exists
            if os.path.exists(metadata_file):
                with open(metadata_file, 'r') as f:
                    data = json.load(f)
                    self.chunks = data.get('chunks', [])
                    self.code_chunks = data.get('code_chunks', [])
                    self.metadata = data.get('metadata', [])

            print(f"Index loaded from {path}")
            return True

        except Exception as e:
            print(f"Error loading index: {e}")
            return False
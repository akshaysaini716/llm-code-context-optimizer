import faiss
import json
import os
import numpy as np
from pathlib import Path
from .embedder import CodeEmbedder
from .chunker import chunk_code_file

class CodeIndexer:
    def __init__(self, code_dir, index_path="faiss.index", meta_path="metadata.json"):
        self.code_dir = Path(code_dir)
        self.index_path = Path(index_path)
        self.meta_path = Path(meta_path)
        self.embedder = CodeEmbedder()
        self.index = None
        self.metadata = []

    def build_index(self):
        all_chunks = []
        for file in self.code_dir.rglob("*"):
            if file.suffix in [".py", ".java", ".kt", ".js", ".ts"]:
                chunks = chunk_code_file(file)
                all_chunks.extend(chunks)

        texts = [c["chunk"] for c in all_chunks]
        self.metadata = all_chunks

        embeddings = self.embedder.encode(texts)
        faiss.normalize_L2(embeddings)

        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)

        faiss.write_index(self.index, str(self.index_path))
        with open(self.meta_path, "w") as f:
            json.dump(self.metadata, f, indent=2)

    def load_index(self):
        self.index = faiss.read_index(str(self.index_path))
        with open(self.meta_path) as f:
            self.metadata = json.load(f)

    def search(self, query, top_k=5):
        query_vec = self.embedder.encode([query])
        faiss.normalize_L2(query_vec)
        D, I = self.index.search(query_vec, top_k)
        return [self.metadata[i] for i in I[0]]

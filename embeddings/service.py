from .indexer import CodeIndexer
from .file_watcher import start_watching

class CodeSearchService:
    def __init__(self, code_dir):
        self.code_dir = code_dir
        self.indexer = CodeIndexer(code_dir)
        self.indexer.build_index()
        start_watching(code_dir, self.indexer)

    def query(self, q: str, k: int = 5):
        self.indexer.load_index()
        return self.indexer.search(q, top_k=k)

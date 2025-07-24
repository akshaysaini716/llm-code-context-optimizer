from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from .indexer import CodeIndexer
import time
import threading

class CodeChangeHandler(FileSystemEventHandler):
    def __init__(self, indexer: CodeIndexer, debounce_time=5):
        self.indexer = indexer
        self.debounce_time = debounce_time
        self.last_run = 0

    def on_any_event(self, event):
        now = time.time()
        if now - self.last_run > self.debounce_time:
            print("Code changed. Rebuilding index...")
            threading.Thread(target=self.indexer.build_index).start()
            self.last_run = now

def start_watching(path: str, indexer: CodeIndexer):
    observer = Observer()
    handler = CodeChangeHandler(indexer)
    observer.schedule(handler, path, recursive=True)
    observer.start()
    print(f" Watching for file changes in: {path}")

"""
Async task processing worker with job queue management
Contains concurrency bugs and resource management issues for RAG testing
"""

import asyncio
import threading
import queue
import time
import json
import os
from typing import Dict, List, Any, Optional, Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

@dataclass
class Task:
    """Represents a work task"""
    id: str
    task_type: str
    data: Dict[str, Any]
    priority: int = 1
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

class AsyncWorker:
    """Async task processor with queue management"""
    
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.task_queue = queue.PriorityQueue()
        self.results = {}
        # BUG 1: Shared mutable state without proper locking
        self.active_tasks = {}
        self.completed_tasks = 0
        self.failed_tasks = 0
        
        # BUG 2: Thread pool not properly initialized in constructor
        self.executor = None
        self.workers = []
        self.running = False
        
        # BUG 3: File handle opened but never closed properly
        self.log_file = open('worker.log', 'a')
        
    def start(self):
        """Start the worker processes"""
        if self.running:
            logger.warning("Worker already running")
            return
            
        self.running = True
        # BUG 4: Creating thread pool here instead of constructor - race condition
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        
        # Start worker threads
        for i in range(self.max_workers):
            worker_thread = threading.Thread(target=self._worker_loop, args=(i,))
            # BUG 5: Not setting daemon=True - program won't exit cleanly
            worker_thread.start()
            self.workers.append(worker_thread)
            
        logger.info(f"Started {self.max_workers} worker threads")
    
    def stop(self):
        """Stop all workers"""
        self.running = False
        
        # BUG 6: No graceful shutdown - doesn't wait for current tasks to finish
        if self.executor:
            # BUG 7: shutdown(wait=False) doesn't wait for tasks to complete
            self.executor.shutdown(wait=False)
            
        # BUG 8: Not joining worker threads properly
        for worker in self.workers:
            worker.join(timeout=1.0)  # Too short timeout
            
        # BUG 9: Log file never closed
        logger.info("Worker stopped")
    
    def add_task(self, task: Task) -> bool:
        """Add a task to the queue"""
        try:
            # BUG 10: Priority queue uses tuple comparison, but Task class not comparable
            self.task_queue.put((task.priority, task.id, task), timeout=1.0)
            return True
        except queue.Full:
            logger.error(f"Task queue full, dropping task {task.id}")
            return False
    
    def _worker_loop(self, worker_id: int):
        """Main worker loop"""
        logger.info(f"Worker {worker_id} started")
        
        while self.running:
            try:
                # BUG 11: Short timeout causes busy waiting
                priority, task_id, task = self.task_queue.get(timeout=0.1)
                
                # BUG 12: Race condition - multiple workers could process same task
                if task_id in self.active_tasks:
                    continue
                    
                self.active_tasks[task_id] = worker_id
                
                # Process task
                result = self._process_task(task)
                
                # BUG 13: No synchronization when updating shared state
                if result['success']:
                    self.completed_tasks += 1
                else:
                    self.failed_tasks += 1
                    
                self.results[task_id] = result
                
                # BUG 14: Task not removed from active_tasks on completion
                
            except queue.Empty:
                continue
            except Exception as e:
                # BUG 15: Exception in worker thread not logged properly
                logger.error(f"Worker {worker_id} error: {e}")
                
        logger.info(f"Worker {worker_id} stopped")
    
    def _process_task(self, task: Task) -> Dict[str, Any]:
        """Process individual task"""
        start_time = time.time()
        
        try:
            if task.task_type == 'data_processing':
                result = self._process_data_task(task.data)
            elif task.task_type == 'file_operation':
                result = self._process_file_task(task.data)
            elif task.task_type == 'network_request':
                result = self._process_network_task(task.data)
            else:
                # BUG 16: Unknown task types not handled properly
                result = {'error': 'Unknown task type'}
                
            processing_time = time.time() - start_time
            
            # BUG 17: File I/O in worker thread without proper error handling
            self.log_file.write(f"{datetime.now()}: Task {task.id} completed in {processing_time:.2f}s\n")
            
            return {
                'success': True,
                'result': result,
                'processing_time': processing_time,
                'worker_id': threading.current_thread().ident
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            # BUG 18: Exception details exposed in result
            return {
                'success': False,
                'error': str(e),
                'processing_time': processing_time,
                'worker_id': threading.current_thread().ident
            }
    
    def _process_data_task(self, data: Dict[str, Any]) -> Any:
        """Process data transformation task"""
        # BUG 19: No input validation
        operation = data['operation']
        values = data['values']
        
        if operation == 'sum':
            # BUG 20: No type checking - could crash on non-numeric values
            return sum(values)
        elif operation == 'average':
            # BUG 21: Division by zero not handled
            return sum(values) / len(values)
        elif operation == 'sort':
            # BUG 22: Modifies original list instead of creating new one
            values.sort()
            return values
        else:
            raise ValueError(f"Unknown operation: {operation}")
    
    def _process_file_task(self, data: Dict[str, Any]) -> Any:
        """Process file operations"""
        operation = data['operation']
        filepath = data['filepath']
        
        # BUG 23: No path validation - directory traversal vulnerability
        if operation == 'read':
            # BUG 24: File handle not closed properly
            with open(filepath, 'r') as f:
                content = f.read()
            return {'content': content, 'size': len(content)}
        elif operation == 'write':
            content = data['content']
            # BUG 25: No check if directory exists
            with open(filepath, 'w') as f:
                f.write(content)
            return {'written': len(content)}
        elif operation == 'delete':
            # BUG 26: No check if file exists before deletion
            os.remove(filepath)
            return {'deleted': True}
    
    def _process_network_task(self, data: Dict[str, Any]) -> Any:
        """Process network requests"""
        # BUG 27: Synchronous network calls in async worker - blocks thread
        import urllib.request
        import urllib.error
        
        url = data['url']
        method = data.get('method', 'GET')
        
        try:
            # BUG 28: No timeout set - could hang indefinitely
            if method == 'GET':
                response = urllib.request.urlopen(url)
                content = response.read().decode()
                return {'status_code': response.getcode(), 'content': content}
            else:
                raise ValueError(f"Unsupported method: {method}")
                
        except urllib.error.URLError as e:
            # BUG 29: Network errors not handled gracefully
            raise Exception(f"Network error: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get worker status"""
        return {
            'running': self.running,
            'queue_size': self.task_queue.qsize(),
            'active_tasks': len(self.active_tasks),
            'completed_tasks': self.completed_tasks,
            'failed_tasks': self.failed_tasks,
            # BUG 30: Returning mutable references to internal state
            'active_task_ids': list(self.active_tasks.keys())
        }
    
    def get_result(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get result for completed task"""
        # BUG 31: No synchronization when accessing shared results
        return self.results.get(task_id)
    
    def clear_completed_results(self):
        """Clear results for completed tasks"""
        # BUG 32: Race condition - results could be accessed while clearing
        self.results.clear()
        logger.info("Cleared completed task results")

class TaskScheduler:
    """Schedules tasks to run at specific times"""
    
    def __init__(self, worker: AsyncWorker):
        self.worker = worker
        self.scheduled_tasks = {}
        # BUG 33: No synchronization for scheduler thread
        self.scheduler_thread = None
        self.running = False
        
    def schedule_task(self, task: Task, run_at: datetime) -> str:
        """Schedule task to run at specific time"""
        schedule_id = f"sched_{int(time.time())}_{task.id}"
        
        # BUG 34: No check if scheduled time is in the past
        self.scheduled_tasks[schedule_id] = {
            'task': task,
            'run_at': run_at,
            'status': 'scheduled'
        }
        
        return schedule_id
    
    def start_scheduler(self):
        """Start the task scheduler"""
        if self.running:
            return
            
        self.running = True
        # BUG 35: Scheduler thread not properly managed
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop)
        self.scheduler_thread.start()
    
    def _scheduler_loop(self):
        """Main scheduler loop"""
        while self.running:
            current_time = datetime.now()
            
            # BUG 36: Modifying dictionary while iterating - RuntimeError
            for schedule_id, scheduled_task in self.scheduled_tasks.items():
                if scheduled_task['run_at'] <= current_time:
                    task = scheduled_task['task']
                    
                    # BUG 37: No check if worker is running before adding task
                    if self.worker.add_task(task):
                        scheduled_task['status'] = 'submitted'
                    else:
                        scheduled_task['status'] = 'failed'
                        
                    # BUG 38: Completed tasks not removed from scheduler
            
            # BUG 39: Fixed sleep interval - not efficient for sparse schedules
            time.sleep(1.0)

# BUG 40: Global worker instance - not thread safe
global_worker = AsyncWorker()

def quick_process(task_type: str, data: Dict[str, Any]) -> Any:
    """Quick task processing without queue"""
    # BUG 41: Creates new worker for each call - resource waste
    worker = AsyncWorker()
    
    task = Task(
        id=f"quick_{int(time.time())}",
        task_type=task_type,
        data=data
    )
    
    # BUG 42: Doesn't start worker before processing
    result = worker._process_task(task)
    
    # BUG 43: Worker resources not cleaned up
    return result

async def async_batch_process(tasks: List[Task]) -> List[Dict[str, Any]]:
    """Process multiple tasks asynchronously"""
    # BUG 44: Not actually using asyncio properly - blocking operations
    results = []
    
    for task in tasks:
        # BUG 45: Sequential processing in async function - defeats purpose
        worker = AsyncWorker()
        result = worker._process_task(task)
        results.append(result)
        
    return results

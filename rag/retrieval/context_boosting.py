"""
Context-Aware Boosting System for RAG Retrieval
Inspired by Cursor IDE's intelligent file prioritization and context awareness
"""
import logging
import time
from typing import Dict, List, Set, Optional, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path
import json
import threading
from collections import defaultdict, deque

from rag.models import RetrievalResult, CodeBaseChunk

logger = logging.getLogger(__name__)

@dataclass
class FileContext:
    """Context information about a file"""
    file_path: str
    last_edited: float = 0.0
    last_opened: float = 0.0
    edit_frequency: int = 0  # Number of edits in session
    open_frequency: int = 0  # Number of opens in session
    time_spent: float = 0.0  # Total time spent in file
    is_currently_open: bool = False
    related_files: Set[str] = field(default_factory=set)  # Files often edited together
    import_relationships: Set[str] = field(default_factory=set)  # Files this imports from

@dataclass
class QueryContext:
    """Context for the current query/session"""
    current_file: Optional[str] = None
    cursor_position: Optional[Tuple[int, int]] = None  # (line, column)
    recent_files: List[str] = field(default_factory=list)  # Last 10 files
    session_files: Set[str] = field(default_factory=set)  # All files in session
    query_history: List[str] = field(default_factory=list)  # Recent queries
    project_root: Optional[str] = None

@dataclass
class BoostConfig:
    """Configuration for different boost factors"""
    # Recency boosts
    recent_edit_boost: float = 1.5      # Files edited in last hour
    recent_open_boost: float = 1.3      # Files opened recently
    current_file_boost: float = 2.0     # Currently viewing file
    
    # Frequency boosts
    high_frequency_boost: float = 1.2   # Frequently accessed files
    session_file_boost: float = 1.15    # Files accessed this session
    
    # Relationship boosts
    import_relationship_boost: float = 1.4  # Files with import relationships
    directory_proximity_boost: float = 1.1  # Files in same directory
    related_files_boost: float = 1.25   # Files often edited together
    
    # Project structure boosts
    main_file_boost: float = 1.1        # Entry points (main.py, index.js)
    config_file_boost: float = 1.05     # Config files when relevant
    test_file_penalty: float = 0.9      # Slight penalty for test files
    
    # Time decay factors
    edit_decay_hours: float = 24.0      # How fast recency decays
    frequency_decay_days: float = 7.0   # How fast frequency decays
    
    # Thresholds
    min_boost: float = 0.5              # Minimum boost multiplier
    max_boost: float = 3.0              # Maximum boost multiplier

class ContextAwareBooster:
    """
    Context-aware boosting system that enhances retrieval based on:
    1. Recently edited/opened files (like Cursor IDE)
    2. File relationships and import dependencies  
    3. Project structure and patterns
    4. User behavior and session context
    """
    
    def __init__(self, config: Optional[BoostConfig] = None, persistence_file: Optional[str] = None):
        self.config = config or BoostConfig()
        self.persistence_file = persistence_file
        
        # File context tracking
        self.file_contexts: Dict[str, FileContext] = {}
        self.query_context = QueryContext()
        
        # Session tracking
        self.session_start_time = time.time()
        self.query_history: deque = deque(maxlen=50)  # Last 50 queries
        self.edit_events: deque = deque(maxlen=1000)  # Last 1000 edit events
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Load persistent context
        self._load_context()
        
        # File pattern recognition
        self.main_file_patterns = [
            'main.py', 'index.js', 'index.ts', 'app.py', 'server.py',
            'main.java', 'Main.java', 'index.html', 'app.js'
        ]
        
        self.config_file_patterns = [
            'config.py', 'settings.py', '.env', 'package.json', 'requirements.txt',
            'pom.xml', 'build.gradle', 'Dockerfile', 'docker-compose.yml'
        ]
        
        self.test_file_patterns = [
            'test_', '_test.py', '.test.js', '.spec.js', 'tests/',
            'Test.java', 'TestCase.java', '__tests__/'
        ]
    
    def track_file_edit(self, file_path: str, edit_time: Optional[float] = None):
        """Track when a file is edited"""
        edit_time = edit_time or time.time()
        
        with self.lock:
            if file_path not in self.file_contexts:
                self.file_contexts[file_path] = FileContext(file_path=file_path)
            
            context = self.file_contexts[file_path]
            context.last_edited = edit_time
            context.edit_frequency += 1
            
            # Track edit event
            self.edit_events.append((file_path, edit_time, 'edit'))
            
            # Update related files (files edited close in time)
            self._update_related_files(file_path, edit_time)
            
            logger.debug(f"Tracked edit for {file_path}")
    
    def track_file_open(self, file_path: str, open_time: Optional[float] = None):
        """Track when a file is opened"""
        open_time = open_time or time.time()
        
        with self.lock:
            if file_path not in self.file_contexts:
                self.file_contexts[file_path] = FileContext(file_path=file_path)
            
            context = self.file_contexts[file_path]
            context.last_opened = open_time
            context.open_frequency += 1
            context.is_currently_open = True
            
            # Update recent files list
            if file_path in self.query_context.recent_files:
                self.query_context.recent_files.remove(file_path)
            self.query_context.recent_files.insert(0, file_path)
            self.query_context.recent_files = self.query_context.recent_files[:10]
            
            # Add to session files
            self.query_context.session_files.add(file_path)
            
            logger.debug(f"Tracked open for {file_path}")
    
    def track_file_close(self, file_path: str, close_time: Optional[float] = None, time_spent: float = 0):
        """Track when a file is closed"""
        with self.lock:
            if file_path in self.file_contexts:
                context = self.file_contexts[file_path]
                context.is_currently_open = False
                context.time_spent += time_spent
    
    def set_current_context(self, current_file: Optional[str] = None, cursor_position: Optional[Tuple[int, int]] = None):
        """Set the current viewing context"""
        with self.lock:
            self.query_context.current_file = current_file
            self.query_context.cursor_position = cursor_position
            
            if current_file:
                self.track_file_open(current_file)
    
    def track_query(self, query: str, results: List[RetrievalResult]):
        """Track a query and its results for learning"""
        with self.lock:
            self.query_history.append({
                'query': query,
                'timestamp': time.time(),
                'result_files': [r.chunk.file_path for r in results[:5]]  # Top 5 results
            })
            
            # Update query context
            self.query_context.query_history.insert(0, query)
            self.query_context.query_history = self.query_context.query_history[:20]
    
    def boost_results(self, results: List[RetrievalResult], query: str) -> List[RetrievalResult]:
        """Apply context-aware boosting to retrieval results"""
        if not results:
            return results
        
        with self.lock:
            boosted_results = []
            current_time = time.time()
            
            for result in results:
                file_path = result.chunk.file_path
                original_score = result.relevance_score
                
                # Calculate total boost
                boost_factor = self._calculate_boost_factor(file_path, query, current_time)
                
                # Apply boost with limits
                boosted_score = original_score * boost_factor
                boosted_score = max(self.config.min_boost * original_score, 
                                  min(self.config.max_boost * original_score, boosted_score))
                
                # Create new result with boosted score
                boosted_result = RetrievalResult(
                    chunk=result.chunk,
                    relevance_score=boosted_score
                )
                boosted_results.append(boosted_result)
            
            # Sort by boosted scores
            boosted_results.sort(key=lambda r: r.relevance_score, reverse=True)
            
            # Track this query
            self.track_query(query, boosted_results)
            
            return boosted_results
    
    def _calculate_boost_factor(self, file_path: str, query: str, current_time: float) -> float:
        """Calculate the total boost factor for a file"""
        total_boost = 1.0
        context = self.file_contexts.get(file_path)
        
        # 1. Recency boosts (like Cursor IDE)
        if context:
            # Recent edit boost
            if context.last_edited > 0:
                hours_since_edit = (current_time - context.last_edited) / 3600
                if hours_since_edit < self.config.edit_decay_hours:
                    recency_factor = max(0.1, 1 - (hours_since_edit / self.config.edit_decay_hours))
                    edit_boost = 1 + (self.config.recent_edit_boost - 1) * recency_factor
                    total_boost *= edit_boost
            
            # Recent open boost
            if context.last_opened > 0:
                hours_since_open = (current_time - context.last_opened) / 3600
                if hours_since_open < 2.0:  # Last 2 hours
                    open_boost = self.config.recent_open_boost * (1 - hours_since_open / 2.0)
                    total_boost *= max(1.0, open_boost)
        
        # 2. Current context boost
        if file_path == self.query_context.current_file:
            total_boost *= self.config.current_file_boost
        
        # 3. Recent files boost
        if file_path in self.query_context.recent_files:
            position = self.query_context.recent_files.index(file_path)
            # More recent = higher boost
            recent_boost = self.config.recent_open_boost * (1 - position / 10)
            total_boost *= max(1.0, recent_boost)
        
        # 4. Frequency boosts
        if context:
            # Edit frequency in session
            if context.edit_frequency > 3:  # Frequently edited
                freq_boost = min(self.config.high_frequency_boost, 
                               1 + 0.05 * context.edit_frequency)
                total_boost *= freq_boost
            
            # Session file boost
            if file_path in self.query_context.session_files:
                total_boost *= self.config.session_file_boost
        
        # 5. Relationship boosts
        current_file = self.query_context.current_file
        if current_file and context:
            # Import relationship boost
            if (current_file in context.import_relationships or 
                file_path in self.file_contexts.get(current_file, FileContext('')).import_relationships):
                total_boost *= self.config.import_relationship_boost
            
            # Directory proximity boost
            if self._are_in_related_directories(file_path, current_file):
                total_boost *= self.config.directory_proximity_boost
            
            # Related files boost (often edited together)
            if file_path in context.related_files:
                total_boost *= self.config.related_files_boost
        
        # 6. Project structure boosts
        file_name = Path(file_path).name.lower()
        
        # Main file boost
        if any(pattern in file_name for pattern in self.main_file_patterns):
            total_boost *= self.config.main_file_boost
        
        # Config file boost (when query might be config-related)
        if (any(pattern in file_name for pattern in self.config_file_patterns) and
            any(word in query.lower() for word in ['config', 'setting', 'env', 'setup'])):
            total_boost *= self.config.config_file_boost
        
        # Test file penalty (unless specifically looking for tests)
        if (any(pattern in file_path.lower() for pattern in self.test_file_patterns) and
            'test' not in query.lower()):
            total_boost *= self.config.test_file_penalty
        
        # 7. Query-specific boosts
        # If query mentions a file name, boost that file
        query_lower = query.lower()
        if Path(file_path).stem.lower() in query_lower:
            total_boost *= 1.3
        
        # If query mentions directory, boost files in that directory
        for part in Path(file_path).parts:
            if part.lower() in query_lower:
                total_boost *= 1.1
                break
        
        return total_boost
    
    def _update_related_files(self, file_path: str, edit_time: float):
        """Update related files based on temporal editing patterns"""
        # Find files edited within 10 minutes
        time_window = 600  # 10 minutes
        
        related_files = set()
        for event in self.edit_events:
            event_file, event_time, event_type = event
            if (abs(event_time - edit_time) < time_window and 
                event_file != file_path and event_type == 'edit'):
                related_files.add(event_file)
        
        # Update relationships bidirectionally
        if file_path not in self.file_contexts:
            self.file_contexts[file_path] = FileContext(file_path=file_path)
        
        self.file_contexts[file_path].related_files.update(related_files)
        
        for related_file in related_files:
            if related_file not in self.file_contexts:
                self.file_contexts[related_file] = FileContext(file_path=related_file)
            self.file_contexts[related_file].related_files.add(file_path)
    
    def _are_in_related_directories(self, file1: str, file2: str) -> bool:
        """Check if two files are in related directories"""
        path1_parts = Path(file1).parts
        path2_parts = Path(file2).parts
        
        # Same directory
        if path1_parts[:-1] == path2_parts[:-1]:
            return True
        
        # Parent/child directory
        min_len = min(len(path1_parts), len(path2_parts))
        if min_len > 1:
            common_parts = sum(1 for a, b in zip(path1_parts[:min_len-1], path2_parts[:min_len-1]) if a == b)
            return common_parts >= min_len - 2
        
        return False
    
    def update_import_relationships(self, file_imports: Dict[str, Set[str]]):
        """Update import relationship information"""
        with self.lock:
            for file_path, imported_files in file_imports.items():
                if file_path not in self.file_contexts:
                    self.file_contexts[file_path] = FileContext(file_path=file_path)
                
                self.file_contexts[file_path].import_relationships = imported_files.copy()
    
    def get_context_summary(self) -> Dict[str, Any]:
        """Get a summary of current context for debugging"""
        with self.lock:
            return {
                'current_file': self.query_context.current_file,
                'recent_files': self.query_context.recent_files[:5],
                'session_files_count': len(self.query_context.session_files),
                'tracked_files_count': len(self.file_contexts),
                'most_edited_files': self._get_most_edited_files(5),
                'recent_queries': self.query_context.query_history[:5]
            }
    
    def _get_most_edited_files(self, limit: int) -> List[Tuple[str, int]]:
        """Get the most frequently edited files"""
        file_frequencies = [(path, ctx.edit_frequency) 
                          for path, ctx in self.file_contexts.items() 
                          if ctx.edit_frequency > 0]
        file_frequencies.sort(key=lambda x: x[1], reverse=True)
        return file_frequencies[:limit]
    
    def _load_context(self):
        """Load persistent context from file"""
        if not self.persistence_file:
            return
        
        try:
            with open(self.persistence_file, 'r') as f:
                data = json.load(f)
                
            # Load file contexts
            for file_path, context_data in data.get('file_contexts', {}).items():
                self.file_contexts[file_path] = FileContext(
                    file_path=file_path,
                    last_edited=context_data.get('last_edited', 0),
                    last_opened=context_data.get('last_opened', 0),
                    edit_frequency=context_data.get('edit_frequency', 0),
                    open_frequency=context_data.get('open_frequency', 0),
                    time_spent=context_data.get('time_spent', 0),
                    related_files=set(context_data.get('related_files', [])),
                    import_relationships=set(context_data.get('import_relationships', []))
                )
                
            logger.info(f"Loaded context for {len(self.file_contexts)} files")
            
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.info(f"No existing context file or invalid format: {e}")
    
    def save_context(self):
        """Save persistent context to file"""
        if not self.persistence_file:
            return
        
        try:
            # Prepare data for serialization
            context_data = {}
            for file_path, context in self.file_contexts.items():
                context_data[file_path] = {
                    'last_edited': context.last_edited,
                    'last_opened': context.last_opened,
                    'edit_frequency': context.edit_frequency,
                    'open_frequency': context.open_frequency,
                    'time_spent': context.time_spent,
                    'related_files': list(context.related_files),
                    'import_relationships': list(context.import_relationships)
                }
            
            data = {
                'file_contexts': context_data,
                'session_info': {
                    'last_session': time.time(),
                    'total_sessions': 1  # This could be incremented
                }
            }
            
            with open(self.persistence_file, 'w') as f:
                json.dump(data, f, indent=2)
                
            logger.info(f"Saved context for {len(self.file_contexts)} files")
            
        except Exception as e:
            logger.error(f"Failed to save context: {e}")
    
    def cleanup_old_context(self, max_age_days: int = 30):
        """Clean up old context data"""
        cutoff_time = time.time() - (max_age_days * 24 * 3600)
        
        with self.lock:
            files_to_remove = []
            for file_path, context in self.file_contexts.items():
                if (context.last_edited < cutoff_time and 
                    context.last_opened < cutoff_time and
                    context.edit_frequency == 0):
                    files_to_remove.append(file_path)
            
            for file_path in files_to_remove:
                del self.file_contexts[file_path]
            
            if files_to_remove:
                logger.info(f"Cleaned up context for {len(files_to_remove)} old files")
    
    def __del__(self):
        """Save context on shutdown"""
        try:
            self.save_context()
        except:
            pass  # Don't raise exceptions in destructor

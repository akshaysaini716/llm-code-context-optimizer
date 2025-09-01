# Context-Aware Boosting Guide

## 🚀 Cursor IDE-Style Context Boosting

I've created a comprehensive context-aware boosting system that mimics how Cursor IDE prioritizes files and context. Here's what it does and how to use it:

## 🎯 Boosting Techniques Implemented

### 1. **Recently Edited Files Boost** (Like Cursor IDE)
- **What it does**: Files edited in the last few hours get higher priority
- **Why**: You're likely working on related functionality
- **Boost factor**: 1.5x - 2.0x for recently edited files
- **Decay**: Gradually decreases over 24 hours

### 2. **Current File Context Boost** 
- **What it does**: Files related to what you're currently viewing get priority
- **Why**: Context is often in related files
- **Boost factor**: 2.0x - 2.5x for current file context
- **Includes**: Same directory, imported files, related modules

### 3. **File Relationship Boosting**
- **Import Dependencies**: Files that import from each other get boosted
- **Directory Proximity**: Files in same/related directories
- **Co-editing Patterns**: Files often edited together in same session

### 4. **Session Context Boosting**
- **Recent Files**: Files opened recently in current session
- **Frequently Accessed**: Files you've opened multiple times
- **Query History**: Files that were useful in previous similar queries

### 5. **Project Structure Intelligence**
- **Entry Points**: `main.py`, `index.js`, `app.py` get slight boost
- **Configuration Files**: Config files boosted when query mentions config
- **Test File Handling**: Test files deprioritized unless explicitly requested

### 6. **Smart Query-Specific Boosting**
- **File Name Mentions**: If query mentions a file name, boost that file
- **Directory Mentions**: If query mentions a directory, boost files in it
- **Symbol Context**: Boost files that define symbols mentioned in query

## 🔧 Integration with Your RAG System

### Option 1: Drop-in Replacement (Easiest)

Replace your multi-stage retriever with the enhanced version:

```python
# In your rag_system.py
from rag.retrieval.enhanced_multi_stage_retriever import create_cursor_like_retriever

class RAGSystem:
    def __init__(self, project_path: str):
        # ... existing code ...
        
        # Replace this:
        # self.multi_stage_retriever = MultiStageRetriever()
        
        # With this:
        self.enhanced_retriever = create_cursor_like_retriever(project_path)
    
    def query(self, query: str, max_context_tokens: int = 8000, top_k: int = 10, 
              current_file: str = None, recent_files: List[str] = None) -> RAGResponse:
        
        try:
            # Enhanced retrieval with context
            retrieval_result = self.enhanced_retriever.retrieve(
                query=query,
                top_k=top_k,
                current_file=current_file,  # Pass current context
                recent_files=recent_files   # Pass recent files
            )
            
            # Rest stays the same
            final_context = self.context_fuser.fuse_context(retrieval_result, max_context_tokens)
            # ... rest of method
            
        except Exception as e:
            logger.error(f"Enhanced retrieval failed: {e}")
            # Fallback to your original system
```

### Option 2: Add Activity Tracking (Recommended)

Track file activity to improve context awareness:

```python
class RAGSystem:
    def track_file_edit(self, file_path: str):
        """Call this whenever a file is edited"""
        self.enhanced_retriever.track_file_activity(file_path, 'edit')
    
    def track_file_open(self, file_path: str):
        """Call this whenever a file is opened"""
        self.enhanced_retriever.track_file_activity(file_path, 'open')
    
    def track_file_close(self, file_path: str, time_spent: float = 0):
        """Call this when a file is closed"""
        self.enhanced_retriever.track_file_activity(
            file_path, 'close', time_spent=time_spent
        )
    
    def update_project_imports(self):
        """Update import relationships for better boosting"""
        # You can implement this to scan your project for import relationships
        import_map = self._scan_project_imports()
        self.enhanced_retriever.update_import_relationships(import_map)
```

## 🧪 Testing the Boosting System

Create a test to see the difference:

```python
# test_context_boosting.py
from rag.retrieval.enhanced_multi_stage_retriever import create_cursor_like_retriever
import time

def test_context_boosting():
    retriever = create_cursor_like_retriever("/path/to/your/project")
    
    # Simulate some file activity
    retriever.track_file_activity("src/auth.py", "edit")
    retriever.track_file_activity("src/user.py", "edit") 
    retriever.track_file_activity("src/database.py", "open")
    
    # Set current context
    test_queries = [
        "authentication methods",
        "user management",
        "database connection"
    ]
    
    print("=== Context Boosting Test ===")
    for query in test_queries:
        print(f"\nQuery: {query}")
        
        # Test without context
        results_no_context = retriever.retrieve(query, top_k=5)
        
        # Test with context (recently edited auth.py)
        results_with_context = retriever.retrieve(
            query, 
            top_k=5,
            current_file="src/auth.py",
            recent_files=["src/auth.py", "src/user.py"]
        )
        
        print("Without context:")
        for i, result in enumerate(results_no_context):
            print(f"  {i+1}. {result.chunk.file_path} (score: {result.relevance_score:.2f})")
        
        print("With context boosting:")  
        for i, result in enumerate(results_with_context):
            print(f"  {i+1}. {result.chunk.file_path} (score: {result.relevance_score:.2f})")
        
        # Show debug info
        debug_info = retriever.get_context_debug_info()
        print(f"Recent files: {debug_info['context_boosting']['recent_files']}")

if __name__ == "__main__":
    test_context_boosting()
```

## 📊 Expected Performance Improvements

### Before Context Boosting:
```
Query: "authentication methods"
1. random_auth_util.py (score: 0.82)
2. old_login_system.py (score: 0.78) 
3. auth_config.py (score: 0.75)
4. current_auth_service.py (score: 0.73)
5. test_auth.py (score: 0.70)
```

### After Context Boosting:
```
Query: "authentication methods"  
1. current_auth_service.py (score: 1.46) ← BOOSTED (recently edited)
2. auth_config.py (score: 1.12) ← BOOSTED (related to current file)  
3. user_service.py (score: 0.95) ← BOOSTED (imports auth_service)
4. middleware/auth.py (score: 0.89) ← BOOSTED (same directory)
5. random_auth_util.py (score: 0.82)
```

Notice how:
- Recently edited files get priority
- Related/imported files are boosted
- Files in same directory are prioritized
- Test files are deprioritized (unless query mentions tests)

## 🚀 Additional Boost Techniques You Can Add

### 1. **Git History Boosting**
```python
def add_git_history_boost(self):
    """Boost files that were recently modified in git"""
    import subprocess
    
    # Get files changed in last 7 days
    result = subprocess.run([
        'git', 'log', '--since=7.days', '--name-only', '--pretty=format:'
    ], capture_output=True, text=True)
    
    recent_git_files = set(result.stdout.strip().split('\n'))
    
    for file_path in recent_git_files:
        if file_path.strip():
            self.track_file_activity(file_path, 'edit')
```

### 2. **Error Context Boosting**
```python
def boost_error_context_files(self, error_traceback: str):
    """Boost files mentioned in error tracebacks"""
    import re
    
    # Extract file paths from traceback
    file_pattern = r'File "([^"]+)"'
    error_files = re.findall(file_pattern, error_traceback)
    
    for file_path in error_files:
        # Boost files that had errors - they're relevant for debugging
        self.track_file_activity(file_path, 'edit')
```

### 3. **Semantic Similarity Boosting**
```python
def boost_semantically_similar_files(self, query: str):
    """Boost files with similar functionality"""
    # This could use file content embeddings to find similar files
    # and boost them based on semantic similarity to the query
    pass
```

### 4. **Time-of-Day Pattern Boosting**
```python
def boost_time_based_patterns(self):
    """Boost files based on time-of-day work patterns"""
    from datetime import datetime
    
    current_hour = datetime.now().hour
    
    # If it's morning (9-11 AM), boost main/entry files
    if 9 <= current_hour <= 11:
        self.config.main_file_boost *= 1.2
    
    # If it's afternoon (2-5 PM), boost feature files  
    elif 14 <= current_hour <= 17:
        self.config.recent_edit_boost *= 1.1
    
    # If it's evening (6+ PM), boost debug/test files
    elif current_hour >= 18:
        self.config.test_file_penalty = 1.0  # Remove test penalty
```

### 5. **Collaboration Context Boosting**
```python
def boost_team_context(self, teammate_recent_files: List[str]):
    """Boost files recently edited by teammates"""
    for file_path in teammate_recent_files:
        # Boost files your teammates are working on
        self.track_file_activity(file_path, 'edit')
```

### 6. **Code Review Context Boosting**
```python
def boost_pr_files(self, pr_files: List[str]):
    """Boost files in current pull request"""
    for file_path in pr_files:
        # Files in PR are highly relevant
        self.track_file_activity(file_path, 'edit')
        # Add extra boost for PR files
        self.context_booster.config.recent_edit_boost *= 1.3
```

## 🎯 Configuration Tuning

You can tune the boost factors based on your workflow:

```python
# For aggressive recency (like Cursor)
cursor_like_config = BoostConfig(
    recent_edit_boost=2.5,      # Very strong recent edit bias
    current_file_boost=3.0,     # Strong current context bias  
    edit_decay_hours=4.0,       # Fast decay - only very recent edits matter
    max_boost=5.0               # Allow high boost factors
)

# For balanced approach
balanced_config = BoostConfig(
    recent_edit_boost=1.4,      # Moderate recent edit bias
    current_file_boost=1.8,     # Moderate current context bias
    edit_decay_hours=12.0,      # Slower decay - longer context window
    max_boost=2.5               # Conservative boost limits
)

# For exploration-heavy workflows  
explorer_config = BoostConfig(
    recent_edit_boost=1.2,      # Light recent bias
    directory_proximity_boost=1.4,  # Boost directory exploration
    related_files_boost=1.6,    # Strong relationship boosting
    test_file_penalty=1.0       # Don't penalize test files
)
```

## 🔄 Next Steps

1. **Start with Option 1** - Replace your multi-stage retriever with the enhanced version
2. **Test with current files** - Pass `current_file` and `recent_files` to see immediate improvements
3. **Add activity tracking** - Implement file edit/open tracking for better context
4. **Tune boost factors** - Adjust the configuration based on your workflow
5. **Add advanced techniques** - Implement git history, error context, or other domain-specific boosting

The context-aware boosting system will learn your patterns and get better over time, just like Cursor IDE does! 🎉

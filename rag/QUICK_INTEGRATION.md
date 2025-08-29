# Quick Integration Guide - Context Boosting

## 🚀 5-Minute Integration

Here's how to add Cursor IDE-style context boosting to your existing RAG system with minimal changes:

### Step 1: Update your `rag_system.py` (2 minutes)

```python
# At the top, replace the import:
# from rag.retrieval.multi_stage_retriever import MultiStageRetriever
from rag.retrieval.enhanced_multi_stage_retriever import create_cursor_like_retriever

class RAGSystem:
    def __init__(self):
        # ... existing initialization code ...
        
        # Replace this line:
        # self.multi_stage_retriever = MultiStageRetriever()
        
        # With this:
        self.enhanced_retriever = create_cursor_like_retriever("your_project_path_here")
    
    def query(self, query: str, max_context_tokens: int = 8000, top_k: int = 10, 
              current_file: str = None, recent_files: List[str] = None) -> RAGResponse:
        start_time = time.time()
        
        try:
            # Replace this line:
            # retrieval_result = self.multi_stage_retriever.retrieve(query, top_k)
            
            # With this:
            retrieval_result = self.enhanced_retriever.retrieve(
                query=query,
                top_k=top_k,
                current_file=current_file,      # NEW: Pass current file context
                recent_files=recent_files       # NEW: Pass recent files
            )
            
            # Everything else stays exactly the same
            if not retrieval_result:
                return RAGResponse(context="", chunks_used=[], total_tokens=0, 
                                 retrieval_time_ms=(time.time() - start_time) * 1000)

            final_context = self.context_fuser.fuse_context(retrieval_result, max_context_tokens)
            chunks_used = [result.chunk for result in retrieval_result]
            
            return RAGResponse(
                context=final_context,
                chunks_used=chunks_used,
                total_tokens=len(self.context_fuser.tokenizer.encode(final_context)),
                retrieval_time_ms=(time.time() - start_time) * 1000
            )
            
        except Exception as e:
            logger.error(f"Enhanced retrieval failed, falling back: {e}")
            # Your original fallback code here
```

### Step 2: Update your API endpoint (2 minutes)

If you have an API endpoint (like in `api/server.py`), update it to pass context:

```python
# Add these fields to your request model
class ChatRequest(BaseModel):
    message: str
    project_path: str
    relevant_only: bool = False
    current_file: Optional[str] = None      # NEW
    recent_files: Optional[List[str]] = []  # NEW

@app.post("/chat")  
def chat(req: ChatRequest):
    # ... existing code ...
    
    if req.include_context and req.project_path:
        # Update this call:
        response = rag_system.query(
            query=req.message,
            max_context_tokens=req.max_tokens,
            current_file=req.current_file,    # NEW  
            recent_files=req.recent_files     # NEW
        )
        
        # ... rest stays the same
```

### Step 3: Test immediately (1 minute)

```python
# Quick test script
def test_context_boosting():
    rag = RAGSystem()
    
    # Test without context
    result1 = rag.query("authentication methods")
    print("Without context:")
    for chunk in result1.chunks_used[:3]:
        print(f"  - {chunk.file_path}")
    
    # Test with context (simulate recently edited auth files)
    result2 = rag.query(
        "authentication methods",
        current_file="src/auth/auth_service.py",
        recent_files=["src/auth/auth_service.py", "src/user/user_model.py"]
    )
    print("\nWith context boosting:")
    for chunk in result2.chunks_used[:3]:
        print(f"  - {chunk.file_path}")

test_context_boosting()
```

## 🎯 What You'll See Immediately

### Before Context Boosting:
```
Query: "authentication methods"
Results:
  - tests/test_old_auth.py
  - utils/random_auth_helper.py  
  - docs/auth_readme.md
```

### After Context Boosting:
```  
Query: "authentication methods"
Results (with current_file="src/auth/auth_service.py"):
  - src/auth/auth_service.py        ← BOOSTED (current file)
  - src/auth/middleware.py          ← BOOSTED (same directory) 
  - src/user/user_model.py          ← BOOSTED (recent file)
```

## 🚀 Optional: Add Activity Tracking (5 more minutes)

If you want even better results, add file activity tracking:

```python
class RAGSystem:
    def track_file_edit(self, file_path: str):
        """Call this when a file is edited"""
        self.enhanced_retriever.track_file_activity(file_path, 'edit')
    
    def track_file_open(self, file_path: str):  
        """Call this when a file is opened"""
        self.enhanced_retriever.track_file_activity(file_path, 'open')

# Usage in your IDE/editor integration:
# rag.track_file_edit("src/auth.py")  # When file is saved
# rag.track_file_open("src/user.py")  # When file is opened
```

## 🎊 That's It!

With these minimal changes, you now have:

✅ **Recently edited files get priority** (like Cursor IDE)  
✅ **Current file context boosting**  
✅ **Directory proximity boosting**  
✅ **Import relationship awareness**  
✅ **Smart test file handling**  
✅ **Session-based learning**  
✅ **Persistent context between sessions**

## 🔧 Advanced Configuration (Optional)

If you want to tune the boosting behavior:

```python
from rag.retrieval.context_boosting import BoostConfig

# Create custom boost config
custom_config = BoostConfig(
    recent_edit_boost=2.0,        # Strong bias toward recently edited files
    current_file_boost=2.5,       # Strong bias toward current context
    test_file_penalty=0.8,        # Slightly reduce test files
    edit_decay_hours=8.0          # Context decays over 8 hours
)

# Use in your RAG system
from rag.retrieval.enhanced_multi_stage_retriever import create_enhanced_retriever_for_project

self.enhanced_retriever = create_enhanced_retriever_for_project(
    project_root="your_project_path", 
    boost_config=custom_config
)
```

## 🐛 Troubleshooting

**Issue**: "No such module enhanced_multi_stage_retriever"  
**Fix**: Make sure you've copied the new files to your `rag/retrieval/` directory

**Issue**: Not seeing much difference in results  
**Fix**: Make sure you're passing `current_file` and `recent_files` parameters

**Issue**: Getting errors about missing dependencies  
**Fix**: The system falls back gracefully, but check that the file paths you're passing actually exist

**Issue**: Want to see what's being boosted  
**Fix**: Use `rag.enhanced_retriever.get_context_debug_info()` to see current context state

Your RAG system now intelligently prioritizes relevant files based on your current context, just like Cursor IDE! 🎉

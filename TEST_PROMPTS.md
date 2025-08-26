# RAG System Test Prompts

## Overview
I've created a comprehensive sample project with 6 Python files containing different types of code patterns to test your enhanced RAG system. Here are the files and suggested test prompts.

## Sample Project Files Created

### 1. `user_management.py` (320 lines)
**Complex class-heavy file with:**
- Multiple classes: `UserManager`, `PasswordManager`, `TokenManager`, `AuthorizedUserManager`
- Enums and dataclasses
- Decorators and inheritance
- JWT authentication logic

### 2. `database_manager.py` (400+ lines)
**Database operations with:**
- `ConnectionPool`, `QueryBuilder`, `DatabaseManager`, `ModelManager` classes
- Method chaining patterns
- Context managers
- Complex SQL query building

### 3. `api_server.py` (350+ lines)
**FastAPI REST API with:**
- Multiple endpoint functions
- Dependency injection
- Pydantic models
- Authentication middleware
- Error handling

### 4. `data_processor.py` (450+ lines)
**Data processing framework with:**
- Abstract base classes
- Multiple transformer classes
- Parallel processing
- Validation utilities

### 5. `utils.py` (400+ lines)
**Utility functions with:**
- Decorators (retry, timing, caching)
- Context managers
- File operations
- Async utilities

### 6. `file_handler.py` (existing)
**File operations**

## Test Prompts by Category

### 🔐 Authentication & User Management

```
1. "How does user authentication work in this system?"
2. "Show me the UserManager class and all its methods"
3. "How do I register a new user through the API?"
4. "What password hashing algorithm is used and how is it implemented?"
5. "How are JWT tokens created and validated?"
6. "What user roles are available and how is authorization handled?"
7. "Show me the complete user login flow from API to token generation"
8. "How do I change a user's password?"
```

### 🗄️ Database Operations

```
9. "How does the database connection pooling work?"
10. "Show me how to use the QueryBuilder to create complex SQL queries"
11. "How do I execute parameterized SQL queries safely?"
12. "What's the difference between execute_query and execute_builder methods?"
13. "How can I backup and restore database tables?"
14. "Show me the ModelManager ORM functionality"
15. "How do I handle database transactions?"
16. "What database statistics can I get?"
```

### 🌐 API & Web Server

```
17. "What are all the available REST API endpoints?"
18. "How is CORS configured in the FastAPI server?"
19. "Show me the authentication middleware implementation"
20. "How are API errors handled and returned to clients?"
21. "What Pydantic models are used for request/response validation?"
22. "How do I protect endpoints with role-based access?"
23. "Show me the health check endpoints"
24. "How is the FastAPI app configured and started?"
```

### 📊 Data Processing

```
25. "What data transformers are available and how do I use them?"
26. "How do I validate email addresses and phone numbers?"
27. "Show me how to process data in parallel vs sequential"
28. "What text transformation options are available?"
29. "How do I calculate statistics on numeric data?"
30. "How can I load and save CSV files with validation?"
31. "Show me the data processing pipeline architecture"
32. "How do I create custom data validators?"
```

### 🛠️ Utilities & Helpers

```
33. "What utility functions are available for file operations?"
34. "How do I use the retry decorator for error handling?"
35. "What caching mechanisms are implemented?"
36. "Show me the timing decorator for performance measurement"
37. "How do I work with configuration files (JSON/YAML)?"
38. "What date/time utilities are available?"
39. "How do I validate URLs and UUIDs?"
40. "Show me the async utility functions"
```

### 🔗 Cross-Component Integration

```
41. "How do the user management and API server components work together?"
42. "Show me the complete user registration flow from API to database"
43. "How does data flow from API request to database storage?"
44. "What's the relationship between UserManager and DatabaseManager?"
45. "How are validation errors propagated from data processing to API responses?"
46. "Show me how authentication tokens are used across different components"
```

### 🏗️ Architecture & Design Patterns

```
47. "What design patterns are used in this codebase?"
48. "Show me all the abstract base classes and their implementations"
49. "How is dependency injection implemented?"
50. "What decorators are available and how do they work?"
51. "Show me the use of context managers in the code"
52. "How is error handling standardized across components?"
```

### 🧪 Testing & Quality

```
53. "How can I test the authentication system?"
54. "What validation checks are performed on user input?"
55. "Show me error handling patterns used throughout the code"
56. "How are database operations made safe from SQL injection?"
57. "What logging is implemented in the system?"
```

## Running the Tests

### Option 1: Run the Comprehensive Test Script
```bash
cd /Users/user/Desktop/Akshay/llm-code-context-optimizer
python test_rag_system.py
```

### Option 2: Test Individual Queries
```python
from rag.core.rag_system import RAGSystem

# Setup RAG system
rag_system = RAGSystem()

# Index the sample project
rag_system.index_codebase(
    project_path="sample_project",
    file_patterns=["*.py"],
    force_reindex=True
)

# Test a query
response = rag_system.query(
    query="How does user authentication work?",
    max_context_tokens=4000,
    top_k=8
)

print(f"Tokens used: {response.total_tokens}")
print(f"Chunks retrieved: {response.chunks_count}")
print(f"Context:\n{response.context}")
```

## What to Look For

### ✅ Enhanced Chunking Benefits
1. **Complete Classes**: UserManager, DatabaseManager should appear as single chunks
2. **Smart Imports**: Only relevant imports included in each chunk
3. **Better Context**: Related methods kept together
4. **Fewer Chunks**: More efficient storage with better context

### ✅ Improved Retrieval
1. **Relevant Results**: Queries should find the right files and classes
2. **Context Quality**: Retrieved context should be complete and useful
3. **Cross-File Understanding**: Queries spanning multiple files should work
4. **Performance**: Fast retrieval with good token efficiency

### ⚠️ Things to Monitor
1. **Token Usage**: Should be efficient compared to old system
2. **Chunk Quality**: Complete classes vs fragmented methods
3. **Retrieval Accuracy**: Finding the right code for each query
4. **Error Handling**: No more Qdrant filtering errors

## Expected Results

With the enhanced system, you should see:
- **30-50% fewer chunks** due to class consolidation
- **Better context quality** with complete class definitions
- **Improved retrieval accuracy** for class-related queries
- **No Qdrant errors** during retrieval
- **Faster query responses** due to better chunking

Try these prompts and let me know how the enhanced RAG system performs!

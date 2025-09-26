# RAG System Test Prompts

## Email Service Debugging Prompts

### Security Issues
1. **"What security vulnerabilities exist in the email service?"**
   - Should find: Hardcoded credentials, HTTP reset links, XSS in HTML emails

2. **"Why is the password reset functionality insecure?"**
   - Should find: HTTP instead of HTTPS, silent failures

3. **"Are there any credential management issues in email_service.py?"**
   - Should find: Hardcoded password, exposed credentials

### Connection Management
4. **"Why am I getting connection errors when sending batch emails?"**
   - Should find: Missing connect() call, resource leaks

5. **"The email service seems to be leaking resources. What's wrong?"**
   - Should find: No disconnect() calls, connection not set to None

6. **"Email sending fails but I don't get any error messages. Why?"**
   - Should find: Silent failures, broad exception handling

### Logic Errors
7. **"Why does email validation reject valid email addresses?"**
   - Should find: Overly restrictive regex pattern

8. **"The email stats show fake data. What's the issue?"**
   - Should find: Hardcoded return values, no actual tracking

9. **"Why doesn't the email service actually clean up logs?"**
   - Should find: Empty method, no implementation

## Data Validation Bugs

### Validation Logic
10. **"Users complain that valid usernames are being rejected. What's wrong?"**
    - Should find: Too restrictive regex, off-by-one error

11. **"Why do some valid email addresses fail validation?"**
    - Should find: Case sensitivity, incomplete regex

12. **"Password validation seems broken. What are the issues?"**
    - Should find: Logic error with 'or' instead of 'and', missing special char check

### Security Vulnerabilities
13. **"What XSS vulnerabilities exist in the data validation?"**
    - Should find: Incomplete sanitization, missing dangerous tags

14. **"Are there any path traversal vulnerabilities in file validation?"**
    - Should find: No check for '../' in filenames

15. **"Why might the credit card validation fail for valid cards?"**
    - Should find: Luhn algorithm bugs, missing card types

### Type Safety
16. **"The age validation crashes sometimes. Why?"**
    - Should find: No type checking, assumes integer input

17. **"What happens if someone uploads a file with negative size?"**
    - Should find: No check for negative file sizes

### Algorithm Errors
18. **"The Luhn algorithm implementation seems incorrect. What's wrong?"**
    - Should find: Off-by-one errors, wrong modulo check

19. **"Phone number validation is too restrictive. Why?"**
    - Should find: US-only format, no international support

## Async Worker Issues

### Concurrency Bugs
20. **"Why do multiple workers sometimes process the same task?"**
    - Should find: Race condition in task assignment, no proper locking

21. **"The worker threads don't exit properly when stopping. What's wrong?"**
    - Should find: No daemon threads, short timeout on join

22. **"Why does the task count sometimes get corrupted?"**
    - Should find: Shared state without synchronization, race conditions

### Resource Management
23. **"The async worker is leaking file handles. Where's the issue?"**
    - Should find: Log file never closed, file handles in worker methods

24. **"Why doesn't the worker shutdown gracefully?"**
    - Should find: No wait for tasks to complete, improper thread management

25. **"Memory usage keeps growing when processing tasks. Why?"**
    - Should find: Results never cleared, tasks not removed from active list

### Design Issues
26. **"The async worker doesn't actually process tasks asynchronously. What's wrong?"**
    - Should find: Blocking operations in async methods, sequential processing

27. **"Why does the scheduler sometimes crash with RuntimeError?"**
    - Should find: Modifying dictionary while iterating

28. **"Task processing fails silently sometimes. How to debug this?"**
    - Should find: Broad exception handling, poor error reporting

## Integration & Architecture Issues

### Cross-Module Problems
29. **"How do all these modules interact and what are the main integration issues?"**
    - Should find: Import dependencies, shared state problems

30. **"What are the main patterns of bugs across the codebase?"**
    - Should find: Resource management, error handling, thread safety

31. **"If I wanted to use these modules in production, what would I need to fix first?"**
    - Should find: Security issues, resource leaks, error handling

### Performance Issues
32. **"Why is the system slow when processing multiple requests?"**
    - Should find: Inefficient resource creation, blocking operations

33. **"What causes high memory usage in long-running processes?"**
    - Should find: Memory leaks, unclosed resources, growing collections

34. **"Are there any scalability issues in the current design?"**
    - Should find: Global instances, thread safety issues, resource management

## Specific Bug Categories

### Error Handling
35. **"What are all the places where exceptions are handled incorrectly?"**
    - Should find: Broad exception handling, silent failures, poor error messages

36. **"Why don't users get proper error messages when things fail?"**
    - Should find: Silent failures, generic error handling

### Input Validation
37. **"Where might malicious input cause problems in the system?"**
    - Should find: XSS vulnerabilities, path traversal, injection possibilities

38. **"What happens if someone provides unexpected data types to the validators?"**
    - Should find: Type checking issues, crash possibilities

### Threading & Concurrency
39. **"What threading bugs exist in the async worker system?"**
    - Should find: Race conditions, shared state issues, improper synchronization

40. **"Why might multiple instances of workers interfere with each other?"**
    - Should find: Global state, shared resources, synchronization issues

## Testing the RAG System Itself

### Expansion Testing
41. **"Show me all the email-related security issues" (then expand with more context)**
    - Test: Does expansion find related issues in other files?

42. **"Find validation bugs" (then expand to get more comprehensive results)**
    - Test: Does expansion include edge cases and related validation code?

43. **"Debug the worker resource leaks" (then expand for full context)**
    - Test: Does expansion include related resource management across files?

### Multi-File Context
44. **"How do the email service and data validator work together?"**
    - Test: Cross-file relationship understanding

45. **"What would happen if I use the async worker to send emails?"**
    - Test: Integration analysis across modules

46. **"Show me all the hardcoded values that should be configurable"**
    - Test: Pattern detection across multiple files

## Expected RAG Behavior

### Standard Queries (expand_window=False)
- Should return 8-12 chunks, focused on specific issues
- 6,000-8,000 tokens of context
- High precision, specific to the query

### Moderate Expansion (expand_window=True, level="moderate")  
- Should return 15-20 chunks with broader context
- 12,000-16,000 tokens
- Include related code patterns and dependencies

### Aggressive Expansion (expand_window=True, level="aggressive")
- Should return 20-30 chunks with comprehensive context
- 20,000-32,000 tokens  
- Include full file context and cross-file relationships

## Sample Usage

```python
# Test with your RAG system
from rag.core.rag_system import RAGSystem

rag = RAGSystem()

# Standard query
response = rag.query(
    "What security vulnerabilities exist in the email service?",
    project_path="sample_project",
    expand_window=False
)

# Expanded query  
expanded = rag.query(
    "What security vulnerabilities exist in the email service?",
    project_path="sample_project", 
    expand_window=True,
    expansion_level="moderate",
    session_id="debug_session_1"
)
```

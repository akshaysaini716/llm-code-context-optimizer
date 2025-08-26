#!/usr/bin/env python3
"""
Comprehensive RAG System Test Script
Tests the enhanced RAG system with various queries and scenarios
"""

import sys
from pathlib import Path
import time

# Add the project root to the path
sys.path.append(str(Path(__file__).parent))

from rag.core.rag_system import RAGSystem
from rag.configs import RAGConfig, ChunkingConfig, RetrievalConfig, FusionConfig

def setup_rag_system():
    """Setup the RAG system with enhanced configuration"""
    print("Setting up RAG system...")
    
    # Configure for enhanced chunking
    config = RAGConfig(
        chunking=ChunkingConfig(
            preserve_class_methods=True,
            smart_imports=True,
            overlap_ratio=0.1,
            include_context_lines=2,
            hierarchical_chunking=True
        ),
        retrieval=RetrievalConfig(
            include_related_chunks=True,  # Disabled due to Qdrant issues
            enable_context_expansion=True,
            expansion_window=200
        ),
        fusion=FusionConfig(
            preserve_structure=True,
            group_by_file=True,
            merge_adjacent=True
        )
    )
    
    rag_system = RAGSystem()
    return rag_system

def index_sample_project(rag_system):
    """Index the sample project"""
    print("\nIndexing sample project...")
    
    try:
        result = rag_system.index_codebase(
            project_path="sample_project",
            file_patterns=["*.py"],
            force_reindex=True
        )
        
        print(f"✅ Indexing completed:")
        print(f"   - Files processed: {result['files_processed']}")
        print(f"   - Chunks created: {result['chunks_created']}")
        print(f"   - Chunks embedded: {result['chunks_embedded']}")
        print(f"   - Indexing time: {result['indexing_time_seconds']:.2f}s")
        
        return True
        
    except Exception as e:
        print(f"❌ Indexing failed: {e}")
        return False

def test_queries(rag_system):
    """Test various queries on the RAG system"""
    
    test_queries = [
        # Authentication and user management queries
        {
            "query": "How does user authentication work in this system?",
            "description": "Testing authentication flow understanding",
            "expected_files": ["user_management.py", "api_server.py"]
        },
        {
            "query": "Show me the UserManager class and its methods",
            "description": "Testing class-aware chunking",
            "expected_files": ["user_management.py"]
        },
        {
            "query": "How do I register a new user through the API?",
            "description": "Testing API endpoint understanding",
            "expected_files": ["api_server.py", "user_management.py"]
        },
        {
            "query": "What password hashing algorithm is used?",
            "description": "Testing specific implementation details",
            "expected_files": ["user_management.py"]
        },
        
        # Database queries
        {
            "query": "How does the database connection pooling work?",
            "description": "Testing database architecture understanding",
            "expected_files": ["database_manager.py"]
        },
        {
            "query": "Show me how to execute SQL queries with parameters",
            "description": "Testing query building functionality",
            "expected_files": ["database_manager.py"]
        },
        {
            "query": "What's the QueryBuilder class and how do I use it?",
            "description": "Testing specific class functionality",
            "expected_files": ["database_manager.py"]
        },
        
        # Data processing queries
        {
            "query": "How do I validate email addresses in the data processor?",
            "description": "Testing data validation functionality",
            "expected_files": ["data_processor.py"]
        },
        {
            "query": "What data transformers are available?",
            "description": "Testing transformer classes understanding",
            "expected_files": ["data_processor.py"]
        },
        {
            "query": "How can I process data in parallel?",
            "description": "Testing parallel processing features",
            "expected_files": ["data_processor.py"]
        },
        
        # API and server queries
        {
            "query": "What are the available REST API endpoints?",
            "description": "Testing API documentation understanding",
            "expected_files": ["api_server.py"]
        },
        {
            "query": "How is CORS configured in the FastAPI server?",
            "description": "Testing specific configuration details",
            "expected_files": ["api_server.py"]
        },
        {
            "query": "Show me the authentication middleware implementation",
            "description": "Testing middleware understanding",
            "expected_files": ["api_server.py"]
        },
        
        # Utility functions queries
        {
            "query": "What utility functions are available for file operations?",
            "description": "Testing utility functions understanding",
            "expected_files": ["utils.py"]
        },
        {
            "query": "How do I use the retry decorator?",
            "description": "Testing decorator functionality",
            "expected_files": ["utils.py"]
        },
        {
            "query": "What caching mechanisms are implemented?",
            "description": "Testing caching functionality",
            "expected_files": ["utils.py"]
        },
        
        # Cross-file queries
        {
            "query": "How do the user management and API server components work together?",
            "description": "Testing cross-file understanding",
            "expected_files": ["user_management.py", "api_server.py"]
        },
        {
            "query": "Show me the complete user registration flow from API to database",
            "description": "Testing end-to-end flow understanding",
            "expected_files": ["api_server.py", "user_management.py", "database_manager.py"]
        },
        
        # Error handling queries
        {
            "query": "How are errors handled in the API endpoints?",
            "description": "Testing error handling understanding",
            "expected_files": ["api_server.py"]
        },
        {
            "query": "What validation errors can occur during data processing?",
            "description": "Testing validation error handling",
            "expected_files": ["data_processor.py"]
        }
    ]
    
    print(f"\n🧪 Testing {len(test_queries)} queries...")
    print("=" * 80)
    
    results = []
    
    for i, test_case in enumerate(test_queries, 1):
        print(f"\n{i}. {test_case['description']}")
        print(f"Query: \"{test_case['query']}\"")
        
        start_time = time.time()
        
        try:
            response = rag_system.query(
                query=test_case['query'],
                max_context_tokens=4000,
                top_k=8
            )
            
            query_time = time.time() - start_time
            
            # Analyze results
            context_length = len(response.context)
            token_count = response.total_tokens
            chunks_used = len(response.chunks_used)
            
            # Check if expected files are represented
            files_found = set()
            for chunk in response.chunks_used:
                file_name = Path(chunk.file_path).name
                files_found.add(file_name)
            
            expected_files = set(test_case.get('expected_files', []))
            files_match = expected_files.intersection(files_found)
            
            print(f"✅ Query completed in {query_time:.2f}s")
            print(f"   - Context length: {context_length:,} chars")
            print(f"   - Tokens used: {token_count}")
            print(f"   - Chunks retrieved: {chunks_used}")
            print(f"   - Files found: {', '.join(sorted(files_found))}")
            
            if expected_files:
                if files_match:
                    print(f"   - ✅ Expected files found: {', '.join(sorted(files_match))}")
                else:
                    print(f"   - ⚠️  Expected files not found: {', '.join(sorted(expected_files))}")
            
            # Show a preview of the context
            if response.context:
                preview = response.context[:200] + "..." if len(response.context) > 200 else response.context
                print(f"   - Context preview: {preview}")
            
            results.append({
                'query': test_case['query'],
                'success': True,
                'query_time': query_time,
                'token_count': token_count,
                'chunks_count': chunks_used,
                'files_found': files_found,
                'expected_files': expected_files,
                'files_match': len(files_match) > 0 if expected_files else True
            })
            
        except Exception as e:
            print(f"❌ Query failed: {e}")
            results.append({
                'query': test_case['query'],
                'success': False,
                'error': str(e)
            })
    
    return results

def print_summary(results):
    """Print test summary"""
    print("\n" + "=" * 80)
    print("📊 TEST SUMMARY")
    print("=" * 80)
    
    successful_queries = [r for r in results if r.get('success', False)]
    failed_queries = [r for r in results if not r.get('success', False)]
    
    print(f"Total queries: {len(results)}")
    print(f"Successful: {len(successful_queries)}")
    print(f"Failed: {len(failed_queries)}")
    
    if successful_queries:
        avg_time = sum(r['query_time'] for r in successful_queries) / len(successful_queries)
        avg_tokens = sum(r['token_count'] for r in successful_queries) / len(successful_queries)
        avg_chunks = sum(r['chunks_count'] for r in successful_queries) / len(successful_queries)
        
        print(f"\nPerformance metrics:")
        print(f"- Average query time: {avg_time:.2f}s")
        print(f"- Average tokens used: {avg_tokens:.0f}")
        print(f"- Average chunks retrieved: {avg_chunks:.1f}")
        
        # File matching accuracy
        queries_with_expectations = [r for r in successful_queries if r.get('expected_files')]
        if queries_with_expectations:
            matching_queries = [r for r in queries_with_expectations if r.get('files_match', False)]
            accuracy = len(matching_queries) / len(queries_with_expectations) * 100
            print(f"- File matching accuracy: {accuracy:.1f}%")
    
    if failed_queries:
        print(f"\n❌ Failed queries:")
        for result in failed_queries:
            print(f"   - \"{result['query'][:50]}...\" - {result.get('error', 'Unknown error')}")
    
    print(f"\n{'✅ All tests passed!' if not failed_queries else '⚠️  Some tests failed'}")

def main():
    """Main test function"""
    print("🚀 RAG System Comprehensive Test")
    print("=" * 80)
    
    # Setup RAG system
    rag_system = setup_rag_system()
    
    # Index the sample project
    if not index_sample_project(rag_system):
        print("❌ Cannot proceed without successful indexing")
        return
    
    # Test queries
    results = test_queries(rag_system)
    
    # Print summary
    print_summary(results)
    
    print(f"\n🎉 Test completed!")
    print(f"You can now test individual queries using:")
    print(f"   response = rag_system.query('your question here')")
    print(f"   print(response.context)")

if __name__ == "__main__":
    main()

# LLM Code Context Optimizer

An intelligent system that optimizes code context for Large Language Models by using AST parsing, dependency graphs, and relevance scoring to provide the most relevant code snippets within token budgets.

## ✨ Features

- 🌳 **Tree-sitter AST Parsing** - Supports Python, JavaScript, TypeScript, Java, Kotlin
- 📊 **Dependency Graph Analysis** - Understands code relationships and importance
- 🎯 **Intelligent Context Ranking** - Prioritizes relevant files based on queries
- 💰 **Token Budget Management** - Respects LLM token limits with smart truncation
- 🔍 **Direct File Query Optimization** - Ultra-focused mode for specific file requests
- ⚡ **Auto-reload Development** - Instant server restarts during development
- 🔄 **Dual Context Modes** - Full project context vs. relevant-only filtering

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set up Gemini API key
export GEMINI_API_KEY="your-gemini-api-key-here"

# 3. Start the server with auto-reload
python api/server.py

# 4. Test the API
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "How does the calculator work?",
    "include_context": true,
    "project_path": "sample_project",
    "relevant_only": true
  }'
```

## Project Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Gemini API Key

You need to set up your Gemini API key to use this service. Choose one of the following methods:

#### Option A: Environment Variable (Recommended)
```bash
export GEMINI_API_KEY="your-gemini-api-key-here"
```

#### Option B: Create a .env file
Create a `.env` file in the project root:
```
GEMINI_API_KEY=your-gemini-api-key-here
```

**To get your Gemini API Key:**
1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Copy the key and use it in one of the methods above

### 3. Run the Application
```bash
python main.py
```

The server will start on `http://localhost:8000`

## Troubleshooting

### Authentication Error
If you see an error like:
```
google.auth.exceptions.DefaultCredentialsError: Your default credentials were not found
```

This means the `GEMINI_API_KEY` environment variable is not set. Follow step 2 above to configure it properly.

### Verify Setup
You can test if your API key is working by making a simple chat request:
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Hello, can you help me with my code?",
    "include_context": false
  }'
```

## 📡 API Endpoints

### POST `/chat` - Chat with LLM using optimized context
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "How does the calculator work?",
    "include_context": true,
    "project_path": "sample_project",
    "relevant_only": true,
    "max_tokens": 8000,
    "use_tree_sitter": true
  }'
```

### GET `/context` - Get optimized context only
```bash
curl "http://localhost:8000/context?path=sample_project&query=calculator&relevant_only=true"
```

### GET `/docs` - Interactive API documentation
Visit `http://localhost:8000/docs` for full API documentation.

## 🎯 Context Modes

### Relevant Mode (`relevant_only=true`)
- **Focused selection** based on query relevance
- **Reduced token usage** (~2000-4000 tokens)
- **Direct file queries** get ultra-focused treatment
- **Best for**: Specific questions, debugging, targeted analysis

### Full Mode (`relevant_only=false`)
- **Complete project context** with all files
- **Maximum token budget** (~6000-8000 tokens)
- **Importance-based sorting** by dependency analysis
- **Best for**: Architecture overview, broad refactoring, comprehensive analysis



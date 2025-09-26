from rag.core.rag_system import RAGSystem
from rag.models import ChatRequest
import os
from utils.file_utils import read_code_files

class ContextService:
    def __init__(self):
        self.rag_system = RAGSystem()

    def get_relevant_context(self, request: ChatRequest) -> str:
        if request.relevant_only:
            # Use the RAG system with full context expansion support
            rag_response = self.rag_system.query(
                query=request.message,
                project_path=request.project_path,
                max_context_tokens=request.max_context_tokens,
                top_k=request.top_k,
                expand_window=request.expand_window,
                expansion_level=request.expansion_level,
                session_id=request.session_id
            )
            return rag_response.context
        else:
            return self.get_all_code_files(request.project_path)

    def get_all_code_files(self, project_path: str) -> str:
        if not os.path.exists(project_path):
            raise FileNotFoundError(f"Project path {project_path} does not exist")

        files_context = read_code_files(project_path)
        if not files_context:
            return ""
        else:
            context_parts = []
            for file_path, context in files_context.items():
                relative_path = os.path.relpath(file_path, project_path)
                context_parts.append(f"== File: {relative_path} ==  \n{context}\n")
            
            return f"Here are all the code files content:\n\n{''.join(context_parts)}"
    
    def get_expanded_context(
        self, 
        message: str, 
        project_path: str, 
        expansion_level: str = "moderate",
        session_id: str = None,
        max_context_tokens: int = 16000
    ) -> str:
        """
        Convenience method for getting expanded context directly.
        Useful when you know you want expanded results from the start.
        """
        rag_response = self.rag_system.query(
            query=message,
            project_path=project_path,
            max_context_tokens=max_context_tokens,
            top_k=15,  # Higher default for expanded queries
            expand_window=True,
            expansion_level=expansion_level,
            session_id=session_id
        )
        return rag_response.context


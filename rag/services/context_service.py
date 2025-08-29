from rag.core.rag_system import RAGSystem
from rag.models import ChatRequest
import os
from utils.file_utils import read_code_files

class ContextService:
    def __init__(self):
        self.rag_system = RAGSystem()

    def get_relevant_context(self, request: ChatRequest) -> str:
        if request.relevant_only:
            return self.rag_system.query(request.message, project_path=request.project_path).context
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


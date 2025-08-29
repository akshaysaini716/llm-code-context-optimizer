"""
Enhanced Multi-Stage Retriever with Context-Aware Boosting
Combines intelligent retrieval strategies with context awareness like Cursor IDE
"""
import logging
from typing import List, Optional, Dict, Any

from rag.models import RetrievalResult
from rag.retrieval.multi_stage_retriever import MultiStageRetriever, QueryAnalysis
from rag.retrieval.context_boosting import ContextAwareBooster, BoostConfig

logger = logging.getLogger(__name__)

class EnhancedMultiStageRetriever(MultiStageRetriever):
    """
    Enhanced multi-stage retriever that combines:
    1. Multi-stage retrieval strategies (symbol, semantic, context expansion)
    2. Context-aware boosting (recent files, relationships, project structure)
    3. User behavior learning (like Cursor IDE)
    """
    
    def __init__(self, boost_config: Optional[BoostConfig] = None, persistence_file: Optional[str] = None):
        super().__init__()
        
        # Initialize context-aware boosting
        self.context_booster = ContextAwareBooster(
            config=boost_config,
            persistence_file=persistence_file
        )
        
        logger.info("Enhanced multi-stage retriever initialized with context boosting")
    
    def retrieve(
        self, 
        query: str, 
        top_k: int = 10,
        current_file: Optional[str] = None,
        cursor_position: Optional[tuple] = None,
        recent_files: Optional[List[str]] = None
    ) -> List[RetrievalResult]:
        """
        Enhanced retrieve method with context awareness
        
        Args:
            query: Search query
            top_k: Number of results to return
            current_file: Currently open/viewing file
            cursor_position: Current cursor position (line, column)
            recent_files: Recently accessed files
        """
        # Update context information
        if current_file:
            self.context_booster.set_current_context(current_file, cursor_position)
        
        if recent_files:
            self.context_booster.query_context.recent_files = recent_files[:10]
        
        # Execute multi-stage retrieval (parent class method)
        results = super().retrieve(query, top_k * 2)  # Get more results for boosting
        
        if not results:
            return []
        
        # Apply context-aware boosting
        boosted_results = self.context_booster.boost_results(results, query)
        
        # Return top-k boosted results
        final_results = boosted_results[:top_k]
        
        logger.info(f"Enhanced retrieval returned {len(final_results)} context-boosted results")
        return final_results
    
    def track_file_activity(self, file_path: str, activity_type: str, **kwargs):
        """
        Track file activity for context learning
        
        Args:
            file_path: Path to the file
            activity_type: 'edit', 'open', 'close'
            **kwargs: Additional parameters (time_spent for close, etc.)
        """
        if activity_type == 'edit':
            self.context_booster.track_file_edit(file_path, kwargs.get('edit_time'))
        elif activity_type == 'open':
            self.context_booster.track_file_open(file_path, kwargs.get('open_time'))
        elif activity_type == 'close':
            self.context_booster.track_file_close(
                file_path, 
                kwargs.get('close_time'), 
                kwargs.get('time_spent', 0)
            )
    
    def update_import_relationships(self, file_imports: Dict[str, List[str]]):
        """
        Update import relationship information for better boosting
        
        Args:
            file_imports: Dict mapping file_path -> [list of imported files]
        """
        # Convert lists to sets for the booster
        import_sets = {file: set(imports) for file, imports in file_imports.items()}
        self.context_booster.update_import_relationships(import_sets)
    
    def get_context_debug_info(self) -> Dict[str, Any]:
        """Get debug information about current context"""
        analysis_info = {}
        
        # Get last query analysis if available
        if hasattr(self, '_last_analysis'):
            analysis_info = {
                'last_query_type': self._last_analysis.query_type.value,
                'last_query_confidence': self._last_analysis.confidence,
                'last_symbols_detected': self._last_analysis.symbols_mentioned
            }
        
        # Get context booster info
        boost_info = self.context_booster.get_context_summary()
        
        return {
            'query_analysis': analysis_info,
            'context_boosting': boost_info,
            'retrieval_stats': {
                'enhanced_retriever': True,
                'context_aware_boosting': True
            }
        }
    
    def save_session_context(self):
        """Save current session context to disk"""
        self.context_booster.save_context()
    
    def cleanup_old_context(self, max_age_days: int = 30):
        """Clean up old context data"""
        self.context_booster.cleanup_old_context(max_age_days)

# Convenience factory functions

def create_enhanced_retriever_for_project(
    project_root: str,
    enable_persistence: bool = True,
    boost_config: Optional[BoostConfig] = None
) -> EnhancedMultiStageRetriever:
    """
    Create an enhanced retriever configured for a specific project
    
    Args:
        project_root: Root directory of the project
        enable_persistence: Whether to save/load context between sessions
        boost_config: Custom boost configuration
    
    Returns:
        Configured EnhancedMultiStageRetriever
    """
    persistence_file = None
    if enable_persistence:
        import os
        persistence_file = os.path.join(project_root, '.rag_context.json')
    
    # Default boost config optimized for code projects
    if boost_config is None:
        boost_config = BoostConfig(
            recent_edit_boost=1.8,      # Strong boost for recently edited files
            current_file_boost=2.2,     # Strong boost for current file context
            import_relationship_boost=1.6,  # Boost related imports
            test_file_penalty=0.8       # Reduce test files unless specifically asked
        )
    
    retriever = EnhancedMultiStageRetriever(
        boost_config=boost_config,
        persistence_file=persistence_file
    )
    
    # Set project context
    retriever.context_booster.query_context.project_root = project_root
    
    return retriever

def create_cursor_like_retriever(project_root: str) -> EnhancedMultiStageRetriever:
    """
    Create a retriever that mimics Cursor IDE's behavior closely
    """
    cursor_config = BoostConfig(
        # Strong recency bias like Cursor
        recent_edit_boost=2.0,
        recent_open_boost=1.5,
        current_file_boost=2.5,
        
        # Relationship awareness
        import_relationship_boost=1.8,
        directory_proximity_boost=1.3,
        related_files_boost=1.4,
        
        # Cursor-like file prioritization
        main_file_boost=1.2,
        test_file_penalty=0.7,  # Cursor tends to deprioritize tests
        
        # Aggressive recency decay (Cursor focuses on very recent activity)
        edit_decay_hours=8.0,   # 8 hours instead of 24
        
        # Higher boost limits
        max_boost=4.0
    )
    
    return create_enhanced_retriever_for_project(
        project_root=project_root,
        enable_persistence=True,
        boost_config=cursor_config
    )

"""
Configuration module for the calculator application
"""

import os
from typing import Dict, Any

class Config:
    """Configuration class for calculator application settings"""
    
    def __init__(self):
        """Initialize configuration with default values"""
        self.history_file = self._get_env_or_default("CALC_HISTORY_FILE", "calculator_history.json")
        self.max_history_size = int(self._get_env_or_default("CALC_MAX_HISTORY", "50"))
        self.decimal_places = int(self._get_env_or_default("CALC_DECIMAL_PLACES", "4"))
        self.auto_save = self._get_env_or_default("CALC_AUTO_SAVE", "true").lower() == "true"
        self.backup_enabled = self._get_env_or_default("CALC_BACKUP", "false").lower() == "true"
        
        # Application metadata
        self.app_name = "Simple Calculator"
        self.version = "1.0.0"
        self.author = "Test Project"
    
    def _get_env_or_default(self, env_var: str, default: str) -> str:
        """Get environment variable or return default value"""
        return os.getenv(env_var, default)
    
    def get_display_precision(self) -> int:
        """Get the number of decimal places for display"""
        return self.decimal_places
    
    def should_auto_save(self) -> bool:
        """Check if auto-save is enabled"""
        return self.auto_save
    
    def is_backup_enabled(self) -> bool:
        """Check if backup is enabled"""
        return self.backup_enabled
    
    def get_history_file_path(self) -> str:
        """Get the full path to the history file"""
        return os.path.abspath(self.history_file)
    
    def validate_settings(self) -> Dict[str, Any]:
        """Validate configuration settings and return status"""
        issues = []
        
        # Check if history file directory exists
        history_dir = os.path.dirname(self.get_history_file_path())
        if not os.path.exists(history_dir):
            issues.append(f"History file directory does not exist: {history_dir}")
        
        # Check decimal places range
        if self.decimal_places < 0 or self.decimal_places > 10:
            issues.append(f"Decimal places should be between 0 and 10, got: {self.decimal_places}")
        
        # Check max history size
        if self.max_history_size < 1 or self.max_history_size > 1000:
            issues.append(f"Max history size should be between 1 and 1000, got: {self.max_history_size}")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "settings": self.to_dict()
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "app_name": self.app_name,
            "version": self.version,
            "author": self.author,
            "history_file": self.history_file,
            "max_history_size": self.max_history_size,
            "decimal_places": self.decimal_places,
            "auto_save": self.auto_save,
            "backup_enabled": self.backup_enabled
        }
    
    def update_from_dict(self, settings: Dict[str, Any]):
        """Update configuration from dictionary"""
        for key, value in settings.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def reset_to_defaults(self):
        """Reset all settings to default values"""
        self.__init__()
    
    def get_supported_operations(self) -> list:
        """Get list of supported calculator operations"""
        return [
            "add", "subtract", "multiply", "divide", 
            "power", "square_root", "average"
        ] 
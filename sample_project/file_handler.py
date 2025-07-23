"""
File handler module for saving and loading calculation history
"""

import json
import os
from typing import List, Dict, Any
from datetime import datetime

class FileHandler:
    """Handles file operations for the calculator application"""
    
    def __init__(self, filename: str = "calculator_history.json"):
        """Initialize file handler with specified filename"""
        self.filename = filename
        self.ensure_file_exists()
    
    def ensure_file_exists(self):
        """Create the history file if it doesn't exist"""
        if not os.path.exists(self.filename):
            self.save_history([])
    
    def save_history(self, history: List[str]) -> bool:
        """Save calculation history to file"""
        try:
            data = {
                "history": history,
                "saved_at": datetime.now().isoformat(),
                "total_calculations": len(history)
            }
            
            with open(self.filename, 'w', encoding='utf-8') as file:
                json.dump(data, file, indent=2, ensure_ascii=False)
            
            return True
            
        except Exception as e:
            print(f"Error saving history: {e}")
            return False
    
    def load_history(self) -> List[str]:
        """Load calculation history from file"""
        try:
            if not os.path.exists(self.filename):
                return []
            
            with open(self.filename, 'r', encoding='utf-8') as file:
                data = json.load(file)
            
            # Handle both old format (list) and new format (dict)
            if isinstance(data, list):
                return data
            elif isinstance(data, dict) and "history" in data:
                return data["history"]
            else:
                return []
                
        except Exception as e:
            print(f"Error loading history: {e}")
            return []
    
    def backup_history(self) -> bool:
        """Create a backup of the current history file"""
        try:
            backup_filename = f"{self.filename}.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            if os.path.exists(self.filename):
                import shutil
                shutil.copy2(self.filename, backup_filename)
                return True
            
            return False
            
        except Exception as e:
            print(f"Error creating backup: {e}")
            return False
    
    def get_file_info(self) -> Dict[str, Any]:
        """Get information about the history file"""
        try:
            if not os.path.exists(self.filename):
                return {"exists": False}
            
            stat = os.stat(self.filename)
            
            return {
                "exists": True,
                "size_bytes": stat.st_size,
                "modified_time": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "filename": self.filename
            }
            
        except Exception as e:
            return {"exists": False, "error": str(e)}
    
    def clear_history_file(self) -> bool:
        """Clear the history file by saving an empty history"""
        return self.save_history([])
    
    def export_to_text(self, output_filename: str = None) -> bool:
        """Export history to a readable text file"""
        try:
            if output_filename is None:
                output_filename = f"calculator_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            
            history = self.load_history()
            
            with open(output_filename, 'w', encoding='utf-8') as file:
                file.write("Calculator History Export\n")
                file.write("=" * 30 + "\n")
                file.write(f"Exported on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                file.write(f"Total calculations: {len(history)}\n\n")
                
                for i, calculation in enumerate(history, 1):
                    file.write(f"{i:3d}. {calculation}\n")
            
            return True
            
        except Exception as e:
            print(f"Error exporting to text: {e}")
            return False 
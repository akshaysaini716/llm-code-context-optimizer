"""
AST Parser service using tree-sitter for multiple programming languages
Extracts symbols, imports, functions, classes, and their relationships
"""

import os
from typing import Dict, List, Set, Optional, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from tree_sitter_language_pack import get_language, get_parser
from tree_sitter import Parser, Node

@dataclass
class Symbol:
    """Represents a symbol (function, class, variable) in the code"""
    name: str
    type: str  # 'function', 'class', 'variable', 'import'
    file_path: str
    start_line: int
    end_line: int
    scope: str = ""  # class or module scope
    parameters: List[str] = field(default_factory=list)
    return_type: Optional[str] = None
    docstring: Optional[str] = None
    dependencies: Set[str] = field(default_factory=set)  # symbols this symbol depends on

@dataclass
class FileAnalysis:
    """Analysis result for a single file"""
    file_path: str
    symbols: List[Symbol]
    imports: List[str]
    exports: List[str]
    dependencies: Set[str]  # other files this file depends on
    language: str

class ASTParser:
    """Tree-sitter based AST parser for multiple languages"""
    
    def __init__(self):
        """Initialize parsers for supported languages using tree-sitter-language-pack"""
        self.languages = {}
        self.parsers = {}
        
        # Language mapping for tree-sitter-language-pack
        language_map = {
            '.py': 'python',
            '.js': 'javascript', 
            '.jsx': 'javascript',
            '.ts': 'typescript',
            '.tsx': 'typescript',
            '.java': 'java',
            '.kt': 'kotlin',
            '.kts': 'kotlin',
        }
        
        for ext, lang_name in language_map.items():
            try:
                # Use the language pack to get language and parser
                language = get_language(lang_name)
                parser = get_parser(lang_name)
                
                self.languages[ext] = language
                self.parsers[ext] = parser
                
                print(f"✅ Successfully initialized {ext} parser for {lang_name}")
                
            except Exception as e:
                print(f"❌ Failed to initialize {ext} parser for {lang_name}: {e}")
                continue
        
        print(f"🎉 Successfully initialized parsers for: {list(self.parsers.keys())}")
    
    def get_file_extension(self, file_path: str) -> str:
        """Get file extension from file path"""
        return Path(file_path).suffix.lower()
    
    def is_supported_file(self, file_path: str) -> bool:
        """Check if file type is supported"""
        return self.get_file_extension(file_path) in self.languages
    
    def parse_file(self, file_path: str, content: str) -> Optional[FileAnalysis]:
        """Parse a single file and extract symbols and dependencies"""
        ext = self.get_file_extension(file_path)
        if ext not in self.parsers:
            return None
        
        parser = self.parsers[ext]
        tree = parser.parse(bytes(content, 'utf8'))
        
        # Extract symbols based on language
        if ext == '.py':
            return self._parse_python(file_path, content, tree.root_node)
        elif ext in ['.js', '.jsx']:
            return self._parse_javascript(file_path, content, tree.root_node)
        elif ext in ['.ts', '.tsx']:
            return self._parse_typescript(file_path, content, tree.root_node)
        elif ext == '.java':
            return self._parse_java(file_path, content, tree.root_node)
        elif ext in ['.kt', '.kts']:
            return self._parse_kotlin(file_path, content, tree.root_node)
        
        return None
    
    def _parse_python(self, file_path: str, content: str, root: Node) -> FileAnalysis:
        """Parse Python file"""
        symbols = []
        imports = []
        exports = []
        dependencies = set()
        
        lines = content.split('\n')
        
        def extract_text(node: Node) -> str:
            return content[node.start_byte:node.end_byte]
        
        def visit_node(node: Node, scope: str = ""):
            """Recursively visit nodes to extract symbols"""
            if node.type == 'import_statement':
                # import module
                for child in node.children:
                    if child.type == 'dotted_name':
                        import_name = extract_text(child)
                        imports.append(import_name)
                        dependencies.add(import_name)
            
            elif node.type == 'import_from_statement':
                # from module import ...
                module_name = None
                imported_items = []
                
                for child in node.children:
                    if child.type == 'dotted_name' and not module_name:
                        module_name = extract_text(child)
                    elif child.type == 'import_list':
                        for item in child.children:
                            if item.type == 'dotted_name':
                                imported_items.append(extract_text(item))
                
                if module_name:
                    imports.append(f"from {module_name} import {', '.join(imported_items)}")
                    dependencies.add(module_name)
            
            elif node.type == 'function_definition':
                # Function definition
                func_name = None
                parameters = []
                docstring = None
                
                for child in node.children:
                    if child.type == 'identifier':
                        func_name = extract_text(child)
                    elif child.type == 'parameters':
                        for param in child.children:
                            if param.type == 'identifier':
                                parameters.append(extract_text(param))
                            elif param.type == 'typed_parameter':
                                for p in param.children:
                                    if p.type == 'identifier':
                                        parameters.append(extract_text(p))
                                        break
                    elif child.type == 'block':
                        # Look for docstring in first statement
                        for stmt in child.children:
                            if stmt.type == 'expression_statement':
                                for expr in stmt.children:
                                    if expr.type == 'string':
                                        docstring = extract_text(expr).strip('"\'')
                                        break
                                break
                
                if func_name:
                    symbol = Symbol(
                        name=func_name,
                        type='function',
                        file_path=file_path,
                        start_line=node.start_point[0] + 1,
                        end_line=node.end_point[0] + 1,
                        scope=scope,
                        parameters=parameters,
                        docstring=docstring
                    )
                    symbols.append(symbol)
                    exports.append(func_name)
            
            elif node.type == 'class_definition':
                # Class definition
                class_name = None
                
                for child in node.children:
                    if child.type == 'identifier':
                        class_name = extract_text(child)
                        break
                
                if class_name:
                    symbol = Symbol(
                        name=class_name,
                        type='class',
                        file_path=file_path,
                        start_line=node.start_point[0] + 1,
                        end_line=node.end_point[0] + 1,
                        scope=scope
                    )
                    symbols.append(symbol)
                    exports.append(class_name)
                    
                    # Parse class methods
                    for child in node.children:
                        if child.type == 'block':
                            for method_node in child.children:
                                visit_node(method_node, class_name)
            
            # Recursively visit children
            for child in node.children:
                if node.type != 'class_definition':  # Avoid double-processing class methods
                    visit_node(child, scope)
        
        visit_node(root)
        
        return FileAnalysis(
            file_path=file_path,
            symbols=symbols,
            imports=imports,
            exports=exports,
            dependencies=dependencies,
            language='python'
        )
    
    def _parse_javascript(self, file_path: str, content: str, root: Node) -> FileAnalysis:
        """Parse JavaScript file"""
        symbols = []
        imports = []
        exports = []
        dependencies = set()
        
        def extract_text(node: Node) -> str:
            return content[node.start_byte:node.end_byte]
        
        def visit_node(node: Node, scope: str = ""):
            if node.type == 'import_statement':
                # ES6 imports
                for child in node.children:
                    if child.type == 'string':
                        module_name = extract_text(child).strip('"\'')
                        imports.append(f"import from {module_name}")
                        dependencies.add(module_name)
            
            elif node.type == 'function_declaration':
                func_name = None
                parameters = []
                
                for child in node.children:
                    if child.type == 'identifier':
                        func_name = extract_text(child)
                    elif child.type == 'formal_parameters':
                        for param in child.children:
                            if param.type == 'identifier':
                                parameters.append(extract_text(param))
                
                if func_name:
                    symbol = Symbol(
                        name=func_name,
                        type='function',
                        file_path=file_path,
                        start_line=node.start_point[0] + 1,
                        end_line=node.end_point[0] + 1,
                        scope=scope,
                        parameters=parameters
                    )
                    symbols.append(symbol)
                    exports.append(func_name)
            
            elif node.type == 'class_declaration':
                class_name = None
                
                for child in node.children:
                    if child.type == 'identifier':
                        class_name = extract_text(child)
                        break
                
                if class_name:
                    symbol = Symbol(
                        name=class_name,
                        type='class',
                        file_path=file_path,
                        start_line=node.start_point[0] + 1,
                        end_line=node.end_point[0] + 1,
                        scope=scope
                    )
                    symbols.append(symbol)
                    exports.append(class_name)
            
            # Recursively visit children
            for child in node.children:
                visit_node(child, scope)
        
        visit_node(root)
        
        return FileAnalysis(
            file_path=file_path,
            symbols=symbols,
            imports=imports,
            exports=exports,
            dependencies=dependencies,
            language='javascript'
        )
    
    def _parse_typescript(self, file_path: str, content: str, root: Node) -> FileAnalysis:
        """Parse TypeScript file (similar to JavaScript with type info)"""
        # For now, use JavaScript parser with some TypeScript-specific handling
        analysis = self._parse_javascript(file_path, content, root)
        analysis.language = 'typescript'
        return analysis
    
    def _parse_java(self, file_path: str, content: str, root: Node) -> FileAnalysis:
        """Parse Java file"""
        symbols = []
        imports = []
        exports = []
        dependencies = set()
        
        def extract_text(node: Node) -> str:
            return content[node.start_byte:node.end_byte]
        
        def visit_node(node: Node, scope: str = ""):
            if node.type == 'import_declaration':
                for child in node.children:
                    if child.type == 'scoped_identifier':
                        import_name = extract_text(child)
                        imports.append(import_name)
                        dependencies.add(import_name.split('.')[0])
            
            elif node.type == 'method_declaration':
                method_name = None
                parameters = []
                
                for child in node.children:
                    if child.type == 'identifier':
                        method_name = extract_text(child)
                    elif child.type == 'formal_parameters':
                        for param in child.children:
                            if param.type == 'formal_parameter':
                                for p in param.children:
                                    if p.type == 'identifier':
                                        parameters.append(extract_text(p))
                
                if method_name:
                    symbol = Symbol(
                        name=method_name,
                        type='function',
                        file_path=file_path,
                        start_line=node.start_point[0] + 1,
                        end_line=node.end_point[0] + 1,
                        scope=scope,
                        parameters=parameters
                    )
                    symbols.append(symbol)
                    exports.append(method_name)
            
            elif node.type == 'class_declaration':
                class_name = None
                
                for child in node.children:
                    if child.type == 'identifier':
                        class_name = extract_text(child)
                        break
                
                if class_name:
                    symbol = Symbol(
                        name=class_name,
                        type='class',
                        file_path=file_path,
                        start_line=node.start_point[0] + 1,
                        end_line=node.end_point[0] + 1,
                        scope=scope
                    )
                    symbols.append(symbol)
                    exports.append(class_name)
                    
                    # Parse class methods
                    for child in node.children:
                        if child.type == 'class_body':
                            for method_node in child.children:
                                visit_node(method_node, class_name)
            
            # Recursively visit children
            for child in node.children:
                if node.type != 'class_declaration':
                    visit_node(child, scope)
        
        visit_node(root)
        
        return FileAnalysis(
            file_path=file_path,
            symbols=symbols,
            imports=imports,
            exports=exports,
            dependencies=dependencies,
            language='java'
        )
    
    def _parse_kotlin(self, file_path: str, content: str, root: Node) -> FileAnalysis:
        """Parse Kotlin file"""
        symbols = []
        imports = []
        exports = []
        dependencies = set()
        
        def extract_text(node: Node) -> str:
            return content[node.start_byte:node.end_byte]
        
        def visit_node(node: Node, scope: str = ""):
            if node.type == 'import_header':
                for child in node.children:
                    if child.type == 'import_identifier':
                        import_name = extract_text(child)
                        imports.append(import_name)
                        dependencies.add(import_name.split('.')[0])
            
            elif node.type == 'function_declaration':
                func_name = None
                parameters = []
                
                for child in node.children:
                    if child.type == 'simple_identifier':
                        func_name = extract_text(child)
                    elif child.type == 'function_value_parameters':
                        for param in child.children:
                            if param.type == 'function_value_parameter':
                                for p in param.children:
                                    if p.type == 'simple_identifier':
                                        parameters.append(extract_text(p))
                                        break
                
                if func_name:
                    symbol = Symbol(
                        name=func_name,
                        type='function',
                        file_path=file_path,
                        start_line=node.start_point[0] + 1,
                        end_line=node.end_point[0] + 1,
                        scope=scope,
                        parameters=parameters
                    )
                    symbols.append(symbol)
                    exports.append(func_name)
            
            elif node.type == 'class_declaration':
                class_name = None
                
                for child in node.children:
                    if child.type == 'simple_identifier':
                        class_name = extract_text(child)
                        break
                
                if class_name:
                    symbol = Symbol(
                        name=class_name,
                        type='class',
                        file_path=file_path,
                        start_line=node.start_point[0] + 1,
                        end_line=node.end_point[0] + 1,
                        scope=scope
                    )
                    symbols.append(symbol)
                    exports.append(class_name)
                    
                    # Parse class methods
                    for child in node.children:
                        if child.type == 'class_body':
                            for method_node in child.children:
                                visit_node(method_node, class_name)
            
            elif node.type == 'object_declaration':
                object_name = None
                
                for child in node.children:
                    if child.type == 'simple_identifier':
                        object_name = extract_text(child)
                        break
                
                if object_name:
                    symbol = Symbol(
                        name=object_name,
                        type='object',
                        file_path=file_path,
                        start_line=node.start_point[0] + 1,
                        end_line=node.end_point[0] + 1,
                        scope=scope
                    )
                    symbols.append(symbol)
                    exports.append(object_name)
            
            # Recursively visit children
            for child in node.children:
                if node.type != 'class_declaration':  # Avoid double-processing class methods
                    visit_node(child, scope)
        
        visit_node(root)
        
        return FileAnalysis(
            file_path=file_path,
            symbols=symbols,
            imports=imports,
            exports=exports,
            dependencies=dependencies,
            language='kotlin'
        )
    
    def analyze_codebase(self, directory: str, extensions: Optional[List[str]] = None) -> Dict[str, FileAnalysis]:
        """Analyze entire codebase and return file analyses"""
        if extensions is None:
            extensions = list(self.languages.keys())
        
        analyses = {}
        
        for root, _, filenames in os.walk(directory):
            for filename in filenames:
                file_path = os.path.join(root, filename)
                ext = self.get_file_extension(file_path)
                
                if ext in extensions and self.is_supported_file(file_path):
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        analysis = self.parse_file(file_path, content)
                        if analysis:
                            analyses[file_path] = analysis
                    
                    except Exception as e:
                        print(f"Error analyzing {file_path}: {e}")
                        continue
        
        return analyses 
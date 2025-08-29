"""
Improved Language Support for Tree-Sitter Chunker
Supports: Python, Java, Kotlin, Scala, TypeScript/JavaScript
Fixes JavaScript/TypeScript and Java chunking, adds module-level extraction
"""
import logging
import re
from typing import List, Tuple, Optional, Set
from pathlib import Path
from tree_sitter import Node, Tree

from rag.models import CodeBaseChunk

logger = logging.getLogger(__name__)

class ImprovedLanguageSupport:
    """Enhanced language-specific chunking implementations"""
    
    @staticmethod
    def chunk_javascript_enhanced(
        chunker_instance, 
        file_path: Path, 
        content: str, 
        tree: Tree
    ) -> List[CodeBaseChunk]:
        """Proper JavaScript/TypeScript chunking"""
        chunks = []
        lines = content.split('\n')
        
        # Extract imports
        imports = ImprovedLanguageSupport._extract_js_imports(content, tree)
        import_symbols = ImprovedLanguageSupport._analyze_js_imports(imports)
        
        # Extract and process classes
        classes = ImprovedLanguageSupport._extract_js_classes(tree, content, chunker_instance)
        for class_name, class_node in classes:
            # Check if we should preserve class methods together
            if chunker_instance.config.preserve_class_methods:
                class_chunk = ImprovedLanguageSupport._create_js_class_chunk(
                    chunker_instance, file_path, content, class_node, class_name, 
                    imports, import_symbols
                )
                if class_chunk:
                    chunks.append(class_chunk)
            else:
                # Split class into separate chunks
                chunk = ImprovedLanguageSupport._create_chunk_from_node(
                    chunker_instance, file_path, content, class_node, class_name, imports
                )
                if chunk:
                    chunks.append(chunk)
        
        # Extract functions (not in classes)
        functions = ImprovedLanguageSupport._extract_js_functions(tree, content)
        standalone_functions = ImprovedLanguageSupport._filter_standalone_functions(
            functions, classes
        )
        
        for func_name, func_node in standalone_functions:
            relevant_imports = ImprovedLanguageSupport._get_relevant_js_imports(
                func_node, imports, import_symbols, content
            )
            chunk = ImprovedLanguageSupport._create_chunk_from_node(
                chunker_instance, file_path, content, func_node, func_name, relevant_imports
            )
            if chunk:
                chunks.append(chunk)
        
        # Extract interfaces (TypeScript)
        if file_path.suffix in ['.ts', '.tsx']:
            interfaces = ImprovedLanguageSupport._extract_ts_interfaces(tree, content)
            for interface_name, interface_node in interfaces:
                chunk = ImprovedLanguageSupport._create_chunk_from_node(
                    chunker_instance, file_path, content, interface_node, 
                    interface_name, imports
                )
                if chunk:
                    chunks.append(chunk)
        
        # Extract module-level code
        module_chunk = ImprovedLanguageSupport._create_js_module_chunk(
            chunker_instance, file_path, content, tree, imports
        )
        if module_chunk:
            chunks.append(module_chunk)
        
        return chunks
    
    @staticmethod  
    def chunk_java_enhanced(
        chunker_instance,
        file_path: Path,
        content: str, 
        tree: Tree
    ) -> List[CodeBaseChunk]:
        """Proper Java chunking implementation"""
        chunks = []
        
        # Extract package and imports
        package_info = ImprovedLanguageSupport._extract_java_package(content)
        imports = ImprovedLanguageSupport._extract_java_imports(content, tree)
        
        # Extract classes and interfaces
        classes = ImprovedLanguageSupport._extract_java_classes(tree, content, chunker_instance)
        interfaces = ImprovedLanguageSupport._extract_java_interfaces(tree, content)
        
        # Process classes
        for class_name, class_node in classes:
            if chunker_instance.config.preserve_class_methods:
                class_chunk = ImprovedLanguageSupport._create_java_class_chunk(
                    chunker_instance, file_path, content, class_node, 
                    class_name, imports, package_info
                )
                if class_chunk:
                    chunks.append(class_chunk)
            else:
                chunk = ImprovedLanguageSupport._create_chunk_from_node(
                    chunker_instance, file_path, content, class_node, 
                    class_name, imports
                )
                if chunk:
                    chunks.append(chunk)
        
        # Process interfaces
        for interface_name, interface_node in interfaces:
            chunk = ImprovedLanguageSupport._create_chunk_from_node(
                chunker_instance, file_path, content, interface_node, 
                interface_name, imports
            )
            if chunk:
                chunks.append(chunk)
        
        # Extract standalone methods (rare in Java, but possible in records/enums)
        methods = ImprovedLanguageSupport._extract_java_methods(tree, content)
        standalone_methods = ImprovedLanguageSupport._filter_standalone_methods(
            methods, classes + interfaces
        )
        
        for method_name, method_node in standalone_methods:
            chunk = ImprovedLanguageSupport._create_chunk_from_node(
                chunker_instance, file_path, content, method_node, 
                method_name, imports
            )
            if chunk:
                chunks.append(chunk)
        
        return chunks
    
    @staticmethod
    def chunk_kotlin_enhanced(
        chunker_instance,
        file_path: Path,
        content: str,
        tree: Tree
    ) -> List[CodeBaseChunk]:
        """Proper Kotlin chunking implementation"""
        chunks = []
        
        # Extract package and imports
        package_info = ImprovedLanguageSupport._extract_kotlin_package(content)
        imports = ImprovedLanguageSupport._extract_kotlin_imports(content, tree)
        
        # Extract classes and interfaces
        classes = ImprovedLanguageSupport._extract_kotlin_classes(tree, content)
        interfaces = ImprovedLanguageSupport._extract_kotlin_interfaces(tree, content)
        
        # Process classes
        for class_name, class_node in classes:
            if chunker_instance.config.preserve_class_methods:
                class_chunk = ImprovedLanguageSupport._create_kotlin_class_chunk(
                    chunker_instance, file_path, content, class_node,
                    class_name, imports, package_info
                )
                if class_chunk:
                    chunks.append(class_chunk)
            else:
                chunk = ImprovedLanguageSupport._create_chunk_from_node(
                    chunker_instance, file_path, content, class_node,
                    class_name, '\n'.join(imports)
                )
                if chunk:
                    chunks.append(chunk)
        
        # Process interfaces
        for interface_name, interface_node in interfaces:
            chunk = ImprovedLanguageSupport._create_chunk_from_node(
                chunker_instance, file_path, content, interface_node,
                interface_name, '\n'.join(imports)
            )
            if chunk:
                chunks.append(chunk)
        
        # Extract standalone functions (top-level functions in Kotlin)
        functions = ImprovedLanguageSupport._extract_kotlin_functions(tree, content)
        standalone_functions = ImprovedLanguageSupport._filter_standalone_functions(
            functions, classes + interfaces
        )
        
        for func_name, func_node in standalone_functions:
            chunk = ImprovedLanguageSupport._create_chunk_from_node(
                chunker_instance, file_path, content, func_node,
                func_name, '\n'.join(imports)
            )
            if chunk:
                chunks.append(chunk)
        
        # Extract object declarations
        objects = ImprovedLanguageSupport._extract_kotlin_objects(tree, content)
        for object_name, object_node in objects:
            chunk = ImprovedLanguageSupport._create_chunk_from_node(
                chunker_instance, file_path, content, object_node,
                object_name, '\n'.join(imports)
            )
            if chunk:
                chunks.append(chunk)
        
        return chunks
    
    @staticmethod
    def chunk_scala_enhanced(
        chunker_instance,
        file_path: Path,
        content: str,
        tree: Tree
    ) -> List[CodeBaseChunk]:
        """Proper Scala chunking implementation"""
        chunks = []
        
        # Extract package and imports
        package_info = ImprovedLanguageSupport._extract_scala_package(content)
        imports = ImprovedLanguageSupport._extract_scala_imports(content, tree)
        
        # Extract classes, traits, and objects
        classes = ImprovedLanguageSupport._extract_scala_classes(tree, content)
        traits = ImprovedLanguageSupport._extract_scala_traits(tree, content)
        objects = ImprovedLanguageSupport._extract_scala_objects(tree, content)
        
        # Process classes
        for class_name, class_node in classes:
            if chunker_instance.config.preserve_class_methods:
                class_chunk = ImprovedLanguageSupport._create_scala_class_chunk(
                    chunker_instance, file_path, content, class_node,
                    class_name, imports, package_info
                )
                if class_chunk:
                    chunks.append(class_chunk)
            else:
                chunk = ImprovedLanguageSupport._create_chunk_from_node(
                    chunker_instance, file_path, content, class_node,
                    class_name, '\n'.join(imports)
                )
                if chunk:
                    chunks.append(chunk)
        
        # Process traits
        for trait_name, trait_node in traits:
            chunk = ImprovedLanguageSupport._create_chunk_from_node(
                chunker_instance, file_path, content, trait_node,
                trait_name, '\n'.join(imports)
            )
            if chunk:
                chunks.append(chunk)
        
        # Process objects
        for object_name, object_node in objects:
            chunk = ImprovedLanguageSupport._create_chunk_from_node(
                chunker_instance, file_path, content, object_node,
                object_name, '\n'.join(imports)
            )
            if chunk:
                chunks.append(chunk)
        
        # Extract standalone functions (rare in Scala, but possible)
        functions = ImprovedLanguageSupport._extract_scala_functions(tree, content)
        standalone_functions = ImprovedLanguageSupport._filter_standalone_functions(
            functions, classes + traits + objects
        )
        
        for func_name, func_node in standalone_functions:
            chunk = ImprovedLanguageSupport._create_chunk_from_node(
                chunker_instance, file_path, content, func_node,
                func_name, '\n'.join(imports)
            )
            if chunk:
                chunks.append(chunk)
        
        return chunks
    
    @staticmethod
    def create_python_module_chunk(
        chunker_instance,
        file_path: Path,
        content: str,
        tree: Tree,
        imports: List[str]
    ) -> Optional[CodeBaseChunk]:
        """Extract meaningful module-level code (constants, globals, main block)"""
        
        # Find module-level statements (not in classes or functions)
        module_statements = []
        lines = content.split('\n')
        
        # Extract module-level assignments and constants
        module_level_code = ImprovedLanguageSupport._extract_module_level_python(tree, content)
        
        if not module_level_code:
            return None
        
        # Combine imports with module-level code
        module_content_parts = []
        if imports:
            module_content_parts.append('\n'.join(imports))
            module_content_parts.append('')  # blank line
        
        module_content_parts.append(module_level_code)
        full_module_content = '\n'.join(module_content_parts)
        
        # Only create chunk if there's significant module-level code
        token_count = len(chunker_instance.tokenizer.encode(module_level_code))
        if token_count < 50:  # Skip if too small
            return None
        
        chunk_id = chunker_instance._generate_chunk_id(file_path, "module", "module")
        
        return CodeBaseChunk(
            id=chunk_id,
            file_path=str(file_path),
            content=full_module_content,
            language=file_path.suffix.lower().lstrip('.'),
            chunk_type="module",
            start_byte=0,
            end_byte=len(content.encode('utf-8')),
            start_line=1,
            end_line=len(lines)
        )
    
    # Helper methods for JavaScript/TypeScript
    
    @staticmethod
    def _extract_js_imports(content: str, tree: Tree) -> List[str]:
        """Extract JavaScript/TypeScript import statements"""
        imports = []
        
        # Use regex as fallback for import extraction
        import_patterns = [
            r'^import\s+.*?from\s+["\'].*?["\'];?\s*$',
            r'^import\s+["\'].*?["\'];?\s*$',
            r'^const\s+.*?=\s+require\(["\'].*?["\']\);?\s*$',
        ]
        
        lines = content.split('\n')
        for line in lines:
            line_stripped = line.strip()
            for pattern in import_patterns:
                if re.match(pattern, line_stripped):
                    imports.append(line_stripped)
                    break
        
        return imports
    
    @staticmethod
    def _analyze_js_imports(imports: List[str]) -> Set[str]:
        """Extract imported symbols from JS imports"""
        symbols = set()
        
        for import_stmt in imports:
            # import { symbol1, symbol2 } from 'module'
            brace_match = re.search(r'\{\s*([^}]+)\s*\}', import_stmt)
            if brace_match:
                symbol_list = brace_match.group(1)
                for symbol in symbol_list.split(','):
                    clean_symbol = symbol.strip().split(' as ')[0]
                    if clean_symbol:
                        symbols.add(clean_symbol)
            
            # import defaultSymbol from 'module'
            default_match = re.search(r'import\s+(\w+)\s+from', import_stmt)
            if default_match:
                symbols.add(default_match.group(1))
        
        return symbols
    
    @staticmethod
    def _extract_js_classes(tree: Tree, content: str, chunker_instance=None) -> List[Tuple[str, Node]]:
        """Extract JavaScript/TypeScript class definitions using tree-sitter queries"""
        classes = []
        
        # Try tree-sitter queries first (more accurate)
        if chunker_instance and hasattr(chunker_instance, 'queries'):
            ext = None
            for possible_ext in ['.js', '.jsx', '.ts', '.tsx']:
                if possible_ext in chunker_instance.queries and 'classes' in chunker_instance.queries[possible_ext]:
                    ext = possible_ext
                    break
            
            if ext:
                try:
                    query = chunker_instance.queries[ext]['classes']
                    from tree_sitter import QueryCursor
                    cursor = QueryCursor()
                    captures = cursor.captures(tree.root_node, query)
                    
                    class_names = []
                    class_defs = []
                    
                    for capture in captures:
                        node, capture_name = capture
                        if capture_name == 'class-name':
                            class_names.append(node)
                        elif capture_name == 'class-def':
                            class_defs.append(node)
                    
                    for name_node, def_node in zip(class_names, class_defs):
                        class_name = content[name_node.start_byte:name_node.end_byte]
                        classes.append((class_name, def_node))
                    
                    if classes:  # If we found classes with queries, return them
                        return classes
                        
                except Exception as e:
                    logger.warning(f"Tree-sitter query failed for JS classes: {e}")
        
        # Fallback to direct tree traversal
        try:
            for node in tree.root_node.children:
                if node.type == 'class_declaration':
                    class_name = ImprovedLanguageSupport._get_class_name_from_node(node, content)
                    if class_name:
                        classes.append((class_name, node))
        except:
            pass
        
        # Regex fallback (last resort)
        if not classes:
            class_pattern = r'class\s+(\w+)(?:\s+extends\s+\w+)?\s*\{'
            for match in re.finditer(class_pattern, content):
                class_name = match.group(1)
                start_pos = match.start()
                end_pos = ImprovedLanguageSupport._find_function_end_js(content, start_pos)
                
                # Create a minimal node-like object
                pseudo_node = type('Node', (), {
                    'start_byte': start_pos,
                    'end_byte': end_pos,
                    'start_point': (content[:start_pos].count('\n'), 0),
                    'end_point': (content[:end_pos].count('\n'), 0),
                    'type': 'class_declaration'
                })()
                
                classes.append((class_name, pseudo_node))
        
        return classes
    
    @staticmethod
    def _extract_js_functions(tree: Tree, content: str) -> List[Tuple[str, Node]]:
        """Extract JavaScript function definitions"""
        functions = []
        
        # Multiple function patterns
        patterns = [
            r'function\s+(\w+)\s*\(',
            r'const\s+(\w+)\s*=\s*(?:async\s+)?(?:function\s*)?\(',
            r'let\s+(\w+)\s*=\s*(?:async\s+)?(?:function\s*)?\(',
            r'var\s+(\w+)\s*=\s*(?:async\s+)?(?:function\s*)?\(',
            r'(\w+)\s*:\s*(?:async\s+)?function\s*\(',
            r'(\w+)\s*:\s*\([^)]*\)\s*=>'
        ]
        
        for pattern in patterns:
            for match in re.finditer(pattern, content):
                func_name = match.group(1)
                start_pos = match.start()
                
                # Simple end detection (could be improved)
                end_pos = ImprovedLanguageSupport._find_function_end_js(content, start_pos)
                
                pseudo_node = type('Node', (), {
                    'start_byte': start_pos,
                    'end_byte': end_pos,
                    'start_point': (content[:start_pos].count('\n'), 0),
                    'end_point': (content[:end_pos].count('\n'), 0),
                    'type': 'function_declaration'
                })()
                
                functions.append((func_name, pseudo_node))
        
        return functions
    
    @staticmethod
    def _find_function_end_js(content: str, start_pos: int) -> int:
        """Find the end of a JavaScript function"""
        # Look for the opening brace
        brace_start = content.find('{', start_pos)
        if brace_start == -1:
            # Arrow function without braces
            line_end = content.find('\n', start_pos)
            return line_end if line_end != -1 else len(content)
        
        # Count braces to find the end
        brace_count = 0
        for i, char in enumerate(content[brace_start:], brace_start):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    return i + 1
        
        return len(content)  # Fallback
    
    @staticmethod
    def _extract_ts_interfaces(tree: Tree, content: str) -> List[Tuple[str, Node]]:
        """Extract TypeScript interface definitions"""
        interfaces = []
        
        interface_pattern = r'interface\s+(\w+)(?:\s+extends\s+[\w,\s]+)?\s*\{'
        for match in re.finditer(interface_pattern, content):
            interface_name = match.group(1)
            start_pos = match.start()
            
            # Find the end of the interface
            brace_count = 0
            end_pos = start_pos
            for i, char in enumerate(content[start_pos:], start_pos):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end_pos = i + 1
                        break
            
            pseudo_node = type('Node', (), {
                'start_byte': start_pos,
                'end_byte': end_pos,
                'start_point': (content[:start_pos].count('\n'), 0),
                'end_point': (content[:end_pos].count('\n'), 0),
                'type': 'interface_declaration'
            })()
            
            interfaces.append((interface_name, pseudo_node))
        
        return interfaces
    
    # Helper methods for Java
    
    @staticmethod
    def _extract_java_package(content: str) -> Optional[str]:
        """Extract Java package declaration"""
        package_match = re.search(r'^package\s+([\w.]+);', content, re.MULTILINE)
        return package_match.group(1) if package_match else None
    
    @staticmethod
    def _extract_java_imports(content: str, tree: Tree) -> List[str]:
        """Extract Java import statements"""
        imports = []
        
        import_pattern = r'^import\s+(?:static\s+)?[\w.]+(?:\.\*)?;'
        lines = content.split('\n')
        
        for line in lines:
            if re.match(import_pattern, line.strip()):
                imports.append(line.strip())
        
        return imports
    
    @staticmethod
    def _extract_java_classes(tree: Tree, content: str, chunker_instance=None) -> List[Tuple[str, Node]]:
        """Extract Java class definitions using tree-sitter queries"""
        classes = []
        
        # Try tree-sitter queries first (more accurate)
        if chunker_instance and hasattr(chunker_instance, 'queries') and '.java' in chunker_instance.queries:
            try:
                query = chunker_instance.queries['.java']['classes']
                from tree_sitter import QueryCursor
                cursor = QueryCursor()
                captures = cursor.captures(tree.root_node, query)
                
                class_names = []
                class_defs = []
                
                for capture in captures:
                    node, capture_name = capture
                    if capture_name == 'class-name':
                        class_names.append(node)
                    elif capture_name == 'class-def':
                        class_defs.append(node)
                
                for name_node, def_node in zip(class_names, class_defs):
                    class_name = content[name_node.start_byte:name_node.end_byte]
                    classes.append((class_name, def_node))
                
                if classes:  # If we found classes with queries, return them
                    return classes
                    
            except Exception as e:
                logger.warning(f"Tree-sitter query failed for Java classes: {e}")
        
        # Regex fallback
        class_pattern = r'(?:public\s+|private\s+|protected\s+)?(?:abstract\s+|final\s+)?class\s+(\w+)'
        for match in re.finditer(class_pattern, content):
            class_name = match.group(1)
            start_pos = match.start()
            end_pos = ImprovedLanguageSupport._find_java_block_end(content, start_pos)
            
            pseudo_node = type('Node', (), {
                'start_byte': start_pos,
                'end_byte': end_pos,
                'start_point': (content[:start_pos].count('\n'), 0),
                'end_point': (content[:end_pos].count('\n'), 0),
                'type': 'class_declaration'
            })()
            
            classes.append((class_name, pseudo_node))
        
        return classes
    
    @staticmethod
    def _extract_java_interfaces(tree: Tree, content: str) -> List[Tuple[str, Node]]:
        """Extract Java interface definitions"""
        interfaces = []
        
        interface_pattern = r'(?:public\s+|private\s+|protected\s+)?interface\s+(\w+)'
        for match in re.finditer(interface_pattern, content):
            interface_name = match.group(1)
            start_pos = match.start()
            end_pos = ImprovedLanguageSupport._find_java_block_end(content, start_pos)
            
            pseudo_node = type('Node', (), {
                'start_byte': start_pos,
                'end_byte': end_pos,
                'start_point': (content[:start_pos].count('\n'), 0),
                'end_point': (content[:end_pos].count('\n'), 0),
                'type': 'interface_declaration'
            })()
            
            interfaces.append((interface_name, pseudo_node))
        
        return interfaces
    
    @staticmethod
    def _extract_java_methods(tree: Tree, content: str) -> List[Tuple[str, Node]]:
        """Extract Java method definitions"""
        methods = []
        
        method_pattern = r'(?:public|private|protected)\s+(?:static\s+)?(?:final\s+)?[\w<>\[\]]+\s+(\w+)\s*\('
        for match in re.finditer(method_pattern, content):
            method_name = match.group(1)
            start_pos = match.start()
            end_pos = ImprovedLanguageSupport._find_java_method_end(content, start_pos)
            
            pseudo_node = type('Node', (), {
                'start_byte': start_pos,
                'end_byte': end_pos,
                'start_point': (content[:start_pos].count('\n'), 0),
                'end_point': (content[:end_pos].count('\n'), 0),
                'type': 'method_declaration'
            })()
            
            methods.append((method_name, pseudo_node))
        
        return methods
    
    # Helper methods for Python module extraction
    
    @staticmethod
    def _extract_module_level_python(tree: Tree, content: str) -> str:
        """Extract module-level Python code"""
        lines = content.split('\n')
        module_lines = []
        
        # Extract constants, global variables, and main blocks
        patterns = [
            r'^[A-Z_][A-Z0-9_]*\s*=',  # Constants
            r'^if\s+__name__\s*==\s*["\']__main__["\']',  # Main block
            r'^\w+\s*=\s*[^(]',  # Global assignments (not function calls)
        ]
        
        in_main_block = False
        main_block_indent = 0
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            # Skip empty lines and comments at module level
            if not stripped or stripped.startswith('#'):
                continue
            
            # Check for main block
            if re.match(patterns[1], line):
                in_main_block = True
                main_block_indent = len(line) - len(line.lstrip())
                module_lines.append(line)
                continue
            
            # If in main block, include indented lines
            if in_main_block:
                current_indent = len(line) - len(line.lstrip())
                if current_indent > main_block_indent or not line.strip():
                    module_lines.append(line)
                    continue
                else:
                    in_main_block = False
            
            # Check for constants and global assignments
            if any(re.match(pattern, line) for pattern in patterns[:1] + patterns[2:]):
                module_lines.append(line)
        
        return '\n'.join(module_lines) if module_lines else ""
    
    # Helper methods for Kotlin
    
    @staticmethod
    def _extract_kotlin_package(content: str) -> Optional[str]:
        """Extract Kotlin package declaration"""
        package_match = re.search(r'^package\s+([\w.]+)', content, re.MULTILINE)
        return package_match.group(1) if package_match else None
    
    @staticmethod
    def _extract_kotlin_imports(content: str, tree: Tree) -> List[str]:
        """Extract Kotlin import statements"""
        imports = []
        
        import_pattern = r'^import\s+[\w.]+(?:\.\*)?(?:\s+as\s+\w+)?'
        lines = content.split('\n')
        
        for line in lines:
            if re.match(import_pattern, line.strip()):
                imports.append(line.strip())
        
        return imports
    
    @staticmethod
    def _extract_kotlin_classes(tree: Tree, content: str) -> List[Tuple[str, Node]]:
        """Extract Kotlin class definitions"""
        classes = []
        
        class_patterns = [
            r'(?:public\s+|private\s+|internal\s+)?(?:abstract\s+|final\s+|open\s+)?class\s+(\w+)',
            r'(?:public\s+|private\s+|internal\s+)?data\s+class\s+(\w+)',
            r'(?:public\s+|private\s+|internal\s+)?enum\s+class\s+(\w+)',
            r'(?:public\s+|private\s+|internal\s+)?sealed\s+class\s+(\w+)'
        ]
        
        for pattern in class_patterns:
            for match in re.finditer(pattern, content):
                class_name = match.group(1)
                start_pos = match.start()
                end_pos = ImprovedLanguageSupport._find_kotlin_block_end(content, start_pos)
                
                pseudo_node = type('Node', (), {
                    'start_byte': start_pos,
                    'end_byte': end_pos,
                    'start_point': (content[:start_pos].count('\n'), 0),
                    'end_point': (content[:end_pos].count('\n'), 0),
                    'type': 'class_declaration'
                })()
                
                classes.append((class_name, pseudo_node))
        
        return classes
    
    @staticmethod
    def _extract_kotlin_interfaces(tree: Tree, content: str) -> List[Tuple[str, Node]]:
        """Extract Kotlin interface definitions"""
        interfaces = []
        
        interface_pattern = r'(?:public\s+|private\s+|internal\s+)?interface\s+(\w+)'
        for match in re.finditer(interface_pattern, content):
            interface_name = match.group(1)
            start_pos = match.start()
            end_pos = ImprovedLanguageSupport._find_kotlin_block_end(content, start_pos)
            
            pseudo_node = type('Node', (), {
                'start_byte': start_pos,
                'end_byte': end_pos,
                'start_point': (content[:start_pos].count('\n'), 0),
                'end_point': (content[:end_pos].count('\n'), 0),
                'type': 'interface_declaration'
            })()
            
            interfaces.append((interface_name, pseudo_node))
        
        return interfaces
    
    @staticmethod
    def _extract_kotlin_functions(tree: Tree, content: str) -> List[Tuple[str, Node]]:
        """Extract Kotlin function definitions"""
        functions = []
        
        function_patterns = [
            r'(?:public\s+|private\s+|internal\s+)?fun\s+(\w+)\s*\(',
            r'(?:public\s+|private\s+|internal\s+)?suspend\s+fun\s+(\w+)\s*\(',
            r'(?:public\s+|private\s+|internal\s+)?inline\s+fun\s+(\w+)\s*\('
        ]
        
        for pattern in function_patterns:
            for match in re.finditer(pattern, content):
                func_name = match.group(1)
                start_pos = match.start()
                end_pos = ImprovedLanguageSupport._find_kotlin_function_end(content, start_pos)
                
                pseudo_node = type('Node', (), {
                    'start_byte': start_pos,
                    'end_byte': end_pos,
                    'start_point': (content[:start_pos].count('\n'), 0),
                    'end_point': (content[:end_pos].count('\n'), 0),
                    'type': 'function_declaration'
                })()
                
                functions.append((func_name, pseudo_node))
        
        return functions
    
    @staticmethod
    def _extract_kotlin_objects(tree: Tree, content: str) -> List[Tuple[str, Node]]:
        """Extract Kotlin object declarations"""
        objects = []
        
        object_pattern = r'(?:public\s+|private\s+|internal\s+)?object\s+(\w+)'
        for match in re.finditer(object_pattern, content):
            object_name = match.group(1)
            start_pos = match.start()
            end_pos = ImprovedLanguageSupport._find_kotlin_block_end(content, start_pos)
            
            pseudo_node = type('Node', (), {
                'start_byte': start_pos,
                'end_byte': end_pos,
                'start_point': (content[:start_pos].count('\n'), 0),
                'end_point': (content[:end_pos].count('\n'), 0),
                'type': 'object_declaration'
            })()
            
            objects.append((object_name, pseudo_node))
        
        return objects
    
    @staticmethod
    def _create_kotlin_class_chunk(
        chunker_instance,
        file_path: Path,
        content: str,
        class_node: Node,
        class_name: str,
        imports: List[str],
        package_info: Optional[str]
    ) -> Optional[CodeBaseChunk]:
        """Create Kotlin class chunk with package and imports"""
        class_content = content[class_node.start_byte:class_node.end_byte]
        
        # Build full content with package and imports
        content_parts = []
        
        if package_info:
            content_parts.append(f"package {package_info}")
            content_parts.append("")
        
        if imports:
            content_parts.extend(imports)
            content_parts.append("")
        
        content_parts.append(class_content)
        
        full_content = '\n'.join(content_parts)
        
        chunk_id = chunker_instance._generate_chunk_id(file_path, class_name, "class")
        
        return CodeBaseChunk(
            id=chunk_id,
            file_path=str(file_path),
            content=full_content,
            language="kotlin",
            chunk_type="class_complete",
            start_byte=class_node.start_byte,
            end_byte=class_node.end_byte,
            start_line=class_node.start_point[0] + 1,
            end_line=class_node.end_point[0] + 1
        )
    
    # Helper methods for Scala
    
    @staticmethod
    def _extract_scala_package(content: str) -> Optional[str]:
        """Extract Scala package declaration"""
        package_match = re.search(r'^package\s+([\w.]+)', content, re.MULTILINE)
        return package_match.group(1) if package_match else None
    
    @staticmethod
    def _extract_scala_imports(content: str, tree: Tree) -> List[str]:
        """Extract Scala import statements"""
        imports = []
        
        import_pattern = r'^import\s+[\w.{}=>,\s_]+'
        lines = content.split('\n')
        
        for line in lines:
            if re.match(import_pattern, line.strip()):
                imports.append(line.strip())
        
        return imports
    
    @staticmethod
    def _extract_scala_classes(tree: Tree, content: str) -> List[Tuple[str, Node]]:
        """Extract Scala class definitions"""
        classes = []
        
        class_patterns = [
            r'(?:private\s+|protected\s+)?(?:abstract\s+|final\s+)?class\s+(\w+)',
            r'(?:private\s+|protected\s+)?case\s+class\s+(\w+)'
        ]
        
        for pattern in class_patterns:
            for match in re.finditer(pattern, content):
                class_name = match.group(1)
                start_pos = match.start()
                end_pos = ImprovedLanguageSupport._find_scala_block_end(content, start_pos)
                
                pseudo_node = type('Node', (), {
                    'start_byte': start_pos,
                    'end_byte': end_pos,
                    'start_point': (content[:start_pos].count('\n'), 0),
                    'end_point': (content[:end_pos].count('\n'), 0),
                    'type': 'class_declaration'
                })()
                
                classes.append((class_name, pseudo_node))
        
        return classes
    
    @staticmethod
    def _extract_scala_traits(tree: Tree, content: str) -> List[Tuple[str, Node]]:
        """Extract Scala trait definitions"""
        traits = []
        
        trait_pattern = r'(?:private\s+|protected\s+)?trait\s+(\w+)'
        for match in re.finditer(trait_pattern, content):
            trait_name = match.group(1)
            start_pos = match.start()
            end_pos = ImprovedLanguageSupport._find_scala_block_end(content, start_pos)
            
            pseudo_node = type('Node', (), {
                'start_byte': start_pos,
                'end_byte': end_pos,
                'start_point': (content[:start_pos].count('\n'), 0),
                'end_point': (content[:end_pos].count('\n'), 0),
                'type': 'trait_declaration'
            })()
            
            traits.append((trait_name, pseudo_node))
        
        return traits
    
    @staticmethod
    def _extract_scala_objects(tree: Tree, content: str) -> List[Tuple[str, Node]]:
        """Extract Scala object definitions"""
        objects = []
        
        object_pattern = r'(?:private\s+|protected\s+)?object\s+(\w+)'
        for match in re.finditer(object_pattern, content):
            object_name = match.group(1)
            start_pos = match.start()
            end_pos = ImprovedLanguageSupport._find_scala_block_end(content, start_pos)
            
            pseudo_node = type('Node', (), {
                'start_byte': start_pos,
                'end_byte': end_pos,
                'start_point': (content[:start_pos].count('\n'), 0),
                'end_point': (content[:end_pos].count('\n'), 0),
                'type': 'object_declaration'
            })()
            
            objects.append((object_name, pseudo_node))
        
        return objects
    
    @staticmethod
    def _extract_scala_functions(tree: Tree, content: str) -> List[Tuple[str, Node]]:
        """Extract Scala function definitions"""
        functions = []
        
        function_patterns = [
            r'(?:private\s+|protected\s+)?def\s+(\w+)\s*\(',
            r'(?:private\s+|protected\s+)?def\s+(\w+)\s*:'
        ]
        
        for pattern in function_patterns:
            for match in re.finditer(pattern, content):
                func_name = match.group(1)
                start_pos = match.start()
                end_pos = ImprovedLanguageSupport._find_scala_function_end(content, start_pos)
                
                pseudo_node = type('Node', (), {
                    'start_byte': start_pos,
                    'end_byte': end_pos,
                    'start_point': (content[:start_pos].count('\n'), 0),
                    'end_point': (content[:end_pos].count('\n'), 0),
                    'type': 'function_declaration'
                })()
                
                functions.append((func_name, pseudo_node))
        
        return functions
    
    @staticmethod
    def _create_scala_class_chunk(
        chunker_instance,
        file_path: Path,
        content: str,
        class_node: Node,
        class_name: str,
        imports: List[str],
        package_info: Optional[str]
    ) -> Optional[CodeBaseChunk]:
        """Create Scala class chunk with package and imports"""
        class_content = content[class_node.start_byte:class_node.end_byte]
        
        # Build full content with package and imports
        content_parts = []
        
        if package_info:
            content_parts.append(f"package {package_info}")
            content_parts.append("")
        
        if imports:
            content_parts.extend(imports)
            content_parts.append("")
        
        content_parts.append(class_content)
        
        full_content = '\n'.join(content_parts)
        
        chunk_id = chunker_instance._generate_chunk_id(file_path, class_name, "class")
        
        return CodeBaseChunk(
            id=chunk_id,
            file_path=str(file_path),
            content=full_content,
            language="scala",
            chunk_type="class_complete",
            start_byte=class_node.start_byte,
            end_byte=class_node.end_byte,
            start_line=class_node.start_point[0] + 1,
            end_line=class_node.end_point[0] + 1
        )
    
    # Common helper methods
    
    @staticmethod
    def _extract_with_query(tree: Tree, content: str, chunker_instance, ext: str, query_name: str, 
                           name_capture: str, def_capture: str) -> List[Tuple[str, Node]]:
        """Generic method to extract elements using tree-sitter queries"""
        results = []
        
        if (chunker_instance and hasattr(chunker_instance, 'queries') and 
            ext in chunker_instance.queries and query_name in chunker_instance.queries[ext]):
            try:
                query = chunker_instance.queries[ext][query_name]
                from tree_sitter import QueryCursor
                cursor = QueryCursor()
                captures = cursor.captures(tree.root_node, query)
                
                names = []
                definitions = []
                
                for capture in captures:
                    node, capture_name = capture
                    if capture_name == name_capture:
                        names.append(node)
                    elif capture_name == def_capture:
                        definitions.append(node)
                
                # Match names with definitions
                for name_node, def_node in zip(names, definitions):
                    element_name = content[name_node.start_byte:name_node.end_byte]
                    results.append((element_name, def_node))
                    
            except Exception as e:
                logger.warning(f"Tree-sitter query failed for {ext} {query_name}: {e}")
        
        return results
    
    @staticmethod
    def _find_java_block_end(content: str, start_pos: int) -> int:
        """Find the end of a Java block (class/interface)"""
        brace_start = content.find('{', start_pos)
        if brace_start == -1:
            return len(content)
        
        brace_count = 0
        for i, char in enumerate(content[brace_start:], brace_start):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    return i + 1
        
        return len(content)
    
    @staticmethod
    def _find_java_method_end(content: str, start_pos: int) -> int:
        """Find the end of a Java method"""
        # Look for either { } block or ; (abstract method)
        brace_start = content.find('{', start_pos)
        semicolon = content.find(';', start_pos)
        
        if brace_start != -1 and (semicolon == -1 or brace_start < semicolon):
            return ImprovedLanguageSupport._find_java_block_end(content, start_pos)
        elif semicolon != -1:
            return semicolon + 1
        
        return len(content)
    
    @staticmethod
    def _get_class_name_from_node(node: Node, content: str) -> Optional[str]:
        """Extract class name from tree-sitter node"""
        try:
            for child in node.children:
                if child.type == 'identifier':
                    return content[child.start_byte:child.end_byte]
        except:
            pass
        return None
    
    @staticmethod
    def _filter_standalone_functions(functions: List[Tuple[str, Node]], classes: List[Tuple[str, Node]]) -> List[Tuple[str, Node]]:
        """Filter out functions that are inside classes"""
        standalone = []
        
        class_ranges = [(node.start_byte, node.end_byte) for _, node in classes]
        
        for func_name, func_node in functions:
            is_standalone = True
            for start, end in class_ranges:
                if start <= func_node.start_byte < end:
                    is_standalone = False
                    break
            if is_standalone:
                standalone.append((func_name, func_node))
        
        return standalone
    
    @staticmethod
    def _filter_standalone_methods(methods: List[Tuple[str, Node]], containers: List[Tuple[str, Node]]) -> List[Tuple[str, Node]]:
        """Filter out methods that are inside classes/interfaces"""
        return ImprovedLanguageSupport._filter_standalone_functions(methods, containers)
    
    @staticmethod
    def _get_relevant_js_imports(node: Node, imports: List[str], symbols: Set[str], content: str) -> str:
        """Get relevant imports for JavaScript code"""
        if not imports:
            return ""
        
        node_content = content[node.start_byte:node.end_byte].lower()
        relevant_imports = []
        
        for import_stmt in imports:
            # Check if any symbol from this import is used
            for symbol in symbols:
                if symbol.lower() in node_content:
                    relevant_imports.append(import_stmt)
                    break
        
        return '\n'.join(relevant_imports)
    
    @staticmethod
    def _create_chunk_from_node(
        chunker_instance, 
        file_path: Path, 
        content: str, 
        node: Node, 
        symbol_name: str, 
        imports: str
    ) -> Optional[CodeBaseChunk]:
        """Create a chunk from a tree-sitter node"""
        node_content = content[node.start_byte:node.end_byte]
        
        # Add context lines if configured
        if chunker_instance.config.include_context_lines > 0:
            lines = content.split('\n')
            start_line = node.start_point[0]
            end_line = node.end_point[0]
            
            context_start = max(0, start_line - chunker_instance.config.include_context_lines)
            context_end = min(len(lines), end_line + 1 + chunker_instance.config.include_context_lines)
            
            context_lines = lines[context_start:context_end]
            node_content = '\n'.join(context_lines)
        
        full_content = f"{imports}\n\n{node_content}" if imports else node_content
        
        chunk_id = chunker_instance._generate_chunk_id(file_path, symbol_name, node.type)
        
        return CodeBaseChunk(
            id=chunk_id,
            file_path=str(file_path),
            content=full_content.strip(),
            language=file_path.suffix.lower().lstrip('.'),
            chunk_type=node.type,
            start_byte=node.start_byte,
            end_byte=node.end_byte,
            start_line=node.start_point[0] + 1,
            end_line=node.end_point[0] + 1
        )
    
    @staticmethod
    def _create_js_class_chunk(
        chunker_instance, 
        file_path: Path, 
        content: str, 
        class_node: Node,
        class_name: str, 
        imports: List[str], 
        import_symbols: Set[str]
    ) -> Optional[CodeBaseChunk]:
        """Create JavaScript class chunk with methods"""
        class_content = content[class_node.start_byte:class_node.end_byte]
        
        # Check token count
        token_count = len(chunker_instance.tokenizer.encode(class_content))
        if token_count > chunker_instance.config.max_class_chunk_size:
            logger.warning(f"JS Class {class_name} has {token_count} tokens, exceeding max")
        
        # Get relevant imports
        relevant_imports = ImprovedLanguageSupport._get_relevant_js_imports(
            class_node, imports, import_symbols, content
        )
        
        full_content = f"{relevant_imports}\n\n{class_content}" if relevant_imports else class_content
        
        chunk_id = chunker_instance._generate_chunk_id(file_path, class_name, "class")
        
        return CodeBaseChunk(
            id=chunk_id,
            file_path=str(file_path),
            content=full_content.strip(),
            language=file_path.suffix.lower().lstrip('.'),
            chunk_type="class_complete",
            start_byte=class_node.start_byte,
            end_byte=class_node.end_byte,
            start_line=class_node.start_point[0] + 1,
            end_line=class_node.end_point[0] + 1
        )
    
    @staticmethod
    def _create_java_class_chunk(
        chunker_instance,
        file_path: Path,
        content: str,
        class_node: Node,
        class_name: str,
        imports: List[str],
        package_info: Optional[str]
    ) -> Optional[CodeBaseChunk]:
        """Create Java class chunk with package and imports"""
        class_content = content[class_node.start_byte:class_node.end_byte]
        
        # Build full content with package and imports
        content_parts = []
        
        if package_info:
            content_parts.append(f"package {package_info};")
            content_parts.append("")
        
        if imports:
            content_parts.extend(imports)
            content_parts.append("")
        
        content_parts.append(class_content)
        
        full_content = '\n'.join(content_parts)
        
        chunk_id = chunker_instance._generate_chunk_id(file_path, class_name, "class")
        
        return CodeBaseChunk(
            id=chunk_id,
            file_path=str(file_path),
            content=full_content,
            language="java",
            chunk_type="class_complete",
            start_byte=class_node.start_byte,
            end_byte=class_node.end_byte,
            start_line=class_node.start_point[0] + 1,
            end_line=class_node.end_point[0] + 1
        )
    
    @staticmethod
    def _create_js_module_chunk(
        chunker_instance,
        file_path: Path,
        content: str,
        tree: Tree,
        imports: List[str]
    ) -> Optional[CodeBaseChunk]:
        """Create JavaScript module-level chunk"""
        lines = content.split('\n')
        module_lines = []
        
        # Extract module-level exports, constants, and other top-level code
        for line in lines:
            stripped = line.strip()
            if (
                stripped.startswith('export ') or
                stripped.startswith('const ') and '=' in stripped or
                stripped.startswith('let ') and '=' in stripped or
                stripped.startswith('var ') and '=' in stripped or
                re.match(r'^[A-Z_][A-Z0-9_]*\s*=', line)  # Constants
            ):
                module_lines.append(line)
        
        if not module_lines:
            return None
        
        module_content = '\n'.join(module_lines)
        token_count = len(chunker_instance.tokenizer.encode(module_content))
        
        if token_count < 30:  # Skip if too small
            return None
        
        # Include imports
        if imports:
            full_content = '\n'.join(imports) + '\n\n' + module_content
        else:
            full_content = module_content
        
        chunk_id = chunker_instance._generate_chunk_id(file_path, "module", "module")
        
        return CodeBaseChunk(
            id=chunk_id,
            file_path=str(file_path),
            content=full_content,
            language=file_path.suffix.lower().lstrip('.'),
            chunk_type="module",
            start_byte=0,
            end_byte=len(content.encode('utf-8')),
            start_line=1,
            end_line=len(lines)
        )
    
    @staticmethod
    def _find_kotlin_block_end(content: str, start_pos: int) -> int:
        """Find the end of a Kotlin block (class/interface/object)"""
        brace_start = content.find('{', start_pos)
        if brace_start == -1:
            return len(content)
        
        brace_count = 0
        for i, char in enumerate(content[brace_start:], brace_start):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    return i + 1
        
        return len(content)
    
    @staticmethod
    def _find_kotlin_function_end(content: str, start_pos: int) -> int:
        """Find the end of a Kotlin function"""
        # Look for either { } block or = assignment
        brace_start = content.find('{', start_pos)
        equals_pos = content.find('=', start_pos)
        
        if brace_start != -1 and (equals_pos == -1 or brace_start < equals_pos):
            return ImprovedLanguageSupport._find_kotlin_block_end(content, start_pos)
        elif equals_pos != -1:
            # Single expression function, find end of line or next function
            line_end = content.find('\n', equals_pos)
            return line_end + 1 if line_end != -1 else len(content)
        
        return len(content)
    
    @staticmethod
    def _find_scala_block_end(content: str, start_pos: int) -> int:
        """Find the end of a Scala block (class/trait/object)"""
        brace_start = content.find('{', start_pos)
        if brace_start == -1:
            return len(content)
        
        brace_count = 0
        for i, char in enumerate(content[brace_start:], brace_start):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    return i + 1
        
        return len(content)
    
    @staticmethod
    def _find_scala_function_end(content: str, start_pos: int) -> int:
        """Find the end of a Scala function"""
        # Look for either { } block or = assignment
        brace_start = content.find('{', start_pos)
        equals_pos = content.find('=', start_pos)
        
        if brace_start != -1 and (equals_pos == -1 or brace_start < equals_pos):
            return ImprovedLanguageSupport._find_scala_block_end(content, start_pos)
        elif equals_pos != -1:
            # Function with body expression
            line_end = content.find('\n', equals_pos)
            return line_end + 1 if line_end != -1 else len(content)
        
        return len(content)

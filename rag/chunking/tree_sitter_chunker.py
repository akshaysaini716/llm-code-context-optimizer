import logging
import tiktoken
import mmap
from pathlib import Path
from typing import List, Tuple, Optional
from rag.models import CodeBaseChunk
from tree_sitter_language_pack import get_language, get_parser
from tree_sitter import Parser, Node, Tree, Query
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

class TreeSitterChunker:
    # TODO: Insert configs here
    def __init__(self):
        self.parsers = {}
        self.languages = {}
        self.queries = {}

        self._initialize_parsers()
        self._initialize_queries()


    def _initialize_parsers(self):
        language_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.jsx': 'javascript',
            '.ts': 'typescript',
            '.tsx': 'typescript',
            '.java': 'java',
            '.kt': 'kotlin',
            '.kts': 'kotlin',
            '.go': 'go'
        }
        for ext, lang_name in language_map.items():
            try:
                language = get_language(lang_name) # type: ignore
                parser = get_parser(lang_name) # type: ignore
                self.languages[ext] = language
                self.parsers[ext] = parser
                logger.info(f"Initialized {ext} parser for {lang_name}")
            except Exception as e:
                logger.warning(f"Failed to initialize parser for {lang_name}: {e}")

    def _initialize_queries(self):
        if '.py' in self.languages:
            python_queries = {
                'functions': '''
                                (function_definition 
                                    name: (identifier) @func-name) @func-def
                            ''',
                'classes': '''
                                (class_definition 
                                    name: (identifier) @class-name) @class-def
                            ''',
                'imports': '''
                                [
                                    (import_statement) @import
                                    (import_from_statement) @import
                                ]
                            ''',
                'methods': '''
                                (class_definition 
                                    body: (block
                                        (function_definition
                                            name: (identifier) @method-name) @method-def))
                            '''
            }
            self.queries['.py'] = {}
            for name, query_text in python_queries.items():
                try:
                    self.queries['.py'][name] = Query(self.languages['.py'], query_text)
                except Exception as e:
                    logger.warning(f"Failed to initialize query for {name}: {e}")


    def chunk_file(self, file_path: Path) -> List[CodeBaseChunk]:
        try:
            if file_path.stat().st_size > 10 * 1024 * 1024:  # 10 MB
                chunks = self._chunk_large_file(file_path)
            else:
                chunks = self._chunk_regular_file(file_path)

            # TODO: Update the metrics here
            return chunks
        except Exception as e:
            logger.error(f"Error chunking file {file_path}: {e}")
            return []

    def _chunk_large_file(self, file_path: Path) -> List[CodeBaseChunk]:
        try:
            with open(file_path, 'rb') as file:
                with mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                    content = mm.read().decode('utf-8', errors='ignore')
                    return self._parse_and_chunk(file_path, content)
        except Exception as e:
            logger.error(f"Error chunking large file {file_path}: {e}")
            return []

    def _chunk_regular_file(self, file_path: Path) -> List[CodeBaseChunk]:
        try:
            content = file_path.read_text(encoding='utf-8', errors='ignore')
            return self._parse_and_chunk(file_path, content)
        except Exception as e:
            logger.error(f"Error chunking regular file {file_path}: {e}")
            return []

    def _parse_and_chunk(self, file_path: Path, content: str) -> List[CodeBaseChunk]:
        # main parsing and chunking logic goes here
        ext = file_path.suffix.lower()
        if ext not in self.parsers:
            return []

        try:
            parser = self.parsers[ext]
            tree = parser.parse(content.encode('utf-8'))
            if ext == '.py':
                return self._chunk_python(file_path, content, tree)
            elif ext in ['.js', '.jsx', '.ts', '.tsx']:
                pass
            elif ext == '.java':
                pass
            elif ext in ['.kt', '.kts']:
                pass
            return [] # TODO: check the implementation
        except Exception as e:
            logger.error(f"Tree-sitter parsing and chunking failed for file {file_path}: {e}")
            return []

    def _chunk_python(self, file_path: Path, content: str, tree: Tree) -> List[CodeBaseChunk]:
        chunks = []
        imports = self._extract_imports_with_query(tree, content, '.py')
        imports_text = '\n'.join(imports)
        functions = self._extract_functions_with_query(tree, content, '.py')
        classes = self._extract_classes_with_query(tree, content, '.py')

        # create chunks
        all_nodes = functions + classes
        all_nodes.sort(key=lambda node: node[1].start_byte)
        for symbol_name, node in all_nodes:
            chunk = self._create_chunk(
                file_path, content, node, symbol_name, imports_text, tree
            )
            if chunk and self._validate_chunk_quality(chunk):
                chunks.append(chunk)

        # add imports chunk if required
        if imports_text and len(imports) > 2:
            imports_chunk = self._create_import_chunk(file_path, content, imports_text)
            if imports_chunk:
                chunks.append(imports_chunk)

        return chunks

    def _extract_imports_with_query(self, tree: Tree, content: str, ext: str) -> List[str]:
        imports = []
        if ext in self.queries and 'imports' in self.queries[ext]:
            query = self.queries[ext]['imports']
            captures = query.captures(tree.root_node)
            if 'import' in captures:
                for node in captures['import']:
                    import_text = content[node.start_byte:node.end_byte]
                    imports.append(import_text)
        return imports


    def _extract_functions_with_query(self, tree: Tree, content: str, ext: str) -> List[Tuple[str, Node]]:
        functions = []
        if ext in self.queries and 'functions' in self.queries[ext]:
            query = self.queries[ext]['functions']
            captures = query.captures(tree.root_node)

            func_names = captures.get('func-name', [])
            func_defs = captures.get('func-def', [])

            for node_name, def_node in zip(func_names, func_defs):
                func_name = content[node_name.start_byte:node_name.end_byte]
                functions.append((func_name, def_node))

        return functions

    def _extract_classes_with_query(self, tree: Tree, content: str, ext: str) -> List[Tuple[str, Node]]:
        classes = []
        if ext in self.queries and 'classes' in self.queries[ext]:
            query = self.queries[ext]['classes']
            captures = query.captures(tree.root_node)

            class_names = captures.get('class-name', [])
            class_defs = captures.get('class-def', [])

            for node_name, def_node in zip(class_names, class_defs):
                class_name = content[node_name.start_byte:node_name.end_byte]
                classes.append((class_name, def_node))

        return classes

    def _create_chunk(self, file_path: Path, content: str, node: Node, symbol_name: str, imports_text: str, tree: Tree) -> Optional[CodeBaseChunk]:
        node_content = content[node.start_byte:node.end_byte]
        need_imports = self._analyze_import_dependencies(node_content, imports_text)
        full_content = f"{imports_text}\n\n{node_content}" if need_imports else node_content

        # TODO: add symbols and metrics here
        return CodeBaseChunk(
            id="",
            file_path=str(file_path),
            content=full_content.strip(),
            language=file_path.suffix.lower().lstrip('.'),
            chunk_type=node.type,
            start_byte=node.start_byte,
            end_byte=node.end_byte,
            start_line=node.start_point[0]+1,
            end_line=node.end_point[0]+1
        )


    def _analyze_import_dependencies(self, content: str, imports_text: str) -> bool:
        # TODO: implement
        return True

    def _validate_chunk_quality(self, chunk: CodeBaseChunk) -> bool:
        # TODO: implement
        return True

    def _create_import_chunk(self, file_path: Path, content: str, imports_text: str) -> Optional[CodeBaseChunk]:
        return CodeBaseChunk(
            id="",
            file_path=str(file_path),
            content=imports_text,
            language=file_path.suffix.lower().lstrip('.'),
            chunk_type="imports",
            start_byte=0,
            end_byte=len(imports_text.encode()),
            start_line=1,
            end_line=len(imports_text.split('\n'))
        )

    def chunk_files_parallel(self, file_paths: List[Path]) -> List[CodeBaseChunk]:
        if len(file_paths) < 2:
            chunks = []
            for path in file_paths:
                chunks.extend(self.chunk_file(path))
            return chunks

        all_chunks = []
        # TODO: implement max workers as config
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_path = {executor.submit(self.chunk_file, path): path for path in file_paths}
            for future in as_completed(future_to_path):
                path = future_to_path[future]
                try:
                    chunks = future.result()
                    all_chunks.extend(chunks)
                except Exception as e:
                    logger.error(f"Error chunking file {path}: {e}")

        return all_chunks

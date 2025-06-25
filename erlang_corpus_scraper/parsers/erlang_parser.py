#!/usr/bin/env python3
"""
Tree-sitter Erlang parser setup for Erlang corpus scraper.
Handles Erlang AST parsing and basic tree navigation.
"""

import os
import sys
import subprocess
import shutil
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from tree_sitter import Language, Parser, Node

# Setup logging
logger = logging.getLogger(__name__)

class ErlangParser:
    """Tree-sitter based Erlang parser."""
    
    def __init__(self, parser_dir: str = "parsers"):
        """Initialize Erlang parser.
        
        Args:
            parser_dir: Directory to store parser files
        """
        self.parser_dir = Path(parser_dir)
        self.parser_dir.mkdir(exist_ok=True)
        
        self.language = None
        self.parser = None
        self._setup_parser()
    
    def _setup_parser(self):
        """Set up tree-sitter Erlang parser."""
        try:
            # Check if language library exists
            lib_path = self.parser_dir / "erlang.so"
            
            if not lib_path.exists():
                logger.info("Building Erlang tree-sitter library...")
                self._build_erlang_library()
            
            # Load language and create parser
            self._load_language(lib_path)
            
            logger.info("✓ Erlang parser initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Erlang parser: {e}")
            raise
    
    def _load_language(self, lib_path: Path):
        """Load the language using the appropriate API."""
        import ctypes
        
        # Try different methods to load the language
        methods = [
            self._try_new_api_with_ctypes,
            self._try_old_api_two_args,
            self._try_old_api_with_set_language
        ]
        
        for i, method in enumerate(methods, 1):
            try:
                method(lib_path)
                logger.info(f"✓ Successfully loaded language using method {i}")
                return
            except Exception as e:
                logger.debug(f"Method {i} failed: {e}")
                continue
        
        raise Exception("All language loading methods failed")
    
    def _try_new_api_with_ctypes(self, lib_path: Path):
        """Try new API with ctypes loading."""
        import ctypes
        
        # Load the shared library
        lib = ctypes.cdll.LoadLibrary(str(lib_path))
        
        # Debug: List all available symbols
        self._debug_library_symbols(lib_path)
        
        # Try common function names for tree-sitter languages
        function_names = ['tree_sitter_erlang', 'tree_sitter_erl', 'language']
        
        for func_name in function_names:
            try:
                logger.info(f"Trying to get function: {func_name}")
                
                # Method 1: Direct getattr
                try:
                    language_func = getattr(lib, func_name)
                    logger.info(f"✓ Found function {func_name} with getattr")
                except AttributeError:
                    logger.info(f"✗ getattr failed for {func_name}")
                    
                    # Method 2: Try with ctypes.CDLL explicit loading
                    try:
                        lib2 = ctypes.CDLL(str(lib_path))
                        language_func = getattr(lib2, func_name)
                        logger.info(f"✓ Found function {func_name} with CDLL")
                        lib = lib2  # Use this library instance
                    except AttributeError:
                        logger.info(f"✗ CDLL also failed for {func_name}")
                        
                        # Method 3: Try manual symbol lookup
                        try:
                            language_func = lib[func_name]
                            logger.info(f"✓ Found function {func_name} with [] notation")
                        except (KeyError, AttributeError):
                            logger.info(f"✗ Manual lookup failed for {func_name}")
                            continue
                
                # Configure function signature
                language_func.restype = ctypes.c_void_p
                language_func.argtypes = []
                
                # Test calling the function
                try:
                    result = language_func()
                    logger.info(f"✓ Successfully called {func_name}(), result: {result}")
                    
                    if result is None or result == 0:
                        logger.warning(f"Function {func_name}() returned null/zero")
                        continue
                        
                except Exception as e:
                    logger.error(f"✗ Failed to call {func_name}(): {e}")
                    continue
                
                # Try to create Language object
                try:
                    self.language = Language(result)
                    logger.info(f"✓ Successfully created Language object")
                except Exception as e:
                    logger.error(f"✗ Failed to create Language object: {e}")
                    continue
                
                # Try to create Parser object
                try:
                    # Check Parser constructor signature
                    import inspect
                    parser_sig = inspect.signature(Parser.__init__)
                    logger.info(f"Parser.__init__ signature: {parser_sig}")
                    
                    if len(parser_sig.parameters) > 1:  # More than just 'self'
                        self.parser = Parser(self.language)
                        logger.info(f"✓ Created Parser with language argument")
                    else:
                        self.parser = Parser()
                        if hasattr(self.parser, 'set_language'):
                            self.parser.set_language(self.language)
                            logger.info(f"✓ Created Parser and set language separately")
                        else:
                            raise Exception("Parser has no set_language method")
                            
                except Exception as e:
                    logger.error(f"✗ Failed to create Parser: {e}")
                    continue
                
                return  # Success!
                
            except Exception as e:
                logger.error(f"✗ Overall failure for {func_name}: {e}")
                continue
        
        raise Exception(f"No valid language function found in {lib_path}")
    
    def _debug_library_symbols(self, lib_path: Path):
        """Debug: Try to list symbols in the library."""
        try:
            # Try using objdump if available
            result = subprocess.run([
                'objdump', '-T', str(lib_path)
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                logger.info("Available symbols in library:")
                lines = result.stdout.split('\n')
                for line in lines:
                    if 'tree_sitter' in line.lower() or 'erlang' in line.lower():
                        logger.info(f"  {line.strip()}")
            else:
                logger.info("objdump failed, trying nm...")
                
                # Try nm as alternative
                result = subprocess.run([
                    'nm', '-D', str(lib_path)
                ], capture_output=True, text=True, timeout=10)
                
                if result.returncode == 0:
                    lines = result.stdout.split('\n')
                    for line in lines:
                        if 'tree_sitter' in line.lower() or 'erlang' in line.lower():
                            logger.info(f"  {line.strip()}")
                            
        except Exception as e:
            logger.info(f"Could not debug symbols: {e}")
        
        # Also try to list what ctypes can see
        try:
            import ctypes
            lib = ctypes.cdll.LoadLibrary(str(lib_path))
            logger.info(f"Library loaded successfully: {lib}")
            
            # Try a few common symbols that should exist
            test_symbols = ['tree_sitter_erlang', '_tree_sitter_erlang', 'tree_sitter_erl']
            for symbol in test_symbols:
                try:
                    func = getattr(lib, symbol)
                    logger.info(f"✓ ctypes can access: {symbol}")
                except AttributeError:
                    logger.info(f"✗ ctypes cannot access: {symbol}")
                    
        except Exception as e:
            logger.info(f"ctypes debug failed: {e}")
    
    def _try_old_api_two_args(self, lib_path: Path):
        """Try old API with two arguments."""
        self.language = Language(str(lib_path), 'erlang')
        self.parser = Parser()
        self.parser.set_language(self.language)
    
    def _try_old_api_with_set_language(self, lib_path: Path):
        """Try another variant of old API."""
        self.language = Language(str(lib_path))
        self.parser = Parser()
        self.parser.set_language(self.language)
    
    def _build_erlang_library(self):
        """Build the tree-sitter Erlang language library."""
        # Clone tree-sitter-erlang if needed
        erlang_repo_path = self.parser_dir / "tree-sitter-erlang"
        
        if not erlang_repo_path.exists():
            logger.info("Cloning tree-sitter-erlang repository...")
            try:
                subprocess.run([
                    "git", "clone", 
                    "https://github.com/WhatsApp/tree-sitter-erlang.git",
                    str(erlang_repo_path)
                ], check=True, capture_output=True, text=True)
                logger.info("✓ Successfully cloned tree-sitter-erlang")
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to clone tree-sitter-erlang: {e}")
                raise
        
        # Build language library using tree-sitter CLI
        lib_path = self.parser_dir / "erlang.so"
        try:
            # Method 1: Try using tree-sitter CLI if available
            try:
                subprocess.run([
                    "tree-sitter", "build", 
                    "--output", str(lib_path),
                    str(erlang_repo_path)
                ], check=True, capture_output=True, text=True, cwd=str(erlang_repo_path))
                logger.info("✓ Successfully built Erlang language library using tree-sitter CLI")
                return
            except (subprocess.CalledProcessError, FileNotFoundError):
                logger.info("tree-sitter CLI not available, trying alternative method...")
            
            # Method 2: Try the old build_library method (for older py-tree-sitter versions)
            try:
                if hasattr(Language, 'build_library'):
                    Language.build_library(
                        str(lib_path),
                        [str(erlang_repo_path)]
                    )
                    logger.info("✓ Successfully built Erlang language library using build_library")
                    return
            except Exception as e:
                logger.info(f"build_library method failed: {e}")
            
            # Method 3: Manual compilation using subprocess
            logger.info("Attempting manual compilation...")
            self._manual_compile_erlang(erlang_repo_path, lib_path)
            
        except Exception as e:
            logger.error(f"Failed to build Erlang language library: {e}")
            raise
    
    def _manual_compile_erlang(self, repo_path: Path, output_path: Path):
        """Manually compile tree-sitter-erlang using gcc/clang."""
        import platform
        
        # Determine compiler and flags
        if platform.system() == "Windows":
            # For Windows, we'd need different approach, but let's focus on Unix first
            raise Exception("Windows compilation not supported yet")
        
        # Unix/Linux/MacOS compilation
        c_files = list(repo_path.glob("src/*.c"))
        if not c_files:
            raise Exception(f"No C source files found in {repo_path}/src/")
        
        # Basic compilation command
        cmd = [
            "gcc" if shutil.which("gcc") else "clang",
            "-shared", "-fPIC", "-O2",
            "-I", str(repo_path / "src"),
            "-o", str(output_path)
        ] + [str(f) for f in c_files]
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info("✓ Successfully compiled Erlang language library manually")
        except subprocess.CalledProcessError as e:
            logger.error(f"Manual compilation failed: {e}")
            logger.error(f"Stdout: {e.stdout}")
            logger.error(f"Stderr: {e.stderr}")
            raise
    
    def parse_string(self, code: str) -> Optional[Node]:
        """Parse Erlang code string into AST.
        
        Args:
            code: Erlang source code string
            
        Returns:
            Root node of the AST, or None if parsing failed
        """
        if not self.parser:
            logger.error("Parser not initialized")
            return None
        
        try:
            tree = self.parser.parse(bytes(code, "utf8"))
            return tree.root_node
        except Exception as e:
            logger.error(f"Failed to parse Erlang code: {e}")
            return None
    
    def parse_file(self, file_path: str) -> Optional[Node]:
        """Parse Erlang file into AST.
        
        Args:
            file_path: Path to Erlang source file
            
        Returns:
            Root node of the AST, or None if parsing failed
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
            return self.parse_string(code)
        except Exception as e:
            logger.error(f"Failed to read/parse file {file_path}: {e}")
            return None
    
    def extract_functions(self, root_node: Node) -> List[Node]:
        """Extract function definition nodes from AST.
        
        Args:
            root_node: Root node of the AST
            
        Returns:
            List of function definition nodes
        """
        functions = []
        
        def visit_node(node: Node):
            # Debug: log what we're seeing
            logger.debug(f"Visiting node: {node.type}")
            
            # In Erlang tree-sitter, function definitions can be various types
            # Common types: 'function_clause', 'function', 'function_definition'
            # But we might also see 'clause' or other patterns
            if node.type in ['function_clause', 'function', 'function_definition', 'clause']:
                logger.debug(f"Found potential function: {node.type}")
                
                # For Erlang, each clause might be separate, but we want to group them
                # For now, let's collect all clauses
                functions.append(node)
            
            # Continue traversing children
            for child in node.children:
                visit_node(child)
        
        visit_node(root_node)
        logger.debug(f"Total function nodes found: {len(functions)}")
        return functions
    
    def get_function_name(self, func_node: Node) -> Optional[str]:
        """Extract function name from function node.
        
        Args:
            func_node: Function definition node
            
        Returns:
            Function name or None if not found
        """
        try:
            # Debug: print node structure
            logger.debug(f"Getting function name from node type: {func_node.type}")
            logger.debug(f"Node children types: {[child.type for child in func_node.children]}")
            
            # Look for function name in various possible locations
            for i, child in enumerate(func_node.children):
                logger.debug(f"Child {i}: type={child.type}, text='{self.node_text(child)}'")
                
                if child.type == 'atom':
                    name = self.node_text(child)
                    logger.debug(f"Found atom: {name}")
                    return name
                elif child.type == 'function_name':
                    name = self.node_text(child)
                    logger.debug(f"Found function_name: {name}")
                    return name
                elif child.type == 'identifier':
                    name = self.node_text(child)
                    logger.debug(f"Found identifier: {name}")
                    return name
            
            # Alternative: look for first atom child
            atoms = [child for child in func_node.children if child.type == 'atom']
            if atoms:
                name = self.node_text(atoms[0])
                logger.debug(f"Found first atom: {name}")
                return name
            
            # Try identifiers
            identifiers = [child for child in func_node.children if child.type == 'identifier']
            if identifiers:
                name = self.node_text(identifiers[0])
                logger.debug(f"Found first identifier: {name}")
                return name
            
            logger.warning(f"Could not find function name in node: {func_node.type}")
            return None
            
        except Exception as e:
            logger.error(f"Failed to extract function name: {e}")
            return None
    
    def get_function_arity(self, func_node: Node) -> int:
        """Extract function arity (number of parameters).
        
        Args:
            func_node: Function definition node
            
        Returns:
            Function arity (number of parameters)
        """
        try:
            # Debug: print the function node structure
            logger.debug(f"Function node type: {func_node.type}")
            logger.debug(f"Function node children: {[child.type for child in func_node.children]}")
            
            # For Erlang, we need to look at the first clause to determine arity
            clauses = self.get_function_clauses(func_node)
            if clauses:
                first_clause = clauses[0]
                logger.debug(f"First clause type: {first_clause.type}")
                logger.debug(f"First clause children: {[child.type for child in first_clause.children]}")
                
                # Look for patterns/parameters in the first clause
                return self._count_parameters_in_clause(first_clause)
            
            # Fallback: try to count patterns directly
            patterns = [child for child in func_node.children if child.type in ['pattern', 'patterns', 'parameters', 'arguments']]
            if patterns:
                # Count parameter children, excluding separators
                params = [c for c in patterns[0].children if c.type not in [',', '(', ')']]
                return len(params)
            
            # Another approach: look for function head pattern
            for child in func_node.children:
                if child.type in ['function_head', 'function_clause']:
                    return self._count_parameters_in_clause(child)
            
            logger.warning(f"Could not determine arity for function node: {func_node.type}")
            return 0
            
        except Exception as e:
            logger.error(f"Failed to extract function arity: {e}")
            return 0
    
    def _count_parameters_in_clause(self, clause_node: Node) -> int:
        """Count parameters in a function clause."""
        try:
            # Look for different patterns in Erlang AST
            # The function head usually looks like: name(param1, param2, ...)
            
            # Method 1: Look for parentheses and count content
            paren_start = None
            paren_end = None
            
            for i, child in enumerate(clause_node.children):
                if self.node_text(child) == '(':
                    paren_start = i
                elif self.node_text(child) == ')':
                    paren_end = i
                    break
            
            if paren_start is not None and paren_end is not None:
                # Count parameters between parentheses
                param_count = 0
                for i in range(paren_start + 1, paren_end):
                    child = clause_node.children[i]
                    # Count non-separator nodes
                    if child.type not in ['comment'] and self.node_text(child) not in [',', ' ', '\n', '\t']:
                        text = self.node_text(child).strip()
                        if text and text != ',':
                            param_count += 1
                
                # Adjust for comma-separated parameters
                # If we have params like "A, B", we counted both A, B and the comma
                # So we need to count more carefully
                param_nodes = []
                for i in range(paren_start + 1, paren_end):
                    child = clause_node.children[i]
                    text = self.node_text(child).strip()
                    # Look for variable-like nodes (identifiers, patterns)
                    if (child.type in ['identifier', 'variable', 'pattern', 'atom'] and 
                        text and text != ',' and not text.isspace()):
                        param_nodes.append(child)
                
                # Alternative: count commas + 1 if there are any non-comma parameters
                comma_count = 0
                has_params = False
                for i in range(paren_start + 1, paren_end):
                    child = clause_node.children[i]
                    text = self.node_text(child).strip()
                    if text == ',':
                        comma_count += 1
                    elif text and not text.isspace():
                        has_params = True
                
                if has_params:
                    return comma_count + 1
                else:
                    return 0
            
            # Method 2: Look for pattern nodes directly
            patterns = [child for child in clause_node.children 
                       if child.type in ['pattern', 'variable', 'identifier'] 
                       and self.node_text(child).strip()]
            
            return len(patterns)
            
        except Exception as e:
            logger.warning(f"Failed to count parameters in clause: {e}")
            return 0
    
    def node_text(self, node: Node) -> str:
        """Get text content of a node.
        
        Args:
            node: AST node
            
        Returns:
            Text content of the node
        """
        return node.text.decode('utf-8') if node.text else ""
    
    def get_function_clauses(self, func_node: Node) -> List[Node]:
        """Get all clauses of a function.
        
        Args:
            func_node: Function definition node
            
        Returns:
            List of function clause nodes
        """
        clauses = []
        
        # If this is already a clause, return it
        if func_node.type == 'function_clause':
            return [func_node]
        
        # Otherwise, find clause children
        for child in func_node.children:
            if child.type == 'function_clause':
                clauses.append(child)
        
        return clauses
    
    def has_guard(self, clause_node: Node) -> bool:
        """Check if a function clause has a guard.
        
        Args:
            clause_node: Function clause node
            
        Returns:
            True if clause has a guard, False otherwise
        """
        for child in clause_node.children:
            if child.type in ['guard', 'when_clause']:
                return True
        return False
    
    def get_guard_node(self, clause_node: Node) -> Optional[Node]:
        """Get the guard node from a function clause.
        
        Args:
            clause_node: Function clause node
            
        Returns:
            Guard node or None if no guard
        """
        for child in clause_node.children:
            if child.type in ['guard', 'when_clause']:
                return child
        return None
    
    def tree_to_variable_index(self, root_node, index_to_code):
        """Extract variable indices from AST (GraphCodeBERT style).
        
        Args:
            root_node: AST node to traverse
            index_to_code: Mapping from position to (token_idx, token_text)
            
        Returns:
            List of position tuples for variables
        """
        if (len(root_node.children) == 0 or root_node.type == 'string') and root_node.type != 'comment':
            index = (root_node.start_point, root_node.end_point)
            if index in index_to_code:
                _, code = index_to_code[index]
                # Check if this is a variable (identifier that's not the node type itself)
                if root_node.type != code and root_node.type == 'identifier':
                    return [index]
            return []
        else:
            code_tokens = []
            for child in root_node.children:
                code_tokens.extend(self.tree_to_variable_index(child, index_to_code))
            return code_tokens
    
    def extract_dataflow_info(self, func_node, file_lines: List[str]):
        """Extract dataflow information for a function (GraphCodeBERT style).
        
        Args:
            func_node: Function AST node
            file_lines: Source file lines
            
        Returns:
            Tuple of (code_tokens, variable_positions, index_to_code_mapping)
        """
        # Get all token positions
        tokens_index = self.tree_to_token_index(func_node)
        
        # Convert positions to tokens
        code_tokens = []
        for pos in tokens_index:
            token = self.index_to_code_token(pos, file_lines)
            code_tokens.append(token)
        
        # Create index_to_code mapping
        index_to_code = {}
        for idx, (index, token) in enumerate(zip(tokens_index, code_tokens)):
            index_to_code[index] = (idx, token)
        
        # Extract variable positions
        variable_positions = self.tree_to_variable_index(func_node, index_to_code)
        
        return code_tokens, variable_positions, index_to_code
    
    def tree_to_token_index(self, node):
        """Extract all token positions from AST (GraphCodeBERT style)."""
        if (len(node.children) == 0 or node.type == 'string') and node.type != 'comment':
            return [(node.start_point, node.end_point)]
        else:
            code_tokens = []
            for child in node.children:
                code_tokens.extend(self.tree_to_token_index(child))
            return code_tokens
    
    def index_to_code_token(self, index, file_lines: List[str]) -> str:
        """Convert position index to actual code token (GraphCodeBERT style)."""
        start_point, end_point = index
        
        if start_point[0] == end_point[0]:
            # Single line token
            line = file_lines[start_point[0]] if start_point[0] < len(file_lines) else ""
            token = line[start_point[1]:end_point[1]]
        else:
            # Multi-line token
            token = ""
            # First line
            if start_point[0] < len(file_lines):
                token += file_lines[start_point[0]][start_point[1]:]
            # Middle lines
            for i in range(start_point[0] + 1, end_point[0]):
                if i < len(file_lines):
                    token += file_lines[i]
            # Last line
            if end_point[0] < len(file_lines):
                token += file_lines[end_point[0]][:end_point[1]]
        
        return token
        """Print AST structure for debugging.
        
        Args:
            node: AST node to print
            indent: Current indentation level
        """
        indent_str = "  " * indent
        node_text = self.node_text(node)
        text_preview = node_text[:50].replace('\n', '\\n') if node_text else ""
        
        print(f"{indent_str}{node.type}: '{text_preview}'")
        
        for child in node.children:
            self.print_ast(child, indent + 1)

    def print_ast(self, node: Node, indent: int = 0) -> None:
        """Print AST structure for debugging.
        
        Args:
            node: AST node to print
            indent: Current indentation level
        """
        indent_str = "  " * indent
        node_text = self.node_text(node)
        text_preview = node_text[:50].replace('\n', '\\n') if node_text else ""
        
        print(f"{indent_str}{node.type}: '{text_preview}'")
        
        for child in node.children:
            self.print_ast(child, indent + 1)

def test_parser():
    """Test the Erlang parser with sample code."""
    logger.info("Testing Erlang parser...")
    
    # Sample Erlang code
    test_code = '''
    %% Test function with multiple clauses
    -spec max(number(), number()) -> number().
    max(A, B) when A > B -> A;
    max(A, B) -> B.
    
    %% Simple function
    hello() -> world.
    '''
    
    try:
        parser = ErlangParser()
        root = parser.parse_string(test_code)
        
        if root:
            logger.info("✓ Successfully parsed test code")
            logger.info(f"Root node type: {root.type}")
            
            # Print AST structure
            print("\nAST Structure:")
            parser.print_ast(root)
            
            # Extract functions
            functions = parser.extract_functions(root)
            logger.info(f"Found {len(functions)} function(s)")
            
            for i, func in enumerate(functions):
                name = parser.get_function_name(func)
                arity = parser.get_function_arity(func)
                logger.info(f"Function {i+1}: {name}/{arity}")
        else:
            logger.error("Failed to parse test code")
            
    except Exception as e:
        logger.error(f"Test failed: {e}")


if __name__ == "__main__":
    # Set up logging for testing
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    test_parser()

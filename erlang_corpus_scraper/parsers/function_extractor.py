#!/usr/bin/env python3
"""
Function extractor for Erlang corpus scraper.
Extracts functions from cloned Erlang repositories and prepares them for GraphCodeBERT training.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

# Import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PARSER_CONFIG, FUNCTION_SCORING, get_output_path
from parsers.erlang_parser import ErlangParser

# Setup logging
logger = logging.getLogger(__name__)

@dataclass
class ErlangFunction:
    """Information about an extracted Erlang function."""
    # Basic info
    idx: str                    # Unique identifier
    name: str                   # Function name
    arity: int                  # Number of parameters
    file_path: str             # Source file path
    line_start: int            # Starting line number
    line_end: int              # Ending line number
    
    # Content
    code: str                  # Raw function code
    code_tokens: List[str]     # Tokenized code
    docstring: Optional[str]   # Documentation/comments
    type_spec: Optional[str]   # Type specification if present
    
    # AST information for data flow extraction
    ast_node: Any              # Original AST node for DFG extraction
    file_lines: List[str]      # File lines for token position mapping
    
    # Metadata
    clauses: int               # Number of function clauses
    has_guards: bool           # Whether function has guards
    has_patterns: bool         # Whether function uses pattern matching
    is_exported: bool          # Whether function is exported
    
    # Scoring
    score: float               # Quality score (0-100)
    score_breakdown: Dict[str, float]  # Detailed scoring
    
    # Repository info
    repo_name: str             # Repository name
    repo_path: str             # Local repository path

class FunctionExtractor:
    """Extracts functions from Erlang source files."""
    
    def __init__(self):
        """Initialize function extractor."""
        self.parser = ErlangParser()
        self.extracted_functions = []
        
    def extract_from_repository(self, repo_path: str, repo_name: str) -> List[ErlangFunction]:
        """Extract functions from a repository.
        
        Args:
            repo_path: Path to cloned repository
            repo_name: Repository name for metadata
            
        Returns:
            List of extracted functions
        """
        logger.info(f"Extracting functions from {repo_name}")
        
        repo_functions = []
        erlang_files = self._find_erlang_files(repo_path)
        
        logger.info(f"Found {len(erlang_files)} Erlang files in {repo_name}")
        
        for file_path in erlang_files:
            try:
                file_functions = self._extract_from_file(file_path, repo_name, repo_path)
                repo_functions.extend(file_functions)
                
            except Exception as e:
                logger.warning(f"Failed to extract from {file_path}: {e}")
                continue
        
        logger.info(f"Extracted {len(repo_functions)} functions from {repo_name}")
        return repo_functions
    
    def _find_erlang_files(self, repo_path: str) -> List[str]:
        """Find all Erlang source files in repository."""
        erlang_files = []
        repo_path = Path(repo_path)
        
        for ext in PARSER_CONFIG["erlang_extensions"]:
            erlang_files.extend(repo_path.rglob(f"*{ext}"))
        
        # Convert to strings and filter
        erlang_files = [str(f) for f in erlang_files if f.is_file()]
        
        # Skip test files, examples, etc.
        filtered_files = []
        skip_patterns = ['test', 'tests', 'example', 'examples', 'demo', 'benchmark']
        
        for file_path in erlang_files:
            skip = False
            for pattern in skip_patterns:
                if pattern in file_path.lower():
                    skip = True
                    break
            if not skip:
                filtered_files.append(file_path)
        
        return filtered_files
    
    def _extract_from_file(self, file_path: str, repo_name: str, repo_path: str) -> List[ErlangFunction]:
        """Extract functions from a single Erlang file."""
        try:
            # Parse file
            root = self.parser.parse_file(file_path)
            if not root:
                logger.warning(f"Failed to parse {file_path}")
                return []
            
            # Read file content for line numbers and tokenization
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                file_content = f.read()
                file_lines = file_content.split('\n')
            
            # Extract functions
            function_nodes = self.parser.extract_functions(root)
            functions = []
            
            for i, func_node in enumerate(function_nodes):
                try:
                    func = self._extract_function_info(
                        func_node, file_path, file_lines, repo_name, repo_path, i
                    )
                    if func and self._should_include_function(func):
                        functions.append(func)
                        
                except Exception as e:
                    logger.warning(f"Failed to extract function {i} from {file_path}: {e}")
                    continue
            
            return functions
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            return []
    
    def _extract_function_info(self, func_node, file_path: str, file_lines: List[str], 
                             repo_name: str, repo_path: str, func_index: int) -> Optional[ErlangFunction]:
        """Extract detailed information about a function."""
        
        # Basic function info
        name = self.parser.get_function_name(func_node)
        if not name:
            return None
            
        arity = self.parser.get_function_arity(func_node)
        
        # Get function code and position  
        start_line = func_node.start_point[0]
        end_line = func_node.end_point[0]
        
        # Get base function code
        func_code = self.parser.node_text(func_node)
        if not func_code.strip():
            return None
            
        # Look for type specs and include them in the code
        type_spec = self._extract_type_spec_from_ast(func_node)
        if type_spec:
            # Prepend type spec to function code
            if not type_spec.startswith('-spec'):
                type_spec = f"-spec {type_spec}"
            func_code = f"{type_spec}\n{func_code}"
        
        # Extract tokens using AST (GraphCodeBERT approach)
        code_tokens = self._extract_ast_tokens(func_node, file_lines)
        
        # If we have a type spec, we need to tokenize the combined code
        if type_spec:
            # Create a temporary combined code for tokenization
            combined_code = f"{type_spec}\n{self.parser.node_text(func_node)}"
            # Parse the combined code to get proper tokens
            temp_root = self.parser.parse_string(combined_code)
            if temp_root:
                code_tokens = self._extract_ast_tokens(temp_root, combined_code.split('\n'))
        
        # Check length constraints
        if (len(code_tokens) < PARSER_CONFIG["min_function_length"] or 
            len(code_tokens) > PARSER_CONFIG["max_function_length"]):
            return None
        
        # Extract clauses and analyze patterns
        clauses = self.parser.get_function_clauses(func_node)
        clause_count = len(clauses)
        
        has_guards = any(self.parser.has_guard(clause) for clause in clauses)
        has_patterns = clause_count > 1  # Multiple clauses indicate pattern matching
        
        # Look for documentation
        docstring = self._extract_documentation(func_node, file_lines, start_line)
        
        # Look for type specs
        type_spec = self._extract_type_spec(file_lines, start_line, name, arity)
        
        # Check if exported (simple heuristic)
        is_exported = self._is_function_exported(file_lines, name, arity)
        
        # Create unique identifier
        rel_path = os.path.relpath(file_path, repo_path)
        idx = f"{repo_name}::{rel_path}::{name}/{arity}::{func_index}"
        
        # Create function object
        func = ErlangFunction(
            idx=idx,
            name=name,
            arity=arity,
            file_path=file_path,
            line_start=start_line,
            line_end=end_line,
            code=func_code,
            code_tokens=code_tokens,
            docstring=docstring,
            type_spec=type_spec,
            ast_node=func_node,  # Store AST node for DFG extraction
            file_lines=file_lines,  # Store file lines for position mapping
            clauses=clause_count,
            has_guards=has_guards,
            has_patterns=has_patterns,
            is_exported=is_exported,
            score=0.0,
            score_breakdown={},
            repo_name=repo_name,
            repo_path=repo_path
        )
        
        # Score the function
        func.score, func.score_breakdown = self._score_function(func)
        
        return func
    
    def _extract_ast_tokens(self, func_node, file_lines: List[str]) -> List[str]:
        """Extract tokens from AST using GraphCodeBERT approach."""
        # Use the parser's GraphCodeBERT-style method
        code_tokens, _, _ = self.parser.extract_dataflow_info(func_node, file_lines)
        
        # Filter out empty tokens
        return [token for token in code_tokens if token.strip()]
    
    def _extract_documentation(self, func_node, file_lines: List[str], start_line: int) -> Optional[str]:
        """Extract documentation/comments for a function from AST."""
        # Try to extract from AST first (more accurate)
        doc_from_ast = self._extract_comments_from_ast(func_node)
        
        if doc_from_ast:
            return doc_from_ast
        
        # Fallback to manual line parsing (in case AST doesn't capture preceding comments)
        return self._extract_comments_from_lines(file_lines, start_line)
    
    def _extract_comments_from_ast(self, func_node) -> Optional[str]:
        """Extract comments from AST nodes."""
        comments = []
        
        # Get the full file AST to look for comments before the function
        # We need to traverse siblings and find comment nodes before this function
        parent = func_node.parent
        if parent:
            func_start_line = func_node.start_point[0]
            
            # Look through all children of parent to find comments before our function
            for sibling in parent.children:
                if sibling == func_node:
                    break  # Found our function, stop looking
                    
                if sibling.type == 'comment':
                    comment_line = sibling.start_point[0]
                    # Include comments that are close to our function (within 10 lines)
                    if func_start_line - comment_line <= 10:
                        comment_text = self.parser.node_text(sibling)
                        # Clean up comment markers
                        if comment_text.startswith('%%'):
                            comment_text = comment_text[2:].strip()
                        elif comment_text.startswith('%'):
                            comment_text = comment_text[1:].strip()
                        
                        if comment_text and not comment_text.startswith('-'):
                            comments.append(comment_text)
        
        # Also look for comments within the function node itself
        comments.extend(self._find_comments_in_node(func_node))
        
        # Don't duplicate type specs in docstring - they're already in code
        
        return ' '.join(comments) if comments else None
    
    def _find_comments_in_node(self, node) -> List[str]:
        """Recursively find comment nodes within a given node."""
        comments = []
        
        if node.type == 'comment':
            comment_text = self.parser.node_text(node)
            if comment_text.startswith('%%'):
                comment_text = comment_text[2:].strip()
            elif comment_text.startswith('%'):
                comment_text = comment_text[1:].strip()
            
            if comment_text and not comment_text.startswith('-'):
                comments.append(comment_text)
        
        # Recursively check children
        for child in node.children:
            comments.extend(self._find_comments_in_node(child))
        
        return comments
    
    def _extract_type_spec_from_ast(self, func_node) -> Optional[str]:
        """Extract type specification from AST."""
        # Look for -spec attributes before the function
        parent = func_node.parent
        if parent:
            func_start_line = func_node.start_point[0]
            
            for sibling in parent.children:
                if sibling == func_node:
                    break
                    
                # Look for attribute nodes that might be type specs
                if sibling.type in ['attribute', 'spec_attribute', 'type_spec']:
                    attr_text = self.parser.node_text(sibling)
                    if attr_text.startswith('-spec'):
                        # Extract the spec content
                        spec_content = attr_text[5:].strip()
                        return spec_content
                        
                # Also check within 5 lines of the function
                if func_start_line - sibling.start_point[0] <= 5:
                    spec_in_node = self._find_spec_in_node(sibling)
                    if spec_in_node:
                        return spec_in_node
        
        return None
    
    def _find_spec_in_node(self, node) -> Optional[str]:
        """Find type spec within a node."""
        if node.type in ['attribute', 'spec_attribute', 'type_spec']:
            text = self.parser.node_text(node)
            if '-spec' in text:
                return text.strip()
        
        for child in node.children:
            result = self._find_spec_in_node(child)
            if result:
                return result
                
        return None
    
    def _extract_comments_from_lines(self, file_lines: List[str], start_line: int) -> Optional[str]:
        """Fallback: Extract documentation from file lines (original approach)."""
        doc_lines = []
        
        # Look for comments before the function (up to 10 lines back)
        for line_idx in range(max(0, start_line - 10), start_line):
            if line_idx < len(file_lines):
                line = file_lines[line_idx].strip()
                if line.startswith('%%'):
                    # Extract comment content
                    comment = line[2:].strip()
                    if comment and not comment.startswith('-'):  # Skip separator lines
                        doc_lines.append(comment)
                elif line.startswith('%') and not line.startswith('%%'):
                    # Single % comments
                    comment = line[1:].strip()
                    if comment:
                        doc_lines.append(comment)
                elif line.strip() and not line.startswith('-'):
                    # Non-comment, non-directive line - stop looking
                    break
        
        return ' '.join(doc_lines) if doc_lines else None
    
    def _extract_type_spec(self, file_lines: List[str], start_line: int, 
                          name: Optional[str], arity: Optional[int]) -> Optional[str]:
        """Extract type specification for a function."""
        # Look for -spec declarations before the function
        for line_idx in range(max(0, start_line - 5), start_line):
            if line_idx < len(file_lines):
                line = file_lines[line_idx].strip()
                if line.startswith('-spec'):
                    # Extract spec content
                    spec_content = line[5:].strip()
                    if name and f"{name}(" in spec_content:
                        return spec_content
                    elif not name:  # Just looking for any spec
                        return spec_content
        
        return None
    
    def _is_function_exported(self, file_lines: List[str], name: str, arity: int) -> bool:
        """Check if function is exported."""
        export_pattern = f"{name}/{arity}"
        
        for line in file_lines:
            line = line.strip()
            if line.startswith('-export(') and export_pattern in line:
                return True
        
        return False
    
    def _score_function(self, func: ErlangFunction) -> Tuple[float, Dict[str, float]]:
        """Score function quality based on our criteria."""
        weights = FUNCTION_SCORING["weights"]
        breakdown = {}
        
        # Size score (25 points)
        token_count = len(func.code_tokens)
        if 10 <= token_count <= 50:
            size_score = 25
        elif 51 <= token_count <= 100:
            size_score = 20
        elif 101 <= token_count <= 200:
            size_score = 10
        elif token_count < 10:
            size_score = 5
        else:
            size_score = 0
        breakdown["size"] = size_score
        
        # Documentation score (25 points)
        if func.docstring:
            words = len(func.docstring.split())
            if words > 20:
                doc_score = 25
            elif words > 10:
                doc_score = 20
            elif words > 3:
                doc_score = 15
            else:
                doc_score = 5
        else:
            doc_score = 0
        breakdown["documentation"] = doc_score
        
        # Language features score (20 points)
        if func.clauses > 1 and func.has_guards:
            features_score = 20
        elif func.has_patterns:
            features_score = 15
        elif func.has_guards:
            features_score = 10
        else:
            features_score = 5
        breakdown["features"] = features_score
        
        # Quality indicators (15 points)
        quality_score = 0
        if func.is_exported:
            quality_score += 5
        if func.type_spec:
            quality_score += 5
        if func.name and not func.name.startswith('_'):  # Not private
            quality_score += 3
        if func.docstring and '@spec' in func.docstring:
            quality_score += 2
        breakdown["quality"] = quality_score
        
        # Complexity balance (15 points)
        if 2 <= func.clauses <= 4:
            complexity_score = 15
        elif func.clauses == 1:
            complexity_score = 10
        elif 5 <= func.clauses <= 8:
            complexity_score = 8
        else:
            complexity_score = 0
        breakdown["complexity"] = complexity_score
        
        # Calculate weighted total
        total_score = (
            breakdown["size"] * weights["size"] +
            breakdown["documentation"] * weights["documentation"] +
            breakdown["features"] * weights["features"] +
            breakdown["quality"] * weights["quality"] +
            breakdown["complexity"] * weights["complexity"]
        )
        
        return min(100, total_score), breakdown
    
    def _should_include_function(self, func: ErlangFunction) -> bool:
        """Determine if function should be included in corpus."""
        # Minimum score threshold
        if func.score < PARSER_CONFIG.get("min_score", 40):
            return False
        
        # Skip obvious test/debug functions
        skip_names = ['test_', 'debug_', 'tmp_', 'temp_']
        if any(func.name.startswith(prefix) for prefix in skip_names):
            return False
        
        # Require documentation if configured
        if PARSER_CONFIG.get("require_docstring", False) and not func.docstring:
            return False
        
        return True

def extract_functions_from_repositories(repo_results: List, max_workers: int = 4) -> List[ErlangFunction]:
    """Extract functions from multiple repositories in parallel."""
    logger.info(f"Starting function extraction from {len(repo_results)} repositories")
    
    all_functions = []
    extractor = FunctionExtractor()
    
    for repo_result in repo_results:
        if not repo_result.success:
            logger.warning(f"Skipping failed clone: {repo_result.repo_info.full_name}")
            continue
        
        try:
            repo_functions = extractor.extract_from_repository(
                repo_result.local_path,
                repo_result.repo_info.full_name
            )
            all_functions.extend(repo_functions)
            
        except Exception as e:
            logger.error(f"Failed to extract from {repo_result.repo_info.full_name}: {e}")
            continue
    
    logger.info(f"Total functions extracted: {len(all_functions)}")
    return all_functions

def save_corpus(functions: List[ErlangFunction], output_file: str = None):
    """Save extracted functions as corpus."""
    if output_file is None:
        output_file = get_output_path("erlang_functions.jsonl")
    
    logger.info(f"Saving {len(functions)} functions to {output_file}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for func in functions:
            # Convert to GraphCodeBERT format
            corpus_entry = {
                "idx": func.idx,
                "repo": func.repo_name,
                "path": os.path.relpath(func.file_path, func.repo_path),
                "func_name": func.name,
                "original_string": func.code,
                "language": "erlang",
                "code": func.code,
                "code_tokens": func.code_tokens,
                "docstring": func.docstring or "",
                "docstring_tokens": func.docstring.split() if func.docstring else [],
                "sha": "unknown",  # Could add git commit hash
                "url": f"https://github.com/{func.repo_name}",
                "partition": "train",  # Will be split later
                
                # Erlang-specific metadata
                "arity": func.arity,
                "clauses": func.clauses,
                "has_guards": func.has_guards,
                "has_patterns": func.has_patterns,
                "is_exported": func.is_exported,
                "type_spec": func.type_spec,
                "score": func.score,
                "score_breakdown": func.score_breakdown,
                
                # Note: AST node and file_lines are not serialized to JSON
                # They will be used later for data flow graph extraction
                # We'll add a separate method for that
            }
            
            f.write(json.dumps(corpus_entry, ensure_ascii=False) + '\n')
    
    logger.info(f"✓ Corpus saved to {output_file}")
    logger.info("Note: AST data preserved in memory for data flow graph extraction")

def main():
    """Test function extraction."""
    logging.basicConfig(level=logging.DEBUG)  # Enable debug logging
    
    # Test with a simple Erlang file
    test_code = '''
    %% Test module for function extraction
    -module(test).
    -export([max/2, factorial/1]).
    
    %% Returns the maximum of two numbers
    -spec max(number(), number()) -> number().
    max(A, B) when A > B -> A;
    max(A, B) -> B.
    
    %% Calculate factorial
    factorial(0) -> 1;
    factorial(N) when N > 0 -> N * factorial(N - 1).
    
    %% Private helper function
    helper() -> ok.
    '''
    
    # Write test file
    test_file = "/tmp/test_erlang.erl"
    with open(test_file, 'w') as f:
        f.write(test_code)
    
    # Test extraction
    extractor = FunctionExtractor()
    functions = extractor._extract_from_file(test_file, "test_repo", "/tmp")
    
    print(f"Extracted {len(functions)} functions:")
    for func in functions:
        print(f"  {func.name}/{func.arity} - Score: {func.score:.1f}")
        print(f"    Docstring: {func.docstring}")
        print(f"    Features: clauses={func.clauses}, guards={func.has_guards}")
        print()

if __name__ == "__main__":
    main()

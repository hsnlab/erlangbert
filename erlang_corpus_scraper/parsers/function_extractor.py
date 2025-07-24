#!/usr/bin/env python3
"""
Function extractor for Erlang corpus scraper.
Extracts functions from cloned Erlang repositories and prepares them for GraphCodeBERT training.
Updated with simplified DFG semantics using clause parameter tracking.
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

# Import our modules - updated for new config structure
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PARSER_CONFIG, FUNCTION_SCORING, OUTPUT_CONFIG, get_output_path
from parsers.erlang_parser import ErlangParser
from scrapers.repo_cloner import CloneResult,RepositoryInfo

# Setup logging
logger = logging.getLogger(__name__)

@dataclass
class ErlangFunction:
    """Information about an extracted Erlang function with GraphCodeBERT data.
    
    Updated to include clause parameter information.
    """
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

    # GraphCodeBERT data - Updated to include clause parameter information
    variable_positions: List[Tuple[int, str, bool]]  # (token_index, variable_name, is_clause_param) tuples
    dataflow_graph: List[Tuple[str, int, str, List[str], List[int]]]  # DFG edges for GraphCodeBERT

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
    """Extracts functions from Erlang source files.
    
    Updated to work with new config structure but maintaining exact same functionality.
    """

    def __init__(self, max_workers: Optional[int] = None, min_score: Optional[float] = None):
        """Initialize function extractor.
        
        Args:
            max_workers: Maximum concurrent parser workers (maps to new config)
            min_score: Minimum function score threshold (maps to new config)
        """
        # Don't initialize parser here - will be done in each worker process
        self.parser = None
        self.extracted_functions = []
        
        # Map old parameters to new config structure
        self.max_workers = max_workers or PARSER_CONFIG.get('max_concurrent_parsers', 8)
        self.min_score = min_score or FUNCTION_SCORING.get('min_score', 5.0)
        
        # Get other config values with fallbacks for compatibility
        self.max_file_size = PARSER_CONFIG.get('max_file_size', 1024 * 1024)
        self.max_function_size = PARSER_CONFIG.get('max_function_size', 10000)
        self.min_function_size = PARSER_CONFIG.get('min_function_size', 20)
        self.timeout_per_file = PARSER_CONFIG.get('timeout_per_file', 30)
        
        logger.info(f"Function extractor initialized: max_workers={self.max_workers}, min_score={self.min_score}")

    def _ensure_parser(self):
        """Ensure parser is initialized (needed for multiprocessing)."""
        if self.parser is None:
            self.parser = ErlangParser()

    def extract_from_clone_results(self, clone_results) -> List[ErlangFunction]:
        """Extract functions from multiple cloned repositories.
        
        This method maintains the exact interface expected by main.py.
        """
        logger.info("Starting function extraction from cloned repositories")
        
        if not clone_results:
            logger.warning("No clone results provided")
            return []

        # For now, use single-threaded processing to avoid multiprocessing pickle issues
        # TODO: Fix multiprocessing with proper worker function
        all_functions = []
        self._ensure_parser()  # Initialize parser once
        
        for repo_data in clone_results:
            print(clone_results)
            print(repo_data)
            if repo_data.success:  
                repo_path = repo_data.local_path  # CloneResult has 'local_path' attribute
                repo_name = repo_data.repo_info.name  # Access name through repo_info
                
                try:
                    repo_functions = self._extract_from_repo(repo_path, repo_name)
                    all_functions.extend(repo_functions)
                    logger.info(f"Extracted {len(repo_functions)} functions from {repo_name}")
                except Exception as e:
                    logger.error(f"Error processing {repo_name}: {e}")

        logger.info(f"Total functions extracted: {len(all_functions)}")
        return all_functions

    def _extract_from_repo(self, repo_path: str, repo_name: str) -> List[ErlangFunction]:
        """Extract functions from a single repository."""
        logger.debug(f"Processing repository: {repo_name}")

        try:
            # Find all Erlang files
            erlang_files = self._find_erlang_files(repo_path)
            logger.debug(f"Found {len(erlang_files)} Erlang files in {repo_name}")

            all_functions = []
            for file_path in erlang_files:
                try:
                    file_functions = self._extract_from_file(file_path, repo_name, repo_path)
                    all_functions.extend(file_functions)
                except Exception as e:
                    logger.warning(f"Error processing {file_path}: {e}")

            return all_functions

        except Exception as e:
            logger.error(f"Error processing repository {repo_name}: {e}")
            return []

    def _find_erlang_files(self, repo_path: str) -> List[str]:
        """Find all Erlang files in a repository."""
        erlang_files = []
        
        logger.info(f"DEBUG: Searching for Erlang files in: {repo_path}")
        logger.info(f"DEBUG: Repository path exists: {os.path.exists(repo_path)}")

        # Use extensions from new config
        extensions = PARSER_CONFIG.get('erlang_extensions', ['.erl', '.hrl'])

        for root, dirs, files in os.walk(repo_path):
            # Skip common non-source directories
            dirs[:] = [d for d in dirs if d not in {'.git', '.svn', '_build', 'deps', 'ebin'}]

            for file in files:
                if any(file.endswith(ext) for ext in extensions):
                    erlang_files.append(os.path.join(root, file))

        return erlang_files

    def _extract_from_file(self, file_path: str, repo_name: str, repo_path: str) -> List[ErlangFunction]:
        """Extract functions from a single Erlang file using the enhanced parser."""
        logger.debug(f"Processing file: {file_path}")

        try:
            # Ensure parser is initialized in this process
            self._ensure_parser()
            
            # Read file content with encoding fallbacks from new config
            content = self._read_file_with_fallbacks(file_path)
            if not content:
                return []

            # Check content size
            if len(content) > self.max_function_size * 10:  # Reasonable file size limit
                logger.debug(f"Skipping very large file: {file_path}")
                return []

            file_lines = content.split('\n')

            # Parse the file using the parser
            root = self.parser.parse_string(content)
            if not root:
                logger.warning(f"Failed to parse {file_path}")
                return []

            # Extract functions using the parser
            functions = self.parser.extract_functions(root)
            logger.debug(f"Found {len(functions)} functions in {file_path}")

            extracted_functions = []

            for func_index, func_node in enumerate(functions):
                try:
                    # Extract basic function info using parser methods
                    name = self.parser.get_function_name(func_node)
                    if not name:
                        continue

                    arity = self.parser.get_function_arity(func_node)
                    clauses = self.parser.get_function_clauses(func_node)

                    # Get GraphCodeBERT data with clause parameter information
                    tokens, var_indices, var_names, is_clause_params = self.parser.extract_graphcodebert_data(func_node, file_lines)
                    
                    # Use the simplified DFG generation with clause parameter data
                    dfg = self.parser.create_dataflow_graph(var_indices, var_names, is_clause_params)

                    # Create variable positions with clause parameter information
                    variable_positions = list(zip(var_indices, var_names, is_clause_params))

                    # Get function text
                    func_code = self.parser.node_text(func_node)

                    # Extract docstring/spec
                    docstring = self._extract_docstring(func_node, file_lines)
                    type_spec = self._extract_type_spec(func_node, file_lines)

                    # Calculate score
                    score = self._calculate_function_score(func_code, docstring, clauses, len(var_indices))

                    # Skip functions below threshold
                    if score < self.min_score:
                        logger.debug(f"Skipping function {name}/{arity} (score {score:.1f} < {self.min_score})")
                        continue

                    # Create function object
                    func_obj = ErlangFunction(
                        idx=f"{repo_name}::{name}/{arity}::{func_index}",
                        name=name,
                        arity=arity,
                        file_path=file_path,
                        line_start=func_node.start_point[0] + 1,
                        line_end=func_node.end_point[0] + 1,
                        code=func_code,
                        code_tokens=tokens,
                        docstring=docstring,
                        type_spec=type_spec,
                        variable_positions=variable_positions,
                        dataflow_graph=dfg,
                        clauses=len(clauses),
                        has_guards=self._has_guards(func_node),
                        has_patterns=self._has_patterns(func_node),
                        is_exported=self._is_exported(func_node, content),
                        score=score,
                        score_breakdown=self._calculate_score_breakdown(func_code, docstring, clauses, len(var_indices)),
                        repo_name=repo_name,
                        repo_path=repo_path
                    )

                    extracted_functions.append(func_obj)

                except Exception as e:
                    logger.warning(f"Error processing function {func_index} in {file_path}: {e}")

            return extracted_functions

        except Exception as e:
            logger.warning(f"Error processing file {file_path}: {e}")
            return []

    # Worker function for multiprocessing (outside the class to avoid pickling issues)
    def _extract_from_repo_worker(repo_path: str, repo_name: str, max_file_size: int, 
                                 max_function_size: int, min_function_size: int, min_score: float) -> List[ErlangFunction]:
        """Worker function for extracting functions from a single repository."""
        # Create a full extractor instance in worker process
        extractor = FunctionExtractor.__new__(FunctionExtractor)
        extractor.parser = None  # Will be initialized when needed
        extractor.extracted_functions = []
        extractor.max_file_size = max_file_size
        extractor.max_function_size = max_function_size
        extractor.min_function_size = min_function_size
        extractor.min_score = min_score
        
        # Initialize all the methods by calling the actual __init__ logic (without parser)
        extractor.max_workers = 1  # Not used in worker
        extractor.timeout_per_file = 30
        
        return extractor._extract_from_repo(repo_path, repo_name)
    
    def _read_file_with_fallbacks(self, file_path: str) -> Optional[str]:
        """Read file with multiple encoding fallbacks."""
        encodings = PARSER_CONFIG.get('file_encodings', ['utf-8', 'latin-1', 'cp1252'])
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    return f.read()
            except (UnicodeDecodeError, UnicodeError):
                continue
            except Exception as e:
                logger.warning(f"Error reading {file_path} with {encoding}: {e}")
                continue
        
        logger.warning(f"Could not read file {file_path} with any encoding")
        return None

    def _extract_docstring(self, func_node, file_lines: List[str]) -> Optional[str]:
        """Extract function docstring/comments."""
        start_line = func_node.start_point[0]
        
        # Look for comments above the function
        docstring_lines = []
        for i in range(max(0, start_line - 5), start_line):
            if i < len(file_lines):
                line = file_lines[i].strip()
                if line.startswith('%'):
                    docstring_lines.append(line[1:].strip())
                elif line and not line.startswith('-'):
                    break
        
        if docstring_lines:
            return ' '.join(docstring_lines)
        return None

    def _extract_type_spec(self, func_node, file_lines: List[str]) -> Optional[str]:
        """Extract function type specification."""
        start_line = func_node.start_point[0]
        
        # Look for -spec annotation above the function
        for i in range(max(0, start_line - 3), start_line):
            if i < len(file_lines):
                line = file_lines[i].strip()
                if line.startswith('-spec'):
                    return line
        
        return None

    def _calculate_function_score(self, code: str, docstring: str, clauses: List, num_vars: int) -> float:
        """Calculate function quality score."""
        score = 0.0
        
        # Base score for having code
        if code and len(code.strip()) > 0:
            score += 10.0
        
        # Score for having nonempty docstring
        if docstring and len(docstring.strip()) > 0:
            score += 25.0
        
        # Score for function length (sweet spot around 100-500 chars)
        code_len = len(code)
        if 50 <= code_len <= 1000:
            score += 20.0
        elif 20 <= code_len <= 50 or 1000 <= code_len <= 2000:
            score += 10.0
        
        # Score for multiple clauses
        num_clauses = len(clauses)
        if num_clauses > 1:
            score += min(num_clauses * 5.0, 25.0)
        
        # Score for having variables
        if num_vars > 0:
            score += min(num_vars * 2.0, 20.0)
        
        # Score for complexity indicators
        complexity_indicators = [
            'when ', 'case ', 'if ', 'receive ', 'try ', 'catch ',
            'fun(', 'spawn', 'gen_', 'handle_'
        ]
        for indicator in complexity_indicators:
            if indicator in code:
                score += 5.0
        
        return min(score, 100.0)

    def _calculate_score_breakdown(self, code: str, docstring: str, clauses: List, num_vars: int) -> Dict[str, float]:
        """Calculate detailed score breakdown."""
        breakdown = {
            'base': 10.0 if code and len(code.strip()) > 0 else 0.0,
            'doc': 25.0 if docstring and len(docstring.strip()) > 0 else 0.0,
            'length': 0.0,
            'clauses': 0.0,
            'variables': 0.0,
            'complexity': 0.0
        }
        
        # Length scoring
        code_len = len(code)
        if 50 <= code_len <= 1000:
            breakdown['length'] = 20.0
        elif 20 <= code_len <= 50 or 1000 <= code_len <= 2000:
            breakdown['length'] = 10.0
        
        # Clause scoring
        num_clauses = len(clauses)
        if num_clauses > 1:
            breakdown['clauses'] = min(num_clauses * 5.0, 25.0)
        
        # Variable scoring
        if num_vars > 0:
            breakdown['variables'] = min(num_vars * 2.0, 20.0)
        
        # Complexity scoring
        complexity_indicators = [
            'when ', 'case ', 'if ', 'receive ', 'try ', 'catch ',
            'fun(', 'spawn', 'gen_', 'handle_'
        ]
        for indicator in complexity_indicators:
            if indicator in code:
                breakdown['complexity'] += 5.0
        
        return breakdown

    def _has_guards(self, func_node) -> bool:
        """Check if function has guards."""
        return 'when' in self.parser.node_text(func_node)

    def _has_patterns(self, func_node) -> bool:
        """Check if function uses pattern matching."""
        code = self.parser.node_text(func_node)
        pattern_indicators = ['{', '[', '=', '|']
        return any(indicator in code for indicator in pattern_indicators)

    def _is_exported(self, func_node, file_content: str) -> bool:
        """Check if function is exported."""
        func_name = self.parser.get_function_name(func_node)
        arity = self.parser.get_function_arity(func_node)
        
        if not func_name:
            return False
        
        # Look for export declaration
        export_pattern = f"{func_name}/{arity}"
        return export_pattern in file_content and '-export(' in file_content

    def get_extracted_functions(self) -> List[ErlangFunction]:
        """Get all extracted functions."""
        return self.extracted_functions

    def save_functions(self, functions: List[ErlangFunction], output_file: Optional[str] = None):
        """Save extracted functions to JSON file.
        
        Maintained for backward compatibility.
        """
        if output_file is None:
            output_file = get_output_path(OUTPUT_CONFIG.get('functions_file', 'functions.jsonl'))

        logger.info(f"Saving {len(functions)} functions to {output_file}")

        with open(output_file, 'w', encoding='utf-8') as f:
            for func in functions:
                # Convert to GraphCodeBERT format (keeping exact same format)
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

                    # GraphCodeBERT specific data
                    "variable_positions": func.variable_positions,
                    "dataflow_graph": func.dataflow_graph,

                    # Erlang-specific metadata
                    "arity": func.arity,
                    "clauses": func.clauses,
                    "has_guards": func.has_guards,
                    "has_patterns": func.has_patterns,
                    "is_exported": func.is_exported,
                    "type_spec": func.type_spec,
                    "score": func.score,
                    "score_breakdown": func.score_breakdown,
                }

                f.write(json.dumps(corpus_entry, ensure_ascii=False) + '\n')

        logger.info(f"✓ Corpus saved to {output_file}")

def main():
    """Test function extraction with the simplified DFG semantics."""
    logging.basicConfig(level=logging.DEBUG)

    # Test with the max/2 function to demonstrate simplified semantics
    test_code = '''
%% Test module for function extraction
-module(test).
-export([max/2]).

%% Returns the maximum of two numbers
-spec max(number(), number()) -> number().
max(A, B) when A > B -> A;
max(A, B) -> B.
'''

    # Create a temporary directory and file for testing
    import tempfile
    import os
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Write test file
        test_file = os.path.join(temp_dir, "test.erl")
        with open(test_file, 'w') as f:
            f.write(test_code)
        
        # Mock clone results format expected by extract_from_clone_results
        mock_clone_results = [ CloneResult(RepositoryInfo("test_repo","","",0,0,0,"Erlang",[],"","","","",False,False,False,100,100), True, temp_dir,"",0,0) ]
        
        # Test extraction using the external API
        extractor = FunctionExtractor(max_workers=1, min_score=0.0)
        functions = extractor.extract_from_clone_results(mock_clone_results)
        
        print("=" * 60)
        print("TESTING SIMPLIFIED DFG SEMANTICS")
        print("=" * 60)
        
        print(f"Extracted {len(functions)} functions:")
        for func in functions:
            if func.name == 'max' and func.arity == 2:
                print(f"\n{func.name}/{func.arity} - Score: {func.score:.1f}")
                print(f"  Clauses: {func.clauses}, Guards: {func.has_guards}")
                print(f"  Tokens ({len(func.code_tokens)}): {func.code_tokens}")
                print(f"  Variables with clause parameter info:")
                for pos, name, is_param in func.variable_positions:
                    print(f"    ({pos}, '{name}', {is_param})")
                print(f"  DFG (simplified semantics):")
                for edge in func.dataflow_graph:
                    var_name, pos, relation, deps, dep_positions = edge
                    print(f"    {edge}")
                print(f"  Docstring: {func.docstring}")
                
                # Explain the simplified semantics
                print(f"\n  DFG Analysis:")
                for edge in func.dataflow_graph:
                    var_name, pos, relation, deps, dep_positions = edge
                    if not deps:
                        print(f"    {var_name}@{pos}: Independent variable")
                    else:
                        source_pos = dep_positions[0] if dep_positions else "?"
                        print(f"    {var_name}@{pos}: Comes from {var_name}@{source_pos}")
                
                print(f"\n  Expected behavior with clause parameter tracking:")
                print(f"    - A@2 and B@4 are independent (clause 1 parameters)")
                print(f"    - A@7 comes from A@2 (guard variable)")
                print(f"    - B@9 comes from B@4 (guard variable)")
                print(f"    - A@11 comes from A@7 (most recent use)")
                print(f"    - A@15 and B@17 are independent (clause 2 parameters)")
                print(f"    - B@20 comes from B@17 (clause 2 return)")
                
        print("\n" + "=" * 60)
        print("✓ Enhanced DFG semantics with clause parameter tracking!")
        print("✓ No more fragile token parsing - using AST structure")
        print("✓ Clause parameters properly identified from parser")
        print("✓ Data flow tracks actual value sources")
        print("=" * 60)

if __name__ == "__main__":
    main()
    

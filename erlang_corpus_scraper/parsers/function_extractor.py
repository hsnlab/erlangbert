#!/usr/bin/env python3
"""
Function extractor for Erlang corpus scraper.
Extracts functions from cloned Erlang repositories and prepares them for GraphCodeBERT training.
Updated to work with the new config.py structure while maintaining 100% functionality compatibility.
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

# Setup logging
logger = logging.getLogger(__name__)

@dataclass
class ErlangFunction:
    """Information about an extracted Erlang function with GraphCodeBERT data.
    
    Keeping the exact same structure for 100% compatibility.
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

    # GraphCodeBERT data - Updated to match new parser format
    variable_positions: List[Tuple[int, str]]  # (token_index, variable_name) pairs
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
        self.parser = ErlangParser()
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

    def extract_from_clone_results(self, clone_results) -> List[ErlangFunction]:
        """Extract functions from multiple cloned repositories.
        
        This method maintains the exact interface expected by main.py.
        """
        all_functions = []
        
        # Filter successful clones
        successful_clones = [r for r in clone_results if r.success and r.local_path]
        logger.info(f"Extracting functions from {len(successful_clones)} successfully cloned repositories")
        
        for clone_result in successful_clones:
            try:
                repo_functions = self.extract_from_repository(
                    clone_result.local_path, 
                    clone_result.repo_info.full_name
                )
                all_functions.extend(repo_functions)
                
                logger.info(f"Extracted {len(repo_functions)} functions from {clone_result.repo_info.full_name}")
                
            except Exception as e:
                logger.error(f"Failed to extract from {clone_result.repo_info.full_name}: {e}")
                continue
        
        logger.info(f"Total functions extracted: {len(all_functions)}")
        
        # Save functions to output file
        self._save_functions_jsonl(all_functions)
        
        return all_functions

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
                # Check file size
                if os.path.getsize(file_path) > self.max_file_size:
                    logger.debug(f"Skipping large file: {file_path}")
                    continue
                    
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
        """Extract functions from a single Erlang file using the new parser format."""
        logger.debug(f"Processing file: {file_path}")

        try:
            # Read file content with encoding fallbacks from new config
            content = self._read_file_with_fallbacks(file_path)
            if not content:
                return []

            # Check content size
            if len(content) > self.max_function_size * 10:  # Reasonable file size limit
                logger.debug(f"Skipping very large file: {file_path}")
                return []

            file_lines = content.split('\n')

            # Parse the file using the new simplified parser
            root = self.parser.parse_string(content)
            if not root:
                logger.warning(f"Failed to parse {file_path}")
                return []

            # Extract functions using the new simplified method
            functions = self.parser.extract_functions(root)
            logger.debug(f"Found {len(functions)} functions in {file_path}")

            extracted_functions = []

            for func_index, func_node in enumerate(functions):
                try:
                    # Extract basic function info using new parser methods
                    name = self.parser.get_function_name(func_node)
                    if not name:
                        continue

                    arity = self.parser.get_function_arity(func_node)
                    clauses = self.parser.get_function_clauses(func_node)

                    # Get GraphCodeBERT data directly from the new parser
                    tokens, var_indices, var_names = self.parser.extract_graphcodebert_data(func_node, file_lines)
                    dfg = self.parser.create_dataflow_graph(var_indices, var_names)

                    # Create variable positions in the format expected by the dataclass
                    variable_positions = list(zip(var_indices, var_names))

                    # Extract position info - use the span from the function node
                    if hasattr(func_node, 'start_point') and hasattr(func_node, 'end_point'):
                        start_line = func_node.start_point[0] + 1  # Convert to 1-based
                        end_line = func_node.end_point[0] + 1
                    else:
                        # Fallback for combined functions
                        start_line = clauses[0].start_point[0] + 1 if clauses else 1
                        end_line = clauses[-1].end_point[0] + 1 if clauses else start_line

                    # Get function code - reconstruct from tokens or extract from source
                    func_code = ' '.join(tokens)

                    # Check function size constraints
                    if len(func_code) < self.min_function_size or len(func_code) > self.max_function_size:
                        continue

                    # Extract docstring (look for comments before the function)
                    docstring = self._extract_docstring(file_lines, start_line)

                    # Extract type spec if present
                    type_spec = self._extract_type_spec(file_lines, start_line, name, arity)

                    # Check if function is exported
                    is_exported = self._is_function_exported(file_lines, name, arity)

                    # Analyze function features
                    has_guards = any(self.parser.has_guard(clause) for clause in clauses)
                    has_patterns = self._has_pattern_matching(tokens)

                    # Create unique identifier
                    rel_path = os.path.relpath(file_path, repo_path)
                    idx = f"{repo_name}::{rel_path}::{name}/{arity}::{func_index}"

                    # Create function object with the new simplified data
                    func = ErlangFunction(
                        idx=idx,
                        name=name,
                        arity=arity,
                        file_path=file_path,
                        line_start=start_line,
                        line_end=end_line,
                        code=func_code,
                        code_tokens=tokens,
                        docstring=docstring,
                        type_spec=type_spec,
                        variable_positions=variable_positions,  # Now using the correct format
                        dataflow_graph=dfg,                    # DFG directly from parser
                        clauses=len(clauses),
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

                    # Filter functions by score
                    if self._should_include_function(func):
                        extracted_functions.append(func)

                except Exception as e:
                    logger.warning(f"Failed to extract function {name if 'name' in locals() else 'unknown'}: {e}")
                    continue

            return extracted_functions

        except Exception as e:
            logger.error(f"Failed to process file {file_path}: {e}")
            return []

    def _read_file_with_fallbacks(self, file_path: str) -> Optional[str]:
        """Read file with encoding fallbacks from config."""
        encoding = PARSER_CONFIG.get('encoding', 'utf-8')
        fallbacks = PARSER_CONFIG.get('encoding_fallbacks', ['latin1', 'cp1252'])
        
        for enc in [encoding] + fallbacks:
            try:
                with open(file_path, 'r', encoding=enc, errors='ignore') as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
            except Exception as e:
                logger.warning(f"Failed to read {file_path} with encoding {enc}: {e}")
                continue
        
        logger.error(f"Failed to read {file_path} with any encoding")
        return None

    def _should_include_function(self, func: ErlangFunction) -> bool:
        """Determine if function should be included in corpus."""
        # Skip obvious test/debug functions
        skip_names = ['test_', 'debug_', 'tmp_', 'temp_']
        if any(func.name.startswith(prefix) for prefix in skip_names):
            return False

        # Minimum score threshold (using instance variable from config)
        if func.score < self.min_score:
            return False

        # Could add more sophisticated filtering based on new config
        # For now, keeping the same logic for compatibility

        return True

    def _extract_docstring(self, file_lines: List[str], func_start_line: int) -> Optional[str]:
        """Extract docstring/comments before a function."""
        comments = []

        # Look backwards from function start for comments
        for i in range(func_start_line - 2, max(-1, func_start_line - 10), -1):
            if i < 0 or i >= len(file_lines):
                continue

            line = file_lines[i].strip()
            if line.startswith('%%') or line.startswith('%'):
                # Remove comment markers and clean up
                comment = line.lstrip('%').strip()
                if comment:
                    comments.insert(0, comment)
            elif line == '':
                continue  # Skip empty lines
            else:
                break  # Stop at non-comment, non-empty line

        return ' '.join(comments) if comments else None

    def _extract_type_spec(self, file_lines: List[str], func_start_line: int, name: str, arity: int) -> Optional[str]:
        """Extract type specification for a function."""
        # Look for -spec lines before the function
        for i in range(max(0, func_start_line - 5), func_start_line):
            if i >= len(file_lines):
                continue

            line = file_lines[i].strip()
            if line.startswith('-spec') and f"{name}(" in line:
                return line

        return None

    def _is_function_exported(self, file_lines: List[str], name: str, arity: int) -> bool:
        """Check if function is exported."""
        export_pattern = f"{name}/{arity}"

        for line in file_lines:
            stripped = line.strip()
            if stripped.startswith('-export(') and export_pattern in line:
                return True

        return False

    def _has_pattern_matching(self, tokens: List[str]) -> bool:
        """Simple heuristic to detect pattern matching."""
        # Look for pattern matching indicators
        pattern_indicators = {'=', '[', '{', '|', '_'}
        return any(token in pattern_indicators for token in tokens)

    def _score_function(self, func: ErlangFunction) -> Tuple[float, Dict[str, float]]:
        """Score function quality based on the configuration criteria.
        
        Updated to use new FUNCTION_SCORING config structure.
        """
        # Get weights from new config structure
        size_weight = FUNCTION_SCORING.get('size_weight', 0.3)
        documentation_weight = FUNCTION_SCORING.get('documentation_weight', 0.2)
        features_weight = FUNCTION_SCORING.get('features_weight', 0.3)
        quality_weight = FUNCTION_SCORING.get('quality_weight', 0.2)
        
        # Get thresholds from new config
        size_thresholds = FUNCTION_SCORING.get('size_thresholds', {
            'small': 50, 'medium': 200, 'large': 500
        })
        feature_bonuses = FUNCTION_SCORING.get('feature_bonuses', {})
        
        breakdown = {}

        # Size score (based on tokens)
        token_count = len(func.code_tokens)
        if 10 <= token_count <= size_thresholds.get('small', 50):
            size_score = 25
        elif token_count <= size_thresholds.get('medium', 200):
            size_score = 20
        elif token_count <= size_thresholds.get('large', 500):
            size_score = 10
        elif token_count < 10:
            size_score = 5
        else:
            size_score = 0
        breakdown["size"] = size_score

        # Documentation score
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

        # Language features score (using new config bonuses)
        features_score = 5  # Base score
        if func.clauses > 1:
            features_score += feature_bonuses.get('multiple_clauses', 3)
        if func.has_guards:
            features_score += feature_bonuses.get('has_guards', 5)
        if func.has_patterns:
            features_score += feature_bonuses.get('has_patterns', 3)
        breakdown["features"] = min(20, features_score)

        # Quality indicators
        quality_score = 0
        if func.is_exported:
            quality_score += feature_bonuses.get('is_exported', 2)
        if func.type_spec:
            quality_score += feature_bonuses.get('has_spec', 8)
        if func.name and not func.name.startswith('_'):  # Not private
            quality_score += 3
        if func.docstring and '@spec' in func.docstring:
            quality_score += 2
        breakdown["quality"] = min(15, quality_score)

        # Complexity balance
        if 2 <= func.clauses <= 4:
            complexity_score = 15
        elif func.clauses == 1:
            complexity_score = 10
        elif 5 <= func.clauses <= 8:
            complexity_score = 8
        else:
            complexity_score = 0
        breakdown["complexity"] = complexity_score

        # Calculate weighted total using new config weights
        total_score = (
            breakdown["size"] * size_weight +
            breakdown["documentation"] * documentation_weight +
            breakdown["features"] * features_weight +
            breakdown["quality"] * quality_weight +
            breakdown["complexity"] * (1.0 - size_weight - documentation_weight - features_weight - quality_weight)
        )

        # Ensure score is within bounds
        max_score = FUNCTION_SCORING.get('max_score', 100.0)
        min_score_bound = FUNCTION_SCORING.get('min_score', 5.0)
        
        final_score = max(min_score_bound, min(max_score, total_score))
        
        return final_score, breakdown

    def _save_functions_jsonl(self, functions: List[ErlangFunction]):
        """Save functions to JSONL format (maintains exact output compatibility)."""
        if not functions:
            logger.warning("No functions to save")
            return
            
        output_file = get_output_path(OUTPUT_CONFIG['functions_file'])
        logger.info(f"Saving {len(functions)} functions to {output_file}")

        with open(output_file, 'w', encoding='utf-8') as f:
            for func in functions:
                # Convert to the exact same format as before for 100% compatibility
                func_dict = {
                    "idx": func.idx,
                    "name": func.name,
                    "arity": func.arity,
                    "file_path": func.file_path,
                    "line_start": func.line_start,
                    "line_end": func.line_end,
                    "code": func.code,
                    "code_tokens": func.code_tokens,
                    "docstring": func.docstring,
                    "type_spec": func.type_spec,
                    "variable_positions": func.variable_positions,
                    "dataflow_graph": func.dataflow_graph,
                    "clauses": func.clauses,
                    "has_guards": func.has_guards,
                    "has_patterns": func.has_patterns,
                    "is_exported": func.is_exported,
                    "score": func.score,
                    "score_breakdown": func.score_breakdown,
                    "repo_name": func.repo_name
                }
                f.write(json.dumps(func_dict, ensure_ascii=False) + '\n')

        logger.info(f"✓ Functions saved to {output_file}")

# Maintain the old function name for compatibility
def save_functions_to_corpus(functions: List[ErlangFunction], output_file: Optional[str] = None):
    """Save extracted functions to JSONL corpus format for GraphCodeBERT.
    
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
    """Test function extraction with the new parser format."""
    logging.basicConfig(level=logging.DEBUG)

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

    # Test extraction with new format
    extractor = FunctionExtractor()
    functions = extractor._extract_from_file(test_file, "test_repo", "/tmp")

    print(f"Extracted {len(functions)} functions:")
    for func in functions:
        print(f"  {func.name}/{func.arity} - Score: {func.score:.1f}")
        print(f"    Clauses: {func.clauses}, Guards: {func.has_guards}")
        print(f"    Tokens ({len(func.code_tokens)}): {func.code_tokens}")
        print(f"    Variables: {func.variable_positions}")
        print(f"    DFG: {func.dataflow_graph}")
        print(f"    Docstring: {func.docstring}")
        print()

if __name__ == "__main__":
    main()

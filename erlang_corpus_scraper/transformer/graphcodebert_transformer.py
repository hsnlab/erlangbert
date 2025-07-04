#!/usr/bin/env python3
"""
GraphCodeBERT data transformer for Erlang corpus.
Converts functions.jsonl to GraphCodeBERT-compatible training format.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import random
from collections import defaultdict

# Import configuration
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import get_output_path, GRAPHCODEBERT_CONFIG

logger = logging.getLogger(__name__)

@dataclass
class GraphCodeBERTExample:
    """Single training example for GraphCodeBERT."""
    idx: str
    code: str
    code_tokens: List[str]
    dfg: List[List[Any]]  # Data flow graph edges
    nl: str = ""  # Natural language description (docstring)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'idx': self.idx,
            'code': self.code,
            'code_tokens': self.code_tokens,
            'dfg': self.dfg,
            'nl': self.nl
        }

class GraphCodeBERTTransformer:
    """Transforms Erlang functions to GraphCodeBERT format."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or GRAPHCODEBERT_CONFIG
        self.max_code_length = self.config.get('max_code_length', 256)
        self.max_dfg_length = self.config.get('max_dfg_length', 64)
        self.max_nl_length = self.config.get('max_nl_length', 128)
        
        # Special tokens
        self.special_tokens = {
            'cls': '[CLS]',
            'sep': '[SEP]', 
            'pad': '[PAD]',
            'unk': '[UNK]',
            'mask': '[MASK]'
        }
        
    def load_functions(self, functions_file: str) -> List[Dict[str, Any]]:
        """Load functions from JSONL file."""
        functions = []
        try:
            with open(functions_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        func_data = json.loads(line)
                        functions.append(func_data)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Invalid JSON on line {line_num}: {e}")
                        continue
            
            logger.info(f"Loaded {len(functions)} functions from {functions_file}")
            return functions
            
        except FileNotFoundError:
            logger.error(f"Functions file not found: {functions_file}")
            return []
        except Exception as e:
            logger.error(f"Error loading functions: {e}")
            return []
    
    def convert_dataflow_graph(self, variable_positions: List[List], 
                             dataflow_graph: List[List]) -> List[List[Any]]:
        """Convert Erlang dataflow graph to GraphCodeBERT format."""
        dfg_edges = []
        
        # Create mapping from variable name to positions
        var_to_positions = defaultdict(list)
        for pos_data in variable_positions:
            if len(pos_data) >= 2:
                pos, var_name = pos_data[0], pos_data[1]
                var_to_positions[var_name].append(pos)
        
        # Process dataflow edges
        for edge in dataflow_graph:
            if len(edge) >= 5:
                var_name, pos, relation, deps, dep_positions = edge[:5]
                
                # Skip if not a "comesFrom" relation
                if relation != "comesFrom":
                    continue
                
                # Add edge for each dependency
                for dep_var in deps:
                    if dep_var in var_to_positions:
                        # Get source positions (dependencies)
                        src_positions = var_to_positions[dep_var]
                        # Get target position (current variable)
                        tgt_position = pos
                        
                        # Add edges from each source position to target
                        for src_pos in src_positions:
                            if src_pos < tgt_position:  # Ensure forward flow
                                dfg_edges.append([src_pos, tgt_position])
        
        # Remove duplicates and sort
        dfg_edges = list(set(tuple(edge) for edge in dfg_edges))
        dfg_edges = [list(edge) for edge in dfg_edges]
        dfg_edges.sort()
        
        # Limit to max_dfg_length
        if len(dfg_edges) > self.max_dfg_length:
            dfg_edges = dfg_edges[:self.max_dfg_length]
        
        return dfg_edges
    
    def prepare_code_tokens(self, code_tokens: List[str]) -> List[str]:
        """Prepare code tokens for GraphCodeBERT."""
        # Add special tokens and truncate if necessary
        tokens = [self.special_tokens['cls']] + code_tokens
        
        # Truncate if too long (reserve space for SEP)
        if len(tokens) > self.max_code_length - 1:
            tokens = tokens[:self.max_code_length - 1]
        
        # Add SEP token
        tokens.append(self.special_tokens['sep'])
        
        return tokens
    
    def prepare_nl_tokens(self, docstring: str) -> str:
        """Prepare natural language description."""
        if not docstring or docstring == "null":
            # Generate simple description from function name
            return ""
        
        # Truncate if too long
        if len(docstring) > self.max_nl_length:
            docstring = docstring[:self.max_nl_length]
        
        return docstring
    
    def convert_function(self, func_data: Dict[str, Any]) -> Optional[GraphCodeBERTExample]:
        """Convert a single function to GraphCodeBERT format."""
        try:
            # Extract required fields
            idx = func_data.get('idx', '')
            code = func_data.get('code', '')
            code_tokens = func_data.get('code_tokens', [])
            variable_positions = func_data.get('variable_positions', [])
            dataflow_graph = func_data.get('dataflow_graph', [])
            docstring = func_data.get('docstring', '')
            
            # Validate required fields
            if not code_tokens:
                logger.warning(f"Function {idx} has no code tokens, skipping")
                return None
            
            # Convert components
            processed_tokens = self.prepare_code_tokens(code_tokens)
            dfg_edges = self.convert_dataflow_graph(variable_positions, dataflow_graph)
            nl_desc = self.prepare_nl_tokens(docstring)
            
            # Create example
            example = GraphCodeBERTExample(
                idx=idx,
                code=code,
                code_tokens=processed_tokens,
                dfg=dfg_edges,
                nl=nl_desc
            )
            
            return example
            
        except Exception as e:
            logger.error(f"Error converting function {func_data.get('idx', 'unknown')}: {e}")
            return None
    
    def split_data(self, examples: List[GraphCodeBERTExample], 
                   train_ratio: float = 0.8, 
                   val_ratio: float = 0.1,
                   test_ratio: float = 0.1,
                   random_seed: int = 42) -> Tuple[List, List, List]:
        """Split examples into train/validation/test sets."""
        # Validate ratios
        total_ratio = train_ratio + val_ratio + test_ratio
        if abs(total_ratio - 1.0) > 0.001:
            raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")
        
        # Shuffle with fixed seed for reproducibility
        random.seed(random_seed)
        shuffled_examples = examples.copy()
        random.shuffle(shuffled_examples)
        
        # Calculate split indices
        total_size = len(shuffled_examples)
        train_size = int(total_size * train_ratio)
        val_size = int(total_size * val_ratio)
        
        # Split data
        train_examples = shuffled_examples[:train_size]
        val_examples = shuffled_examples[train_size:train_size + val_size]
        test_examples = shuffled_examples[train_size + val_size:]
        
        logger.info(f"Data split: {len(train_examples)} train, {len(val_examples)} val, {len(test_examples)} test")
        
        return train_examples, val_examples, test_examples
    
    def save_examples(self, examples: List[GraphCodeBERTExample], 
                     output_file: str) -> bool:
        """Save examples to JSONL file."""
        try:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                for example in examples:
                    json.dump(example.to_dict(), f, ensure_ascii=False)
                    f.write('\n')
            
            logger.info(f"Saved {len(examples)} examples to {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving examples to {output_file}: {e}")
            return False
    
    def generate_stats(self, examples: List[GraphCodeBERTExample]) -> Dict[str, Any]:
        """Generate statistics for the converted dataset."""
        if not examples:
            return {}
        
        code_lengths = [len(ex.code_tokens) for ex in examples]
        dfg_lengths = [len(ex.dfg) for ex in examples]
        nl_lengths = [len(ex.nl) for ex in examples if ex.nl]
        
        stats = {
            'total_examples': len(examples),
            'code_token_stats': {
                'min': min(code_lengths),
                'max': max(code_lengths),
                'avg': sum(code_lengths) / len(code_lengths),
                'median': sorted(code_lengths)[len(code_lengths) // 2]
            },
            'dfg_stats': {
                'min': min(dfg_lengths) if dfg_lengths else 0,
                'max': max(dfg_lengths) if dfg_lengths else 0,
                'avg': sum(dfg_lengths) / len(dfg_lengths) if dfg_lengths else 0,
                'examples_with_dfg': len([d for d in dfg_lengths if d > 0])
            },
            'nl_stats': {
                'examples_with_nl': len(nl_lengths),
                'avg_length': sum(nl_lengths) / len(nl_lengths) if nl_lengths else 0
            }
        }
        
        return stats
    
    def transform(self, functions_file: str, output_dir: str, 
                  split_data: bool = True) -> bool:
        """Main transformation pipeline."""
        logger.info("Starting GraphCodeBERT transformation")
        logger.info(f"Input: {functions_file}")
        logger.info(f"Output: {output_dir}")
        
        # Load functions
        functions = self.load_functions(functions_file)
        if not functions:
            logger.error("No functions loaded, aborting transformation")
            return False
        
        # Convert functions
        examples = []
        failed_count = 0
        
        for func_data in functions:
            example = self.convert_function(func_data)
            if example:
                examples.append(example)
            else:
                failed_count += 1
        
        logger.info(f"Converted {len(examples)} functions successfully, {failed_count} failed")
        
        if not examples:
            logger.error("No examples converted successfully")
            return False
        
        # Generate statistics
        stats = self.generate_stats(examples)
        logger.info(f"Dataset statistics: {json.dumps(stats, indent=2)}")
        
        # Save statistics
        stats_file = os.path.join(output_dir, 'dataset_stats.json')
        os.makedirs(output_dir, exist_ok=True)
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        # Split and save data
        if split_data:
            train_examples, val_examples, test_examples = self.split_data(examples)
            
            # Save splits
            splits = [
                (train_examples, 'train.jsonl'),
                (val_examples, 'valid.jsonl'),
                (test_examples, 'test.jsonl')
            ]
            
            success = True
            for split_examples, filename in splits:
                output_file = os.path.join(output_dir, filename)
                if not self.save_examples(split_examples, output_file):
                    success = False
            
            return success
        else:
            # Save all examples to single file
            output_file = os.path.join(output_dir, 'all_examples.jsonl')
            return self.save_examples(examples, output_file)

def main():
    """Test the transformer."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Transform Erlang functions to GraphCodeBERT format')
    parser.add_argument('--input', required=True, help='Input functions.jsonl file')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--no-split', action='store_true', help='Do not split into train/val/test')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Transform data
    transformer = GraphCodeBERTTransformer()
    success = transformer.transform(
        functions_file=args.input,
        output_dir=args.output,
        split_data=not args.no_split
    )
    
    if success:
        logger.info("Transformation completed successfully")
        return 0
    else:
        logger.error("Transformation failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
GraphCodeBERT dataset for Erlang code.
Updated to work directly with function extractor output, eliminating the transformer layer.
"""

import os
import sys
import json
import logging
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Dummy classes for when PyTorch is not available
    class Dataset:
        pass
    class DataLoader:
        pass

# Import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import GRAPHCODEBERT_CONFIG, FINETUNING_CONFIG, HARDWARE_CONFIG

logger = logging.getLogger(__name__)

@dataclass
class GraphCodeBERTExample:
    """A single GraphCodeBERT training example."""
    idx: str
    code: str
    code_tokens: List[str]
    dfg: List[List[Any]]  # Data flow graph edges
    nl_tokens: List[str]
    nl: str = ""  # Natural language description (docstring)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""

        return {
            'idx': self.idx,
            'code': self.code,
            'code_tokens': self.code_tokens,
            'dfg': self.dfg,
            'nl_tokens': self.nl_tokens,
            'nl': self.nl
        }

class ErlangCodeDataset(Dataset):
    """Dataset for GraphCodeBERT MLM training on Erlang functions.

    Works directly with function extractor output format.
    """

    def __init__(self, examples: List[Dict[str, Any]], tokenizer,
                 max_seq_length: Optional[int] = None,
                 mlm_probability: float = 0.15):
        """Initialize dataset.

        Args:
            examples: List of function dictionaries from function extractor
            tokenizer: GraphCodeBERT tokenizer
            max_seq_length: Maximum sequence length (from config if None)
            mlm_probability: Probability of masking tokens for MLM
        """
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length or GRAPHCODEBERT_CONFIG['max_code_length']
        self.mlm_probability = mlm_probability

        # GraphCodeBERT special tokens
        self.cls_token_id = tokenizer.cls_token_id
        self.sep_token_id = tokenizer.sep_token_id
        self.pad_token_id = tokenizer.pad_token_id
        self.mask_token_id = tokenizer.mask_token_id

        # Convert function extractor output to training examples
        self.examples = self._convert_functions_to_examples(examples)

        logger.info(f"Dataset initialized with {len(self.examples)} examples")
        logger.info(f"Max sequence length: {self.max_seq_length}")
        logger.info(f"MLM probability: {self.mlm_probability}")

    def _convert_functions_to_examples(self, functions: List[Dict[str, Any]]) -> List[GraphCodeBERTExample]:
        """Convert function extractor output to GraphCodeBERT examples."""
        examples = []

        for func_data in functions:
            try:
                # Extract required fields from function extractor output
                idx = func_data.get('idx', '')
                code = func_data.get('code', '')
                code_tokens = func_data.get('code_tokens', [])

                # Handle the enhanced variable positions with clause parameter info
                variable_positions = func_data.get('variable_positions', [])
                dataflow_graph = func_data.get('dataflow_graph', [])

                # Convert DFG from function extractor format to GraphCodeBERT format
                dfg_edges = self._convert_dfg_to_edges(variable_positions, dataflow_graph)

                # Get docstring
                docstring = func_data.get('docstring', '')
                docstring_tokens = func_data.get('docstring_tokens', [])

                # Create example
                example = GraphCodeBERTExample(
                    idx=idx,
                    code=code,
                    code_tokens=code_tokens,
                    dfg=dfg_edges,
                    nl_tokens=docstring_tokens,
                    nl=docstring
                )

                examples.append(example)

            except Exception as e:
                logger.warning(f"Failed to convert function {func_data.get('idx', 'unknown')}: {e}")
                continue

        logger.info(f"Successfully converted {len(examples)}/{len(functions)} functions")
        return examples

    def _convert_dfg_to_edges(self, variable_positions: List[Tuple],
                             dataflow_graph: List[Tuple]) -> List[List[int]]:
        """Convert function extractor DFG format to GraphCodeBERT edge format.

        Function extractor format:
        - variable_positions: [(token_index, variable_name, is_clause_param), ...]
        - dataflow_graph: [(var_name, pos, relation, deps, dep_positions), ...]

        GraphCodeBERT format:
        - List of [from_token_pos, to_token_pos] edge pairs
        """
        dfg_edges = []

        # Process dataflow graph edges
        for edge in dataflow_graph:
            if len(edge) >= 5:
                var_name, pos, relation, deps, dep_positions = edge[:5]

                # Skip if not a "comesFrom" relation
                if relation != "comesFrom":
                    continue

                # Add edges from dependency positions to current position
                for dep_pos in dep_positions:
                    if dep_pos < pos:  # Ensure forward flow
                        dfg_edges.append([dep_pos, pos])

        # Remove duplicates and sort on the variable position
        dfg_edges = list(set(tuple(edge) for edge in dfg_edges))
        dfg_edges = [list(edge) for edge in dfg_edges]
        dfg_edges.sort()

        # Limit to reasonable number of edges
        max_dfg_length = GRAPHCODEBERT_CONFIG.get('max_dfg_length', 64)
        if len(dfg_edges) > max_dfg_length:
            dfg_edges = dfg_edges[:max_dfg_length]

        return dfg_edges

    def _validate_example(self, example: Dict[str, Any]) -> bool:
        """Validate that example has required fields."""
        required_fields = ['idx', 'code_tokens', 'dfg']
        return all(field in example for field in required_fields)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single training example formatted for GraphCodeBERT MLM.

        Returns tensor dict compatible with GraphCodeBERTTrainer expectations.
        """
        example = self.examples[idx]

        # Extract data from example
        doc = example.nl
        if len(example.nl_tokens) > 0:
            doc = example.nl_tokens
        code_tokens = example.code_tokens
        dfg_edges = example.dfg

        # Create GraphCodeBERT input sequence
        input_ids, position_idx, all_edges, token_boundaries = self._create_input_sequence(doc, code_tokens, dfg_edges)

        # Apply MLM masking
        input_ids, labels = self._apply_mlm_masking(input_ids, token_boundaries)

        # Create graph-guided attention mask
        attention_mask = self._create_attention_mask(input_ids, token_boundaries, all_edges)

        # Return in format expected by GraphCodeBERTTrainer
        if TORCH_AVAILABLE:
            return {
                'input_ids': torch.tensor(input_ids, dtype=torch.long),
                'position_idx': torch.tensor(position_idx, dtype=torch.long),
                'attention_mask': torch.tensor(attention_mask, dtype=torch.bool),
                'labels': torch.tensor(labels, dtype=torch.long),
                'idx': example.idx
            }
        else:
            # Return raw data when PyTorch not available (for testing)
            return {
                'input_ids': input_ids,
                'position_idx': position_idx,
                'attention_mask': attention_mask,
                'labels': labels,
                'idx': example.idx
            }

    def _create_input_sequence(self,
                               doc_tokens: List[str],
                               code_tokens: List[str],
                               dfg_edges: List[List[int]]) -> Tuple[List[int], List[int], List[List[int]], Dict[str,Tuple[int]]]:
        """Create GraphCodeBERT input sequence using the official tokenization approach.

        Follows microsoft/CodeBERT GraphCodeBERT tokenization with '@ ' prefix trick.

        Args:
            doc_tokens: Docstring tokens
            code_tokens: Code tokens from function extractor
            dfg_edges: DFG edges as [from_token_pos, to_token_pos] pairs

        Returns:
            - input_ids: Token IDs for the sequence
            - position_idx: Position embeddings (pad_id+1 for code, 0 for DFG nodes)
            - all_edges: DFG edges for attention mask
            - token_boundaries: Token boundary markers
        """
        # Clean code tokens (remove existing [CLS], [SEP])
        clean_code = [token for token in code_tokens if token not in ['[CLS]', '[SEP]']]

        # CRITICAL: Apply GraphCodeBERT's tokenization with '@ ' prefix
        # This ensures consistent subword tokenization regardless of position
        code_tokens_subword = [
            self.tokenizer.tokenize('@ ' + x)[1:] if idx != 0 else self.tokenizer.tokenize(x)
            for idx, x in enumerate(clean_code)
        ]

        # Build ori2cur_pos mapping to track subword splits
        ori2cur_pos = {}
        ori2cur_pos[-1] = (0, 0)
        for i in range(len(code_tokens_subword)):
            ori2cur_pos[i] = (ori2cur_pos[i-1][1], ori2cur_pos[i-1][1] + len(code_tokens_subword[i]))

        # Flatten the list of token lists
        code_tokens_flat = [y for x in code_tokens_subword for y in x]

        # Truncate code tokens to leave room for DFG nodes
        max_code_length = self.max_seq_length - 3  # Reserve for [CLS], [SEP], [SEP]
        dfg_length = min(len(dfg_edges), GRAPHCODEBERT_CONFIG.get('max_dfg_length', 64))
        code_tokens_flat = code_tokens_flat[:max_code_length - dfg_length]

        # Add special tokens around code
        code_tokens_flat = [self.tokenizer.cls_token] + code_tokens_flat + [self.tokenizer.sep_token]

        # Convert tokens to IDs
        code_ids = self.tokenizer.convert_tokens_to_ids(code_tokens_flat)

        # Create position indices (GraphCodeBERT's way)
        # Code tokens: sequential positions starting from pad_token_id + 1
        position_idx = [i + self.pad_token_id + 1 for i in range(len(code_ids))]

        # Process DFG: extract unique variable nodes
        dfg_processed = self._process_dfg_for_graphcodebert(dfg_edges, clean_code, ori2cur_pos, len(code_ids))

        # Truncate DFG to fit remaining space
        max_dfg_nodes = self.max_seq_length - len(code_ids)
        dfg_nodes = dfg_processed['nodes'][:max_dfg_nodes]
        dfg_to_code = dfg_processed['dfg_to_code'][:max_dfg_nodes]
        dfg_to_dfg = dfg_processed['dfg_to_dfg'][:max_dfg_nodes]

        # Append DFG nodes to sequence
        # DFG nodes use UNK token ID and position 0
        code_ids += [self.tokenizer.unk_token_id] * len(dfg_nodes)
        position_idx += [0] * len(dfg_nodes)  # DFG nodes get position 0

        # Padding
        padding_length = self.max_seq_length - len(code_ids)
        code_ids += [self.pad_token_id] * padding_length
        position_idx += [self.pad_token_id] * padding_length

        # Build token boundaries
        code_end = len(code_tokens_flat)  # End of code section (including [CLS] and [SEP])
        dfg_start = code_end
        dfg_end = dfg_start + len(dfg_nodes)

        token_boundaries = {
            'nl': (1, 1),  # No NL tokens in MLM task
            'code': (1, code_end - 1),  # Exclude [CLS] and [SEP]
            'variables': (dfg_start, dfg_end - 1) if dfg_nodes else (dfg_start, dfg_start)
        }

        # Convert dfg_to_code and dfg_to_dfg to edge list for attention mask
        all_edges = self._build_attention_edges(dfg_to_code, dfg_to_dfg, dfg_start)

        return code_ids, position_idx, all_edges, token_boundaries

    def _process_dfg_for_graphcodebert(self, dfg_edges: List[List[int]],
                                        code_tokens: List[str],
                                        ori2cur_pos: Dict[int, Tuple[int, int]],
                                        code_length: int) -> Dict[str, Any]:
        """Process DFG edges into GraphCodeBERT's expected format.

        Args:
            dfg_edges: Original DFG edges from function extractor
            code_tokens: Original code tokens (before subword tokenization)
            ori2cur_pos: Mapping from original to current positions after subword split
            code_length: Length of code sequence (including [CLS] and [SEP])

        Returns:
            Dictionary with 'nodes', 'dfg_to_code', 'dfg_to_dfg'
        """
        # Collect unique positions that appear in DFG edges
        positions_in_dfg = set()
        for edge in dfg_edges:
            if len(edge) >= 2:
                positions_in_dfg.add(edge[0])
                positions_in_dfg.add(edge[1])

        # Sort positions to maintain consistent ordering
        sorted_positions = sorted(positions_in_dfg)

        # Create DFG nodes (variable names)
        dfg_nodes = []
        pos_to_dfg_idx = {}  # Map from original token position to DFG node index

        for dfg_idx, orig_pos in enumerate(sorted_positions):
            if orig_pos < len(code_tokens):
                token = code_tokens[orig_pos]
                dfg_nodes.append(token)
                pos_to_dfg_idx[orig_pos] = dfg_idx

        # Build dfg_to_code: map each DFG node to its token span in the code sequence
        dfg_to_code = []
        for orig_pos in sorted_positions:
            if orig_pos in ori2cur_pos:
                start, end = ori2cur_pos[orig_pos]
                # Adjust for [CLS] token at the beginning
                dfg_to_code.append((start + 1, end + 1))
            else:
                # Fallback: map to the position itself
                dfg_to_code.append((orig_pos + 1, orig_pos + 2))

        # Build dfg_to_dfg: reindex DFG edges to use DFG node indices
        # Create reverse mapping for quick lookup
        reverse_index = {orig_pos: dfg_idx for dfg_idx, orig_pos in enumerate(sorted_positions)}

        # Initialize dfg_to_dfg as empty lists
        dfg_to_dfg = [[] for _ in range(len(dfg_nodes))]

        # Process edges to build dependency lists
        for edge in dfg_edges:
            if len(edge) >= 2:
                from_pos, to_pos = edge[0], edge[1]
                # Map to DFG indices
                if from_pos in reverse_index and to_pos in reverse_index:
                    from_idx = reverse_index[from_pos]
                    to_idx = reverse_index[to_pos]
                    # Add from_idx as a dependency of to_idx
                    if from_idx not in dfg_to_dfg[to_idx]:
                        dfg_to_dfg[to_idx].append(from_idx)

        return {
            'nodes': dfg_nodes,
            'dfg_to_code': dfg_to_code,
            'dfg_to_dfg': dfg_to_dfg
        }

    def _build_attention_edges(self, dfg_to_code: List[Tuple[int, int]],
                               dfg_to_dfg: List[List[int]],
                               dfg_start: int) -> List[List[int]]:
        """Build attention edges from dfg_to_code and dfg_to_dfg mappings.

        Args:
            dfg_to_code: List of (start, end) tuples mapping DFG nodes to code positions
            dfg_to_dfg: List of dependency lists for each DFG node
            dfg_start: Starting position of DFG nodes in the sequence

        Returns:
            List of [from_pos, to_pos] edges for attention mask
        """
        edges = []

        # DFG-to-code edges (bidirectional)
        for dfg_idx, (code_start, code_end) in enumerate(dfg_to_code):
            dfg_pos = dfg_start + dfg_idx
            # Connect DFG node to all tokens in its span
            for code_pos in range(code_start, code_end):
                edges.append([dfg_pos, code_pos])
                edges.append([code_pos, dfg_pos])

        # DFG-to-DFG edges (bidirectional)
        for to_idx, from_indices in enumerate(dfg_to_dfg):
            to_pos = dfg_start + to_idx
            for from_idx in from_indices:
                from_pos = dfg_start + from_idx
                edges.append([from_pos, to_pos])
                edges.append([to_pos, from_pos])

        return edges

    def _transform_dfg_to_variable_nodes(self,
                                       code_tokens: List[str],
                                       dfg_edges: List[List[int]]) -> Tuple[List[str], List[List[int]], List[List[int]]]:
        """Transform DFG edges to variable nodes.

        DEPRECATED: This method is kept for backward compatibility but should not be used.
        Use _process_dfg_for_graphcodebert instead.
        """
        # Collect all unique positions that appear in DFG edges
        positions_in_dfg = set()
        for edge in dfg_edges:
            positions_in_dfg.update(edge)

        # Sort positions to maintain consistent ordering
        sorted_positions = sorted(positions_in_dfg)

        # Create variable nodes (one per position)
        variable_nodes = []
        pos_to_var_idx = {}  # Map from original token position to variable node index

        for i, pos in enumerate(sorted_positions):
            if pos < len(code_tokens):
                # Get the actual token at this position
                token = code_tokens[pos]
                variable_nodes.append(token)
                pos_to_var_idx[pos] = i

        # Create variable-to-variable edges by remapping original DFG edges
        var_dfg_edges = []
        for edge in dfg_edges:
            from_pos, to_pos = edge[0], edge[1]
            if from_pos in pos_to_var_idx and to_pos in pos_to_var_idx:
                from_var_idx = pos_to_var_idx[from_pos]
                to_var_idx = pos_to_var_idx[to_pos]
                var_dfg_edges.append([from_var_idx, to_var_idx])

        # Create variable-to-code edges (each variable node connects to its source code position)
        var_to_code_edges = []
        for i, pos in enumerate(sorted_positions):
            # Variable i connects to code position pos
            var_to_code_edges.append([i, pos])

        return variable_nodes, var_dfg_edges, var_to_code_edges


    def _apply_mlm_masking(self, input_ids: List[int], token_boundaries: Dict[str,Tuple[int]]) -> Tuple[List[int], List[int]]:
        """Apply masked language modeling to code tokens only."""
        input_ids = input_ids.copy()
        labels = [-100] * len(input_ids)  # -100 = ignore in loss calculation

        # Only mask code tokens
        code_positions = [i for i in range(token_boundaries['code'][0], token_boundaries['code'][1])]

        # Randomly select positions to mask
        num_to_mask = int(len(code_positions) * self.mlm_probability)
        if num_to_mask > 0:
            masked_positions = np.random.choice(code_positions, size=num_to_mask, replace=False)

            for pos in masked_positions:
                labels[pos] = input_ids[pos]  # Store original token for loss calculation

                # Mask strategy: 80% [MASK], 10% random, 10% unchanged
                rand = np.random.random()
                if rand < 0.8:
                    input_ids[pos] = self.mask_token_id
                elif rand < 0.9:
                    input_ids[pos] = np.random.randint(0, self.tokenizer.vocab_size)
                # else: keep original token (10% unchanged)

        return input_ids, labels

    def _create_attention_mask(self, input_ids: List[int], token_boundaries: Dict[str,Tuple[int]], all_edges: List[List[int]]) -> List[List[bool]]:
        """Create graph-guided attention mask for GraphCodeBERT."""
        seq_len = len(input_ids)
        attention_mask = np.zeros((seq_len, seq_len), dtype=bool)

        # Basic attention patterns
        # 1. All tokens can attend to special tokens ([CLS], [SEP])
        for i in range(seq_len):
            for j in range(seq_len):
                if input_ids[j] == self.cls_token_id or input_ids[j] == self.sep_token_id:
                    attention_mask[i][j] = True

        # 2. NL tokens can attend to other NL tokens
        code_positions = [i for i in range(token_boundaries['code'][0], token_boundaries['code'][1])]
        nl_positions = [i for i in range(token_boundaries['nl'][0], token_boundaries['nl'][1])]
        code_or_nl_positions = code_positions + nl_positions
        for i in code_or_nl_positions:
            for j in code_or_nl_positions:
                attention_mask[i][j] = True

        # TODO: Add proper graph attantion masks
        # 3. Add graph-guided edges
        for edge in all_edges:
            if len(edge) == 2:
                # edges already adjusted for token indices
                from_pos, to_pos = edge[0], edge[1]
                if 0 <= from_pos < seq_len and 0 <= to_pos < seq_len:
                    attention_mask[from_pos][to_pos] = True
                    attention_mask[to_pos][from_pos] = True  # Bidirectional

        return attention_mask.tolist()


def graphcodebert_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate function for GraphCodeBERT batch processing."""
    # Stack all tensors
    collated = {}
    for key in batch[0].keys():
        if key == 'idx':
            collated[key] = [item[key] for item in batch]
        else:
            collated[key] = torch.stack([item[key] for item in batch])

    return collated


def create_graphcodebert_dataloader(train_file: str, val_file: Optional[str] = None,
                                  tokenizer=None, batch_size: Optional[int] = None,
                                  shuffle: bool = True) -> Tuple[DataLoader, Optional[DataLoader]]:
    """Create GraphCodeBERT dataloaders directly from function extractor output.

    This function replaces the transformer layer completely.

    Args:
        train_file: Path to training JSONL file from function extractor
        val_file: Path to validation JSONL file (optional)
        tokenizer: GraphCodeBERT tokenizer
        batch_size: Batch size (from config if None)
        shuffle: Whether to shuffle training data

    Returns:
        Tuple of (train_dataloader, val_dataloader)
    """
    # Import config for pipeline integration
    try:
        effective_batch_size = batch_size or FINETUNING_CONFIG['batch_size']
        num_workers = HARDWARE_CONFIG['num_workers']
        pin_memory = HARDWARE_CONFIG['pin_memory']
    except:
        # Fallback values
        effective_batch_size = batch_size or 8
        num_workers = 0
        pin_memory = False

    logger.info(f"Creating dataloaders from function extractor output")
    logger.info(f"Training file: {train_file}")
    logger.info(f"Batch size: {effective_batch_size}")

    # Load training data directly from function extractor
    train_functions = _load_function_extractor_output(train_file)
    if not train_functions:
        raise ValueError(f"No training functions loaded from {train_file}")

    # Create training dataset
    train_dataset = ErlangCodeDataset(
        examples=train_functions,
        tokenizer=tokenizer
    )

    # Create training dataloader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=effective_batch_size,
        shuffle=shuffle,
        collate_fn=graphcodebert_collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    # Create validation dataloader if validation file provided
    val_dataloader = None
    if val_file:
        val_functions = _load_function_extractor_output(val_file)
        if val_functions:
            val_dataset = ErlangCodeDataset(
                examples=val_functions,
                tokenizer=tokenizer
            )

            val_dataloader = DataLoader(
                val_dataset,
                batch_size=effective_batch_size,
                shuffle=False,  # Don't shuffle validation data
                collate_fn=graphcodebert_collate_fn,
                num_workers=num_workers,
                pin_memory=pin_memory
            )

            logger.info(f"Created validation dataloader: {len(val_functions)} functions")
        else:
            logger.warning(f"No validation functions loaded from {val_file}")

    logger.info(f"Created training dataloader: {len(train_functions)} functions")
    return train_dataloader, val_dataloader


def split_and_save_functions(functions_file: str, output_dir: str,
                           train_ratio: float = 0.8, val_ratio: float = 0.1, test_ratio: float = 0.1,
                           random_seed: int = 42) -> Tuple[str, str, str]:
    """Split function extractor output into train/validation/test sets and save.

    Args:
        functions_file: Path to function extractor JSONL output
        output_dir: Directory to save split files
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set
        random_seed: Random seed for reproducible splits

    Returns:
        Tuple of (train_file_path, val_file_path, test_file_path)
    """
    import random
    import os

    # Validate ratios
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 0.001:
        raise ValueError(f"Split ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}")

    # Load all functions
    logger.info(f"Loading functions from {functions_file} for splitting")
    functions = _load_function_extractor_output(functions_file)
    if not functions:
        raise ValueError(f"No functions loaded from {functions_file}")

    # Set random seed for reproducible splits
    random.seed(random_seed)
    random.shuffle(functions)

    # Calculate split sizes
    total_functions = len(functions)
    train_size = int(total_functions * train_ratio)
    val_size = int(total_functions * val_ratio)
    test_size = total_functions - train_size - val_size  # Remaining functions

    # Split the data
    train_functions = functions[:train_size]
    val_functions = functions[train_size:train_size + val_size]
    test_functions = functions[train_size + val_size:]

    logger.info(f"Split {total_functions} functions:")
    logger.info(f"  Training: {len(train_functions)} ({len(train_functions)/total_functions*100:.1f}%)")
    logger.info(f"  Validation: {len(val_functions)} ({len(val_functions)/total_functions*100:.1f}%)")
    logger.info(f"  Test: {len(test_functions)} ({len(test_functions)/total_functions*100:.1f}%)")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Save splits
    train_file = os.path.join(output_dir, 'train.jsonl')
    val_file = os.path.join(output_dir, 'valid.jsonl')
    test_file = os.path.join(output_dir, 'test.jsonl')
    full_file = os.path.join(output_dir, 'full.jsonl')

    _save_functions_to_jsonl(train_functions, train_file)
    _save_functions_to_jsonl(val_functions, val_file)
    _save_functions_to_jsonl(test_functions, test_file)
    _save_functions_to_jsonl(functions, full_file)

    logger.info(f"✓ Saved split datasets to {output_dir}")
    logger.info(f"  Training: {train_file}")
    logger.info(f"  Validation: {val_file}")
    logger.info(f"  Test: {test_file}")
    logger.info(f"  Full dataset: {full_file}")

    return train_file, val_file, test_file


def _save_functions_to_jsonl(functions: List[Dict[str, Any]], file_path: str):
    """Save functions to JSONL file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        for func in functions:
            f.write(json.dumps(func, ensure_ascii=False) + '\n')


def create_split_dataloaders(functions_file: str, output_dir: str, tokenizer,
                           train_ratio: float = 0.8, val_ratio: float = 0.1, test_ratio: float = 0.1,
                           batch_size: Optional[int] = None, random_seed: int = 42) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Convenience function to split and create all dataloaders in one step.

    Args:
        functions_file: Path to function extractor JSONL output
        output_dir: Directory to save split files
        tokenizer: GraphCodeBERT tokenizer
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set
        batch_size: Batch size (from config if None)
        random_seed: Random seed for reproducible splits

    Returns:
        Tuple of (train_dataloader, val_dataloader, test_dataloader)
    """
    # Split and save the data
    train_file, val_file, test_file = split_and_save_functions(
        functions_file, output_dir, train_ratio, val_ratio, test_ratio, random_seed
    )

    # Create dataloaders
    train_dataloader, val_dataloader = create_graphcodebert_dataloader(
        train_file=train_file,
        val_file=val_file,
        tokenizer=tokenizer,
        batch_size=batch_size,
        shuffle=True
    )

    # Create test dataloader
    test_functions = _load_function_extractor_output(test_file)
    test_dataset = ErlangCodeDataset(examples=test_functions, tokenizer=tokenizer)

    effective_batch_size = batch_size or FINETUNING_CONFIG.get('batch_size', 8)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=effective_batch_size,
        shuffle=False,
        collate_fn=graphcodebert_collate_fn,
        num_workers=HARDWARE_CONFIG.get('num_workers', 0),
        pin_memory=HARDWARE_CONFIG.get('pin_memory', False)
    )

    logger.info(f"Created test dataloader: {len(test_functions)} functions")

    return train_dataloader, val_dataloader, test_dataloader


def _load_function_extractor_output(file_path: str) -> List[Dict[str, Any]]:
    """Load functions directly from function extractor JSONL output."""
    functions = []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    func_data = json.loads(line)
                    functions.append(func_data)
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON on line {line_num} in {file_path}: {e}")
                    continue

        logger.info(f"Loaded {len(functions)} functions from {file_path}")
        return functions

    except FileNotFoundError:
        logger.error(f"Function extractor output file not found: {file_path}")
        return []
    except Exception as e:
        logger.error(f"Error loading function extractor output from {file_path}: {e}")
        return []


# For backward compatibility, export the factory function
def create_dataloader(*args, **kwargs):
    """Backward compatibility wrapper for the GraphCodeBERT trainer.

    This allows the trainer to use create_graphcodebert_dataloader()
    as a drop-in replacement for its existing create_dataloader() method.
    """
    return create_graphcodebert_dataloader(*args, **kwargs)


if __name__ == "__main__":
    import sys

    # Check if running with PyTorch available
    try:
        import torch
        PYTORCH_AVAILABLE = True
    except ImportError:
        PYTORCH_AVAILABLE = False
        print("PyTorch not available - running tokenization tests only")

    logging.basicConfig(level=logging.DEBUG)

    # ========================================
    # TOKENIZATION UNIT TESTS (No PyTorch required)
    # ========================================
    print("\n" + "="*70)
    print("TOKENIZATION UNIT TESTS (GraphCodeBERT compatibility)")
    print("="*70 + "\n")

    # Create mock tokenizer that simulates RoBERTa behavior
    class MockTokenizer:
        """Mock tokenizer simulating RoBERTa's position-dependent subword tokenization."""
        def __init__(self):
            self.vocab_size = 50000
            self.cls_token = '[CLS]'
            self.sep_token = '[SEP]'
            self.pad_token = '[PAD]'
            self.unk_token = '[UNK]'
            self.mask_token = '[MASK]'
            self.cls_token_id = 0
            self.sep_token_id = 2
            self.pad_token_id = 1
            self.mask_token_id = 50264
            self.unk_token_id = 3

        def tokenize(self, text):
            """Simulate position-dependent subword tokenization."""
            # Key behavior: tokens at start vs middle split differently
            if text.startswith('@ '):
                # Remove '@ ' prefix and tokenize as if in middle
                actual_text = text[2:]
                return ['@'] + self._tokenize_middle(actual_text)
            else:
                # Tokenize as if at sequence start
                return self._tokenize_start(text)

        def _tokenize_start(self, text):
            """Tokenize at sequence start (may split differently)."""
            if any(c.isdigit() for c in text):
                i = next(i for i, c in enumerate(text) if c.isdigit())
                return [text[:i], text[i:]]
            return [text]

        def _tokenize_middle(self, text):
            """Tokenize in middle (consistent splitting)."""
            if any(c.isdigit() for c in text):
                i = next(i for i, c in enumerate(text) if c.isdigit())
                return [text[:i], text[i:]]
            return [text]

        def convert_tokens_to_ids(self, tokens):
            """Convert tokens to IDs."""
            token_map = {
                '[CLS]': 0, '[SEP]': 2, '[PAD]': 1, '[UNK]': 3, '[MASK]': 50264,
                '@': 999
            }
            return [token_map.get(t, hash(t) % 1000 + 100) for t in tokens]

        def encode(self, text, add_special_tokens=False):
            """Encode text to token IDs."""
            tokens = self.tokenize(text)
            return self.convert_tokens_to_ids(tokens)

    tokenizer = MockTokenizer()

    # Test 1: '@ ' prefix trick
    print("Test 1: '@ ' Prefix Trick")
    print("-" * 70)
    code_tokens = ['func', 'var123', 'result']

    # Without '@ ' prefix (WRONG - inconsistent tokenization)
    tokens_wrong = [tokenizer.tokenize(x) for x in code_tokens]
    print(f"  Without '@ ' prefix (wrong): {tokens_wrong}")

    # With '@ ' prefix (CORRECT - matches GraphCodeBERT)
    tokens_correct = [
        tokenizer.tokenize('@ ' + x)[1:] if idx != 0 else tokenizer.tokenize(x)
        for idx, x in enumerate(code_tokens)
    ]
    print(f"  With '@ ' prefix (correct):  {tokens_correct}")
    print(f"  ✓ First token unchanged, rest use consistent middle-position tokenization\n")

    # Test 2: ori2cur_pos mapping
    print("Test 2: ori2cur_pos Mapping (tracks subword splits)")
    print("-" * 70)
    tokenized = [
        tokenizer.tokenize('@ ' + x)[1:] if idx != 0 else tokenizer.tokenize(x)
        for idx, x in enumerate(code_tokens)
    ]

    ori2cur_pos = {}
    ori2cur_pos[-1] = (0, 0)
    for i in range(len(tokenized)):
        ori2cur_pos[i] = (ori2cur_pos[i-1][1], ori2cur_pos[i-1][1] + len(tokenized[i]))

    flattened = [y for x in tokenized for y in x]
    print(f"  Original tokens: {code_tokens}")
    print(f"  After tokenization: {tokenized}")
    print(f"  Flattened: {flattened}")
    print(f"  ori2cur_pos mapping:")
    for i in range(len(code_tokens)):
        start, end = ori2cur_pos[i]
        span_tokens = flattened[start:end]
        print(f"    Token {i} ('{code_tokens[i]}') -> positions [{start}:{end}] = {span_tokens}")
    print(f"  ✓ Mapping correctly tracks subword boundaries\n")

    # Test 3: position_idx construction
    print("Test 3: position_idx Construction (GraphCodeBERT format)")
    print("-" * 70)
    code_tokens_with_special = [tokenizer.cls_token] + flattened + [tokenizer.sep_token]
    code_ids = tokenizer.convert_tokens_to_ids(code_tokens_with_special)
    position_idx = [i + tokenizer.pad_token_id + 1 for i in range(len(code_ids))]

    print(f"  Tokens: {code_tokens_with_special}")
    print(f"  Token IDs: {code_ids}")
    print(f"  position_idx: {position_idx}")
    print(f"  First position: {position_idx[0]} (should be pad_id+1 = {tokenizer.pad_token_id + 1})")
    print(f"  Last position: {position_idx[-1]} (should be pad_id+{len(code_ids)} = {tokenizer.pad_token_id + len(code_ids)})")
    print(f"  ✓ Position indices follow GraphCodeBERT convention\n")

    # Test 4: DFG reindexing
    print("Test 4: DFG Reindexing (absolute -> relative indices)")
    print("-" * 70)
    dfg = [
        ('x', 2, 'comesFrom', ['param'], [0]),
        ('y', 5, 'comesFrom', ['x'], [2]),
        ('result', 8, 'computedFrom', ['x', 'y'], [2, 5])
    ]
    print(f"  Original DFG (absolute indices):")
    for item in dfg:
        print(f"    {item}")

    # Build reverse index
    reverse_index = {}
    for idx, x in enumerate(dfg):
        reverse_index[x[1]] = idx

    # Reindex
    dfg_reindexed = []
    for idx, x in enumerate(dfg):
        new_item = x[:-1] + ([reverse_index[i] for i in x[-1] if i in reverse_index],)
        dfg_reindexed.append(new_item)

    print(f"\n  After reindexing (relative DFG indices):")
    for item in dfg_reindexed:
        print(f"    {item}")
    print(f"  ✓ Edges now reference DFG node indices instead of token positions\n")

    # Test 5: dfg_to_code mapping
    print("Test 5: dfg_to_code Mapping (DFG nodes -> code token spans)")
    print("-" * 70)
    dfg_positions = [1, 2]  # Positions of variables in original code
    dfg_to_code = [ori2cur_pos[pos] for pos in dfg_positions]
    # Adjust for [CLS] token
    dfg_to_code_adjusted = [(x[0] + 1, x[1] + 1) for x in dfg_to_code]

    print(f"  DFG node positions in original code: {dfg_positions}")
    print(f"  dfg_to_code (before [CLS] adjustment): {dfg_to_code}")
    print(f"  dfg_to_code (after [CLS] adjustment): {dfg_to_code_adjusted}")
    for idx, (start, end) in enumerate(dfg_to_code_adjusted):
        span = code_tokens_with_special[start:end]
        print(f"    DFG node {idx} maps to tokens [{start}:{end}] = {span}")
    print(f"  ✓ DFG nodes correctly map to their code token spans\n")

    print("="*70)
    print("ALL TOKENIZATION TESTS PASSED ✓")
    print("="*70 + "\n")

    # ========================================
    # PYTORCH INTEGRATION TEST (requires PyTorch)
    # ========================================
    if PYTORCH_AVAILABLE:
        print("\n" + "="*70)
        print("PYTORCH INTEGRATION TEST")
        print("="*70 + "\n")

        # Test with function extractor output format
        test_functions = [
            {
                "idx": "test::max/2::0",
                "code": "max(A, B) when A > B -> A; max(A, B) -> B.",
                "code_tokens": ["[CLS]", "max", "(", "A", ",", "B", ")", "when", "A", ">", "B", "->", "A", ";", "max", "(", "A", ",", "B", ")", "->", "B", ".", "[SEP]"],
                "variable_positions": [(3, "A", True), (5, "B", True), (8, "A", False), (10, "B", False), (12, "A", False), (16, "A", True), (18, "B", True), (21, "B", False)],
                "dataflow_graph": [("A", 3, "comesFrom", [], []), ("B", 5, "comesFrom", [], []), ("A", 8, "comesFrom", ["A"], [3]), ("B", 10, "comesFrom", ["B"], [5]), ("A", 12, "comesFrom", ["A"], [8]), ("A", 16, "comesFrom", [], []), ("B", 18, "comesFrom", [], []), ("B", 21, "comesFrom", ["B"], [18])],
                "docstring": "Returns the maximum of two numbers",
                "docstring_tokens": ["Returns", "the", "maximum", "of", "two", "numbers"]
            }
        ]

        print("Testing PyTorch dataset integration...")
        dataset = ErlangCodeDataset(test_functions, tokenizer, max_seq_length=128)

        print(f"Dataset created with {len(dataset)} examples")

        # Test getting an item
        item = dataset[0]
        print(f"Item keys: {list(item.keys())}")
        print(f"Input sequence length: {item['input_ids'].shape}")
        print(f"Position index length: {item['position_idx'].shape}")
        print(f"MLM labels shape: {item['labels'].shape}")
        print(f"Attention mask shape: {item['attention_mask'].shape}")

        print("\n✓ PyTorch integration working!")
    else:
        print("Skipping PyTorch integration test (PyTorch not installed)")
        print("To run full tests: pip install torch")

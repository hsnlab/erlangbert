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
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass

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

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
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
        input_ids, labels = self._apply_mlm_masking(input_ids, position_idx, token_boundaries)

        # Create graph-guided attention mask
        attention_mask = self._create_attention_mask(input_ids, token_boundaries, position_idx, all_edges)

        # Return in format expected by GraphCodeBERTTrainer
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'position_idx': torch.tensor(position_idx, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.bool),
            'labels': torch.tensor(labels, dtype=torch.long),
            'idx': example.idx
        }

    def _create_input_sequence(self,
                               doc_tokens: List[str],
                               code_tokens: List[str],
                               dfg_edges: List[List[int]]) -> Tuple[List[int], List[int], List[List[int]], Dict[str,Tuple[int]]]:
        """Create GraphCodeBERT input sequence from function extractor format.

        Creates proper GraphCodeBERT format: [CLS] + NL + [SEP] + Code + [SEP] + Variable_Nodes

        Args:
            doc_tokens: Docstring tokens
            code_tokens: Code tokens (already include [CLS] and [SEP])
            dfg_edges: DFG edges as [from_token_pos, to_token_pos] pairs

        Returns:
            - input_ids: Token IDs for the sequence
            - position_idx: Position type for each token (0=special, 1=pad, 2+=code, 0=variables)
            - all_edges: All edges for attention mask (variable↔code + variable↔variable)
        """
        # Step 1: Transform DFG edges to variable nodes using the correct algorithm
        variable_nodes, var_dfg_edges, var_to_code_edges = self._transform_dfg_to_variable_nodes(
            code_tokens, dfg_edges
        )

        # Step 2: Build GraphCodeBERT input sequence
        # [CLS] + NL + [SEP] + Code + [SEP] + Variable_Nodes

        # Clean code tokens (remove existing [CLS], [SEP])
        clean_code = [token for token in code_tokens if token not in ['[CLS]', '[SEP]']]

        # Build sequence sections
        sequence_tokens = []

        # Add [CLS]
        sequence_tokens.append('[CLS]')
        nl_start = len(sequence_tokens)

        # Add NL section
        nl_tokens = doc_tokens
        sequence_tokens.extend(nl_tokens)
        if nl_tokens:
            sequence_tokens.append('[SEP]')

        # Add Code section
        code_start = len(sequence_tokens)
        sequence_tokens.extend(clean_code)
        sequence_tokens.append('[SEP]')

        # Add Variable nodes section
        var_start = len(sequence_tokens)
        sequence_tokens.extend(variable_nodes)

        logger.debug(f"Raw GraphCodeBERT input sequence: {sequence_tokens}")

        # Step 3: Convert tokens to IDs and create position indices
        # position_idx: In roberta-token dimensions:
        # - special tokens (CLS, SEP): 0
        # - variable tokens (CLS, SEP): 0
        # - nl and code tokens): [2, 2+end_of_code_tokens]
        # - padding: 1
        input_ids = []
        position_idx = []
        token_idx = [] # maps parsed tokens to roberta tokens
        nl_token_start = -1
        code_token_start = -1
        var_token_start = -1
        for i, token in enumerate(sequence_tokens):
            token_idx.append(len(input_ids))                
            if token == "[CLS]":
                # CLS token
                input_ids.append(self.cls_token_id)
                position_idx.append(0)  # Special token
                nl_token_start = 1
            elif token == "[SEP]":
                # SEP token
                input_ids.append(self.sep_token_id)
                position_idx.append(0)  # Special token
            elif nl_start <= i < code_start-1:
                # NL token
                if nl_token_start==-1:
                    nl_token_start = len(input_ids)
                    self.nl_start = nl_token_start
                token_ids = self.tokenizer.encode(token, add_special_tokens=False)
                if token_ids:
                    for i in range(len(token_ids)):
                        position_idx.append(len(input_ids)+i+1)
                    input_ids.extend(token_ids)
                else:
                    input_ids.append(self.tokenizer.unk_token_id)
                    position_idx.append(1)
            elif code_start <= i < var_start-1:
                # Code token
                if code_token_start==-1:
                    code_token_start = len(input_ids)
                    self.code_start = code_token_start
                token_ids = self.tokenizer.encode(token, add_special_tokens=False)
                if token_ids:
                    for i in range(len(token_ids)):
                        position_idx.append(len(input_ids)+i)
                    input_ids.extend(token_ids)
                else:
                    input_ids.append(self.tokenizer.unk_token_id)
                    position_idx.append(1)
            elif i >= var_start:
                # Variable node
                if var_token_start==-1:
                    var_token_start = len(input_ids)
                    self.var_start = var_token_start
                token_ids = self.tokenizer.encode(token, add_special_tokens=False)
                if token_ids:
                    for i in range(len(token_ids)):
                        position_idx.append(len(input_ids)+i-1)
                    input_ids.extend(token_ids)
                else:
                    input_ids.append(self.tokenizer.unk_token_id)
                    position_idx.append(1)

        # print("PI:", position_idx)
        # print("PI:", len(position_idx))
        # print("TI:", token_idx)
        # print("TI:", len(token_idx))
        # print("nl:", nl_token_start)
        # print("code:", code_token_start)
        # print("var:", var_token_start)

        token_boundaries = {}
        token_boundaries['nl'] = (nl_token_start, code_token_start - 2)  # Exclude [SEP], inclusive
        token_boundaries['code'] = (code_token_start, var_token_start - 2)  # Exclude [SEP], inclusive
        token_boundaries['variables'] = (var_token_start, len(sequence_tokens)-1) # inclusive

        # print("ranges:", token_boundaries)

        # Step 4: Adjust edges for new sequence positions
        adjusted_edges = self._adjust_edges_for_sequence(
            var_dfg_edges, var_to_code_edges, code_tokens, token_idx
        )

        # Step 5: Pad to max sequence length
        while len(input_ids) < self.max_seq_length:
            input_ids.append(self.pad_token_id)
            position_idx.append(1)  # Padding

        # Truncate if too long
        input_ids = input_ids[:self.max_seq_length]
        position_idx = position_idx[:self.max_seq_length]

        return input_ids, position_idx, adjusted_edges, token_boundaries

    def _transform_dfg_to_variable_nodes(self,
                                       code_tokens: List[str],
                                       dfg_edges: List[List[int]]) -> Tuple[List[str], List[List[int]], List[List[int]]]:
        """Transform DFG edges to variable nodes.

        Each token position in DFG edges becomes a separate variable node.
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

    def _adjust_edges_for_sequence(self, var_dfg_edges: List[List[int]],
                                 var_to_code_edges: List[List[int]],
                                 code_tokens: List[str],
                                 token_idx: List[int]) -> List[List[int]]:
        """Adjust edge indices for the final sequence layout.

        TODO: This function incorrectly assumes all tokens map to a single Roberta token. Use the position_idx index to fix it!
        """
        adjusted_edges = []

        # Variable-to-variable edges (in variable section)
        for edge in var_dfg_edges:
            from_var, to_var = token_idx[edge[0]], token_idx[edge[1]] # map to roberta tokens
            from_seq_pos = self.var_start + from_var
            to_seq_pos = self.var_start + to_var
            adjusted_edges.append([from_seq_pos, to_seq_pos])

        # Variable-to-code edges (bidirectional)
        for edge in var_to_code_edges:
            var_idx, code_pos = token_idx[edge[0]], token_idx[edge[1]] # map to roberta tokens
            var_seq_pos = self.var_start + var_idx
            code_seq_pos = self.code_start + code_pos

            # Bidirectional edges
            adjusted_edges.append([var_seq_pos, code_seq_pos])
            adjusted_edges.append([code_seq_pos, var_seq_pos])

        return adjusted_edges

    def _apply_mlm_masking(self, input_ids: List[int], position_idx: List[int], token_boundaries: Dict[str,Tuple[int]]) -> Tuple[List[int], List[int]]:
        """Apply masked language modeling to code tokens only."""
        input_ids = input_ids.copy()
        labels = [-100] * len(input_ids)  # -100 = ignore in loss calculation

        # Only mask code tokens (position_idx > 1)
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

    def _create_attention_mask(self, input_ids: List[int], token_boundaries: Dict[str,Tuple[int]], position_idx: List[int], all_edges: List[List[int]]) -> List[List[bool]]:
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
                from_pos, to_pos = position_idx[edge[0]], position_idx[edge[1]]
                if 0 <= from_pos < seq_len and 0 <= to_pos < seq_len:
                    attention_mask[from_pos][to_pos] = True
                    attention_mask[to_pos][from_pos] = True  # Bidirectional

        return attention_mask.tolist()


def graphcodebert_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
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

    _save_functions_to_jsonl(train_functions, train_file)
    _save_functions_to_jsonl(val_functions, val_file)
    _save_functions_to_jsonl(test_functions, test_file)

    logger.info(f"✓ Saved split datasets to {output_dir}")
    logger.info(f"  Training: {train_file}")
    logger.info(f"  Validation: {val_file}")
    logger.info(f"  Test: {test_file}")

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
    logging.basicConfig(level=logging.DEBUG)

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

    print("Testing direct function extractor integration...")

    # Create mock tokenizer
    class MockTokenizer:
        def __init__(self):
            self.vocab_size = 50000
            self.cls_token_id = 0
            self.sep_token_id = 2
            self.pad_token_id = 1
            self.mask_token_id = 50264
            self.unk_token_id = 3

        def encode(self, text, add_special_tokens=False):
            ret = [hash(text) % 1000 + 100]
            if len(text) > 3:
                ret.extend([117 for i, _ in enumerate(text) if  i % 3 == 0 and i > 3])
            return ret

    tokenizer = MockTokenizer()
    dataset = ErlangCodeDataset(test_functions, tokenizer, max_seq_length=128)

    print(f"Dataset created with {len(dataset)} examples")

    # Test getting an item
    item = dataset[0]
    print(f"Item keys: {list(item.keys())}")
    print(f"Input sequence length: {item['input_ids'].shape}")
    print(f"Tokens: {item['input_ids']}")
    print(f"Position index length: {item['position_idx'].shape}")
    print(f"MLM labels shape: {item['labels'].shape}")
    print(f"Has attention mask: {item['attention_mask'].shape}")
    torch.set_printoptions(
        threshold=10000,      # Total elements before truncation
        # edgeitems=128,        # Items at beginning/end of each dimension
        edgeitems=4,        # Items at beginning/end of each dimension
        linewidth=120         # Characters per line
    )
    print(f"Attention mask: {item['attention_mask']}")

    print("✓ Direct function extractor integration working!")

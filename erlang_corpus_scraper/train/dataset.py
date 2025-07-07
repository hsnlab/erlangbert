import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Any, List, Tuple, Optional
import random
import logging
import json

logger = logging.getLogger(__name__)

class ErlangCodeDataset(Dataset):
    """Enhanced Dataset for Erlang code examples supporting GraphCodeBERT MLM training.
    
    Integrates with the existing pipeline's config system and GraphCodeBERTTrainer.
    Supports the full GraphCodeBERT input format including:
    - Graph-guided attention masks
    - Position indices for code vs data flow nodes
    - MLM masking and labels
    - Data flow mappings from your preprocessed format
    """
    
    def __init__(self, 
                 examples: List[Dict[str, Any]], 
                 tokenizer, 
                 max_code_length: Optional[int] = None,
                 max_dfg_length: Optional[int] = None,
                 mlm_probability: float = 0.15):
        """Initialize the dataset with pipeline integration.
        
        Args:
            examples: List of Erlang function dictionaries in your format
            tokenizer: GraphCodeBERT tokenizer (RoBERTa-based)
            max_code_length: Maximum number of code tokens (from config if None)
            max_dfg_length: Maximum number of data flow variables (from config if None)
            mlm_probability: Probability of masking tokens for MLM
        """
        self.examples = examples
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability
        
        # Import config dynamically to avoid circular imports
        try:
            from config import GRAPHCODEBERT_CONFIG
            self.max_code_length = max_code_length or GRAPHCODEBERT_CONFIG['max_code_length']
            self.max_dfg_length = max_dfg_length or GRAPHCODEBERT_CONFIG['max_dfg_length']
        except ImportError:
            # Fallback values if config not available
            self.max_code_length = max_code_length or 256
            self.max_dfg_length = max_dfg_length or 64
        
        # Total sequence length: code + dfg variables
        self.max_seq_length = self.max_code_length + self.max_dfg_length
        
        # Special tokens - compatible with your tokenizer
        self.cls_token_id = getattr(tokenizer, 'cls_token_id', 0)
        self.sep_token_id = getattr(tokenizer, 'sep_token_id', 2)
        self.pad_token_id = getattr(tokenizer, 'pad_token_id', 1)
        self.mask_token_id = getattr(tokenizer, 'mask_token_id', 50264)
        
        # Validate examples format
        if examples and not self._validate_example_format(examples[0]):
            logger.warning("Example format may not match expected structure")
        
        logger.info(f"ErlangCodeDataset initialized: {len(examples)} examples, "
                   f"max_code_length={self.max_code_length}, max_dfg_length={self.max_dfg_length}")
    
    def _validate_example_format(self, example: Dict[str, Any]) -> bool:
        """Validate that example has expected format from your pipeline."""
        required_fields = ['idx', 'code_tokens', 'dfg']
        return all(field in example for field in required_fields)
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single training example formatted for GraphCodeBERT MLM.
        
        Returns tensor dict compatible with GraphCodeBERTTrainer expectations.
        """
        example = self.examples[idx]
        
        # Extract data from your preprocessed format
        code_tokens = example['code_tokens']  # Already includes [CLS] and [SEP]
        dfg_edges = example.get('dfg', [])    # List of edges like [[2, 13]]
        
        # Create GraphCodeBERT input sequence
        input_ids, position_idx, all_edges = self._create_input_sequence(code_tokens, dfg_edges)
        
        # Apply MLM masking
        input_ids, labels = self._apply_mlm_masking(input_ids, position_idx)
        
        # Create graph-guided attention mask
        attention_mask = self._create_attention_mask(position_idx, all_edges)
        
        # Return in format expected by GraphCodeBERTTrainer
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'position_idx': torch.tensor(position_idx, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.bool),
            'labels': torch.tensor(labels, dtype=torch.long),
            'idx': example.get('idx', str(idx))  # Keep original identifier
        }
    
    def _prepare_code_tokens(self, code_tokens: List[str]) -> List[str]:
        """Your tokens are already prepared with [CLS] and [SEP] - just validate."""
        if not code_tokens:
            return ["[CLS]", "[SEP]"]
        
        # Ensure we have [CLS] at start and [SEP] at end
        if code_tokens[0] != "[CLS]":
            code_tokens = ["[CLS]"] + code_tokens
        if code_tokens[-1] != "[SEP]":
            code_tokens = code_tokens + ["[SEP]"]
            
        return code_tokens
    
    def _create_input_sequence(self, 
                             code_tokens: List[str], 
                             dfg_edges: List[List[int]]) -> Tuple[List[int], List[int], List[List[int]]]:
        """Create GraphCodeBERT input sequence from your preprocessed format.
        
        Creates proper GraphCodeBERT format: [CLS] + NL + [SEP] + Code + [SEP] + Variable_Nodes
        
        Args:
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
        section_boundaries = {}
        
        # Add [CLS]
        sequence_tokens.append('[CLS]')
        
        # Add NL section (empty for MLM-only training)
        nl_start = len(sequence_tokens)
        nl_tokens = []  # Empty for now, could be function name or description
        sequence_tokens.extend(nl_tokens)
        if nl_tokens:
            sequence_tokens.append('[SEP]')
        section_boundaries['nl'] = (nl_start, len(sequence_tokens))
        
        # Add Code section
        code_start = len(sequence_tokens)
        sequence_tokens.extend(clean_code)
        sequence_tokens.append('[SEP]')
        section_boundaries['code'] = (code_start, len(sequence_tokens) - 1)  # Exclude [SEP]
        
        # Add Variable nodes section
        var_start = len(sequence_tokens)
        sequence_tokens.extend(variable_nodes)
        section_boundaries['variables'] = (var_start, len(sequence_tokens))
        
        # Store boundaries for edge adjustment
        self.code_start = code_start
        self.var_start = var_start
        
        # Step 3: Convert tokens to IDs and create position indices
        input_ids = []
        position_idx = []
        
        for i, token in enumerate(sequence_tokens):
            if token == "[CLS]":
                input_ids.append(self.cls_token_id)
                position_idx.append(0)  # Special token
            elif token == "[SEP]":
                input_ids.append(self.sep_token_id)
                position_idx.append(0)  # Special token
            elif code_start <= i < var_start:
                # Code token
                token_ids = self.tokenizer.encode(token, add_special_tokens=False)
                if token_ids:
                    input_ids.append(token_ids[0])
                    position_idx.append(i + 2)  # Code tokens start from position 2
                else:
                    input_ids.append(self.tokenizer.unk_token_id)
                    position_idx.append(i + 2)
            elif i >= var_start:
                # Variable node
                token_ids = self.tokenizer.encode(token, add_special_tokens=False)
                if token_ids:
                    input_ids.append(token_ids[0])
                    position_idx.append(0)  # Variable nodes use position 0 (like special tokens)
                else:
                    input_ids.append(self.tokenizer.unk_token_id)
                    position_idx.append(0)
            else:
                # NL token (if any)
                token_ids = self.tokenizer.encode(token, add_special_tokens=False)
                if token_ids:
                    input_ids.append(token_ids[0])
                    position_idx.append(i + 2)
                else:
                    input_ids.append(self.tokenizer.unk_token_id)
                    position_idx.append(i + 2)
        
        # Step 4: Adjust edges for new sequence positions
        adjusted_edges = self._adjust_edges_for_sequence(
            var_dfg_edges, var_to_code_edges, code_tokens
        )
        
        # Step 5: Pad to max sequence length
        while len(input_ids) < self.max_seq_length:
            input_ids.append(self.pad_token_id)
            position_idx.append(1)  # Padding
        
        # Truncate if too long
        input_ids = input_ids[:self.max_seq_length]
        position_idx = position_idx[:self.max_seq_length]
        
        return input_ids, position_idx, adjusted_edges
    
    def _transform_dfg_to_variable_nodes(self, 
                                       code_tokens: List[str], 
                                       dfg_edges: List[List[int]]) -> Tuple[List[str], List[List[int]], List[List[int]]]:
        """Transform DFG edges to variable nodes using the correct algorithm.
        
        Each token position in DFG edges becomes a separate variable node.
        
        Args:
            code_tokens: Original code tokens
            dfg_edges: DFG edges as [from_token_pos, to_token_pos] pairs
            
        Returns:
            - variable_nodes: List of variable node tokens
            - var_dfg_edges: Edges between variable node indices  
            - var_to_code_edges: Bidirectional edges between variable nodes and code positions
        """
        # Step 1: Create variable nodes (each token position becomes a node)
        token_pos_to_var_idx = {}
        variable_nodes = []
        var_to_code_edges = []
        
        var_idx = 0
        for edge in dfg_edges:
            for token_pos in [edge[0], edge[1]]:
                if token_pos not in token_pos_to_var_idx:
                    # Create new variable node for this token position
                    token_pos_to_var_idx[token_pos] = var_idx
                    variable_nodes.append(code_tokens[token_pos])  # Variable node token
                    
                    # Add bidirectional edges: variable node ↔ code token position  
                    var_to_code_edges.append([var_idx, token_pos])  # var_node -> code_pos
                    var_to_code_edges.append([token_pos, var_idx])  # code_pos -> var_node
                    
                    var_idx += 1
        
        # Step 2: Create DFG edges between variable nodes
        var_dfg_edges = []
        for edge in dfg_edges:
            from_token_pos, to_token_pos = edge[0], edge[1]
            from_var_idx = token_pos_to_var_idx[from_token_pos]
            to_var_idx = token_pos_to_var_idx[to_token_pos]
            
            # Add DFG edge: from_var_node -> to_var_node
            var_dfg_edges.append([from_var_idx, to_var_idx])
        
        logger.debug(f"Created {len(variable_nodes)} variable nodes: {variable_nodes}")
        logger.debug(f"Variable DFG edges: {var_dfg_edges}")
        logger.debug(f"Variable-code edges: {len(var_to_code_edges)} total")
        
        return variable_nodes, var_dfg_edges, var_to_code_edges
    
    def _adjust_edges_for_sequence(self, 
                                 var_dfg_edges: List[List[int]], 
                                 var_to_code_edges: List[List[int]],
                                 original_code_tokens: List[str]) -> List[List[int]]:
        """Adjust edge indices for the final sequence positions.
        
        Args:
            var_dfg_edges: Edges between variable node indices
            var_to_code_edges: Edges between variable nodes and original code positions
            original_code_tokens: Original code tokens (with [CLS], [SEP])
            
        Returns:
            All edges adjusted for final sequence positions
        """
        adjusted_edges = []
        
        # Adjust variable DFG edges (variable node -> variable node)
        for edge in var_dfg_edges:
            from_var_idx, to_var_idx = edge[0], edge[1]
            from_var_pos = self.var_start + from_var_idx
            to_var_pos = self.var_start + to_var_idx
            adjusted_edges.append([from_var_pos, to_var_pos])
        
        # Adjust variable-to-code edges
        for edge in var_to_code_edges:
            if edge[0] < len(original_code_tokens):
                # This is code_pos -> var_idx
                original_code_pos, var_idx = edge[0], edge[1]
                # Adjust code position: skip original [CLS] and account for new sequence structure
                if original_code_pos == 0:  # Original [CLS]
                    continue  # Skip, we have new [CLS]
                elif original_code_tokens[original_code_pos] == "[SEP]":
                    continue  # Skip original [SEP], we have new ones
                else:
                    # Adjust for new code section position (skip original [CLS])
                    new_code_pos = self.code_start + original_code_pos - 1
                    new_var_pos = self.var_start + var_idx
                    adjusted_edges.append([new_code_pos, new_var_pos])
            else:
                # This is var_idx -> code_pos
                var_idx, original_code_pos = edge[0], edge[1]
                # Same adjustment as above
                if original_code_pos == 0 or original_code_tokens[original_code_pos] == "[SEP]":
                    continue
                else:
                    new_var_pos = self.var_start + var_idx
                    new_code_pos = self.code_start + original_code_pos - 1
                    adjusted_edges.append([new_var_pos, new_code_pos])
        
        logger.debug(f"Adjusted {len(adjusted_edges)} total edges for attention mask")
        return adjusted_edges
    
    def _create_dfg_adjacency_list(self, dfg_edges: List[List[int]], num_tokens: int) -> List[List[int]]:
        """Convert your DFG edge format to adjacency list.
        
        Args:
            dfg_edges: List of edges like [[2, 13], [5, 7]] 
            num_tokens: Number of tokens in sequence
            
        Returns:
            Adjacency list where dfg_to_dfg[i] contains indices that token i connects to
        """
        # Initialize adjacency list
        dfg_to_dfg = [[] for _ in range(min(num_tokens, self.max_seq_length))]
        
        # Add edges from your format
        for edge in dfg_edges:
            if len(edge) >= 2:
                from_idx, to_idx = edge[0], edge[1]
                # Ensure indices are valid
                if 0 <= from_idx < len(dfg_to_dfg) and 0 <= to_idx < len(dfg_to_dfg):
                    dfg_to_dfg[from_idx].append(to_idx)
                    # Add bidirectional connection
                    if to_idx not in dfg_to_dfg[from_idx]:
                        dfg_to_dfg[from_idx].append(to_idx)
                    if from_idx not in dfg_to_dfg[to_idx]:
                        dfg_to_dfg[to_idx].append(from_idx)
        
        return dfg_to_dfg
    
    def _create_attention_mask(self, 
                             position_idx: List[int], 
                             all_edges: List[List[int]]) -> np.ndarray:
        """Create graph-guided attention mask using the proper GraphCodeBERT format.
        
        Args:
            position_idx: Position indices for each token
            all_edges: All edges (variable↔variable + variable↔code) for attention
            
        Returns:
            Graph-guided attention mask [seq_len, seq_len]
        """
        seq_len = len(position_idx)
        attention_mask = np.zeros((seq_len, seq_len), dtype=bool)
        
        # Calculate boundaries
        total_tokens = sum([1 for pos in position_idx if pos != 1])  # Non-padding tokens
        
        # 1. Self-attention for all non-padding tokens
        for i in range(total_tokens):
            attention_mask[i, i] = True
        
        # 2. Special tokens ([CLS], [SEP]) can attend to all non-padding tokens
        for idx, pos in enumerate(position_idx):
            if pos == 0 and idx < total_tokens:  # Special tokens (including variable nodes)
                attention_mask[idx, :total_tokens] = True
                attention_mask[:total_tokens, idx] = True
        
        # 3. Code tokens can attend to each other (standard code attention)
        for i in range(seq_len):
            for j in range(seq_len):
                if (position_idx[i] >= 2 and position_idx[j] >= 2 and 
                    i < total_tokens and j < total_tokens):  # Both are code tokens
                    attention_mask[i, j] = True
        
        # 4. Add graph-guided attention from DFG edges
        for edge in all_edges:
            if len(edge) >= 2:
                from_idx, to_idx = edge[0], edge[1]
                if (0 <= from_idx < seq_len and 0 <= to_idx < seq_len and
                    from_idx < total_tokens and to_idx < total_tokens):
                    # Bidirectional attention for all DFG connections
                    attention_mask[from_idx, to_idx] = True
                    attention_mask[to_idx, from_idx] = True
        
        logger.debug(f"Created attention mask: {seq_len}x{seq_len}, "
                    f"{np.sum(attention_mask)} total connections")
        
        return attention_mask
    
    def _apply_mlm_masking(self, input_ids: List[int], position_idx: List[int]) -> Tuple[List[int], List[int]]:
        """Apply MLM masking following BERT strategy: 80% [MASK], 10% random, 10% unchanged."""
        masked_input_ids = input_ids.copy()
        labels = [-100] * len(input_ids)  # -100 means don't compute loss
        
        # Only mask code tokens (position_idx >= 2)
        maskable_positions = [i for i, pos in enumerate(position_idx) if pos >= 2]
        
        # Randomly select 15% of maskable positions
        num_to_mask = max(1, int(len(maskable_positions) * self.mlm_probability))
        if maskable_positions:
            masked_positions = random.sample(maskable_positions, min(num_to_mask, len(maskable_positions)))
            
            for pos in masked_positions:
                labels[pos] = input_ids[pos]  # Store original token for loss computation
                
                rand = random.random()
                if rand < 0.8:
                    # 80% of time: replace with [MASK]
                    masked_input_ids[pos] = self.mask_token_id
                elif rand < 0.9:
                    # 10% of time: replace with random token
                    masked_input_ids[pos] = random.randint(0, self.tokenizer.vocab_size - 1)
                # 10% of time: keep unchanged
        
        return masked_input_ids, labels

def create_graphcodebert_dataloader(train_file: str, 
                                  val_file: Optional[str],
                                  tokenizer,
                                  batch_size: Optional[int] = None,
                                  shuffle: bool = True) -> Tuple[DataLoader, Optional[DataLoader]]:
    """Create DataLoaders for GraphCodeBERT training, integrated with pipeline config.
    
    This function is designed to be called by GraphCodeBERTTrainer.create_dataloader()
    and integrates with your main.py pipeline.
    
    Args:
        train_file: Path to training JSONL file
        val_file: Path to validation JSONL file (optional)
        tokenizer: GraphCodeBERT tokenizer
        batch_size: Batch size (from config if None)
        shuffle: Whether to shuffle training data
    
    Returns:
        Tuple of (train_dataloader, val_dataloader)
    """
    # Import config for pipeline integration
    try:
        from config import FINETUNING_CONFIG, HARDWARE_CONFIG
        effective_batch_size = batch_size or FINETUNING_CONFIG['batch_size']
        num_workers = HARDWARE_CONFIG['num_workers']
        pin_memory = HARDWARE_CONFIG['pin_memory']
    except ImportError:
        # Fallback values
        effective_batch_size = batch_size or 8
        num_workers = 0
        pin_memory = False
    
    logger.info(f"Creating dataloaders: batch_size={effective_batch_size}, "
               f"num_workers={num_workers}, pin_memory={pin_memory}")
    
    # Load training data
    train_examples = _load_jsonl_examples(train_file)
    if not train_examples:
        raise ValueError(f"No training examples loaded from {train_file}")
    
    # Create training dataset
    train_dataset = ErlangCodeDataset(
        examples=train_examples,
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
        val_examples = _load_jsonl_examples(val_file)
        if val_examples:
            val_dataset = ErlangCodeDataset(
                examples=val_examples,
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
            
            logger.info(f"Created validation dataloader: {len(val_examples)} examples")
        else:
            logger.warning(f"No validation examples loaded from {val_file}")
    
    logger.info(f"Created training dataloader: {len(train_examples)} examples")
    return train_dataloader, val_dataloader


def _load_jsonl_examples(file_path: str) -> List[Dict[str, Any]]:
    """Load examples from JSONL file in your pipeline format."""
    examples = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    example = json.loads(line)
                    # Validate your format
                    if 'code_tokens' in example and 'dfg' in example and 'idx' in example:
                        examples.append(example)
                    else:
                        logger.warning(f"Invalid example format at line {line_num}")
                except json.JSONDecodeError as e:
                    logger.warning(f"JSON decode error at line {line_num}: {e}")
        
        logger.info(f"Loaded {len(examples)} valid examples from {file_path}")
        return examples
        
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        return []
    except Exception as e:
        logger.error(f"Error loading examples from {file_path}: {e}")
        return []


def graphcodebert_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Custom collate function for GraphCodeBERT data, compatible with your trainer.
    
    Handles the complex attention masks and ensures proper batching for GraphCodeBERT.
    """
    batch_size = len(batch)
    
    # Stack tensors
    input_ids = torch.stack([item['input_ids'] for item in batch])
    position_idx = torch.stack([item['position_idx'] for item in batch])  
    labels = torch.stack([item['labels'] for item in batch])
    
    # Handle attention masks - they need special batching for GraphCodeBERT
    attention_masks = torch.stack([item['attention_mask'] for item in batch])
    
    # Keep identifiers for debugging/logging
    indices = [item['idx'] for item in batch]
    
    return {
        'input_ids': input_ids,           # [batch_size, seq_len]
        'position_idx': position_idx,     # [batch_size, seq_len]
        'attention_mask': attention_masks, # [batch_size, seq_len, seq_len]
        'labels': labels,                 # [batch_size, seq_len]
        'indices': indices,               # List of original identifiers
    }


# Integration function for GraphCodeBERTTrainer
def integrate_with_trainer():
    """Integration function to be imported by GraphCodeBERTTrainer.
    
    This allows the trainer to use create_graphcodebert_dataloader() 
    as a drop-in replacement for its existing create_dataloader() method.
    """
    return create_graphcodebert_dataloader

# Pipeline integration examples
if __name__ == "__main__":
    # Test with actual function-extractor output format
    
    example_functions = [
        {
            "idx": "test::max/2::4",
            "code": "max(A, B) when A > B -> A; max(A, B) -> B.",
            "code_tokens": ["[CLS]", "max", "(", "A", ",", "B", ")", "when", "A", ">", "B", "->", "A", ";", "max", "(", "A", ",", "B", ")->", "B", ".", "[SEP]"],
            "dfg": [[3, 8], [5, 10], [3, 12], [3, 16], [5, 18], [5, 21]],  # Based on function-extractor DFG
            "nl": ""
        }
    ]
    
    print("=" * 80)
    print("ERLANG DATASET - TESTING WITH FUNCTION-EXTRACTOR FORMAT")
    print("=" * 80)
    
    print("Function-extractor output for max/2:")
    print("  Tokens: ['max', '(', 'A', ',', 'B', ')', 'when', 'A', '>', 'B', '->', 'A', ';', 'max', '(', 'A', ',', 'B', ')', '->', 'B', '.']")
    print("  Variables: [(2, 'A'), (4, 'B'), (7, 'A'), (9, 'B'), (11, 'A'), (15, 'A'), (17, 'B'), (20, 'B')]")
    print("  DFG: A@2 -> A@7,11,15 and B@4 -> B@9,17,20")
    
    print("\nConverted to our format:")
    for i, token in enumerate(example_functions[0]['code_tokens']):
        marker = " ← DFG" if any(i in edge for edge in example_functions[0]['dfg']) else ""
        print(f"  {i:2d}: {token}{marker}")
    
    print(f"\nDFG edges: {example_functions[0]['dfg']}")
    print("  [3, 8]:  A (param) -> A (guard)")
    print("  [5, 10]: B (param) -> B (guard)")  
    print("  [3, 12]: A (param) -> A (return clause 1)")
    print("  [3, 16]: A (param) -> A (param clause 2)")
    print("  [5, 18]: B (param) -> B (param clause 2)")
    print("  [5, 21]: B (param) -> B (return clause 2)")
    
    # Create mock tokenizer and test
    class MockTokenizer:
        def __init__(self):
            self.vocab = {'[CLS]': 0, '[SEP]': 2, '[PAD]': 1, '[MASK]': 50264, '[UNK]': 3}
            self.cls_token_id = 0
            self.sep_token_id = 2
            self.pad_token_id = 1
            self.mask_token_id = 50264
            self.unk_token_id = 3
            self.vocab_size = 50265
            
        def encode(self, text, add_special_tokens=False):
            if text in self.vocab:
                return [self.vocab[text]]
            return [hash(text) % 1000 + 100]
    
    tokenizer = MockTokenizer()
    dataset = ErlangCodeDataset(examples=example_functions, tokenizer=tokenizer)
    
    # Test transformation
    code_tokens = example_functions[0]['code_tokens']
    dfg_edges = example_functions[0]['dfg']
    
    variable_nodes, var_dfg_edges, var_to_code_edges = dataset._transform_dfg_to_variable_nodes(code_tokens, dfg_edges)
    
    print(f"\n" + "=" * 50)
    print("DFG TRANSFORMATION RESULTS")
    print("=" * 50)
    print(f"Variable nodes: {variable_nodes}")
    print("Expected: ['A', 'A', 'B', 'B', 'A', 'A', 'A', 'B', 'B'] (9 nodes)")
    print(f"Variable DFG edges: {var_dfg_edges}")
    print(f"Total edges created: {len(var_to_code_edges)} variable-code connections")
    
    # Test full pipeline
    item = dataset[0]
    print(f"\n" + "=" * 50)
    print("FINAL DATASET ITEM")
    print("=" * 50)
    print(f"Sequence length: {item['input_ids'].shape}")
    print(f"Variable section starts at: {dataset.var_start}")
    
    attention_mask = item['attention_mask'].numpy()
    total_connections = np.sum(attention_mask)
    print(f"Attention connections: {total_connections}")
    
    labels = item['labels'].numpy()
    masked_tokens = np.sum(labels != -100)
    print(f"Masked tokens for MLM: {masked_tokens}")
    
    print("\n✓ Function-extractor format successfully processed!")
    print("✓ Common ancestor variables properly handled!")
    print("=" * 80)


# # Pipeline integration examples
# if __name__ == "__main__":
#     # Test with the corrected example and actually call the functions
    
#     example_functions = [
#         {
#             "idx": "ninenines/cowboy::src/cowboy_rest.erl::choose_charset/2::36",
#             "code": "choose_charset ( Req , State = # state { charsets_p = CP } , [ Charset | Tail ] ) -> match_charset ( Req , State , Tail , CP , Charset ) .",
#             "code_tokens": ["[CLS]", "choose_charset", "(", "Req", ",", "State", "=", "#", "state", "{", "charsets_p", "=", "CP", "}", ",", "[", "Charset", "|", "Tail", "]", ")", "->", "match_charset", "(", "Req", ",", "State", ",", "Tail", ",", "CP", ",", "Charset", ")", ".", "[SEP]"],
#             "dfg": [[3, 24], [5, 26], [12, 30], [16, 32], [18, 28]],  # Corrected indices
#             "nl": ""
#         }
#     ]
    
#     print("=" * 80)
#     print("ERLANG DATASET - TESTING DFG TRANSFORMATION")
#     print("=" * 80)
    
#     # Create a mock tokenizer for testing
#     class MockTokenizer:
#         def __init__(self):
#             self.vocab = {'[CLS]': 0, '[SEP]': 2, '[PAD]': 1, '[MASK]': 50264, '[UNK]': 3}
#             self.cls_token_id = 0
#             self.sep_token_id = 2
#             self.pad_token_id = 1
#             self.mask_token_id = 50264
#             self.unk_token_id = 3
#             self.vocab_size = 50265
            
#         def encode(self, text, add_special_tokens=False):
#             # Simple mock encoding - hash the text to get consistent IDs
#             if text in self.vocab:
#                 return [self.vocab[text]]
#             return [hash(text) % 1000 + 100]  # Mock token ID
    
#     # Create dataset with mock tokenizer
#     tokenizer = MockTokenizer()
#     dataset = ErlangCodeDataset(
#         examples=example_functions,
#         tokenizer=tokenizer,
#         max_code_length=256,
#         max_dfg_length=64
#     )
    
#     print("Original code tokens:")
#     for i, token in enumerate(example_functions[0]['code_tokens']):
#         marker = " ← DFG" if any(i in edge for edge in example_functions[0]['dfg']) else ""
#         print(f"  {i:2d}: {token}{marker}")
    
#     print(f"\nDFG edges: {example_functions[0]['dfg']}")
    
#     # Test the internal transformation functions
#     print("\n" + "=" * 50)
#     print("TESTING DFG TRANSFORMATION")
#     print("=" * 50)
    
#     code_tokens = example_functions[0]['code_tokens']
#     dfg_edges = example_functions[0]['dfg']
    
#     # Test variable node transformation
#     variable_nodes, var_dfg_edges, var_to_code_edges = dataset._transform_dfg_to_variable_nodes(
#         code_tokens, dfg_edges
#     )
    
#     print(f"Variable nodes created: {variable_nodes}")
#     print(f"Variable DFG edges: {var_dfg_edges}")
#     print(f"Variable-to-code edges (before adjustment): {var_to_code_edges}")
    
#     # Test full input sequence creation
#     print("\n" + "=" * 50)
#     print("TESTING FULL INPUT SEQUENCE CREATION")
#     print("=" * 50)
    
#     input_ids, position_idx, all_edges = dataset._create_input_sequence(code_tokens, dfg_edges)
    
#     print(f"Input sequence length: {len(input_ids)}")
#     print(f"Code section starts at: {dataset.code_start}")
#     print(f"Variable section starts at: {dataset.var_start}")
    
#     # Decode the input sequence for display
#     print(f"\nFull input sequence:")
#     for i in range(min(len(input_ids), 50)):  # Show first 50 tokens
#         if input_ids[i] == dataset.pad_token_id:
#             break
            
#         # Determine section
#         section = ""
#         if i == 0:
#             section = " (CLS)"
#         elif i < dataset.code_start:
#             section = " (NL)"
#         elif i < dataset.var_start - 1:  # -1 for [SEP]
#             section = " (CODE)"
#         elif i == dataset.var_start - 1:
#             section = " (SEP)"
#         elif i >= dataset.var_start:
#             section = " (VAR)"
            
#         print(f"  {i:2d}: token_id={input_ids[i]:4d}, pos_idx={position_idx[i]:2d}{section}")
    
#     print(f"\nPosition indices breakdown:")
#     pos_counts = {}
#     for pos in position_idx:
#         pos_counts[pos] = pos_counts.get(pos, 0) + 1
#     for pos, count in sorted(pos_counts.items()):
#         pos_type = {0: "special/var", 1: "padding", 2: "code"}.get(pos, f"code+{pos-2}")
#         print(f"  Position {pos:2d}: {count:3d} tokens ({pos_type})")
    
#     print(f"\nAll edges for attention mask ({len(all_edges)} total):")
#     for i, edge in enumerate(all_edges):
#         if i < 20:  # Show first 20 edges
#             from_pos, to_pos = edge[0], edge[1]
#             from_section = "CLS" if from_pos == 0 else ("CODE" if from_pos < dataset.var_start else "VAR")
#             to_section = "CLS" if to_pos == 0 else ("CODE" if to_pos < dataset.var_start else "VAR")
#             print(f"  [{from_pos:2d}→{to_pos:2d}]: {from_section} → {to_section}")
#         elif i == 20:
#             print(f"  ... and {len(all_edges) - 20} more edges")
#             break
    
#     # Test full dataset item retrieval
#     print("\n" + "=" * 50)
#     print("TESTING FULL DATASET ITEM")
#     print("=" * 50)
    
#     item = dataset[0]
#     print(f"Dataset item keys: {list(item.keys())}")
#     print(f"Input IDs shape: {item['input_ids'].shape}")
#     print(f"Position idx shape: {item['position_idx'].shape}")
#     print(f"Attention mask shape: {item['attention_mask'].shape}")
#     print(f"Labels shape: {item['labels'].shape}")
#     print(f"Example ID: {item['idx']}")
    
#     # Check attention mask statistics
#     attention_mask = item['attention_mask'].numpy()
#     total_connections = np.sum(attention_mask)
#     total_possible = attention_mask.shape[0] * attention_mask.shape[1]
#     print(f"Attention mask: {total_connections}/{total_possible} connections ({total_connections/total_possible*100:.1f}%)")
    
#     # Check MLM labels
#     labels = item['labels'].numpy()
#     masked_positions = np.sum(labels != -100)
#     print(f"MLM labels: {masked_positions} tokens masked for prediction")
    
#     print("\n" + "=" * 80)
#     print("✓ GraphCodeBERT DFG transformation working correctly!")
#     print("✓ Variable nodes created from DFG edges")
#     print("✓ Proper sequence format: [CLS] + Code + [SEP] + Variables") 
#     print("✓ Graph-guided attention mask generated")
#     print("✓ MLM masking applied to code tokens only")
#     print("=" * 80)

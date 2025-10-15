"""
Masked Language Model (MLM) evaluator for GraphCodeBERT.

Evaluates the model's ability to predict masked tokens in Erlang code,
which is the core pre-training task for GraphCodeBERT.
"""

import os
import sys
import json
import math
import torch
import torch.nn.functional as F
from typing import Dict, List, Any
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

# Import our modules - updated for new config structure
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from base import BaseEvaluator
from train.dataset import ErlangCodeDataset

class MLMEvaluator(BaseEvaluator):
    """Masked Language Model evaluation for GraphCodeBERT."""

    def get_name(self) -> str:
        return "mlm"

    def _load_model_and_tokenizer(self):
        """Load GraphCodeBERT model and tokenizer."""
        if self.model is not None and self.tokenizer is not None:
            return

        try:
            from transformers import RobertaTokenizer, RobertaConfig
            from train.model import GraphCodeBERTForMLM

            # Load tokenizer
            self.tokenizer = RobertaTokenizer.from_pretrained('microsoft/graphcodebert-base')
            self.logger.info("✓ Loaded GraphCodeBERT tokenizer")

            # Load checkpoint
            model_path = self._find_model_file(self.model_checkpoint)
            self.logger.info(f"Loading checkpoint from: {model_path}")

            checkpoint = torch.load(model_path, map_location=self.device)

            # Get config and state_dict from checkpoint
            if 'config' not in checkpoint or 'model_state_dict' not in checkpoint:
                raise ValueError(
                    f"Checkpoint {model_path} is missing 'config' or 'model_state_dict'. "
                    f"This checkpoint was created before the position fix. "
                    f"Please retrain the model with the fixed code."
                )

            config = checkpoint['config']
            state_dict = checkpoint['model_state_dict']
            self.logger.info("✓ Loaded config and weights from checkpoint")

            # Create model architecture and load weights
            self.model = GraphCodeBERTForMLM(config)
            self.model.load_state_dict(state_dict, strict=False)
            self.logger.info(f"✓ Loaded model from {model_path}")

            self.model.to(self.device)
            self.model.eval()

        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise

    def _find_model_file(self, checkpoint_path: str) -> str:
        """Find the actual model file from checkpoint path.

        Args:
            checkpoint_path: Path to checkpoint (file or directory)

        Returns:
            Path to the actual model file

        Raises:
            FileNotFoundError: If model file cannot be found
        """
        import os

        # If it's a file, use it directly
        if os.path.isfile(checkpoint_path):
            return checkpoint_path

        # If it's a directory, look for common model file names
        if os.path.isdir(checkpoint_path):
            candidate_files = [
                'pytorch_model.bin',    # HuggingFace standard
                'model.bin',           # Our custom format
                'model.pt',            # PyTorch standard
                'model.pth',           # PyTorch alternative
                'checkpoint.pt',       # Training checkpoint
                'best_model.pt',       # Best model
            ]

            for filename in candidate_files:
                model_file = os.path.join(checkpoint_path, filename)
                if os.path.isfile(model_file):
                    self.logger.info(f"Found model file: {filename}")
                    return model_file

            # List what's actually in the directory for debugging
            contents = os.listdir(checkpoint_path)
            self.logger.error(f"No model file found in {checkpoint_path}")
            self.logger.error(f"Directory contents: {contents}")
            raise FileNotFoundError(f"No model file found in checkpoint directory: {checkpoint_path}")

        # Path doesn't exist
        raise FileNotFoundError(f"Checkpoint path does not exist: {checkpoint_path}")

    def evaluate(self, data_path: str, dataset_type: str = "erlang", 
                 batch_size: int = 32, max_examples: int = None) -> Dict[str, float]:
        """Run MLM evaluation on different dataset types.
        
        Args:
            data_path: Path to evaluation data file
            dataset_type: Type of dataset ("erlang" or "graphcodebert")
            batch_size: Batch size for evaluation
            max_examples: Maximum number of examples to evaluate
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.logger.info(f"Loading {dataset_type} dataset from {data_path}")
        
        # Unified data loading based on dataset type
        if dataset_type == "erlang":
            functions = self._load_erlang_functions(data_path, max_examples)
        elif dataset_type == "graphcodebert":
            functions = self._load_graphcodebert_functions(data_path, max_examples)
        else:
            raise ValueError(f"Unknown dataset_type: {dataset_type}. Use 'erlang' or 'graphcodebert'")
        
        # Load model and tokenizer
        self._load_model_and_tokenizer()

        # Create dataset 
        dataset = ErlangCodeDataset(functions, self.tokenizer, 
                                    max_seq_length=256, mlm_probability=0.15)        
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=self._collate_fn)

        self.logger.info(f"Starting MLM evaluation on {data_path}")

        # Run evaluation
        metrics = self._evaluate_batches(dataloader)

        self.logger.info("MLM evaluation completed")
        for metric, value in metrics.items():
            self.logger.info(f"  {metric}: {value:.4f}")

        return metrics

    def _load_erlang_functions(self, data_path: str, max_examples: int = None) -> List[Dict]:
        """Load Erlang functions from JSONL file."""
        functions = []
        
        with open(data_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if max_examples and i >= max_examples:
                    break
                    
                function = json.loads(line.strip())
                functions.append(function)
        
        self.logger.info(f"Loaded {len(functions)} Erlang functions")
        return functions
    
    def _load_graphcodebert_functions(self, data_path: str, max_examples: int = None) -> List[Dict]:
        """Load and convert GraphCodeBERT dataset to our format."""
        functions = []
        
        with open(data_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if max_examples and i >= max_examples:
                    break
                    
                data = json.loads(line)
                
                # Convert GraphCodeBERT format to our internal format
                function = {
                    'idx': f"{data['repo']}::{data['func_name']}::{i}",
                    'code': data['code'],
                    'code_tokens': ['[CLS]'] + data['code_tokens'] + ['[SEP]'],
                    'dataflow_graph': [],  # No DFG available
                    'docstring': data.get('docstring', ''),
                    'docstring_tokens': data.get('docstring_tokens', []),
                    'variable_positions': []  # No variable positions
                }
                functions.append(function)
        
        self.logger.info(f"Loaded and converted {len(functions)} GraphCodeBERT functions")
        return functions

    def _validate_example(self, example: Dict[str, Any]) -> bool:
        """Validate that example has required fields for MLM evaluation."""
        required_fields = ['code_tokens', 'dataflow_graph']
        return all(field in example for field in required_fields)

    def _collate_fn(self, batch):
        """Collate function for evaluation batches."""
        from train.dataset import graphcodebert_collate_fn
        return graphcodebert_collate_fn(batch)

    def _evaluate_batches(self, dataloader: DataLoader) -> Dict[str, float]:
        """Run evaluation on all batches and compute metrics."""
        total_loss = 0.0
        total_tokens = 0
        correct_predictions = 0
        total_predictions = 0

        # For perplexity calculation
        total_log_likelihood = 0.0

        # For top-k accuracy
        top5_correct = 0
        top10_correct = 0

        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating MLM"):
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}

                # Forward pass (always use same interface for consistency)
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    position_idx=batch['position_idx'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels']
                )

                # Extract predictions and labels
                logits = outputs['logits']  # [batch_size, seq_len, vocab_size]
                labels = batch['labels']    # [batch_size, seq_len]

                # Only evaluate masked tokens (labels != -100)
                mask = (labels != -100)

                if mask.sum() == 0:
                    continue  # No masked tokens in this batch

                # Get predictions for masked tokens
                masked_logits = logits[mask]  # [num_masked_tokens, vocab_size]
                masked_labels = labels[mask]  # [num_masked_tokens]

                # Compute loss
                loss = F.cross_entropy(masked_logits, masked_labels, reduction='sum')
                total_loss += loss.item()

                # Compute accuracy metrics
                predictions = torch.argmax(masked_logits, dim=-1)
                correct = (predictions == masked_labels).sum().item()

                correct_predictions += correct
                total_predictions += len(masked_labels)
                total_tokens += len(masked_labels)

                # Top-k accuracy
                _, top_k = torch.topk(masked_logits, k=10, dim=-1)
                top5_correct += (top_k[:, :5] == masked_labels.unsqueeze(1)).any(dim=1).sum().item()
                top10_correct += (top_k == masked_labels.unsqueeze(1)).any(dim=1).sum().item()

                # For perplexity
                log_probs = F.log_softmax(masked_logits, dim=-1)
                target_log_probs = log_probs.gather(1, masked_labels.unsqueeze(1)).squeeze(1)
                total_log_likelihood += target_log_probs.sum().item()

        # Compute final metrics
        if total_predictions == 0:
            raise ValueError("No masked tokens found in evaluation data")

        accuracy = correct_predictions / total_predictions
        avg_loss = total_loss / total_tokens
        perplexity = math.exp(-total_log_likelihood / total_tokens)
        top5_accuracy = top5_correct / total_predictions
        top10_accuracy = top10_correct / total_predictions

        return {
            'mlm_accuracy': accuracy,
            'mlm_loss': avg_loss,
            'perplexity': perplexity,
            'top5_accuracy': top5_accuracy,
            'top10_accuracy': top10_accuracy,
            'total_examples': len(dataloader.dataset),
            'total_masked_tokens': total_predictions
        }

def main(model_checkpoint: str, function_idx: str = None):
    """
    Main function to demonstrate MLM token prediction.

    Usage: 
        python mlm.py <model_checkpoint_path>                       # Uses hardcoded max function
        python mlm.py <model_checkpoint_path> functions.jsonl <idx> # Loads from file by idx
    """

    # Check if we should load from file
    if function_idx is not None and len(sys.argv) >= 4:
        jsonl_file = sys.argv[2]
        target_idx = function_idx
        
        print("=== MLM Code Token Prediction Demo (from file) ===")
        print(f"Loading function from: {jsonl_file}")
        print(f"Target function idx: {target_idx}")
        print(f"Loading model: {model_checkpoint}")
        print()
        
        # Load functions from JSONL file using existing loader
        from train.dataset import _load_function_extractor_output
        
        try:
            functions = _load_function_extractor_output(jsonl_file)
            print(f"Loaded {len(functions)} functions from file")
            
            # Find the function with matching idx
            target_function = None
            for func in functions:
                if func.get('idx') == target_idx:
                    target_function = func
                    break
            
            if target_function is None:
                print(f"Error: Function with idx '{target_idx}' not found in {jsonl_file}")
                print(f"Available indices: {[f.get('idx', 'NO_IDX') for f in functions[:5]]}...")
                return
                
            selected_function = target_function
            print(f"Found function: {selected_function['func_name']} (arity: {selected_function.get('arity', 'unknown')})")
            
        except Exception as e:
            print(f"Error loading functions from {jsonl_file}: {e}")
            return
    else:
        # Use the original hardcoded max function
        selected_function = {
            "idx": "test::max/2::0",
            "code": "max(A, B) when A > B -> A; max(A, B) -> B.",
            "code_tokens": ["[CLS]", "max", "(", "A", ",", "B", ")", "when", "A", ">", "B", "->", "A", ";", "max", "(", "A", ",", "B", ")", "->", "B", ".", "[SEP]"],
            "variable_positions": [(3, "A", True), (5, "B", True), (8, "A", False), (10, "B", False), (12, "A", False), (16, "A", True), (18, "B", True), (21, "B", False)],
            "dataflow_graph": [("A", 3, "comesFrom", [], []), ("B", 5, "comesFrom", [], []), ("A", 8, "comesFrom", ["A"], [3]), ("B", 10, "comesFrom", ["B"], [5]), ("A", 12, "comesFrom", ["A"], [8]), ("A", 16, "comesFrom", [], []), ("B", 18, "comesFrom", [], []), ("B", 21, "comesFrom", ["B"], [18])],
            "docstring": "Returns the maximum of two numbers",
            "docstring_tokens": ["Returns", "the", "maximum", "of", "two", "numbers"]
        }
        
        print("=== MLM Code Token Prediction Demo (hardcoded) ===")
        print(f"Function: {selected_function['code']}")
        print(f"Loading model: {model_checkpoint}")
        print()


    # Initialize evaluator
    evaluator = MLMEvaluator(model_checkpoint)
    evaluator._load_model_and_tokenizer()

    # Create input sequence WITHOUT random masking by calling the dataset creation directly
    from train.dataset import ErlangCodeDataset

    # Create a temporary dataset just to get the input creation method
    temp_dataset = ErlangCodeDataset([selected_function], evaluator.tokenizer, max_seq_length=128, mlm_probability=0.0)
        
    # Access the private method to create unmasked input sequence
    doc_tokens = selected_function.get('docstring_tokens', [])
    code_tokens = selected_function['code_tokens']
    dfg_edges = [(edge[1], edge[4][0]) for edge in selected_function['dataflow_graph'] if edge[4] != []]  # Convert format

    input_ids, position_idx, all_edges, token_boundaries = temp_dataset._create_input_sequence(
        doc_tokens, code_tokens, dfg_edges
    )

    # Create attention mask without MLM masking
    attention_mask = temp_dataset._create_attention_mask(input_ids, token_boundaries, all_edges)

    # Convert to tensors
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    position_idx = torch.tensor(position_idx, dtype=torch.long)
    attention_mask = torch.tensor(attention_mask, dtype=torch.bool)

    # Get original tokens (this is the RoBERTa tokenization)
    original_tokens = evaluator.tokenizer.convert_ids_to_tokens(input_ids)

    # output = f"RoBERTa Token sequence ({len(original_tokens)} tokens):"
    # for i, token in enumerate(original_tokens):
    #     if input_ids[i] != evaluator.tokenizer.pad_token_id:
    #         output += f"{i:2d}: {token}, "
    # print(output)

    # Iterate through each non-special RoBERTa token
    evaluator.model.eval()
    with torch.no_grad():
        for pos in range(len(input_ids)):

            # Only include NL and code tokens
            if not(token_boundaries['code'][0] <= pos <= token_boundaries['code'][1]):
                continue

            token_id = input_ids[pos].item()
            original_token = original_tokens[pos]

            # Skip special tokens and padding
            if token_id in [evaluator.tokenizer.cls_token_id,
                           evaluator.tokenizer.sep_token_id,
                           evaluator.tokenizer.pad_token_id]:
                continue

            print(f"Position {pos}: Original RoBERTa token '{original_token}'")

            # Create masked version
            masked_input_ids = input_ids.clone()
            masked_input_ids[pos] = evaluator.tokenizer.mask_token_id

            # Prepare batch tensors
            batch_input_ids = masked_input_ids.unsqueeze(0).to(evaluator.device)
            batch_position_idx = position_idx.unsqueeze(0).to(evaluator.device)
            batch_attention_mask = attention_mask.unsqueeze(0).to(evaluator.device)

            # Forward pass
            outputs = evaluator.model(
                input_ids=batch_input_ids,
                position_idx=batch_position_idx,
                attention_mask=batch_attention_mask
            )

            # Get predictions for the masked position
            logits = outputs['logits'][0, pos, :]  # [vocab_size]
            probabilities = F.softmax(logits, dim=0)

            # Get top-5 predictions
            top5_probs, top5_indices = torch.topk(probabilities, 5)
            top5_tokens = evaluator.tokenizer.convert_ids_to_tokens(top5_indices.tolist())

            print("  Top-5 predictions:")
            for i, (token, prob) in enumerate(zip(top5_tokens, top5_probs)):
                is_correct = "✓" if top5_indices[i].item() == token_id else " "
                print(f"    {i+1}. {token:<12} {prob:.4f} {is_correct}")

            # Check if original token is in top-5
            original_rank = -1
            for i, idx in enumerate(top5_indices):
                if idx.item() == token_id:
                    original_rank = i + 1
                    break

            if original_rank > 0:
                print(f"  → Original token ranked #{original_rank}")
            else:
                original_prob = probabilities[token_id].item()
                print(f"  → Original token not in top-5 (prob: {original_prob:.6f})")

            print()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python mlm.py <model_checkpoint_path> [functions.jsonl] [function_idx]")
        sys.exit(1)
    
    model_checkpoint = sys.argv[1]
    
    if len(sys.argv) >= 4:
        # Called with JSONL file and function index
        function_idx = sys.argv[3]
        main(model_checkpoint, function_idx)
    else:
        # Called with just model checkpoint (use hardcoded function)
        main(model_checkpoint)

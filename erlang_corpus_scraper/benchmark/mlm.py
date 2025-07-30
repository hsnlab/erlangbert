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

class MLMEvaluator(BaseEvaluator):
    """Masked Language Model evaluation for GraphCodeBERT."""

    def get_name(self) -> str:
        return "mlm"

    def _load_model_and_tokenizer(self):
        """Load GraphCodeBERT model and tokenizer."""
        if self.model is not None and self.tokenizer is not None:
            return

        try:
            from transformers import RobertaTokenizer
            from train.model import create_graphcodebert_model

            # Load tokenizer
            self.tokenizer = RobertaTokenizer.from_pretrained('microsoft/graphcodebert-base')
            self.logger.info("✓ Loaded GraphCodeBERT tokenizer")

            # Load model
            self.model = create_graphcodebert_model('microsoft/graphcodebert-base')

            # Find the actual model file
            model_path = self._find_model_file(self.model_checkpoint)
            self.logger.info(f"Loading checkpoint from: {model_path}")

            # Load checkpoint weights
            checkpoint = torch.load(model_path, map_location=self.device)

            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint

            self.model.load_state_dict(state_dict, strict=False)
            self.model.to(self.device)
            self.model.eval()

            self.logger.info(f"✓ Loaded GraphCodeBERT model from {model_path}")

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

    def evaluate(self, data_path: str, batch_size: int = 32, max_examples: int = None) -> Dict[str, float]:
        """Evaluate MLM performance on Erlang code.

        Args:
            data_path: Path to evaluation JSONL data
            batch_size: Evaluation batch size
            max_examples: Maximum number of examples to evaluate (None for all)

        Returns:
            Dictionary with MLM metrics
        """
        self.logger.info(f"Starting MLM evaluation on {data_path}")

        # Load model and tokenizer
        self._load_model_and_tokenizer()

        # Load evaluation data
        eval_data = self._load_evaluation_data(data_path, max_examples)
        self.logger.info(f"Loaded {len(eval_data)} evaluation examples")

        if not eval_data:
            raise ValueError(f"No evaluation data found in {data_path}")

        # Create evaluation dataset and dataloader
        eval_dataset = self._create_eval_dataset(eval_data)
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self._collate_fn
        )

        # Run evaluation
        metrics = self._evaluate_batches(eval_dataloader)

        self.logger.info("MLM evaluation completed")
        for metric, value in metrics.items():
            self.logger.info(f"  {metric}: {value:.4f}")

        return metrics

    def _load_evaluation_data(self, data_path: str, max_examples: int = None) -> List[Dict[str, Any]]:
        """Load evaluation data from JSONL file."""
        self.logger.info(f"Loading evaluation data from: {data_path}")

        # Check if file exists and get size
        import os
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Evaluation data file not found: {data_path}")

        file_size = os.path.getsize(data_path)
        self.logger.info(f"File size: {file_size} bytes")

        if file_size == 0:
            raise ValueError(f"Evaluation data file is empty: {data_path}")

        data = []
        total_lines = 0
        valid_lines = 0
        invalid_json_lines = 0
        invalid_field_lines = 0

        with open(data_path, 'r') as f:
            for i, line in enumerate(f):
                total_lines += 1

                if max_examples is not None and valid_lines >= max_examples:
                    break

                line = line.strip()
                if not line:
                    continue  # Skip empty lines

                try:
                    example = json.loads(line)

                    # Validate required fields
                    if self._validate_example(example):
                        data.append(example)
                        valid_lines += 1
                    else:
                        invalid_field_lines += 1
                        if invalid_field_lines <= 3:  # Log first few invalid examples
                            self.logger.warning(f"Line {i+1}: Missing required fields. Has: {list(example.keys())}")

                except json.JSONDecodeError as e:
                    invalid_json_lines += 1
                    if invalid_json_lines <= 3:  # Log first few JSON errors
                        self.logger.warning(f"Line {i+1}: Invalid JSON - {e}")

        # Log summary
        self.logger.info(f"Data loading summary:")
        self.logger.info(f"  Total lines: {total_lines}")
        self.logger.info(f"  Valid examples: {valid_lines}")
        self.logger.info(f"  Invalid JSON lines: {invalid_json_lines}")
        self.logger.info(f"  Invalid field lines: {invalid_field_lines}")

        if not data:
            self.logger.error("No valid evaluation data found!")
            self.logger.error("Required fields: code_tokens, dataflow_graph")

            # Show first few lines for debugging
            self.logger.error("First 3 lines of file:")
            with open(data_path, 'r') as f:
                for i, line in enumerate(f):
                    if i >= 3:
                        break
                    self.logger.error(f"  Line {i+1}: {line[:100]}...")

        return data

    def _validate_example(self, example: Dict[str, Any]) -> bool:
        """Validate that example has required fields for MLM evaluation."""
        required_fields = ['code_tokens', 'dataflow_graph']
        return all(field in example for field in required_fields)

    def _create_eval_dataset(self, eval_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create evaluation dataset with MLM masking."""
        from train.dataset import ErlangCodeDataset

        # Create dataset using existing infrastructure
        dataset = ErlangCodeDataset(
            examples=eval_data,
            tokenizer=self.tokenizer,
            max_seq_length=256,  # From config
            mlm_probability=0.15  # Standard MLM masking
        )

        return dataset

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

                # Forward pass
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

def main(model_checkpoint: str):
    """
    Main function to demonstrate MLM token prediction on max/2 function.

    Usage: python mlm.py <model_checkpoint_path>
    """

    # Sample max/2 function from function extractor format
    max_function = {
        "idx": "test::max/2::0",
        "code": "max(A, B) when A > B -> A; max(A, B) -> B.",
        "code_tokens": ["[CLS]", "max", "(", "A", ",", "B", ")", "when", "A", ">", "B", "->", "A", ";", "max", "(", "A", ",", "B", ")", "->", "B", ".", "[SEP]"],
        "variable_positions": [(3, "A", True), (5, "B", True), (8, "A", False), (10, "B", False), (12, "A", False), (16, "A", True), (18, "B", True), (21, "B", False)],
        "dataflow_graph": [("A", 3, "comesFrom", [], []), ("B", 5, "comesFrom", [], []), ("A", 8, "comesFrom", ["A"], [3]), ("B", 10, "comesFrom", ["B"], [5]), ("A", 12, "comesFrom", ["A"], [8]), ("A", 16, "comesFrom", [], []), ("B", 18, "comesFrom", [], []), ("B", 21, "comesFrom", ["B"], [18])],
        "docstring": "Returns the maximum of two numbers",
        "docstring_tokens": ["Returns", "the", "maximum", "of", "two", "numbers"]
    }

    print("=== MLM Code Token Prediction Demo ===")
    print(f"Function: {max_function['code']}")
    print(f"Loading model: {model_checkpoint}")
    print()

    # Initialize evaluator
    evaluator = MLMEvaluator(model_checkpoint)
    evaluator._load_model_and_tokenizer()

    # Create input sequence WITHOUT random masking by calling the dataset creation directly
    from train.dataset import ErlangCodeDataset

    # Create a temporary dataset just to get the input creation method
    temp_dataset = ErlangCodeDataset([max_function], evaluator.tokenizer, max_seq_length=128, mlm_probability=0.0)

    # Access the private method to create unmasked input sequence
    doc_tokens = max_function.get('docstring_tokens', [])
    code_tokens = max_function['code_tokens']
    dfg_edges = [(edge[1], edge[4][0]) for edge in max_function['dataflow_graph'] if edge[4] != []]  # Convert format

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
    import sys
    if len(sys.argv) != 2:
        print("Usage: python mlm.py <model_checkpoint_path>")
        sys.exit(1)

    model_checkpoint = sys.argv[1]
    main(model_checkpoint)

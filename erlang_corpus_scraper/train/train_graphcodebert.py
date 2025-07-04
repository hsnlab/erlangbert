#!/usr/bin/env python3
"""
Training script for GraphCodeBERT fine-tuning on Erlang corpus.
Supports both direct fine-tuning and LoRA adaptation.
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional
import random
import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    from transformers import (
        RobertaConfig, RobertaModel, RobertaTokenizer,
        AdamW, get_linear_schedule_with_warmup,
        TrainingArguments, Trainer
    )
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

# Import configuration
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import GRAPHCODEBERT_CONFIG, FINETUNING_CONFIG, LORA_CONFIG, HARDWARE_CONFIG

logger = logging.getLogger(__name__)

class ErlangCodeDataset(Dataset):
    """Dataset for Erlang code examples."""
    
    def __init__(self, examples: List[Dict[str, Any]], tokenizer, max_length: int = 256):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Tokenize code
        code_tokens = example['code_tokens']
        code_text = ' '.join(code_tokens)
        
        # Tokenize with transformer tokenizer
        encoding = self.tokenizer(
            code_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Prepare data flow graph (simplified)
        dfg = example.get('dfg', [])
        dfg_tensor = torch.zeros(self.max_length, self.max_length)
        for edge in dfg:
            if len(edge) >= 2 and edge[0] < self.max_length and edge[1] < self.max_length:
                dfg_tensor[edge[0], edge[1]] = 1
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'dfg_matrix': dfg_tensor,
            'idx': example.get('idx', str(idx))
        }

class GraphCodeBERTModel(nn.Module):
    """GraphCodeBERT model for Erlang code representation."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Load pre-trained RoBERTa as base
        self.roberta = RobertaModel.from_pretrained(
            GRAPHCODEBERT_CONFIG['model_name']
        )
        
        # Additional layers for graph processing
        hidden_size = self.roberta.config.hidden_size
        self.graph_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            batch_first=True
        )
        
        self.projection = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, input_ids, attention_mask, dfg_matrix=None):
        # Get RoBERTa embeddings
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        sequence_output = outputs.last_hidden_state
        
        # Apply graph attention if DFG is provided
        if dfg_matrix is not None:
            # Convert DFG matrix to attention mask
            graph_attention_mask = dfg_matrix.bool()
            
            # Apply graph-guided attention
            attended_output, _ = self.graph_attention(
                sequence_output,
                sequence_output,
                sequence_output,
                attn_mask=graph_attention_mask
            )
            
            # Combine with original output
            sequence_output = sequence_output + self.dropout(attended_output)
        
        # Project to final representation
        pooled_output = self.projection(sequence_output[:, 0])  # Use [CLS] token
        
        return {
            'last_hidden_state': sequence_output,
            'pooler_output': pooled_output
        }

class GraphCodeBERTTrainer:
    """Trainer for GraphCodeBERT fine-tuning."""
    
    def __init__(self, model_name: str = None, use_lora: bool = False):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch and transformers required for training")
        
        self.model_name = model_name or GRAPHCODEBERT_CONFIG['model_name']
        self.use_lora = use_lora
        self.tokenizer = None
        self.model = None
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() and HARDWARE_CONFIG['use_gpu'] else 'cpu')
        logger.info(f"Using device: {self.device}")
    
    def setup_model_and_tokenizer(self):
        """Initialize model and tokenizer."""
        # Load tokenizer
        self.tokenizer = RobertaTokenizer.from_pretrained(
            GRAPHCODEBERT_CONFIG['tokenizer_name']
        )
        
        # Load model configuration
        config = RobertaConfig.from_pretrained(self.model_name)
        
        # Create model
        self.model = GraphCodeBERTModel(config)
        
        # Apply LoRA if requested
        if self.use_lora:
            if not PEFT_AVAILABLE:
                raise ImportError("PEFT library required for LoRA training")
            
            lora_config = LoraConfig(
                r=LORA_CONFIG['r'],
                lora_alpha=LORA_CONFIG['alpha'],
                lora_dropout=LORA_CONFIG['dropout'],
                target_modules=LORA_CONFIG['target_modules'],
                bias=LORA_CONFIG['bias'],
                task_type=TaskType.FEATURE_EXTRACTION
            )
            
            self.model = get_peft_model(self.model, lora_config)
            logger.info("Applied LoRA adaptation")
        
        self.model.to(self.device)
        logger.info(f"Model loaded: {self.model_name}")
    
    def load_dataset(self, data_file: str) -> List[Dict[str, Any]]:
        """Load dataset from JSONL file."""
        examples = []
        try:
            with open(data_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        examples.append(json.loads(line))
            
            logger.info(f"Loaded {len(examples)} examples from {data_file}")
            return examples
        
        except Exception as e:
            logger.error(f"Error loading dataset {data_file}: {e}")
            return []
    
    def create_dataloader(self, examples: List[Dict[str, Any]], 
                         batch_size: int, shuffle: bool = True) -> DataLoader:
        """Create DataLoader for training/evaluation."""
        dataset = ErlangCodeDataset(
            examples=examples,
            tokenizer=self.tokenizer,
            max_length=GRAPHCODEBERT_CONFIG['max_code_length']
        )
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=HARDWARE_CONFIG['num_workers'],
            pin_memory=HARDWARE_CONFIG['pin_memory']
        )
    
    def train(self, train_file: str, val_file: str, output_dir: str):
        """Train the model."""
        logger.info("Starting training...")
        
        # Setup model and tokenizer
        self.setup_model_and_tokenizer()
        
        # Load datasets
        train_examples = self.load_dataset(train_file)
        val_examples = self.load_dataset(val_file)
        
        if not train_examples:
            raise ValueError("No training examples loaded")
        
        # Create data loaders
        train_dataloader = self.create_dataloader(
            train_examples, 
            FINETUNING_CONFIG['batch_size'], 
            shuffle=True
        )
        
        val_dataloader = self.create_dataloader(
            val_examples, 
            FINETUNING_CONFIG['batch_size'], 
            shuffle=False
        ) if val_examples else None
        
        # Setup optimizer
        optimizer = AdamW(
            self.model.parameters(),
            lr=FINETUNING_CONFIG['learning_rate'],
            eps=FINETUNING_CONFIG['adam_epsilon'],
            weight_decay=FINETUNING_CONFIG['weight_decay']
        )
        
        # Setup scheduler
        total_steps = len(train_dataloader) * FINETUNING_CONFIG['num_epochs']
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=FINETUNING_CONFIG['warmup_steps'],
            num_training_steps=total_steps
        )
        
        # Training loop
        self.model.train()
        best_loss = float('inf')
        
        for epoch in range(FINETUNING_CONFIG['num_epochs']):
            logger.info(f"Epoch {epoch + 1}/{FINETUNING_CONFIG['num_epochs']}")
            
            total_loss = 0
            for batch_idx, batch in enumerate(train_dataloader):
                # Move to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                dfg_matrix = batch['dfg_matrix'].to(self.device)
                
                # Forward pass
                optimizer.zero_grad()
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    dfg_matrix=dfg_matrix
                )
                
                # Simple contrastive loss for representation learning
                embeddings = outputs['pooler_output']
                loss = self.contrastive_loss(embeddings)
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    FINETUNING_CONFIG['max_grad_norm']
                )
                optimizer.step()
                scheduler.step()
                
                total_loss += loss.item()
                
                if batch_idx % FINETUNING_CONFIG['logging_steps'] == 0:
                    logger.info(f"Batch {batch_idx}, Loss: {loss.item():.4f}")
            
            avg_loss = total_loss / len(train_dataloader)
            logger.info(f"Epoch {epoch + 1} average loss: {avg_loss:.4f}")
            
            # Validation
            if val_dataloader:
                val_loss = self.evaluate(val_dataloader)
                logger.info(f"Validation loss: {val_loss:.4f}")
                
                # Save best model
                if val_loss < best_loss:
                    best_loss = val_loss
                    self.save_model(output_dir, epoch)
            else:
                self.save_model(output_dir, epoch)
        
        logger.info("Training completed!")
    
    def contrastive_loss(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Simple contrastive loss for representation learning."""
        # Normalize embeddings
        embeddings = torch.nn.functional.normalize(embeddings, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(embeddings, embeddings.T)
        
        # Create positive pairs (same function, different representation)
        batch_size = embeddings.size(0)
        labels = torch.arange(batch_size).to(embeddings.device)
        
        # InfoNCE loss
        temperature = 0.1
        similarity_matrix = similarity_matrix / temperature
        loss = torch.nn.functional.cross_entropy(similarity_matrix, labels)
        
        return loss
    
    def evaluate(self, dataloader: DataLoader) -> float:
        """Evaluate the model."""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                dfg_matrix = batch['dfg_matrix'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    dfg_matrix=dfg_matrix
                )
                
                embeddings = outputs['pooler_output']
                loss = self.contrastive_loss(embeddings)
                total_loss += loss.item()
        
        self.model.train()
        return total_loss / len(dataloader)
    
    def save_model(self, output_dir: str, epoch: int):
        """Save the trained model."""
        os.makedirs(output_dir, exist_ok=True)
        
        model_path = os.path.join(output_dir, f"model_epoch_{epoch}")
        
        if self.use_lora:
            # Save LoRA adapter
            self.model.save_pretrained(model_path)
        else:
            # Save full model
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'tokenizer': self.tokenizer,
                'config': self.model.config
            }, os.path.join(model_path, 'pytorch_model.bin'))
        
        logger.info(f"Model saved to {model_path}")

def main():
    """Main training script."""
    parser = argparse.ArgumentParser(description='Train GraphCodeBERT on Erlang corpus')
    
    parser.add_argument('--train-file', required=True, help='Training data file (JSONL)')
    parser.add_argument('--val-file', help='Validation data file (JSONL)')
    parser.add_argument('--output-dir', required=True, help='Output directory for trained model')
    parser.add_argument('--use-lora', action='store_true', help='Use LoRA adaptation')
    parser.add_argument('--model-name', default=GRAPHCODEBERT_CONFIG['model_name'], 
                       help='Pre-trained model name')
    parser.add_argument('--batch-size', type=int, default=FINETUNING_CONFIG['batch_size'],
                       help='Training batch size')
    parser.add_argument('--learning-rate', type=float, default=FINETUNING_CONFIG['learning_rate'],
                       help='Learning rate')
    parser.add_argument('--num-epochs', type=int, default=FINETUNING_CONFIG['num_epochs'],
                       help='Number of training epochs')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Check dependencies
    if not TORCH_AVAILABLE:
        logger.error("PyTorch and transformers are required for training")
        logger.error("Install with: pip install torch transformers")
        return 1
    
    if args.use_lora and not PEFT_AVAILABLE:
        logger.error("PEFT library is required for LoRA training")
        logger.error("Install with: pip install peft")
        return 1
    
    # Update config with command line arguments
    if args.batch_size:
        FINETUNING_CONFIG['batch_size'] = args.batch_size
    if args.learning_rate:
        FINETUNING_CONFIG['learning_rate'] = args.learning_rate
    if args.num_epochs:
        FINETUNING_CONFIG['num_epochs'] = args.num_epochs
    
    # Validate input files
    if not os.path.exists(args.train_file):
        logger.error(f"Training file not found: {args.train_file}")
        return 1
    
    if args.val_file and not os.path.exists(args.val_file):
        logger.error(f"Validation file not found: {args.val_file}")
        return 1
    
    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    if TORCH_AVAILABLE:
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
    
    try:
        # Create trainer
        trainer = GraphCodeBERTTrainer(
            model_name=args.model_name,
            use_lora=args.use_lora
        )
        
        # Start training
        trainer.train(
            train_file=args.train_file,
            val_file=args.val_file,
            output_dir=args.output_dir
        )
        
        logger.info("Training completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())

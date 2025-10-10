"""
GraphCodeBERT Trainer for MLM Training on Erlang Code

This module provides the training pipeline for GraphCodeBERT MLM training,
integrated with the existing pipeline's config system and command-line interface.
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from tqdm import tqdm
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import (
    RobertaTokenizer, RobertaConfig,
    get_linear_schedule_with_warmup
)

# Optional LoRA support
try:
    from peft import LoraConfig, get_peft_model, TaskType
    HAS_LORA = True
except ImportError:
    HAS_LORA = False

# Import our modules
from train.model import (
    GraphCodeBERTForMLM, create_graphcodebert_model, check_device_setup
)
from train.dataset import create_graphcodebert_dataloader

# Import config - handle gracefully if not available
try:
    from config import GRAPHCODEBERT_CONFIG, FINETUNING_CONFIG, LORA_CONFIG, HARDWARE_CONFIG
except ImportError:
    # Fallback configuration if config.py not available
    GRAPHCODEBERT_CONFIG = {
        'model_name': 'microsoft/graphcodebert-base',
        'tokenizer_name': 'microsoft/graphcodebert-base',
        'max_code_length': 256,
        'max_dfg_length': 64,
    }
    FINETUNING_CONFIG = {
        'batch_size': 8,
        'learning_rate': 2e-5,
        'num_epochs': 10,
        'warmup_steps': 1000,
        'weight_decay': 0.01,
        'adam_epsilon': 1e-8,
        'max_grad_norm': 1.0,
    }
    LORA_CONFIG = {
        'r': 16,
        'alpha': 32,
        'dropout': 0.1,
        'target_modules': ['query', 'value'],
        'bias': 'none',
    }
    HARDWARE_CONFIG = {
        'num_workers': 0,
        'pin_memory': False,
    }

logger = logging.getLogger(__name__)
if not HAS_LORA:
    logger.warning("PEFT/LoRA not available - falling back to full fine-tuning")

class GraphCodeBERTTrainer:
    """Trainer for GraphCodeBERT fine-tuning on Erlang code.
    
    Integrates seamlessly with the existing pipeline and supports:
    - MLM training on Erlang code with data flow graphs
    - Both full fine-tuning and LoRA adaptation
    - CPU and GPU training
    - Integration with main.py command-line interface
    """
    
    def __init__(self, 
                 model_name: Optional[str] = None, 
                 use_lora: bool = False,
                 output_dir: str = "./models/erlang_graphcodebert"):
        """Initialize trainer.
        
        Args:
            model_name: Pre-trained model name (from config if None)
            use_lora: Whether to use LoRA adaptation
            output_dir: Directory to save trained model
        """
        self.device = check_device_setup()
        self.use_lora = use_lora and HAS_LORA
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if use_lora and not HAS_LORA:
            logger.warning("LoRA requested but PEFT not available - using full fine-tuning")
            logger.info("Install PEFT with: pip install peft")
        
        self.model_name = model_name or GRAPHCODEBERT_CONFIG['model_name']
        self.model = None
        self.tokenizer = None
        
        # Training state
        self.global_step = 0
        self.best_loss = float('inf')
        
        logger.info(f"GraphCodeBERT Trainer initialized")
        logger.info(f"  Model: {self.model_name}")
        logger.info(f"  LoRA: {self.use_lora}")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Output: {self.output_dir}")
    
    def setup_model_and_tokenizer(self):
        """Initialize model and tokenizer with pipeline integration."""
        logger.info("Setting up model and tokenizer...")
        
        # Load tokenizer
        self.tokenizer = RobertaTokenizer.from_pretrained(
            GRAPHCODEBERT_CONFIG['tokenizer_name']
        )
        
        # Create GraphCodeBERT model
        self.model = create_graphcodebert_model(self.model_name)
        
        # Apply LoRA if requested
        if self.use_lora:
            if not HAS_LORA:
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
        
        # Move to device
        self.model.to(self.device)
        
        # Log model info
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Model loaded: {total_params:,} total parameters, {trainable_params:,} trainable")
    
    def train(self, 
              train_file: str, 
              val_file: Optional[str] = None, 
              batch_size: Optional[int] = None,
              num_epochs: Optional[int] = None,
              learning_rate: Optional[float] = None) -> Dict[str, Any]:
        """Train the GraphCodeBERT model.
        
        Args:
            train_file: Path to training JSONL file
            val_file: Path to validation JSONL file (optional)
            batch_size: Training batch size (from config if None)
            num_epochs: Number of training epochs (from config if None)
            learning_rate: Learning rate (from config if None)
            
        Returns:
            Training results dictionary
        """
        logger.info("Starting GraphCodeBERT MLM training...")
        
        # Setup model and tokenizer if not already done
        if self.model is None:
            self.setup_model_and_tokenizer()
        
        # Save initial checkpoint before training
        logger.info("Saving initial checkpoint before training...")
        self.save_model("initial")
        logger.info("✓ Initial checkpoint saved")

        # Use config values with overrides
        effective_batch_size = batch_size or FINETUNING_CONFIG['batch_size']
        effective_num_epochs = num_epochs or FINETUNING_CONFIG['num_epochs']
        effective_learning_rate = learning_rate or FINETUNING_CONFIG['learning_rate']
        
        # Create dataloaders using our integrated function
        train_dataloader, val_dataloader = create_graphcodebert_dataloader(
            train_file=train_file,
            val_file=val_file,
            tokenizer=self.tokenizer,
            batch_size=effective_batch_size,
            shuffle=True
        )
        
        if not train_dataloader:
            raise ValueError(f"No training data loaded from {train_file}")
        
        # Setup optimizer and scheduler
        optimizer = AdamW(
            self.model.parameters(),
            lr=effective_learning_rate,
            eps=FINETUNING_CONFIG['adam_epsilon'],
            weight_decay=FINETUNING_CONFIG['weight_decay']
        )
        
        total_steps = len(train_dataloader) * effective_num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=FINETUNING_CONFIG['warmup_steps'],
            num_training_steps=total_steps
        )
        
        # Training loop
        logger.info(f"Training configuration:")
        logger.info(f"  Epochs: {effective_num_epochs}")
        logger.info(f"  Batch size: {effective_batch_size}")
        logger.info(f"  Learning rate: {effective_learning_rate}")
        logger.info(f"  Total steps: {total_steps}")
        logger.info(f"  Warmup steps: {FINETUNING_CONFIG['warmup_steps']}")
        
        results = self._training_loop(
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            num_epochs=effective_num_epochs
        )
        
        # Save final model
        self.save_model()
        
        logger.info("Training completed successfully!")
        return results
    
    def _training_loop(self, 
                      train_dataloader: DataLoader,
                      val_dataloader: Optional[DataLoader],
                      optimizer: torch.optim.Optimizer,
                      scheduler: torch.optim.lr_scheduler.LambdaLR,
                      num_epochs: int) -> Dict[str, Any]:
        """Main training loop."""
        self.model.train()
        
        epoch_losses = []
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch + 1}/{num_epochs}")
            
            # Training
            epoch_loss = self._train_epoch(train_dataloader, optimizer, scheduler)
            epoch_losses.append(epoch_loss)
            
            logger.info(f"Epoch {epoch + 1} - Training loss: {epoch_loss:.4f}")
            
            # Validation
            if val_dataloader:
                val_loss = self._validate_epoch(val_dataloader)
                logger.info(f"Epoch {epoch + 1} - Validation loss: {val_loss:.4f}")
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_model(suffix="best")
                    logger.info(f"New best validation loss: {val_loss:.4f}")
            
            # Save checkpoint every epoch
            self.save_model(suffix=f"epoch_{epoch + 1}")
        
        return {
            'epoch_losses': epoch_losses,
            'best_validation_loss': best_val_loss if val_dataloader else None,
            'final_loss': epoch_losses[-1] if epoch_losses else None,
        }
    
    def _train_epoch(self,
                    dataloader: DataLoader,
                    optimizer: torch.optim.Optimizer,
                    scheduler: torch.optim.lr_scheduler.LambdaLR) -> float:
        """Train for one epoch with structure-aware losses."""
        self.model.train()
        total_loss = 0.0
        total_mlm_loss = 0.0
        total_edge_loss = 0.0
        total_alignment_loss = 0.0
        num_batches = len(dataloader)

        progress_bar = tqdm(dataloader, desc="Training", disable=False)

        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}

            # Forward pass
            optimizer.zero_grad()

            outputs = self.model(
                input_ids=batch['input_ids'],
                position_idx=batch['position_idx'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels'],
                edge_candidates=batch.get('edge_candidates'),
                edge_labels=batch.get('edge_labels'),
                alignment_candidates=batch.get('alignment_candidates'),
                alignment_labels=batch.get('alignment_labels'),
                dfg_start_idx=batch.get('dfg_start_idx')
            )

            loss = outputs['loss']

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), FINETUNING_CONFIG['max_grad_norm'])
            optimizer.step()
            scheduler.step()

            # Update metrics
            total_loss += loss.item()
            if outputs.get('mlm_loss') is not None:
                total_mlm_loss += outputs['mlm_loss'].item()
            if outputs.get('edge_loss') is not None:
                total_edge_loss += outputs['edge_loss'].item()
            if outputs.get('alignment_loss') is not None:
                total_alignment_loss += outputs['alignment_loss'].item()

            self.global_step += 1

            # Update progress bar with individual loss components
            avg_loss = total_loss / (batch_idx + 1)
            postfix = {
                'loss': f'{loss.item():.4f}',
                'avg': f'{avg_loss:.4f}',
                'lr': f'{scheduler.get_last_lr()[0]:.2e}'
            }

            # Add individual losses if available
            if outputs.get('mlm_loss') is not None:
                postfix['mlm'] = f'{outputs["mlm_loss"].item():.3f}'
            if outputs.get('edge_loss') is not None:
                postfix['edge'] = f'{outputs["edge_loss"].item():.3f}'
            if outputs.get('alignment_loss') is not None:
                postfix['align'] = f'{outputs["alignment_loss"].item():.3f}'

            progress_bar.set_postfix(postfix)

        # Log average losses
        avg_epoch_loss = total_loss / num_batches
        logger.info(f"  Average losses - Total: {avg_epoch_loss:.4f}, "
                   f"MLM: {total_mlm_loss/num_batches:.4f}, "
                   f"Edge: {total_edge_loss/num_batches:.4f}, "
                   f"Align: {total_alignment_loss/num_batches:.4f}")

        return avg_epoch_loss
    
    def _validate_epoch(self, dataloader: DataLoader) -> float:
        """Validate for one epoch with structure-aware losses."""
        self.model.eval()
        total_loss = 0.0
        total_mlm_loss = 0.0
        total_edge_loss = 0.0
        total_alignment_loss = 0.0
        num_batches = len(dataloader)

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validation", disable=False):
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}

                # Forward pass
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    position_idx=batch['position_idx'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels'],
                    edge_candidates=batch.get('edge_candidates'),
                    edge_labels=batch.get('edge_labels'),
                    alignment_candidates=batch.get('alignment_candidates'),
                    alignment_labels=batch.get('alignment_labels'),
                    dfg_start_idx=batch.get('dfg_start_idx')
                )

                loss = outputs['loss']
                total_loss += loss.item()

                # Track individual losses
                if outputs.get('mlm_loss') is not None:
                    total_mlm_loss += outputs['mlm_loss'].item()
                if outputs.get('edge_loss') is not None:
                    total_edge_loss += outputs['edge_loss'].item()
                if outputs.get('alignment_loss') is not None:
                    total_alignment_loss += outputs['alignment_loss'].item()

        # Log average validation losses
        avg_val_loss = total_loss / num_batches
        logger.info(f"  Validation losses - Total: {avg_val_loss:.4f}, "
                   f"MLM: {total_mlm_loss/num_batches:.4f}, "
                   f"Edge: {total_edge_loss/num_batches:.4f}, "
                   f"Align: {total_alignment_loss/num_batches:.4f}")

        return avg_val_loss
    
    def save_model(self, suffix: str = "final"):
        """Save model and tokenizer."""
        save_dir = self.output_dir / f"checkpoint-{suffix}"
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        if self.use_lora:
            # Save LoRA weights
            self.model.save_pretrained(save_dir)
        else:
            # Save full model
            torch.save(self.model.state_dict(), save_dir / "model.bin")
            
        # Save tokenizer
        self.tokenizer.save_pretrained(save_dir)
        
        # Save config
        config_dict = {
            'model_name': self.model_name,
            'use_lora': self.use_lora,
            'global_step': self.global_step,
        }
        with open(save_dir / "training_config.json", 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        logger.info(f"Model saved to {save_dir}")
    
    def load_model(self, checkpoint_path: str):
        """Load model from checkpoint."""
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Load config
        config_file = checkpoint_path / "training_config.json"
        if config_file.exists():
            with open(config_file, 'r') as f:
                config = json.load(f)
                self.use_lora = config.get('use_lora', False)
                self.global_step = config.get('global_step', 0)
        
        # Setup model and tokenizer
        self.setup_model_and_tokenizer()
        
        # Load weights
        if self.use_lora:
            # Load LoRA weights
            self.model = self.model.from_pretrained(checkpoint_path)
        else:
            # Load full model
            model_path = checkpoint_path / "model.bin"
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
        
        logger.info(f"Model loaded from {checkpoint_path}")


def main():
    """Main training function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Train GraphCodeBERT for MLM on Erlang code",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data arguments
    parser.add_argument("--train-file", type=str, required=True,
                       help="Path to training JSONL file")
    parser.add_argument("--val-file", type=str, default=None,
                       help="Path to validation JSONL file")
    parser.add_argument("--output-dir", type=str, default="./models/erlang_graphcodebert",
                       help="Directory to save trained model")
    
    # Model arguments
    parser.add_argument("--model-name", type=str, default=None,
                       help="Pre-trained model name (from config if not specified)")
    parser.add_argument("--use-lora", action="store_true",
                       help="Use LoRA adaptation instead of full fine-tuning")
    
    # Training arguments
    parser.add_argument("--batch-size", type=int, default=None,
                       help="Training batch size (from config if not specified)")
    parser.add_argument("--num-epochs", type=int, default=None,
                       help="Number of training epochs (from config if not specified)")
    parser.add_argument("--learning-rate", type=float, default=None,
                       help="Learning rate (from config if not specified)")
    
    # Other arguments
    parser.add_argument("--resume-from", type=str, default=None,
                       help="Resume training from checkpoint")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug logging")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(Path(args.output_dir) / "training.log")
        ]
    )
    
    logger.info("=" * 80)
    logger.info("GRAPHCODEBERT MLM TRAINING ON ERLANG CODE")
    logger.info("=" * 80)
    
    # Log arguments
    logger.info("Training arguments:")
    for arg, value in vars(args).items():
        logger.info(f"  {arg}: {value}")
    
    try:
        # Create trainer
        trainer = GraphCodeBERTTrainer(
            model_name=args.model_name,
            use_lora=args.use_lora,
            output_dir=args.output_dir
        )
        
        # Resume from checkpoint if specified
        if args.resume_from:
            trainer.load_model(args.resume_from)
            logger.info(f"Resumed training from {args.resume_from}")
        
        # Start training
        start_time = time.time()
        
        results = trainer.train(
            train_file=args.train_file,
            val_file=args.val_file,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate
        )
        
        end_time = time.time()
        training_time = end_time - start_time
        
        # Log results
        logger.info("=" * 80)
        logger.info("TRAINING COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        logger.info(f"Training time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
        
        if results['epoch_losses']:
            logger.info(f"Final training loss: {results['final_loss']:.4f}")
            logger.info(f"Best training loss: {min(results['epoch_losses']):.4f}")
        
        if results['best_validation_loss']:
            logger.info(f"Best validation loss: {results['best_validation_loss']:.4f}")
        
        logger.info(f"Model saved to: {trainer.output_dir}")
        
        # Save training results
        results_file = trainer.output_dir / "training_results.json"
        results['training_time_seconds'] = training_time
        results['arguments'] = vars(args)
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Training results saved to: {results_file}")
        
        return 0
        
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
        return 130
        
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

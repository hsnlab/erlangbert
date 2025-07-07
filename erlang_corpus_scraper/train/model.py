"""
GraphCodeBERT Model Implementation for MLM Training on Erlang Code

This module provides the GraphCodeBERT model architecture specifically designed
for Masked Language Modeling (MLM) pre-training on Erlang code. It implements
the full GraphCodeBERT architecture with graph-guided masked attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, Tuple
import logging

from transformers import (
    RobertaModel, RobertaConfig, RobertaForMaskedLM,
    RobertaTokenizer
)

logger = logging.getLogger(__name__)

class GraphCodeBERTModel(nn.Module):
    """GraphCodeBERT model for Erlang code MLM training.
    
    Implements the full GraphCodeBERT architecture as described in the paper:
    - RoBERTa base for token representations
    - Graph-guided masked attention for incorporating data flow
    - MLM head for masked language modeling
    """
    
    def __init__(self, config: RobertaConfig):
        """Initialize GraphCodeBERT model.
        
        Args:
            config: RoBERTa configuration object
        """
        super().__init__()
        self.config = config
        
        # Load pre-trained RoBERTa as base encoder
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        
        # MLM head for masked language modeling
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = self.roberta.embeddings.word_embeddings.weight
        
        logger.info(f"GraphCodeBERT model initialized with {config.hidden_size}d hidden size")
    
    def forward(self, 
                input_ids: torch.Tensor,
                position_idx: torch.Tensor, 
                attention_mask: torch.Tensor,
                labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass through GraphCodeBERT model.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            position_idx: Position indices [batch_size, seq_len] 
            attention_mask: Graph-guided attention mask [batch_size, seq_len, seq_len]
            labels: MLM labels [batch_size, seq_len] (-100 for non-masked tokens)
            
        Returns:
            Dictionary containing:
                - logits: MLM prediction logits [batch_size, seq_len, vocab_size]
                - loss: MLM loss (if labels provided)
                - hidden_states: Last hidden states [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len = input_ids.shape
        
        # Create embeddings with position information
        embeddings = self._create_embeddings(input_ids, position_idx)
        
        # Convert 3D attention mask to 4D for multi-head attention
        # GraphCodeBERT uses custom attention pattern
        extended_attention_mask = self._prepare_attention_mask(attention_mask)
        
        # Pass through RoBERTa encoder with custom attention
        encoder_outputs = self.roberta.encoder(
            embeddings,
            attention_mask=extended_attention_mask,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_values=None,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        
        hidden_states = encoder_outputs.last_hidden_state
        
        # MLM prediction head
        logits = self.lm_head(hidden_states)
        
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            # Only compute loss on masked tokens (labels != -100)
            loss_fct = nn.CrossEntropyLoss()
            
            # Flatten for loss computation
            masked_lm_loss = loss_fct(
                logits.view(-1, self.config.vocab_size),
                labels.view(-1)
            )
            loss = masked_lm_loss
        
        return {
            'logits': logits,
            'loss': loss,
            'hidden_states': hidden_states,
        }
    
    def _create_embeddings(self, input_ids: torch.Tensor, position_idx: torch.Tensor) -> torch.Tensor:
        """Create embeddings with GraphCodeBERT position encoding.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            position_idx: Position indices [batch_size, seq_len]
            
        Returns:
            Input embeddings [batch_size, seq_len, hidden_size]
        """
        # Get word embeddings
        embeddings = self.roberta.embeddings.word_embeddings(input_ids)
        
        # Create position embeddings based on position_idx
        # position_idx: 0=special tokens, 1=padding, 2+=code tokens
        seq_len = input_ids.size(1)
        
        # Create position IDs for transformer
        # Use position_idx directly but ensure proper range
        position_ids = torch.clamp(position_idx, 0, self.config.max_position_embeddings - 1)
        
        # Get position embeddings
        position_embeddings = self.roberta.embeddings.position_embeddings(position_ids)
        
        # Get token type embeddings (all zeros for MLM)
        token_type_ids = torch.zeros_like(input_ids, dtype=torch.long, device=input_ids.device)
        token_type_embeddings = self.roberta.embeddings.token_type_embeddings(token_type_ids)
        
        # Combine embeddings
        embeddings = embeddings + position_embeddings + token_type_embeddings
        embeddings = self.roberta.embeddings.LayerNorm(embeddings)
        embeddings = self.roberta.embeddings.dropout(embeddings)
        
        return embeddings
    
    def _prepare_attention_mask(self, attention_mask: torch.Tensor) -> torch.Tensor:
        """Prepare 3D attention mask for multi-head attention.
        
        Args:
            attention_mask: Graph-guided mask [batch_size, seq_len, seq_len]
            
        Returns:
            Extended attention mask [batch_size, 1, seq_len, seq_len]
        """
        batch_size, seq_len, _ = attention_mask.shape
        
        # Convert boolean mask to float with -inf for masked positions
        # GraphCodeBERT attention: True=attend, False=mask
        extended_attention_mask = attention_mask.unsqueeze(1).float()  # [batch, 1, seq, seq]
        
        # Convert to attention scores: 0.0 for attend, -inf for mask
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        
        return extended_attention_mask
    
    def get_input_embeddings(self):
        """Get input embeddings for compatibility."""
        return self.roberta.embeddings.word_embeddings
    
    def set_input_embeddings(self, value):
        """Set input embeddings for compatibility."""
        self.roberta.embeddings.word_embeddings = value
        self.lm_head.weight = value
    
    def get_output_embeddings(self):
        """Get output embeddings for compatibility."""
        return self.lm_head
    
    def set_output_embeddings(self, new_embeddings):
        """Set output embeddings for compatibility."""
        self.lm_head = new_embeddings


class GraphCodeBERTForMLM(nn.Module):
    """Wrapper class that provides RobertaForMaskedLM-compatible interface.
    
    This class wraps GraphCodeBERTModel to provide the same interface as
    RobertaForMaskedLM while supporting graph-guided attention.
    """
    
    def __init__(self, config: RobertaConfig):
        super().__init__()
        self.config = config
        self.roberta = GraphCodeBERTModel(config)
    
    def forward(self, 
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                position_idx: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                **kwargs) -> Dict[str, torch.Tensor]:
        """Forward pass compatible with HuggingFace Trainer.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Graph-guided attention mask [batch_size, seq_len, seq_len]
            position_idx: Position indices [batch_size, seq_len]
            labels: MLM labels [batch_size, seq_len]
            **kwargs: Additional arguments (ignored)
            
        Returns:
            ModelOutput-like dictionary with loss and logits
        """
        if attention_mask is None:
            # Create default attention mask if not provided
            attention_mask = torch.ones(
                input_ids.shape + (input_ids.shape[1],), 
                dtype=torch.bool, 
                device=input_ids.device
            )
        
        if position_idx is None:
            # Create default position indices
            position_idx = torch.arange(
                input_ids.shape[1], 
                dtype=torch.long, 
                device=input_ids.device
            ).unsqueeze(0).expand(input_ids.shape)
        
        return self.roberta(
            input_ids=input_ids,
            position_idx=position_idx,
            attention_mask=attention_mask,
            labels=labels
        )
    
    def get_input_embeddings(self):
        return self.roberta.get_input_embeddings()
    
    def set_input_embeddings(self, value):
        self.roberta.set_input_embeddings(value)
    
    def get_output_embeddings(self):
        return self.roberta.get_output_embeddings()
    
    def set_output_embeddings(self, new_embeddings):
        self.roberta.set_output_embeddings(new_embeddings)


def create_graphcodebert_model(model_name: str = "microsoft/graphcodebert-base") -> GraphCodeBERTForMLM:
    """Factory function to create GraphCodeBERT model for MLM.
    
    Args:
        model_name: Pre-trained model name or path
        
    Returns:
        Initialized GraphCodeBERT model for MLM training
    """
    try:
        # Load configuration
        config = RobertaConfig.from_pretrained(model_name)
        
        # Create model
        model = GraphCodeBERTForMLM(config)
        
        # Load pre-trained weights into RoBERTa components
        pretrained_model = RobertaForMaskedLM.from_pretrained(model_name)
        
        # Transfer weights from pre-trained model
        model.roberta.roberta.load_state_dict(
            pretrained_model.roberta.state_dict(), 
            strict=False
        )
        model.roberta.lm_head.load_state_dict(
            pretrained_model.lm_head.state_dict(),
            strict=False
        )
        
        logger.info(f"Created GraphCodeBERT model from {model_name}")
        return model
        
    except Exception as e:
        logger.error(f"Failed to create GraphCodeBERT model: {e}")
        raise


def check_device_setup():
    """Check device setup and warn about CPU training."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if device.type == "cpu":
        logger.info("Running on CPU - training will be slower but functional")
        logger.info("Tip: Consider reducing batch size for better performance")
    else:
        logger.info(f"Using GPU - {torch.cuda.get_device_name()}")
    
    return device


if __name__ == "__main__":
    # Test model creation and basic functionality
    print("Testing GraphCodeBERT model creation...")
    
    # Create model
    try:
        model = create_graphcodebert_model()
        device = check_device_setup()
        model = model.to(device)
        
        # Test forward pass
        batch_size, seq_len = 2, 64
        input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=device)
        position_idx = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        attention_mask = torch.ones((batch_size, seq_len, seq_len), dtype=torch.bool, device=device)
        labels = torch.randint(0, 1000, (batch_size, seq_len), device=device)
        
        # Forward pass
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                position_idx=position_idx,
                attention_mask=attention_mask,
                labels=labels
            )
        
        print(f"✓ Model forward pass successful!")
        print(f"  - Logits shape: {outputs['logits'].shape}")
        print(f"  - Loss: {outputs['loss'].item():.4f}")
        print(f"  - Hidden states shape: {outputs['hidden_states'].shape}")
        print(f"  - Device: {device}")
        
    except Exception as e:
        print(f"✗ Model test failed: {e}")
        raise
    
    print("\n✓ GraphCodeBERT model ready for MLM training!")

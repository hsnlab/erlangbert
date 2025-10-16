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
from transformers.models.roberta.modeling_roberta import RobertaLMHead

from .checker import run_all_checks

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

        # Load pre-trained RoBERTa as base encoder (named 'encoder' to avoid double 'roberta.' prefix)
        self.encoder = RobertaModel(config, add_pooling_layer=False)

        # MLM head for masked language modeling (same structure as RobertaForMaskedLM)
        self.lm_head = RobertaLMHead(config)

        logger.info(f"GraphCodeBERT model initialized with {config.hidden_size}d hidden size")
    
    def forward(self,
                input_ids: torch.Tensor,
                position_idx: torch.Tensor,
                attention_mask: torch.Tensor,
                labels: Optional[torch.Tensor] = None,
                edge_candidates: Optional[torch.Tensor] = None,
                edge_labels: Optional[torch.Tensor] = None,
                alignment_candidates: Optional[torch.Tensor] = None,
                alignment_labels: Optional[torch.Tensor] = None,
                dfg_start_idx: Optional[int] = None,
                **kwargs) -> Dict[str, torch.Tensor]:
        """Forward pass through GraphCodeBERT model with structure-aware losses.

        Implements three pre-training tasks from GraphCodeBERT paper (Section 4.3):
        1. Masked Language Modeling (MLM)
        2. Edge Prediction (Equation 7)
        3. Node Alignment (Equation 8)

        Args:
            input_ids: Token IDs [batch_size, seq_len]
            position_idx: Position indices [batch_size, seq_len]
            attention_mask: Graph-guided attention mask [batch_size, seq_len, seq_len]
            labels: MLM labels [batch_size, seq_len] (-100 for non-masked tokens)
            edge_candidates: Edge candidate pairs [batch_size, num_edge_cand, 2]
            edge_labels: Edge existence labels [batch_size, num_edge_cand]
            alignment_candidates: Node-token alignment pairs [batch_size, num_align_cand, 2]
            alignment_labels: Alignment labels [batch_size, num_align_cand]
            dfg_start_idx: Starting position of DFG nodes in sequence (can be int or list)
            **kwargs: Additional arguments (ignored for compatibility)

        Returns:
            Dictionary containing:
                - logits: MLM prediction logits [batch_size, seq_len, vocab_size]
                - loss: Combined loss (if any labels provided)
                - mlm_loss: MLM loss component (for logging)
                - edge_loss: Edge prediction loss component (for logging)
                - alignment_loss: Node alignment loss component (for logging)
                - hidden_states: Last hidden states [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len = input_ids.shape
        
        # Create embeddings with position information
        embeddings = self._create_embeddings(input_ids, position_idx)
        
        # Convert 3D attention mask to 4D for multi-head attention
        # GraphCodeBERT uses custom attention pattern
        extended_attention_mask = self._prepare_attention_mask(attention_mask)
        
        # Pass through RoBERTa encoder with custom attention
        encoder_outputs = self.encoder.encoder(
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

        # Compute individual losses for the three pre-training tasks
        mlm_loss = None
        edge_loss = None
        alignment_loss = None

        # 1. Masked Language Modeling loss
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            mlm_loss = loss_fct(
                logits.view(-1, self.config.vocab_size),
                labels.view(-1)
            )

        # 2. Edge Prediction loss (Equation 7 from paper)
        if edge_candidates is not None and edge_labels is not None:
            # Handle dfg_start_idx as either int or list
            if isinstance(dfg_start_idx, (list, tuple)):
                # Batch processing: use first element (assumes same structure across batch)
                dfg_start = dfg_start_idx[0] if dfg_start_idx else 0
            else:
                dfg_start = dfg_start_idx if dfg_start_idx is not None else 0

            edge_loss = self._compute_edge_prediction_loss(
                hidden_states, edge_candidates, edge_labels, dfg_start
            )

        # 3. Node Alignment loss (Equation 8 from paper)
        if alignment_candidates is not None and alignment_labels is not None:
            # Handle dfg_start_idx as either int or list
            if isinstance(dfg_start_idx, (list, tuple)):
                dfg_start = dfg_start_idx[0] if dfg_start_idx else 0
            else:
                dfg_start = dfg_start_idx if dfg_start_idx is not None else 0

            alignment_loss = self._compute_node_alignment_loss(
                hidden_states, alignment_candidates, alignment_labels, dfg_start
            )

        # Combine losses (equal weighting as per paper)
        loss = None
        active_losses = []
        if mlm_loss is not None:
            active_losses.append(mlm_loss)
        if edge_loss is not None:
            active_losses.append(edge_loss)
        if alignment_loss is not None:
            active_losses.append(alignment_loss)

        if active_losses:
            loss = sum(active_losses) / len(active_losses)

        return {
            'logits': logits,
            'loss': loss,
            'mlm_loss': mlm_loss,
            'edge_loss': edge_loss,
            'alignment_loss': alignment_loss,
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
        embeddings = self.encoder.embeddings.word_embeddings(input_ids)

        # Create position embeddings based on position_idx
        # position_idx: 0=special tokens, 1=padding, 2+=code tokens
        seq_len = input_ids.size(1)

        # Create position IDs for transformer
        # Use position_idx directly but ensure proper range
        position_ids = torch.clamp(position_idx, 0, self.config.max_position_embeddings - 1)

        # Get position embeddings
        position_embeddings = self.encoder.embeddings.position_embeddings(position_ids)

        # Get token type embeddings (all zeros for MLM)
        token_type_ids = torch.zeros_like(input_ids, dtype=torch.long, device=input_ids.device)
        token_type_embeddings = self.encoder.embeddings.token_type_embeddings(token_type_ids)

        # Combine embeddings
        embeddings = embeddings + position_embeddings + token_type_embeddings
        embeddings = self.encoder.embeddings.LayerNorm(embeddings)
        embeddings = self.encoder.embeddings.dropout(embeddings)
        
        return embeddings

    def _compute_edge_prediction_loss(self,
                                     hidden_states: torch.Tensor,
                                     edge_candidates: torch.Tensor,
                                     edge_labels: torch.Tensor,
                                     dfg_start_idx: int) -> torch.Tensor:
        """Compute edge prediction loss (GraphCodeBERT paper Equation 7).

        Predicts whether edges exist between pairs of DFG nodes using dot product + sigmoid.

        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            edge_candidates: [batch_size, num_candidates, 2] - pairs of DFG node indices
            edge_labels: [batch_size, num_candidates] - binary labels (1=edge exists, 0=no edge)
            dfg_start_idx: Starting position of DFG nodes in sequence

        Returns:
            Edge prediction loss (scalar)
        """
        batch_size, num_candidates, _ = edge_candidates.shape
        seq_len = hidden_states.size(1)

        # Extract hidden states for DFG nodes
        # edge_candidates contains DFG node indices (0-indexed within DFG)
        # Need to offset by dfg_start_idx to get sequence positions
        from_indices = edge_candidates[:, :, 0] + dfg_start_idx  # [batch, num_candidates]
        to_indices = edge_candidates[:, :, 1] + dfg_start_idx    # [batch, num_candidates]

        # Clamp indices to valid range to prevent index out of bounds
        from_indices = torch.clamp(from_indices, 0, seq_len - 1)
        to_indices = torch.clamp(to_indices, 0, seq_len - 1)

        # Gather hidden states for the candidate pairs
        batch_indices = torch.arange(batch_size, device=hidden_states.device).unsqueeze(1).expand(-1, num_candidates)

        from_hidden = hidden_states[batch_indices, from_indices]  # [batch, num_candidates, hidden_size]
        to_hidden = hidden_states[batch_indices, to_indices]      # [batch, num_candidates, hidden_size]

        # Compute edge scores using scaled dot product (paper: p_ij = sigmoid(h_i^T · h_j))
        # Scale by sqrt(hidden_size) to prevent saturation (like scaled dot-product attention)
        hidden_size = hidden_states.size(-1)
        edge_scores = torch.sum(from_hidden * to_hidden, dim=-1) / (hidden_size ** 0.5)  # [batch, num_candidates]
        edge_probs = torch.sigmoid(edge_scores)  # [batch, num_candidates]

        # Binary cross-entropy loss: -(y*log(p) + (1-y)*log(1-p))
        # Mask out padded candidates (where label is 0 and both indices are 0)
        valid_mask = (from_indices != dfg_start_idx) | (to_indices != dfg_start_idx) | (edge_labels > 0)

        loss = F.binary_cross_entropy(edge_probs, edge_labels, reduction='none')

        # Compute masked average, return 0 if no valid candidates
        num_valid = valid_mask.float().sum()
        if num_valid < 1.0:
            return torch.tensor(0.0, device=loss.device, dtype=loss.dtype)

        loss = (loss * valid_mask.float()).sum() / num_valid

        return loss

    def _compute_node_alignment_loss(self,
                                    hidden_states: torch.Tensor,
                                    alignment_candidates: torch.Tensor,
                                    alignment_labels: torch.Tensor,
                                    dfg_start_idx: int) -> torch.Tensor:
        """Compute node alignment loss (GraphCodeBERT paper Equation 8).

        Predicts whether DFG nodes align with specific code tokens using dot product + sigmoid.

        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            alignment_candidates: [batch_size, num_candidates, 2] - pairs of (node_idx, code_token_idx)
            alignment_labels: [batch_size, num_candidates] - binary labels (1=aligned, 0=not aligned)
            dfg_start_idx: Starting position of DFG nodes in sequence

        Returns:
            Node alignment loss (scalar)
        """
        batch_size, num_candidates, _ = alignment_candidates.shape
        seq_len = hidden_states.size(1)

        # Extract indices
        node_indices = alignment_candidates[:, :, 0] + dfg_start_idx  # DFG node positions in sequence
        code_indices = alignment_candidates[:, :, 1]                   # Code token positions (already absolute)

        # Clamp indices to valid range to prevent index out of bounds
        node_indices = torch.clamp(node_indices, 0, seq_len - 1)
        code_indices = torch.clamp(code_indices, 0, seq_len - 1)

        # Gather hidden states
        batch_indices = torch.arange(batch_size, device=hidden_states.device).unsqueeze(1).expand(-1, num_candidates)

        node_hidden = hidden_states[batch_indices, node_indices]  # [batch, num_candidates, hidden_size]
        code_hidden = hidden_states[batch_indices, code_indices]  # [batch, num_candidates, hidden_size]

        # Compute alignment scores using scaled dot product (paper: p_ij = sigmoid(h_node^T · h_token))
        # Scale by sqrt(hidden_size) to prevent saturation (like scaled dot-product attention)
        hidden_size = hidden_states.size(-1)
        alignment_scores = torch.sum(node_hidden * code_hidden, dim=-1) / (hidden_size ** 0.5)  # [batch, num_candidates]
        alignment_probs = torch.sigmoid(alignment_scores)  # [batch, num_candidates]

        # Binary cross-entropy loss with padding mask
        valid_mask = (node_indices != dfg_start_idx) | (code_indices != 0) | (alignment_labels > 0)

        loss = F.binary_cross_entropy(alignment_probs, alignment_labels, reduction='none')

        # Compute masked average, return 0 if no valid candidates
        num_valid = valid_mask.float().sum()
        if num_valid < 1.0:
            return torch.tensor(0.0, device=loss.device, dtype=loss.dtype)

        loss = (loss * valid_mask.float()).sum() / num_valid

        return loss

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
        return self.encoder.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        """Set input embeddings for compatibility."""
        self.encoder.embeddings.word_embeddings = value
        # RobertaLMHead uses decoder.weight, not weight
        self.lm_head.decoder.weight = value
    
    def get_output_embeddings(self):
        """Get output embeddings for compatibility."""
        return self.lm_head
    
    def set_output_embeddings(self, new_embeddings):
        """Set output embeddings for compatibility."""
        self.lm_head = new_embeddings


class GraphCodeBERTForMLM(nn.Module):
    """Wrapper class that provides RobertaForMaskedLM-compatible interface.

    This class wraps GraphCodeBERTModel to provide the same interface as
    RobertaForMaskedLM while supporting graph-guided attention and structure-aware
    pre-training tasks (edge prediction and node alignment).
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
                edge_candidates: Optional[torch.Tensor] = None,
                edge_labels: Optional[torch.Tensor] = None,
                alignment_candidates: Optional[torch.Tensor] = None,
                alignment_labels: Optional[torch.Tensor] = None,
                dfg_start_idx: Optional[int] = None,
                **kwargs) -> Dict[str, torch.Tensor]:
        """Forward pass compatible with HuggingFace Trainer with structure-aware tasks.

        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Graph-guided attention mask [batch_size, seq_len, seq_len]
            position_idx: Position indices [batch_size, seq_len]
            labels: MLM labels [batch_size, seq_len]
            edge_candidates: Edge candidate pairs [batch_size, num_edge_cand, 2]
            edge_labels: Edge existence labels [batch_size, num_edge_cand]
            alignment_candidates: Node-token alignment pairs [batch_size, num_align_cand, 2]
            alignment_labels: Alignment labels [batch_size, num_align_cand]
            dfg_start_idx: Starting position of DFG nodes in sequence
            **kwargs: Additional arguments (ignored for compatibility)

        Returns:
            ModelOutput-like dictionary with loss, logits, and individual loss components
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
            labels=labels,
            edge_candidates=edge_candidates,
            edge_labels=edge_labels,
            alignment_candidates=alignment_candidates,
            alignment_labels=alignment_labels,
            dfg_start_idx=dfg_start_idx
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
    """
    Create GraphCodeBERT model for MLM with proper weight transfer.
    
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

        # Transfer RoBERTa encoder weights to the inner model (now named 'encoder')
        model.roberta.encoder.load_state_dict(pretrained_model.roberta.state_dict(), strict=True)

        # Transfer full LM head weights (RobertaLMHead with dense, layer_norm, decoder)
        model.roberta.lm_head.load_state_dict(pretrained_model.lm_head.state_dict(), strict=True)

        # UNCOMMENT THIS TO CHECK THE LOADED MODEL
        #
        # tokenizer = RobertaTokenizer.from_pretrained(model_name)
        # issues, baseline_preds = run_all_checks(model, tokenizer, model_name)
    
        # if issues:
        #     logger.warning(f"Model validation found issues: {issues}")
        # else:
        #     logger.info("✓ Model validation passed")
        
        logger.info(f"✓ GraphCodeBERT model created from {model_name}")
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

"""
Checkpoint saving and loading utilities for GraphCodeBERT models.

Handles conversion between GraphCodeBERTForMLM checkpoint format and
RobertaForMaskedLM-compatible format for evaluation.
"""

import torch
from pathlib import Path
from typing import Dict, Any, Union
import logging

logger = logging.getLogger(__name__)


def save_checkpoint(model, config: Any, save_path: Union[str, Path]) -> None:
    """Save GraphCodeBERT model checkpoint in RobertaForMaskedLM-compatible format.

    Converts internal model structure to be compatible with RobertaForMaskedLM:
    - roberta.encoder.* → roberta.*
    - roberta.lm_head.* → lm_head.*

    Args:
        model: GraphCodeBERTForMLM model instance
        config: Model configuration object
        save_path: Path to save checkpoint (will be named model.bin)
    """
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    # Get model state dict
    state_dict = model.state_dict()

    # Fix keys to be compatible with RobertaForMaskedLM
    fixed_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('roberta.encoder.'):
            # New format: roberta.encoder.* → roberta.*
            k = k.replace('roberta.encoder.', 'roberta.', 1)
        elif k.startswith('roberta.lm_head.'):
            # roberta.lm_head.* → lm_head.*
            k = k.replace('roberta.lm_head.', 'lm_head.', 1)
        fixed_state_dict[k] = v

    checkpoint = {
        'model_state_dict': fixed_state_dict,
        'config': config,
    }

    torch.save(checkpoint, save_path / "model.bin")
    logger.info(f"Saved checkpoint to {save_path / 'model.bin'}")


def load_checkpoint_for_eval(checkpoint_path: Union[str, Path],
                             model_class,
                             device: str = 'cpu') -> Any:
    """Load checkpoint for evaluation with RobertaForMaskedLM.

    Handles both old and new checkpoint formats:
    - Old: roberta.roberta.* (from double nesting)
    - New: roberta.encoder.* (from encoder naming)

    Both are converted to roberta.* for RobertaForMaskedLM compatibility.

    Args:
        checkpoint_path: Path to checkpoint directory or file
        model_class: Model class to load into (usually RobertaForMaskedLM)
        device: Device to load checkpoint on

    Returns:
        Loaded model instance

    Raises:
        FileNotFoundError: If checkpoint file not found
        ValueError: If checkpoint format is invalid
    """
    checkpoint_path = Path(checkpoint_path)

    # Find model file
    if checkpoint_path.is_file():
        model_file = checkpoint_path
    elif checkpoint_path.is_dir():
        # Look for common model file names
        candidates = ['model.bin', 'pytorch_model.bin', 'model.pt']
        model_file = None
        for candidate in candidates:
            candidate_path = checkpoint_path / candidate
            if candidate_path.exists():
                model_file = candidate_path
                logger.info(f"Found model file: {candidate}")
                break
        if model_file is None:
            raise FileNotFoundError(f"No model file found in {checkpoint_path}")
    else:
        raise FileNotFoundError(f"Checkpoint path does not exist: {checkpoint_path}")

    # Load checkpoint
    logger.info(f"Loading checkpoint from: {model_file}")
    checkpoint = torch.load(model_file, map_location=device, weights_only=False)

    # Check format
    if not isinstance(checkpoint, dict) or 'model_state_dict' not in checkpoint:
        raise ValueError(
            f"Invalid checkpoint format. Expected dict with 'model_state_dict', "
            f"got: {list(checkpoint.keys()) if isinstance(checkpoint, dict) else type(checkpoint)}"
        )

    state_dict = checkpoint['model_state_dict']
    logger.info(f"Loaded checkpoint with {len(state_dict)} keys")

    # Load baseline model
    logger.info("Loading baseline model...")
    model = model_class.from_pretrained('microsoft/graphcodebert-base')

    # Extract RoBERTa-compatible weights
    roberta_state = {k: v for k, v in state_dict.items()
                    if k.startswith('roberta.') or k.startswith('lm_head.')}

    logger.info(f"Extracted {len(roberta_state)} RoBERTa-compatible keys")

    # Fix keys to match RobertaForMaskedLM structure
    # Handles both old (roberta.roberta.*) and new (roberta.encoder.*) formats
    fixed_state = {}
    for k, v in roberta_state.items():
        original_k = k
        if k.startswith('roberta.roberta.'):
            # Old checkpoint format: roberta.roberta.* → roberta.*
            k = k.replace('roberta.roberta.', 'roberta.', 1)
        elif k.startswith('roberta.encoder.'):
            # New checkpoint format: roberta.encoder.* → roberta.*
            k = k.replace('roberta.encoder.', 'roberta.', 1)
        elif k.startswith('roberta.lm_head.'):
            # Both formats: roberta.lm_head.* → lm_head.*
            k = k.replace('roberta.lm_head.', 'lm_head.', 1)
        fixed_state[k] = v

    logger.info(f"Fixed keys: {len(fixed_state)} keys after prefix normalization")

    # Load weights
    missing, unexpected = model.load_state_dict(fixed_state, strict=False)

    # Log results
    logger.info(f"✓ Loaded {len(fixed_state)} weights from checkpoint")
    if unexpected:
        logger.warning(f"Unexpected keys (ignored): {len(unexpected)} - {unexpected[:3]}")
    if missing:
        lm_head_missing = [k for k in missing if 'lm_head' in k]
        other_missing = [k for k in missing if 'lm_head' not in k]
        if lm_head_missing:
            logger.warning(f"Missing lm_head keys: {len(lm_head_missing)} - {lm_head_missing[:3]}")
        if other_missing:
            logger.warning(f"Missing non-lm_head keys: {len(other_missing)} - {other_missing[:3]}")

    return model

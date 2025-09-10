import torch
import logging
from transformers import RobertaForMaskedLM

logger = logging.getLogger(__name__)

def check_weight_transfer(our_model, reference_model):
    """Check if weights were transferred correctly from pretrained model."""
    logger.info("=== WEIGHT TRANSFER CHECK ===")
    
    issues = []
    
    # Check 1: RoBERTa embeddings should match exactly
    our_embeddings = our_model.roberta.roberta.embeddings.word_embeddings.weight
    ref_embeddings = reference_model.roberta.embeddings.word_embeddings.weight
    
    if torch.allclose(our_embeddings, ref_embeddings, atol=1e-6):
        logger.info("✓ RoBERTa embeddings match perfectly")
    else:
        logger.error("✗ RoBERTa embeddings DON'T match!")
        diff = torch.abs(our_embeddings - ref_embeddings).max().item()
        logger.error(f"  Max difference: {diff}")
        issues.append("embeddings_mismatch")
    
    # Check 2: First transformer layer should match
    our_first_layer = our_model.roberta.roberta.encoder.layer[0].attention.self.query.weight
    ref_first_layer = reference_model.roberta.encoder.layer[0].attention.self.query.weight
    
    if torch.allclose(our_first_layer, ref_first_layer, atol=1e-6):
        logger.info("✓ First transformer layer matches")
    else:
        logger.error("✗ First transformer layer DON'T match!")
        diff = torch.abs(our_first_layer - ref_first_layer).max().item()
        logger.error(f"  Max difference: {diff}")
        issues.append("transformer_mismatch")
    
    # Check 3: LM head should match reference
    our_lm_head = our_model.roberta.lm_head.weight
    ref_lm_head = reference_model.lm_head.decoder.weight
    
    if torch.allclose(our_lm_head, ref_lm_head, atol=1e-6):
        logger.info("✓ LM head matches reference")
    else:
        logger.error("✗ LM head DON'T match reference!")
        diff = torch.abs(our_lm_head - ref_lm_head).max().item()
        logger.error(f"  Max difference: {diff}")
        issues.append("lm_head_mismatch")
    
    return issues


def check_weight_sharing(model):
    """Check if weight sharing is properly established."""
    logger.info("=== WEIGHT SHARING CHECK ===")
    
    issues = []
    
    # Check if LM head and embeddings share the same tensor (not just equal values)
    embeddings_weight = model.roberta.roberta.embeddings.word_embeddings.weight
    lm_head_weight = model.roberta.lm_head.weight
    
    # Check 1: Same memory location (is operator)
    if embeddings_weight is lm_head_weight:
        logger.info("✓ LM head and embeddings share the same tensor")
    else:
        logger.error("✗ LM head and embeddings are separate tensors!")
        issues.append("no_weight_sharing")
        
        # If not sharing, check if at least values are equal
        if torch.allclose(embeddings_weight, lm_head_weight, atol=1e-6):
            logger.info("  (Values are equal, but not shared)")
        else:
            logger.error("  (Values are also different!)")
            diff = torch.abs(embeddings_weight - lm_head_weight).max().item()
            logger.error(f"    Max difference: {diff}")
            issues.append("different_values")
    
    # Check 2: Gradient flow will be shared
    if embeddings_weight.requires_grad and lm_head_weight.requires_grad:
        logger.info("✓ Both tensors require gradients")
    else:
        logger.warning("⚠ Gradient requirements differ")
        logger.info(f"  Embeddings requires_grad: {embeddings_weight.requires_grad}")
        logger.info(f"  LM head requires_grad: {lm_head_weight.requires_grad}")
    
    return issues


def check_model_structure(model):
    """Check model structure and parameter counts."""
    logger.info("=== MODEL STRUCTURE CHECK ===")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Check key components exist
    checks = [
        ("roberta.roberta", "RoBERTa encoder"),
        ("roberta.lm_head", "LM head"),
        ("roberta.roberta.embeddings", "Embeddings"),
        ("roberta.roberta.encoder", "Transformer encoder"),
    ]
    
    issues = []
    for path, name in checks:
        try:
            component = model
            for attr in path.split('.'):
                component = getattr(component, attr)
            logger.info(f"✓ {name} found")
        except AttributeError:
            logger.error(f"✗ {name} NOT found at {path}")
            issues.append(f"missing_{name.lower().replace(' ', '_')}")
    
    return issues


def check_forward_pass(model, tokenizer):
    """Test that model forward pass works and produces reasonable outputs."""
    logger.info("=== FORWARD PASS CHECK ===")
    
    test_code = "func max(a, b int) int { if a > <mask> { return a } return b }"
    
    try:
        # Tokenize
        encoding = tokenizer(test_code, return_tensors="pt", truncation=True, max_length=64)
        input_ids = encoding.input_ids
        
        # Create required inputs
        batch_size, seq_len = input_ids.shape
        position_idx = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
        attention_mask = torch.ones((batch_size, seq_len, seq_len), dtype=torch.bool)
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                position_idx=position_idx,
                attention_mask=attention_mask
            )
        
        # Check outputs
        if 'logits' not in outputs:
            logger.error("✗ No 'logits' in model output")
            return ["no_logits"]
        
        logits = outputs['logits']
        logger.info(f"✓ Forward pass successful")
        logger.info(f"  Logits shape: {logits.shape}")
        logger.info(f"  Logits range: [{logits.min().item():.3f}, {logits.max().item():.3f}]")
        
        # Check for mask token and predictions
        mask_token_id = tokenizer.mask_token_id
        mask_positions = (input_ids == mask_token_id).nonzero(as_tuple=True)
        
        if len(mask_positions[0]) == 0:
            logger.warning("⚠ No mask token found in test")
            return []
        
        # Get predictions for first mask
        mask_pos = mask_positions[1][0].item()
        mask_logits = logits[0, mask_pos]
        top_token_ids = torch.topk(mask_logits, 5).indices
        top_tokens = tokenizer.convert_ids_to_tokens(top_token_ids.tolist())
        
        logger.info(f"✓ Top predictions for mask: {top_tokens}")
        
        # Check if predictions look reasonable (not random)
        reasonable = any(token.lower().strip('ġ') in ['b', 'a', 'int', 'return'] for token in top_tokens)
        if reasonable:
            logger.info("✓ Predictions look reasonable")
            return []
        else:
            logger.warning(f"⚠ Predictions may be unreasonable: {top_tokens}")
            return ["unreasonable_predictions"]
        
    except Exception as e:
        logger.error(f"✗ Forward pass failed: {e}")
        return ["forward_pass_failed"]


def check_pretrained_baseline(tokenizer, model_name="microsoft/graphcodebert-base"):
    """Test the original pretrained model as baseline."""
    logger.info("=== PRETRAINED BASELINE CHECK ===")
    
    try:
        # Load original model
        original_model = RobertaForMaskedLM.from_pretrained(model_name)
        original_model.eval()
        
        test_code = "func max(a, b int) int { if a > <mask> { return a } return b }"
        encoding = tokenizer(test_code, return_tensors="pt", truncation=True, max_length=64)
        
        with torch.no_grad():
            outputs = original_model(**encoding)
            logits = outputs.logits
        
        # Get predictions
        mask_token_id = tokenizer.mask_token_id
        mask_positions = (encoding.input_ids == mask_token_id).nonzero(as_tuple=True)
        
        if len(mask_positions[0]) > 0:
            mask_pos = mask_positions[1][0].item()
            mask_logits = logits[0, mask_pos]
            top_token_ids = torch.topk(mask_logits, 5).indices
            top_tokens = tokenizer.convert_ids_to_tokens(top_token_ids.tolist())
            
            logger.info(f"✓ Original model predictions: {top_tokens}")
            return top_tokens
        else:
            logger.warning("⚠ No mask in baseline test")
            return []
            
    except Exception as e:
        logger.error(f"✗ Baseline check failed: {e}")
        return []


def run_all_checks(model, tokenizer, model_name="microsoft/graphcodebert-base"):
    """Run all validation checks and return summary."""
    logger.info("Starting comprehensive model validation...")
    
    all_issues = []
    
    # Check 1: Weight transfer
    try:
        reference_model = RobertaForMaskedLM.from_pretrained(model_name)
        issues = check_weight_transfer(model, reference_model)
        all_issues.extend(issues)
    except Exception as e:
        logger.error(f"Weight transfer check failed: {e}")
        all_issues.append("weight_check_failed")
    
    # Check 2: Weight sharing
    issues = check_weight_sharing(model)
    all_issues.extend(issues)
    
    # Check 3: Model structure
    issues = check_model_structure(model)
    all_issues.extend(issues)
    
    # Check 4: Forward pass
    issues = check_forward_pass(model, tokenizer)
    all_issues.extend(issues)
    
    # Check 5: Baseline comparison
    baseline_predictions = check_pretrained_baseline(tokenizer, model_name)
    
    # Summary
    logger.info("=== VALIDATION SUMMARY ===")
    if not all_issues:
        logger.info("✓ All checks passed!")
    else:
        logger.error(f"✗ Found {len(all_issues)} issues:")
        for issue in all_issues:
            logger.error(f"  - {issue}")
    
    return all_issues, baseline_predictions

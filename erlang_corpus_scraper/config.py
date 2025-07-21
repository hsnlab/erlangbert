#!/usr/bin/env python3
"""
Configuration file for Erlang corpus scraper and GraphCodeBERT transformer.
"""

import os
from pathlib import Path

# === Output Configuration ===
OUTPUT_CONFIG = {
    'base_dir': './output',
    'repos_file': 'repositories.json',
    'clone_results_file': 'clone_results.json', 
    'functions_file': 'functions.jsonl',
    'graphcodebert_dir': 'graphcodebert_data',  # New: GraphCodeBERT output
    'failed_functions_file': 'failed_functions.json',
    'corpus_summary_file': 'corpus_summary.json'
}

# === GitHub API Configuration ===
GITHUB_CONFIG = {
    'base_url': 'https://api.github.com',
    'search_endpoint': '/search/repositories',
    'rate_limit_per_hour': 5000,  # For authenticated requests
    'request_timeout': 30,
    'max_retries': 3,
    'retry_delay': 1.0,
    'user_agent': 'Erlang-Corpus-Scraper/1.0'
}

# === Repository Discovery Configuration ===
DISCOVERY_CONFIG = {
    'search_queries': [
        'language:erlang stars:>10',
        'language:erlang forks:>5',
        'language:erlang pushed:>2020-01-01',
        'language:erlang topic:erlang',
        'language:erlang topic:otp',
        'language:erlang topic:distributed',
        'language:erlang topic:actor-model',
        'language:erlang topic:elixir',
        'language:erlang topic:phoenix'
    ],
    'min_stars': 5,
    'min_size': 100,  # KB
    'max_size': 100000,  # KB (100MB)
    'exclude_forks': False,
    'exclude_archived': True,
    'max_repos_per_query': 100,
    'max_total_repos': 1000
}

# === Cloning Configuration ===
CLONE_CONFIG = {
    'base_clone_dir': './cloned_repos',
    'max_concurrent_clones': 4,
    'clone_timeout': 300,  # 5 minutes
    'max_retries': 2,
    'retry_delay': 5.0,
    'depth': 1,  # Shallow clone
    'cleanup_on_failure': True
}

# === Parser Configuration ===
PARSER_CONFIG = {
    'erlang_extensions': ['.erl', '.hrl'],
    'max_file_size': 1024 * 1024,  # 1MB
    'encoding': 'utf-8',
    'encoding_fallbacks': ['latin1', 'cp1252'],
    'max_concurrent_parsers': 8,
    'timeout_per_file': 30,
    'max_function_size': 10000,  # characters
    'min_function_size': 20,     # characters
}

# === Function Scoring Configuration ===
FUNCTION_SCORING = {
    'size_weight': 0.3,
    'documentation_weight': 0.2,
    'features_weight': 0.3,
    'quality_weight': 0.2,
    'min_score': 5.0,
    'max_score': 100.0,
    'size_thresholds': {
        'small': 50,    # tokens
        'medium': 200,  # tokens
        'large': 500    # tokens
    },
    'feature_bonuses': {
        'has_guards': 5,
        'has_patterns': 3,
        'has_comprehensions': 4,
        'has_try_catch': 6,
        'is_exported': 2,
        'has_spec': 8,
        'multiple_clauses': 3
    }
}

# === GraphCodeBERT Configuration ===
GRAPHCODEBERT_CONFIG = {
    'max_code_length': 256,       # Maximum number of code tokens
    'max_dfg_length': 64,         # Maximum number of data flow edges
    'max_nl_length': 128,         # Maximum natural language description length
    'model_name': 'microsoft/graphcodebert-base',
    'tokenizer_name': 'microsoft/graphcodebert-base',
    'special_tokens': {
        'cls': '[CLS]',
        'sep': '[SEP]', 
        'pad': '[PAD]',
        'unk': '[UNK]',
        'mask': '[MASK]'
    },
    'data_splits': {
        'train_ratio': 0.8,
        'val_ratio': 0.1,
        'test_ratio': 0.1,
        'random_seed': 42
    }
}

# === Fine-tuning Configuration ===
FINETUNING_CONFIG = {
    'batch_size': 32,
    'learning_rate': 2e-5,
    'num_epochs': 10,
    'warmup_steps': 1000,
    'max_grad_norm': 1.0,
    'weight_decay': 0.01,
    'adam_epsilon': 1e-8,
    'save_steps': 1000,
    'eval_steps': 500,
    'logging_steps': 100,
    'save_total_limit': 3,
    'fp16': True,  # Mixed precision training
    'gradient_accumulation_steps': 1
}

# === LoRA Configuration ===
LORA_CONFIG = {
    'r': 16,                    # Rank of adaptation
    'alpha': 32,                # LoRA scaling parameter  
    'dropout': 0.1,             # Dropout probability
    'target_modules': [         # Which modules to apply LoRA to
        'query', 'key', 'value', 'dense'
    ],
    'bias': 'none',             # Whether to train bias parameters
    'task_type': 'FEATURE_EXTRACTION'
}

# === Evaluation Configuration ===
EVALUATION_CONFIG = {
    'metrics': ['bleu', 'rouge', 'codebleu', 'exact_match'],
    'eval_batch_size': 64,
    'beam_size': 5,
    'max_decode_length': 256,
    'save_predictions': True,
    'compute_similarity': True
}

# === Hardware Configuration ===
HARDWARE_CONFIG = {
    'use_gpu': True,
    'gpu_memory_limit': None,   # None for no limit
    'mixed_precision': True,
    'distributed_training': False,
    'num_workers': 4,           # For data loading
    'pin_memory': True
}

# === Logging Configuration ===
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file_logging': True,
    'log_file': 'erlang_corpus.log',
    'max_log_size': 10 * 1024 * 1024,  # 10MB
    'backup_count': 5
}

def get_output_path(filename: str) -> str:
    """Get full path for output file."""
    base_dir = Path(OUTPUT_CONFIG['base_dir'])
    base_dir.mkdir(exist_ok=True)
    return str(base_dir / filename)

def get_graphcodebert_output_path(filename: str) -> str:
    """Get full path for GraphCodeBERT output file."""
    base_dir = Path(OUTPUT_CONFIG['base_dir']) / OUTPUT_CONFIG['graphcodebert_dir']
    base_dir.mkdir(parents=True, exist_ok=True)
    return str(base_dir / filename)

def get_clone_path(repo_name: str) -> str:
    """Get full path for cloned repository."""
    clone_dir = CLONE_CONFIG['base_clone_dir']
    os.makedirs(clone_dir, exist_ok=True)
    
    # Replace problematic characters in repo name
    safe_name = repo_name.replace('/', '_').replace('\\', '_')
    return os.path.join(clone_dir, safe_name)

def validate_config():
    """Validate configuration settings."""
    errors = []
    
    # Validate split ratios
    splits = GRAPHCODEBERT_CONFIG['data_splits']
    total_ratio = splits['train_ratio'] + splits['val_ratio'] + splits['test_ratio']
    if abs(total_ratio - 1.0) > 0.001:
        errors.append(f"Data split ratios must sum to 1.0, got {total_ratio}")
    
    # Validate positive values
    if FINETUNING_CONFIG['learning_rate'] <= 0:
        errors.append("Learning rate must be positive")
    
    if GRAPHCODEBERT_CONFIG['max_code_length'] <= 0:
        errors.append("Max code length must be positive")
    
    # Validate LoRA configuration
    if LORA_CONFIG['r'] <= 0 or LORA_CONFIG['alpha'] <= 0:
        errors.append("LoRA rank and alpha must be positive")
    
    if errors:
        raise ValueError("Configuration validation failed:\n" + "\n".join(errors))
    
    return True

# Validate on import
validate_config()
